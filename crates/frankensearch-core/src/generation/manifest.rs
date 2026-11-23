use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::SearchError;

use super::CommitRange;
use super::codec::validate_identity_text;
use super::embedding_identity::EmbedderRevision;

// ---------------------------------------------------------------------------
// Artifact descriptors
// ---------------------------------------------------------------------------

/// Descriptor for a single FSVI vector index shard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorArtifact {
    /// Relative path within the generation directory (e.g. `"vectors/shard_0.fsvi"`).
    pub path: String,
    /// Byte size of the artifact file.
    pub size_bytes: u64,
    /// Hex-encoded checksum (SHA-256) of the file contents.
    pub checksum: String,
    /// Number of vectors stored in this shard.
    pub vector_count: u64,
    /// Vector dimensionality.
    pub dimension: u32,
    /// Which embedder tier produced these vectors.
    pub embedder_tier: EmbedderTierTag,
}

/// Tag identifying which tier of the two-tier system produced an artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbedderTierTag {
    /// Fast tier (e.g. potion-128M, ~0.57ms).
    Fast,
    /// Quality tier (e.g. MiniLM-L6-v2, ~128ms).
    Quality,
}

/// Descriptor for a Tantivy lexical index segment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexicalArtifact {
    /// Relative path within the generation directory (e.g. `"lexical/segment_0"`).
    pub path: String,
    /// Byte size of all files in the segment directory.
    pub size_bytes: u64,
    /// Hex-encoded checksum (SHA-256) of the concatenated segment files.
    pub checksum: String,
    /// Number of documents indexed in this segment.
    pub document_count: u64,
}

/// Metadata for `RaptorQ` repair symbols protecting an artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepairDescriptor {
    /// Path of the protected artifact (matches a `VectorArtifact.path` or `LexicalArtifact.path`).
    pub protected_artifact: String,
    /// Path to the `.fec` sidecar file containing repair symbols.
    pub sidecar_path: String,
    /// Number of source symbols the artifact was split into.
    pub source_symbols: u32,
    /// Number of repair symbols generated.
    pub repair_symbols: u32,
    /// Overhead ratio (`repair_symbols` / `source_symbols`).
    pub overhead_ratio: f64,
}

// ---------------------------------------------------------------------------
// Activation invariants
// ---------------------------------------------------------------------------

/// A predicate that must hold before a generation can be activated for serving.
///
/// Activation invariants enforce all-or-nothing readiness: every invariant must
/// pass, or the generation is rejected and the previous generation continues serving.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivationInvariant {
    /// Machine-readable invariant identifier.
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// The kind of check this invariant represents.
    pub kind: InvariantKind,
}

/// Classification of activation invariant checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvariantKind {
    /// All listed artifacts must be present and pass checksum verification.
    AllArtifactsVerified,
    /// Embedder revision must match the node's runtime embedder.
    EmbedderRevisionMatch,
    /// Total vector count must match document count (no missing embeddings).
    VectorCountConsistency {
        /// Expected total vectors across all shards.
        expected_total: u64,
    },
    /// Generation must cover a commit range that is contiguous with the previous
    /// activated generation (no gaps in commit history).
    CommitContinuity {
        /// The `high` value of the previous generation's commit range.
        previous_high: u64,
    },
    /// Custom predicate supplied by the deployment.
    Custom {
        /// Name of the custom check.
        check_name: String,
    },
}

// ---------------------------------------------------------------------------
// Generation manifest
// ---------------------------------------------------------------------------

/// Complete manifest for a search generation.
///
/// This is the unit of replication and activation in Native Mode distributed search.
/// A node fetches the manifest, verifies all artifacts, checks invariants, and
/// atomically swaps the active generation pointer on success.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationManifest {
    /// Exact schema version for fail-closed parsing.
    pub schema_version: u32,
    /// Unique identifier for this generation (content-derived or monotonic).
    pub generation_id: String,
    /// Hex-encoded hash (SHA-256) of the canonical serialized manifest body,
    /// computed with this field set to the empty string.
    pub manifest_hash: String,
    /// Commit range that produced this generation.
    pub commit_range: CommitRange,
    /// Timestamp (Unix millis) when generation build started.
    pub build_started_at: u64,
    /// Timestamp (Unix millis) when generation build completed.
    pub build_completed_at: u64,
    /// Embedder revisions used (keyed by tier tag stringified).
    pub embedders: BTreeMap<String, EmbedderRevision>,
    /// Vector index artifacts in this generation.
    pub vector_artifacts: Vec<VectorArtifact>,
    /// Lexical index artifacts in this generation.
    pub lexical_artifacts: Vec<LexicalArtifact>,
    /// Repair symbol descriptors for durability.
    pub repair_descriptors: Vec<RepairDescriptor>,
    /// Activation invariants that must all pass before serving.
    pub activation_invariants: Vec<ActivationInvariant>,
    /// Total document count across all artifacts.
    pub total_documents: u64,
    /// Optional free-form metadata (deployment tags, build host, etc.).
    pub metadata: BTreeMap<String, String>,
}

/// Current schema version for [`GenerationManifest`].
///
/// Version 2 replaces the ambiguous display-name/weights/dimension embedder
/// descriptor with the complete frozen embedding-identity bundle.
pub const MANIFEST_SCHEMA_VERSION: u32 = 2;

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Result of validating a [`GenerationManifest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationResult {
    /// Individual validation findings (empty means valid).
    pub findings: Vec<ValidationFinding>,
}

impl ValidationResult {
    /// Whether the manifest passes all validation checks.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.findings
            .iter()
            .all(|f| f.severity != FindingSeverity::Error)
    }

    /// Collect only error-severity findings.
    #[must_use]
    pub fn errors(&self) -> Vec<&ValidationFinding> {
        self.findings
            .iter()
            .filter(|f| f.severity == FindingSeverity::Error)
            .collect()
    }

    /// Collect only warning-severity findings.
    #[must_use]
    pub fn warnings(&self) -> Vec<&ValidationFinding> {
        self.findings
            .iter()
            .filter(|f| f.severity == FindingSeverity::Warning)
            .collect()
    }
}

/// A single validation finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationFinding {
    /// Which check produced this finding.
    pub check: &'static str,
    /// Severity of the finding.
    pub severity: FindingSeverity,
    /// Human-readable description.
    pub message: String,
}

/// Severity of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FindingSeverity {
    /// Informational, does not block activation.
    Info,
    /// Suspicious but not blocking.
    Warning,
    /// Blocks activation.
    Error,
}

/// Validates a [`GenerationManifest`] for structural and semantic correctness.
///
/// Returns a [`ValidationResult`] with all findings. Call [`ValidationResult::is_valid`]
/// to check whether the manifest is safe to activate.
#[must_use]
pub fn validate_manifest(manifest: &GenerationManifest) -> ValidationResult {
    let mut findings = Vec::new();

    check_schema_version(manifest, &mut findings);
    check_generation_id(manifest, &mut findings);
    check_manifest_hash(manifest, &mut findings);
    check_commit_range(manifest, &mut findings);
    check_timestamps(manifest, &mut findings);
    check_embedders(manifest, &mut findings);
    check_vector_artifacts(manifest, &mut findings);
    check_lexical_artifacts(manifest, &mut findings);
    check_repair_descriptors(manifest, &mut findings);
    check_activation_invariants(manifest, &mut findings);
    check_document_count_consistency(manifest, &mut findings);

    ValidationResult { findings }
}

/// Computes the canonical manifest hash for a generation manifest.
///
/// The canonical hash is SHA-256 over JSON serialization of the manifest with
/// `manifest_hash` cleared.
///
/// # Errors
///
/// Returns `SearchError::SubsystemError` if serialization fails.
pub fn compute_manifest_hash(manifest: &GenerationManifest) -> crate::SearchResult<String> {
    let mut canonical = manifest.clone();
    canonical.manifest_hash.clear();
    let serialized =
        serde_json::to_vec(&canonical).map_err(|source| SearchError::SubsystemError {
            subsystem: "generation_manifest",
            source: Box::new(source),
        })?;
    Ok(lower_hex(Sha256::digest(serialized)))
}

fn lower_hex(bytes: impl AsRef<[u8]>) -> String {
    let bytes = bytes.as_ref();
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(&mut hex, "{byte:02x}");
    }
    hex
}

/// Convert a validation result into a `SearchResult`, producing an error
/// if any error-severity findings exist.
///
/// # Errors
///
/// Returns `SearchError::InvalidConfig` when validation fails.
pub fn require_valid(result: &ValidationResult) -> crate::SearchResult<()> {
    if result.is_valid() {
        return Ok(());
    }
    let messages: Vec<String> = result.errors().iter().map(|f| f.message.clone()).collect();
    Err(SearchError::InvalidConfig {
        field: "generation_manifest".into(),
        value: String::new(),
        reason: messages.join("; "),
    })
}

// ---------------------------------------------------------------------------
// Individual checks
// ---------------------------------------------------------------------------

fn check_schema_version(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    if m.schema_version != MANIFEST_SCHEMA_VERSION {
        f.push(ValidationFinding {
            check: "schema_version",
            severity: FindingSeverity::Error,
            message: format!(
                "schema_version {} is unsupported; expected exactly {}",
                m.schema_version, MANIFEST_SCHEMA_VERSION
            ),
        });
    }
}

fn check_generation_id(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    if m.generation_id.is_empty() {
        f.push(ValidationFinding {
            check: "generation_id",
            severity: FindingSeverity::Error,
            message: "generation_id must not be empty".into(),
        });
    }
}

fn check_manifest_hash(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    if m.manifest_hash.is_empty() {
        f.push(ValidationFinding {
            check: "manifest_hash",
            severity: FindingSeverity::Error,
            message: "manifest_hash must not be empty".into(),
        });
        return;
    }
    if !is_valid_sha256_hex(&m.manifest_hash) {
        f.push(ValidationFinding {
            check: "manifest_hash",
            severity: FindingSeverity::Error,
            message: "manifest_hash must be 64 lowercase/uppercase hex chars".into(),
        });
        return;
    }

    match compute_manifest_hash(m) {
        Ok(expected) => {
            if !m.manifest_hash.eq_ignore_ascii_case(&expected) {
                f.push(ValidationFinding {
                    check: "manifest_hash",
                    severity: FindingSeverity::Error,
                    message: format!(
                        "manifest_hash does not match canonical manifest body (expected {expected})"
                    ),
                });
            }
        }
        Err(err) => {
            f.push(ValidationFinding {
                check: "manifest_hash",
                severity: FindingSeverity::Error,
                message: format!("failed to recompute manifest_hash: {err}"),
            });
        }
    }
}

fn is_valid_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.chars().all(|c| c.is_ascii_hexdigit())
}

fn check_commit_range(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    if m.commit_range.is_empty() {
        f.push(ValidationFinding {
            check: "commit_range",
            severity: FindingSeverity::Error,
            message: format!(
                "commit_range is invalid: high ({}) < low ({})",
                m.commit_range.high, m.commit_range.low
            ),
        });
    }
}

fn check_timestamps(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    if m.build_started_at == 0 {
        f.push(ValidationFinding {
            check: "build_started_at",
            severity: FindingSeverity::Error,
            message: "build_started_at must be a positive Unix timestamp".into(),
        });
    }
    if m.build_completed_at == 0 {
        f.push(ValidationFinding {
            check: "build_completed_at",
            severity: FindingSeverity::Error,
            message: "build_completed_at must be a positive Unix timestamp".into(),
        });
    }
    if m.build_completed_at < m.build_started_at {
        f.push(ValidationFinding {
            check: "build_timestamps",
            severity: FindingSeverity::Error,
            message: format!(
                "build_completed_at ({}) is before build_started_at ({})",
                m.build_completed_at, m.build_started_at
            ),
        });
    }
}

fn check_embedders(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    if m.embedders.is_empty() {
        f.push(ValidationFinding {
            check: "embedders",
            severity: FindingSeverity::Error,
            message: "at least one embedder revision must be specified".into(),
        });
    }
    for (key, rev) in &m.embedders {
        if let Err(error) = validate_identity_text("embedder_tier", key) {
            f.push(ValidationFinding {
                check: "embedder_tier",
                severity: FindingSeverity::Error,
                message: format!("embedder tier label is invalid: {error}"),
            });
            continue;
        }
        if let Err(error) = rev.validate() {
            f.push(ValidationFinding {
                check: "embedder_identity",
                severity: FindingSeverity::Error,
                message: format!("embedder '{key}' has invalid identity: {error}"),
            });
        }
    }
}

fn check_vector_artifacts(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    for (i, art) in m.vector_artifacts.iter().enumerate() {
        if art.path.is_empty() {
            f.push(ValidationFinding {
                check: "vector_artifact_path",
                severity: FindingSeverity::Error,
                message: format!("vector_artifacts[{i}] has empty path"),
            });
        }
        if art.checksum.is_empty() {
            f.push(ValidationFinding {
                check: "vector_artifact_checksum",
                severity: FindingSeverity::Error,
                message: format!("vector_artifacts[{i}] '{}' has empty checksum", art.path),
            });
        }
        if art.dimension == 0 {
            f.push(ValidationFinding {
                check: "vector_artifact_dimension",
                severity: FindingSeverity::Error,
                message: format!("vector_artifacts[{i}] '{}' has dimension 0", art.path),
            });
        }
    }

    // Check for duplicate paths.
    let mut seen = std::collections::HashSet::new();
    for art in &m.vector_artifacts {
        if !seen.insert(&art.path) {
            f.push(ValidationFinding {
                check: "vector_artifact_duplicate",
                severity: FindingSeverity::Error,
                message: format!("duplicate vector artifact path: '{}'", art.path),
            });
        }
    }
}

fn check_lexical_artifacts(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    for (i, art) in m.lexical_artifacts.iter().enumerate() {
        if art.path.is_empty() {
            f.push(ValidationFinding {
                check: "lexical_artifact_path",
                severity: FindingSeverity::Error,
                message: format!("lexical_artifacts[{i}] has empty path"),
            });
        }
        if art.checksum.is_empty() {
            f.push(ValidationFinding {
                check: "lexical_artifact_checksum",
                severity: FindingSeverity::Error,
                message: format!("lexical_artifacts[{i}] '{}' has empty checksum", art.path),
            });
        }
    }

    let mut seen = std::collections::HashSet::new();
    for art in &m.lexical_artifacts {
        if !seen.insert(&art.path) {
            f.push(ValidationFinding {
                check: "lexical_artifact_duplicate",
                severity: FindingSeverity::Error,
                message: format!("duplicate lexical artifact path: '{}'", art.path),
            });
        }
    }
}

fn check_repair_descriptors(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    let all_artifact_paths: std::collections::HashSet<&str> = m
        .vector_artifacts
        .iter()
        .map(|a| a.path.as_str())
        .chain(m.lexical_artifacts.iter().map(|a| a.path.as_str()))
        .collect();

    for (i, rd) in m.repair_descriptors.iter().enumerate() {
        if !all_artifact_paths.contains(rd.protected_artifact.as_str()) {
            f.push(ValidationFinding {
                check: "repair_descriptor_target",
                severity: FindingSeverity::Error,
                message: format!(
                    "repair_descriptors[{i}] references unknown artifact '{}'",
                    rd.protected_artifact
                ),
            });
        }
        if rd.source_symbols == 0 {
            f.push(ValidationFinding {
                check: "repair_descriptor_symbols",
                severity: FindingSeverity::Error,
                message: format!(
                    "repair_descriptors[{i}] for '{}' has 0 source symbols",
                    rd.protected_artifact
                ),
            });
        }
        if rd.overhead_ratio.is_nan() || rd.overhead_ratio < 0.0 || rd.overhead_ratio > 10.0 {
            f.push(ValidationFinding {
                check: "repair_descriptor_overhead",
                severity: FindingSeverity::Warning,
                message: format!(
                    "repair_descriptors[{i}] overhead ratio {} is outside expected range [0, 10]",
                    rd.overhead_ratio
                ),
            });
        }
    }
}

fn check_activation_invariants(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    let mut seen_ids = std::collections::HashSet::new();
    for inv in &m.activation_invariants {
        if inv.id.is_empty() {
            f.push(ValidationFinding {
                check: "invariant_id",
                severity: FindingSeverity::Error,
                message: "activation invariant has empty id".into(),
            });
        }
        if !seen_ids.insert(&inv.id) {
            f.push(ValidationFinding {
                check: "invariant_duplicate",
                severity: FindingSeverity::Error,
                message: format!("duplicate activation invariant id: '{}'", inv.id),
            });
        }
    }
}

fn check_document_count_consistency(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    let vector_total: u64 = m.vector_artifacts.iter().map(|a| a.vector_count).sum();
    let lexical_total: u64 = m.lexical_artifacts.iter().map(|a| a.document_count).sum();

    // Vector count should match declared total (per tier, so may be 2x for two-tier).
    // We only warn if there's a gross mismatch.
    if m.total_documents == 0 && (!m.vector_artifacts.is_empty() || !m.lexical_artifacts.is_empty())
    {
        f.push(ValidationFinding {
            check: "total_documents",
            severity: FindingSeverity::Error,
            message: "total_documents is 0 but artifacts are present".into(),
        });
    }

    if !m.lexical_artifacts.is_empty() && lexical_total != m.total_documents {
        f.push(ValidationFinding {
            check: "lexical_document_count",
            severity: FindingSeverity::Warning,
            message: format!(
                "lexical document count ({lexical_total}) != total_documents ({})",
                m.total_documents
            ),
        });
    }

    // For two-tier, vector_total may be 2 * total_documents (fast + quality tier).
    // Flag only if it doesn't match any reasonable multiple.
    if !m.vector_artifacts.is_empty()
        && vector_total != m.total_documents
        && vector_total != m.total_documents * 2
    {
        f.push(ValidationFinding {
            check: "vector_count_consistency",
            severity: FindingSeverity::Warning,
            message: format!(
                "vector count ({vector_total}) doesn't match total_documents ({}) or 2x (two-tier)",
                m.total_documents
            ),
        });
    }
}
