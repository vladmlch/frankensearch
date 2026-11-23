//! Non-interfering lexical shadow-oracle support.
//!
//! [`ShadowLexical`] wraps the split lexical seam — [`crate::traits::LexicalRead`]
//! for queries and [`crate::traits::LexicalWrite`] for mutations. The serving
//! backend is always awaited and returned unchanged. Eligible shadow work is
//! detached into the caller's asupersync region behind a bounded admission
//! gate, so sampling, pressure shedding, a full gate, spawn failure, shadow
//! failure, or artifact failure cannot change the serving result.

mod artifacts;
mod runtime;
mod types;
mod wrapper;

pub use artifacts::{ShadowArtifactSummary, append_shadow_degradation};
pub use types::{
    AlwaysAdmitShadowLoad, AtomicShadowLoadProbe, ShadowDegradationKind, ShadowDegradationRecord,
    ShadowDivergenceClass, ShadowDivergenceRecord, ShadowDocument, ShadowLexicalConfig,
    ShadowLoadProbe, ShadowObservationRecord, ShadowQuery, ShadowRankedHit, ShadowShedReason,
    ShadowStatus, compute_shadow_corpus_hash,
};
pub use wrapper::{ShadowLexical, ShadowLexicalObserver};

/// Version of one production divergence JSONL record.
pub const SHADOW_DIVERGENCE_SCHEMA_VERSION: u32 = 2;
/// Version of one typed shadow degradation JSONL record.
pub const SHADOW_DEGRADATION_SCHEMA_VERSION: u32 = 1;
/// Version of one completed shadow observation JSONL record.
pub const SHADOW_OBSERVATION_SCHEMA_VERSION: u32 = 1;
/// Fixed-point sampling denominator. One basis point is 1/10,000.
pub const SHADOW_SAMPLE_DENOMINATOR: u64 = 10_000;
/// Directory beneath an index root that owns shadow evidence.
pub const SHADOW_ARTIFACT_DIRECTORY: &str = ".quill-shadow";
/// Append-only classified divergence stream.
pub const SHADOW_DIVERGENCES_FILE: &str = "divergences.jsonl";
/// Append-only typed degradation stream.
pub const SHADOW_DEGRADATIONS_FILE: &str = "degradations.jsonl";
/// Append-only completed-comparison evidence stream.
pub const SHADOW_OBSERVATIONS_FILE: &str = "observations.jsonl";

#[cfg(test)]
mod tests;
