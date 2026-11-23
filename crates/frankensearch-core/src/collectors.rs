//! Core telemetry collectors and canonical event payloads.
//!
//! This module provides the runtime-safe collector layer used by control-plane
//! integrations. It is intentionally lightweight:
//!
//! - default sampling cadence is 1000ms (from the ops config contract)
//! - minimum allowed cadence is 100ms (guardrail against accidental hot polling)
//! - per-event overhead is O(1): one atomic increment plus payload assembly
//!
//! Collectors emit canonical `search`, `embedding`, `index`, and `resource`
//! payloads aligned with `schemas/telemetry-event-v1.schema.json`.

mod collector;
mod stream;
mod types;

pub use collector::RuntimeMetricsCollector;
pub use stream::{
    LiveSearchFrame, LiveSearchStreamEmitter, SearchStreamHealth, SearchStreamPublishOutcome,
};
pub use types::{
    CollectorConfig, CollectorSnapshot, EmbedderTier, EmbeddingCollectorSample, EmbeddingStage,
    EmbeddingStatus, IndexCollectorSample, IndexInventory, IndexOperation, IndexStatus,
    LifecycleSeverity, LifecycleState, PressureProfile, QuantizationMode, ResourceCollectorSample,
    SearchCollectorSample, SearchEventPhase, SearchStreamConfig, SearchStreamMode,
    TelemetryCorrelation, TelemetryEmbedderInfo, TelemetryEmbeddingJob, TelemetryEnvelope,
    TelemetryEvent, TelemetryInstance, TelemetryQueryClass, TelemetryResourceSample,
    TelemetrySearchMetrics, TelemetrySearchQuery, TelemetrySearchResults,
};

/// Canonical telemetry envelope schema version.
pub const TELEMETRY_SCHEMA_VERSION: u8 = 1;

/// Default collector interval (1s) from `docs/ops-config-contract.md`.
pub const DEFAULT_COLLECTION_INTERVAL_MS: u64 = 1_000;

/// Minimum allowed interval to bound overhead.
pub const MIN_COLLECTION_INTERVAL_MS: u64 = 100;

/// Default in-memory stream buffer size for live search events.
pub const DEFAULT_SEARCH_STREAM_CAPACITY: usize = 1_024;

#[cfg(test)]
use collector::{
    EMPTY_QUERY_PLACEHOLDER, MAX_QUERY_TEXT_CHARS, normalize_non_negative_f64, sanitize_cpu_pct,
    sanitize_query_text,
};

#[cfg(test)]
mod tests;
