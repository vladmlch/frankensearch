use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::{
    DEFAULT_COLLECTION_INTERVAL_MS, DEFAULT_SEARCH_STREAM_CAPACITY, MIN_COLLECTION_INTERVAL_MS,
    TELEMETRY_SCHEMA_VERSION,
};
use crate::error::{SearchError, SearchResult};
use crate::query_class::QueryClass;

/// Runtime collector configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectorConfig {
    /// Telemetry collection interval in milliseconds.
    pub collection_interval_ms: u64,
}

impl Default for CollectorConfig {
    fn default() -> Self {
        Self {
            collection_interval_ms: DEFAULT_COLLECTION_INTERVAL_MS,
        }
    }
}

impl CollectorConfig {
    /// Validate collector settings.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when interval is below the
    /// minimum supported value.
    pub fn validate(&self) -> SearchResult<()> {
        if self.collection_interval_ms < MIN_COLLECTION_INTERVAL_MS {
            return Err(SearchError::InvalidConfig {
                field: "collector.collection_interval_ms".to_owned(),
                value: self.collection_interval_ms.to_string(),
                reason: format!(
                    "must be >= {MIN_COLLECTION_INTERVAL_MS}ms to bound collector overhead"
                ),
            });
        }
        Ok(())
    }

    /// Return the configured interval as [`Duration`].
    #[must_use]
    pub const fn interval(&self) -> Duration {
        Duration::from_millis(self.collection_interval_ms)
    }
}

/// Point-in-time collector counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectorSnapshot {
    /// Configured collection interval in milliseconds.
    pub collection_interval_ms: u64,
    /// Number of search events emitted.
    pub search_events_emitted: u64,
    /// Number of embedding events emitted.
    pub embedding_events_emitted: u64,
    /// Number of index events emitted.
    pub index_events_emitted: u64,
    /// Number of resource events emitted.
    pub resource_events_emitted: u64,
    /// Total events emitted across all families.
    pub total_events_emitted: u64,
}

/// Bounded-buffer policy for live search streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchStreamMode {
    /// Drop oldest queued event when buffer is full; never block producer.
    Lossy,
    /// Reject new event when buffer is full; caller must handle backpressure.
    NonLossy,
}

/// Configuration for the live search stream emitter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchStreamConfig {
    /// Maximum number of queued search events.
    pub capacity: usize,
    /// Backpressure/drop policy.
    pub mode: SearchStreamMode,
}

impl Default for SearchStreamConfig {
    fn default() -> Self {
        Self {
            capacity: DEFAULT_SEARCH_STREAM_CAPACITY,
            mode: SearchStreamMode::Lossy,
        }
    }
}

impl SearchStreamConfig {
    /// Validate stream configuration.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] if capacity is zero.
    pub fn validate(&self) -> SearchResult<()> {
        if self.capacity == 0 {
            return Err(SearchError::InvalidConfig {
                field: "search_stream.capacity".to_owned(),
                value: self.capacity.to_string(),
                reason: "must be >= 1 for bounded buffering".to_owned(),
            });
        }
        Ok(())
    }
}

/// Canonical telemetry instance identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetryInstance {
    pub instance_id: String,
    pub project_key: String,
    pub host_name: String,
    pub pid: Option<u32>,
}

/// Canonical telemetry correlation IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetryCorrelation {
    pub event_id: String,
    pub root_request_id: String,
    pub parent_event_id: Option<String>,
}

/// Telemetry envelope used by the control plane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetryEnvelope {
    pub v: u8,
    pub ts: String,
    pub event: TelemetryEvent,
}

impl TelemetryEnvelope {
    /// Build a v1 telemetry envelope.
    #[must_use]
    pub fn new(ts: impl Into<String>, event: TelemetryEvent) -> Self {
        Self {
            v: TELEMETRY_SCHEMA_VERSION,
            ts: ts.into(),
            event,
        }
    }
}

/// Canonical telemetry event families.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TelemetryEvent {
    Search {
        instance: TelemetryInstance,
        correlation: TelemetryCorrelation,
        query: TelemetrySearchQuery,
        results: TelemetrySearchResults,
        metrics: TelemetrySearchMetrics,
    },
    Embedding {
        instance: TelemetryInstance,
        correlation: TelemetryCorrelation,
        job: TelemetryEmbeddingJob,
        embedder: TelemetryEmbedderInfo,
        status: EmbeddingStatus,
        duration_ms: u64,
    },
    Index {
        instance: TelemetryInstance,
        correlation: TelemetryCorrelation,
        operation: IndexOperation,
        inventory: IndexInventory,
        dimension: usize,
        quantization: QuantizationMode,
        status: IndexStatus,
        duration_ms: u64,
    },
    Resource {
        instance: TelemetryInstance,
        correlation: TelemetryCorrelation,
        sample: TelemetryResourceSample,
    },
    Lifecycle {
        instance: TelemetryInstance,
        correlation: TelemetryCorrelation,
        state: LifecycleState,
        severity: LifecycleSeverity,
        reason: Option<String>,
        uptime_ms: Option<u64>,
    },
}

/// Search query sub-payload for `search` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetrySearchQuery {
    pub text: String,
    pub class: TelemetryQueryClass,
    pub phase: SearchEventPhase,
}

/// Search result-count sub-payload for `search` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetrySearchResults {
    pub result_count: usize,
    pub lexical_count: usize,
    pub semantic_count: usize,
    /// Hash/fnv/JL control-vector hits. Absent from older payloads.
    #[serde(default)]
    pub hash_control_count: usize,
}

/// Search metric sub-payload for `search` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetrySearchMetrics {
    pub latency_us: u64,
    pub memory_bytes: Option<u64>,
}

/// Embedding job sub-payload for `embedding` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetryEmbeddingJob {
    pub job_id: String,
    pub queue_depth: usize,
    pub doc_count: usize,
    pub stage: EmbeddingStage,
}

/// Embedder identity sub-payload for `embedding` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetryEmbedderInfo {
    pub id: String,
    pub tier: EmbedderTier,
    pub dimension: usize,
}

/// Index inventory sub-payload for `index` events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexInventory {
    pub words: u64,
    pub tokens: u64,
    pub lines: u64,
    pub bytes: u64,
    pub docs: u64,
}

/// Resource sample sub-payload for `resource` events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetryResourceSample {
    pub cpu_pct: f64,
    pub rss_bytes: u64,
    pub io_read_bytes: u64,
    pub io_write_bytes: u64,
    pub interval_ms: u64,
    pub load_avg_1m: Option<f64>,
    pub pressure_profile: Option<PressureProfile>,
}

/// Query class values normalized for telemetry contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TelemetryQueryClass {
    Empty,
    Identifier,
    ShortKeyword,
    NaturalLanguage,
}

impl From<QueryClass> for TelemetryQueryClass {
    fn from(value: QueryClass) -> Self {
        match value {
            QueryClass::Empty => Self::Empty,
            QueryClass::Identifier => Self::Identifier,
            QueryClass::ShortKeyword => Self::ShortKeyword,
            QueryClass::NaturalLanguage => Self::NaturalLanguage,
        }
    }
}

/// Search phase values for telemetry contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchEventPhase {
    Initial,
    Refined,
    Reranked,
    RefinementFailed,
}

/// Embedding job stage values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingStage {
    Fast,
    Quality,
    Background,
}

/// Embedding tier values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbedderTier {
    Hash,
    Fast,
    Quality,
}

/// Embedding lifecycle status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Index operation kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexOperation {
    Build,
    Rebuild,
    Append,
    Compact,
    Repair,
    Snapshot,
}

/// Quantization modes used by index telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationMode {
    F32,
    F16,
    Int8,
    Pq,
}

/// Index operation status values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexStatus {
    Started,
    Completed,
    Failed,
}

/// Resource pressure profile values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PressureProfile {
    Strict,
    Performance,
    Degraded,
}

/// Lifecycle event state values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleState {
    Started,
    Stopped,
    Healthy,
    Degraded,
    Stale,
    Recovering,
}

/// Lifecycle severity values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LifecycleSeverity {
    Info,
    Warn,
    Error,
}

/// Input sample for `search` event collection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchCollectorSample {
    pub query_text: String,
    pub query_class: QueryClass,
    pub phase: SearchEventPhase,
    pub result_count: usize,
    pub lexical_count: usize,
    pub semantic_count: usize,
    /// Hash/fnv/JL control-vector hits. Absent from older payloads.
    #[serde(default)]
    pub hash_control_count: usize,
    pub latency_us: u64,
    pub memory_bytes: Option<u64>,
}

/// Input sample for `embedding` event collection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingCollectorSample {
    pub job_id: String,
    pub queue_depth: usize,
    pub doc_count: usize,
    pub stage: EmbeddingStage,
    pub embedder_id: String,
    pub tier: EmbedderTier,
    pub dimension: usize,
    pub status: EmbeddingStatus,
    pub duration_ms: u64,
}

/// Input sample for `index` event collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexCollectorSample {
    pub operation: IndexOperation,
    pub inventory: IndexInventory,
    pub dimension: usize,
    pub quantization: QuantizationMode,
    pub status: IndexStatus,
    pub duration_ms: u64,
}

/// Input sample for `resource` event collection.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResourceCollectorSample {
    pub cpu_pct: f64,
    pub rss_bytes: u64,
    pub io_read_bytes: u64,
    pub io_write_bytes: u64,
    pub interval_ms: u64,
    pub load_avg_1m: Option<f64>,
    pub pressure_profile: Option<PressureProfile>,
}
