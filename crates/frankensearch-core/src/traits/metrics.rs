use std::fmt;
use std::sync::Arc;

use crate::error::SearchError;
use crate::types::{EmbeddingMetrics, IndexMetrics, SearchMetrics};

// ─── Metrics Exporter Trait ─────────────────────────────────────────────────

/// Trait for exporting search/index/embed telemetry to external consumers.
///
/// Implementations must be non-blocking and fast, because callbacks are invoked
/// directly from hot paths.
pub trait MetricsExporter: fmt::Debug + Send + Sync {
    /// Called when a search request completes.
    fn on_search_completed(&self, metrics: &SearchMetrics);

    /// Called when an embedding operation completes.
    fn on_embedding_completed(&self, metrics: &EmbeddingMetrics);

    /// Called when index state changes after an update/commit.
    fn on_index_updated(&self, metrics: &IndexMetrics);

    /// Called when a search pipeline error is observed.
    fn on_error(&self, error: &SearchError);
}

/// Shared handle for dynamic telemetry exporters.
pub type SharedMetricsExporter = Arc<dyn MetricsExporter>;

/// No-op exporter used when no telemetry sink is attached.
///
/// This is intentionally empty so callers can cheaply opt out of telemetry.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoOpMetricsExporter;

impl MetricsExporter for NoOpMetricsExporter {
    fn on_search_completed(&self, _: &SearchMetrics) {}

    fn on_embedding_completed(&self, _: &EmbeddingMetrics) {}

    fn on_index_updated(&self, _: &IndexMetrics) {}

    fn on_error(&self, _: &SearchError) {}
}
