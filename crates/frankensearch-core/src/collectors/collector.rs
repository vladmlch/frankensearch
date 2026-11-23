use std::sync::atomic::{AtomicU64, Ordering};

use super::types::{
    CollectorConfig, CollectorSnapshot, EmbeddingCollectorSample, IndexCollectorSample,
    ResourceCollectorSample, SearchCollectorSample, TelemetryCorrelation, TelemetryEmbedderInfo,
    TelemetryEmbeddingJob, TelemetryEnvelope, TelemetryEvent, TelemetryInstance,
    TelemetryQueryClass, TelemetryResourceSample, TelemetrySearchMetrics, TelemetrySearchQuery,
    TelemetrySearchResults,
};
use crate::error::SearchResult;

pub(super) const EMPTY_QUERY_PLACEHOLDER: &str = "<empty>";
pub(super) const MAX_QUERY_TEXT_CHARS: usize = 500;

/// O(1) runtime collector for canonical telemetry payloads.
#[derive(Debug)]
pub struct RuntimeMetricsCollector {
    config: CollectorConfig,
    search_events: AtomicU64,
    embedding_events: AtomicU64,
    index_events: AtomicU64,
    resource_events: AtomicU64,
}

impl Default for RuntimeMetricsCollector {
    fn default() -> Self {
        Self {
            config: CollectorConfig::default(),
            search_events: AtomicU64::new(0),
            embedding_events: AtomicU64::new(0),
            index_events: AtomicU64::new(0),
            resource_events: AtomicU64::new(0),
        }
    }
}

impl RuntimeMetricsCollector {
    /// Create a collector with explicit configuration.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::SearchError::InvalidConfig`] when `config` is invalid.
    pub fn new(config: CollectorConfig) -> SearchResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            ..Self::default()
        })
    }

    /// Read-only access to collector configuration.
    #[must_use]
    pub const fn config(&self) -> CollectorConfig {
        self.config
    }

    /// Emit a canonical `search` event envelope.
    #[must_use]
    pub fn emit_search(
        &self,
        ts: impl Into<String>,
        instance: TelemetryInstance,
        correlation: TelemetryCorrelation,
        sample: SearchCollectorSample,
    ) -> TelemetryEnvelope {
        self.search_events.fetch_add(1, Ordering::Relaxed);
        let SearchCollectorSample {
            query_text,
            query_class,
            phase,
            result_count,
            lexical_count,
            semantic_count,
            hash_control_count,
            latency_us,
            memory_bytes,
        } = sample;
        let query_text = sanitize_query_text(query_text);

        TelemetryEnvelope::new(
            ts,
            TelemetryEvent::Search {
                instance,
                correlation,
                query: TelemetrySearchQuery {
                    text: query_text,
                    class: TelemetryQueryClass::from(query_class),
                    phase,
                },
                results: TelemetrySearchResults {
                    result_count,
                    lexical_count,
                    semantic_count,
                    hash_control_count,
                },
                metrics: TelemetrySearchMetrics {
                    latency_us,
                    memory_bytes,
                },
            },
        )
    }

    /// Emit a canonical `embedding` event envelope.
    #[must_use]
    pub fn emit_embedding(
        &self,
        ts: impl Into<String>,
        instance: TelemetryInstance,
        correlation: TelemetryCorrelation,
        sample: EmbeddingCollectorSample,
    ) -> TelemetryEnvelope {
        self.embedding_events.fetch_add(1, Ordering::Relaxed);

        TelemetryEnvelope::new(
            ts,
            TelemetryEvent::Embedding {
                instance,
                correlation,
                job: TelemetryEmbeddingJob {
                    job_id: sample.job_id,
                    queue_depth: sample.queue_depth,
                    doc_count: sample.doc_count,
                    stage: sample.stage,
                },
                embedder: TelemetryEmbedderInfo {
                    id: sample.embedder_id,
                    tier: sample.tier,
                    dimension: sample.dimension,
                },
                status: sample.status,
                duration_ms: sample.duration_ms,
            },
        )
    }

    /// Emit a canonical `index` event envelope.
    #[must_use]
    pub fn emit_index(
        &self,
        ts: impl Into<String>,
        instance: TelemetryInstance,
        correlation: TelemetryCorrelation,
        sample: IndexCollectorSample,
    ) -> TelemetryEnvelope {
        self.index_events.fetch_add(1, Ordering::Relaxed);

        TelemetryEnvelope::new(
            ts,
            TelemetryEvent::Index {
                instance,
                correlation,
                operation: sample.operation,
                inventory: sample.inventory,
                dimension: sample.dimension,
                quantization: sample.quantization,
                status: sample.status,
                duration_ms: sample.duration_ms,
            },
        )
    }

    /// Emit a canonical `resource` event envelope.
    #[must_use]
    pub fn emit_resource(
        &self,
        ts: impl Into<String>,
        instance: TelemetryInstance,
        correlation: TelemetryCorrelation,
        sample: ResourceCollectorSample,
    ) -> TelemetryEnvelope {
        self.resource_events.fetch_add(1, Ordering::Relaxed);
        let cpu_pct = sanitize_cpu_pct(sample.cpu_pct);
        let interval_ms = sample.interval_ms.max(1);
        let load_avg_1m = sample.load_avg_1m.and_then(normalize_non_negative_f64);

        TelemetryEnvelope::new(
            ts,
            TelemetryEvent::Resource {
                instance,
                correlation,
                sample: TelemetryResourceSample {
                    cpu_pct,
                    rss_bytes: sample.rss_bytes,
                    io_read_bytes: sample.io_read_bytes,
                    io_write_bytes: sample.io_write_bytes,
                    interval_ms,
                    load_avg_1m,
                    pressure_profile: sample.pressure_profile,
                },
            },
        )
    }

    /// Snapshot collector counters.
    #[must_use]
    pub fn snapshot(&self) -> CollectorSnapshot {
        let search_events = self.search_events.load(Ordering::Relaxed);
        let embedding_events = self.embedding_events.load(Ordering::Relaxed);
        let index_events = self.index_events.load(Ordering::Relaxed);
        let resource_events = self.resource_events.load(Ordering::Relaxed);

        CollectorSnapshot {
            collection_interval_ms: self.config.collection_interval_ms,
            search_events_emitted: search_events,
            embedding_events_emitted: embedding_events,
            index_events_emitted: index_events,
            resource_events_emitted: resource_events,
            total_events_emitted: search_events
                .saturating_add(embedding_events)
                .saturating_add(index_events)
                .saturating_add(resource_events),
        }
    }
}

pub(super) fn sanitize_query_text(mut query_text: String) -> String {
    let leading_trimmed = query_text.trim_start();
    let start = query_text.len() - leading_trimmed.len();
    let trimmed_len = leading_trimmed.trim_end().len();
    if trimmed_len == 0 {
        query_text.clear();
        query_text.push_str(EMPTY_QUERY_PLACEHOLDER);
        return query_text;
    }
    if start == 0 && trimmed_len == query_text.len() && query_text.len() <= MAX_QUERY_TEXT_CHARS {
        return query_text;
    }

    let retained_len = query_text[start..start + trimmed_len]
        .char_indices()
        .nth(MAX_QUERY_TEXT_CHARS)
        .map_or(trimmed_len, |(offset, _)| offset);
    query_text.truncate(start + retained_len);
    if start != 0 {
        drop(query_text.drain(..start));
    }
    query_text
}

pub(super) const fn sanitize_cpu_pct(cpu_pct: f64) -> f64 {
    if !cpu_pct.is_finite() {
        return 0.0;
    }
    cpu_pct.clamp(0.0, 100.0)
}

pub(super) fn normalize_non_negative_f64(value: f64) -> Option<f64> {
    if value.is_finite() && value >= 0.0 {
        Some(value)
    } else {
        None
    }
}
