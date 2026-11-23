use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use frankensearch_core::config::TwoTierMetrics;
use frankensearch_core::error::SearchError;
use frankensearch_core::host_adapter::{AdapterLifecycleEvent, HostAdapter};
use frankensearch_core::query_class::QueryClass;
use frankensearch_core::traits::{Embedder, ModelCategory};
use frankensearch_core::types::{EmbeddingMetrics, SearchMetrics, SearchMode};
use frankensearch_core::{
    EmbedderTier, EmbeddingCollectorSample, EmbeddingStage, EmbeddingStatus, LifecycleSeverity,
    LifecycleState, ResourceCollectorSample, SearchCollectorSample, SearchEventPhase,
    TelemetryCorrelation, TelemetryEnvelope, TelemetryEvent, TelemetryInstance,
};

use super::TwoTierSearcher;

static TELEMETRY_EVENT_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy)]
pub(super) struct CpuJiffiesSnapshot {
    pub(super) process_jiffies: u64,
    pub(super) total_jiffies: u64,
}

impl TwoTierSearcher {
    pub(super) fn export_search_metrics(
        &self,
        query_class: QueryClass,
        metrics: &TwoTierMetrics,
        result_count: usize,
        refined: bool,
    ) {
        let Some(exporter) = self.config.metrics_exporter.as_ref() else {
            return;
        };
        let payload = SearchMetrics {
            mode: SearchMode::TwoTier,
            query_class: Some(query_class),
            total_latency_ms: metrics.phase1_total_ms
                + if refined {
                    metrics.phase2_total_ms
                } else {
                    0.0
                },
            phase1_latency_ms: Some(metrics.phase1_total_ms),
            phase2_latency_ms: if refined {
                Some(metrics.phase2_total_ms)
            } else {
                None
            },
            result_count,
            lexical_candidates: metrics.lexical_candidates,
            semantic_candidates: metrics.semantic_candidates,
            hash_control_candidates: metrics.hash_control_candidates,
            refined,
        };
        exporter.on_search_completed(&payload);
    }

    pub(super) fn export_embedding_metrics(
        &self,
        embedder: &dyn Embedder,
        batch_size: usize,
        duration_ms: f64,
    ) {
        let Some(exporter) = self.config.metrics_exporter.as_ref() else {
            return;
        };
        let payload = EmbeddingMetrics {
            embedder_id: embedder.id().to_owned(),
            batch_size,
            duration_ms,
            dimension: embedder.dimension(),
            is_semantic: embedder.is_semantic(),
        };
        exporter.on_embedding_completed(&payload);
    }

    pub(super) fn export_error(&self, error: &SearchError) {
        if let Some(exporter) = self.config.metrics_exporter.as_ref() {
            exporter.on_error(error);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_search_telemetry(
        &self,
        query_text: &str,
        query_class: QueryClass,
        phase: SearchEventPhase,
        result_count: usize,
        lexical_count: usize,
        semantic_count: usize,
        hash_control_count: usize,
        latency: Duration,
        root_request_id: &str,
        parent_event_id: Option<String>,
    ) -> Option<String> {
        let host_adapter = self.host_adapter.as_ref()?;

        let event_id = next_telemetry_identifier("evt");
        let telemetry_instance = telemetry_instance_for_adapter(host_adapter.as_ref());
        let telemetry_correlation = TelemetryCorrelation {
            event_id: event_id.clone(),
            root_request_id: root_request_id.to_owned(),
            parent_event_id,
        };

        let envelope = self.runtime_metrics_collector.emit_search(
            telemetry_timestamp_now(),
            telemetry_instance,
            telemetry_correlation,
            SearchCollectorSample {
                query_text: query_text.to_owned(),
                query_class,
                phase,
                result_count,
                lexical_count,
                semantic_count,
                hash_control_count,
                latency_us: u64::try_from(latency.as_micros()).unwrap_or(u64::MAX),
                memory_bytes: None,
            },
        );

        if let Err(err) = self
            .live_search_stream_emitter
            .publish_search(envelope.clone())
        {
            self.export_error(&err);
            tracing::warn!(error = %err, "live search stream publish failed");
        }

        if let Err(err) = host_adapter.emit_telemetry(&envelope) {
            self.export_error(&err);
            tracing::warn!(error = %err, "host adapter telemetry emission failed");
        }

        Some(event_id)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_embedding_telemetry(
        &self,
        embedder: &dyn Embedder,
        stage: EmbeddingStage,
        status: EmbeddingStatus,
        duration: Duration,
        root_request_id: &str,
        parent_event_id: Option<String>,
    ) -> Option<String> {
        let host_adapter = self.host_adapter.as_ref()?;

        let event_id = next_telemetry_identifier("evt");
        let telemetry_instance = telemetry_instance_for_adapter(host_adapter.as_ref());
        let telemetry_correlation = TelemetryCorrelation {
            event_id: event_id.clone(),
            root_request_id: root_request_id.to_owned(),
            parent_event_id,
        };

        let envelope = self.runtime_metrics_collector.emit_embedding(
            telemetry_timestamp_now(),
            telemetry_instance,
            telemetry_correlation,
            EmbeddingCollectorSample {
                job_id: format!("embed-{event_id}"),
                queue_depth: 0,
                doc_count: 1,
                stage,
                embedder_id: embedder.id().to_owned(),
                tier: embedder_tier_for_stage(stage, embedder.category()),
                dimension: embedder.dimension(),
                status,
                duration_ms: u64::try_from(duration.as_millis()).unwrap_or(u64::MAX),
            },
        );

        if let Err(err) = host_adapter.emit_telemetry(&envelope) {
            self.export_error(&err);
            tracing::warn!(error = %err, "host adapter telemetry emission failed");
        }

        Some(event_id)
    }

    pub(super) fn emit_host_lifecycle_hook(&self, event: &AdapterLifecycleEvent) {
        let Some(host_adapter) = self.host_adapter.as_ref() else {
            return;
        };
        if let Err(err) = host_adapter.on_lifecycle_event(event) {
            self.export_error(&err);
            tracing::warn!(error = %err, "host adapter lifecycle hook failed");
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_lifecycle_telemetry(
        &self,
        root_request_id: &str,
        parent_event_id: Option<String>,
        state: LifecycleState,
        severity: LifecycleSeverity,
        reason: Option<String>,
        uptime_ms: Option<u64>,
    ) -> Option<String> {
        let host_adapter = self.host_adapter.as_ref()?;

        let event_id = next_telemetry_identifier("evt");
        let telemetry_instance = telemetry_instance_for_adapter(host_adapter.as_ref());
        let telemetry_correlation = TelemetryCorrelation {
            event_id: event_id.clone(),
            root_request_id: root_request_id.to_owned(),
            parent_event_id,
        };
        let envelope = TelemetryEnvelope::new(
            telemetry_timestamp_now(),
            TelemetryEvent::Lifecycle {
                instance: telemetry_instance,
                correlation: telemetry_correlation,
                state,
                severity,
                reason,
                uptime_ms,
            },
        );

        if let Err(err) = host_adapter.emit_telemetry(&envelope) {
            self.export_error(&err);
            tracing::warn!(error = %err, "host adapter telemetry emission failed");
        }

        Some(event_id)
    }

    fn emit_resource_telemetry(
        &self,
        root_request_id: &str,
        parent_event_id: Option<String>,
    ) -> Option<String> {
        let host_adapter = self.host_adapter.as_ref()?;

        let event_id = next_telemetry_identifier("evt");
        let telemetry_instance = telemetry_instance_for_adapter(host_adapter.as_ref());
        let telemetry_correlation = TelemetryCorrelation {
            event_id: event_id.clone(),
            root_request_id: root_request_id.to_owned(),
            parent_event_id,
        };
        let sample = self.collect_resource_sample();
        let envelope = self.runtime_metrics_collector.emit_resource(
            telemetry_timestamp_now(),
            telemetry_instance,
            telemetry_correlation,
            sample,
        );

        if let Err(err) = host_adapter.emit_telemetry(&envelope) {
            self.export_error(&err);
            tracing::warn!(error = %err, "host adapter telemetry emission failed");
        }

        Some(event_id)
    }

    fn collect_resource_sample(&self) -> ResourceCollectorSample {
        let interval_ms = self
            .runtime_metrics_collector
            .config()
            .collection_interval_ms;
        let process_jiffies = read_proc_process_jiffies();
        let total_jiffies = read_proc_total_jiffies();
        let cpu_pct = self.update_cpu_pct_estimate(process_jiffies, total_jiffies);
        let rss_bytes = read_proc_status_rss_bytes().unwrap_or(0);
        let (io_read_bytes, io_write_bytes) = read_proc_io_bytes().unwrap_or((0, 0));
        let load_avg_1m = read_proc_load_avg_1m();

        ResourceCollectorSample {
            cpu_pct,
            rss_bytes,
            io_read_bytes,
            io_write_bytes,
            interval_ms,
            load_avg_1m,
            pressure_profile: None,
        }
    }

    fn update_cpu_pct_estimate(
        &self,
        process_jiffies: Option<u64>,
        total_jiffies: Option<u64>,
    ) -> f64 {
        let Some(process_jiffies) = process_jiffies else {
            return 0.0;
        };
        let Some(total_jiffies) = total_jiffies else {
            return 0.0;
        };

        let current = CpuJiffiesSnapshot {
            process_jiffies,
            total_jiffies,
        };
        let mut cpu_state = self
            .resource_cpu_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let cpu_pct = cpu_pct_from_jiffies(*cpu_state, current);
        *cpu_state = Some(current);
        cpu_pct
    }

    pub(super) fn emit_phase_health_telemetry(
        &self,
        root_request_id: &str,
        parent_event_id: Option<String>,
        uptime: Duration,
    ) {
        self.emit_host_lifecycle_hook(&AdapterLifecycleEvent::HealthTick {
            ts: telemetry_timestamp_now(),
        });
        let uptime_ms = Some(u64::try_from(uptime.as_millis()).unwrap_or(u64::MAX));
        let _ = self.emit_lifecycle_telemetry(
            root_request_id,
            parent_event_id.clone(),
            LifecycleState::Healthy,
            LifecycleSeverity::Info,
            None,
            uptime_ms,
        );
        let _ = self.emit_resource_telemetry(root_request_id, parent_event_id);
    }

    pub(super) fn emit_session_stop_telemetry(
        &self,
        root_request_id: &str,
        parent_event_id: Option<String>,
        state: LifecycleState,
        severity: LifecycleSeverity,
        reason: Option<String>,
        uptime: Duration,
    ) {
        self.emit_host_lifecycle_hook(&AdapterLifecycleEvent::SessionStop {
            ts: telemetry_timestamp_now(),
        });
        let uptime_ms = Some(u64::try_from(uptime.as_millis()).unwrap_or(u64::MAX));
        let _ = self.emit_lifecycle_telemetry(
            root_request_id,
            parent_event_id,
            state,
            severity,
            reason,
            uptime_ms,
        );
    }
}

pub(super) fn next_telemetry_identifier(prefix: &str) -> String {
    const CROCKFORD_BASE32: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";
    const TIMESTAMP_SHIFTS: [u32; 10] = [45, 40, 35, 30, 25, 20, 15, 10, 5, 0];
    const RANDOM_SHIFTS: [u32; 16] = [75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0];

    // ULID timestamp component is 48 bits of milliseconds since Unix epoch.
    let timestamp_ms = telemetry_timestamp_ms() & 0x0000_FFFF_FFFF_FFFF;

    // Deterministic, process-local entropy for uniqueness without extra deps.
    let sequence = TELEMETRY_EVENT_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = u64::from(std::process::id());
    let entropy = sequence ^ timestamp_ms.rotate_left(17) ^ (pid << 32);

    let pid_low16 = u16::try_from(pid % (u64::from(u16::MAX) + 1)).unwrap_or_default();
    let mut random_bytes = [0_u8; 10];
    random_bytes[..2].copy_from_slice(&pid_low16.to_be_bytes());
    random_bytes[2..].copy_from_slice(&entropy.to_be_bytes());

    let random_component = random_bytes
        .iter()
        .fold(0_u128, |acc, byte| (acc << 8) | u128::from(*byte));

    let mut id = String::with_capacity(prefix.len() + 1 + 26);
    if !prefix.is_empty() {
        id.push_str(prefix);
        id.push('_');
    }

    for shift in TIMESTAMP_SHIFTS {
        let index = usize::try_from((timestamp_ms >> shift) & 0x1F).unwrap_or_default();
        id.push(char::from(CROCKFORD_BASE32[index]));
    }
    for shift in RANDOM_SHIFTS {
        let index = usize::try_from((random_component >> shift) & 0x1F).unwrap_or_default();
        id.push(char::from(CROCKFORD_BASE32[index]));
    }

    id
}

fn telemetry_timestamp_ms() -> u64 {
    let nanos = frankensearch_core::rfc3339::now_unix_nanos();
    if nanos <= 0 {
        return 0;
    }
    let nanos_u128 = u128::try_from(nanos).unwrap_or_default();
    let millis = nanos_u128 / 1_000_000;
    u64::try_from(millis).unwrap_or(u64::MAX)
}

pub(super) fn telemetry_timestamp_now() -> String {
    frankensearch_core::rfc3339::format_unix_nanos(frankensearch_core::rfc3339::now_unix_nanos())
}

fn telemetry_instance_for_adapter(host_adapter: &dyn HostAdapter) -> TelemetryInstance {
    let identity = host_adapter.identity();
    let host_name = std::env::var("HOSTNAME")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown-host".to_owned());
    let instance_id = identity
        .instance_uuid
        .unwrap_or_else(|| format!("{}-{}", identity.adapter_id, std::process::id()));
    TelemetryInstance {
        instance_id,
        project_key: identity.host_project,
        host_name,
        pid: Some(std::process::id()),
    }
}

#[allow(clippy::cast_precision_loss)]
pub(super) fn cpu_pct_from_jiffies(
    previous: Option<CpuJiffiesSnapshot>,
    current: CpuJiffiesSnapshot,
) -> f64 {
    let Some(previous) = previous else {
        return 0.0;
    };

    let process_delta = current
        .process_jiffies
        .saturating_sub(previous.process_jiffies);
    let total_delta = current.total_jiffies.saturating_sub(previous.total_jiffies);
    if process_delta == 0 || total_delta == 0 {
        return 0.0;
    }

    let raw = (process_delta as f64 / total_delta as f64) * 100.0;
    raw.clamp(0.0, 100.0)
}

fn read_proc_total_jiffies() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let raw = std::fs::read_to_string("/proc/stat").ok()?;
        parse_proc_total_jiffies(raw.as_str())
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn read_proc_process_jiffies() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let raw = std::fs::read_to_string("/proc/self/stat").ok()?;
        parse_proc_process_jiffies(raw.as_str())
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn read_proc_status_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let raw = std::fs::read_to_string("/proc/self/status").ok()?;
        parse_proc_status_rss_bytes(raw.as_str())
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn read_proc_io_bytes() -> Option<(u64, u64)> {
    #[cfg(target_os = "linux")]
    {
        let raw = std::fs::read_to_string("/proc/self/io").ok()?;
        parse_proc_io_bytes(raw.as_str())
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn read_proc_load_avg_1m() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        let raw = std::fs::read_to_string("/proc/loadavg").ok()?;
        parse_proc_load_avg_1m(raw.as_str())
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(any(target_os = "linux", test))]
pub(super) fn parse_proc_total_jiffies(raw: &str) -> Option<u64> {
    let cpu_line = raw.lines().find(|line| line.starts_with("cpu "))?;
    let mut total_jiffies = 0_u64;
    for token in cpu_line.split_whitespace().skip(1) {
        let value = token.parse::<u64>().ok()?;
        total_jiffies = total_jiffies.checked_add(value)?;
    }
    Some(total_jiffies)
}

#[cfg(any(target_os = "linux", test))]
pub(super) fn parse_proc_process_jiffies(raw: &str) -> Option<u64> {
    let close_paren = raw.rfind(')')?;
    let stats_after_comm = raw.get(close_paren + 2..)?;
    let mut stats = stats_after_comm.split_whitespace();
    let utime = stats.nth(11)?.parse::<u64>().ok()?;
    let stime = stats.next()?.parse::<u64>().ok()?;
    utime.checked_add(stime)
}

#[cfg(any(target_os = "linux", test))]
pub(super) fn parse_proc_status_rss_bytes(raw: &str) -> Option<u64> {
    let line = raw
        .lines()
        .find(|line| line.trim_start().starts_with("VmRSS:"))?;
    let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    kib.checked_mul(1024)
}

#[cfg(any(target_os = "linux", test))]
pub(super) fn parse_proc_io_bytes(raw: &str) -> Option<(u64, u64)> {
    let mut read_bytes = None;
    let mut write_bytes = None;
    for line in raw.lines() {
        if let Some((key, value)) = line.split_once(':') {
            let parsed = value.trim().parse::<u64>().ok()?;
            match key.trim() {
                "read_bytes" => read_bytes = Some(parsed),
                "write_bytes" => write_bytes = Some(parsed),
                _ => {}
            }
        }
    }
    Some((read_bytes?, write_bytes?))
}

#[cfg(any(target_os = "linux", test))]
pub(super) fn parse_proc_load_avg_1m(raw: &str) -> Option<f64> {
    let value = raw.split_whitespace().next()?.parse::<f64>().ok()?;
    if value.is_finite() && value >= 0.0 {
        Some(value)
    } else {
        None
    }
}

pub(super) const fn embedder_tier_for_stage(
    stage: EmbeddingStage,
    category: ModelCategory,
) -> EmbedderTier {
    match stage {
        EmbeddingStage::Quality => EmbedderTier::Quality,
        EmbeddingStage::Fast | EmbeddingStage::Background => match category {
            ModelCategory::HashEmbedder => EmbedderTier::Hash,
            ModelCategory::StaticEmbedder => EmbedderTier::Fast,
            ModelCategory::TransformerEmbedder | ModelCategory::ApiEmbedder => {
                EmbedderTier::Quality
            }
        },
    }
}

#[cfg(test)]
pub(super) const TELEMETRY_TIMESTAMP_FALLBACK_RFC3339: &str = "1970-01-01T00:00:00Z";
