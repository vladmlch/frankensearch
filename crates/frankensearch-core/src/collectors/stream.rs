use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use serde::{Deserialize, Serialize};

use super::TELEMETRY_SCHEMA_VERSION;
use super::types::{SearchStreamConfig, SearchStreamMode, TelemetryEnvelope, TelemetryEvent};
use crate::error::{SearchError, SearchResult};

/// A buffered live-search frame for timeline/live-feed consumers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiveSearchFrame {
    /// Opaque resume cursor.
    pub cursor: String,
    /// Monotonic sequence number.
    pub sequence: u64,
    /// Number of events dropped since the prior emitted frame.
    pub dropped_since_last: u64,
    /// Canonical search telemetry event.
    pub event: TelemetryEnvelope,
}

/// Result of publishing one search event into the stream emitter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchStreamPublishOutcome {
    /// Cursor assigned to this frame.
    pub cursor: String,
    /// Sequence assigned to this frame.
    pub sequence: u64,
    /// Current queue depth after publish.
    pub buffered: usize,
    /// Number of dropped events attached to this frame.
    pub dropped_since_last: u64,
}

/// Point-in-time stream health counters for dashboards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchStreamHealth {
    /// Configured buffering mode.
    pub mode: SearchStreamMode,
    /// Maximum buffer capacity.
    pub capacity: usize,
    /// Current queued frame count.
    pub buffered: usize,
    /// Total successfully emitted frames.
    pub emitted_total: u64,
    /// Total dropped frames (lossy mode only).
    pub dropped_total: u64,
    /// Total rejected publishes due to full buffer (non-lossy mode).
    pub backpressure_rejections: u64,
}

/// Bounded live-search stream emitter with explicit drop/backpressure accounting.
///
/// This is intentionally synchronous and lightweight so pipeline code can publish
/// search events without async runtime coupling.
#[derive(Debug)]
pub struct LiveSearchStreamEmitter {
    config: SearchStreamConfig,
    queue: Mutex<VecDeque<LiveSearchFrame>>,
    next_sequence: AtomicU64,
    emitted_total: AtomicU64,
    dropped_total: AtomicU64,
    backpressure_rejections: AtomicU64,
    pending_dropped_since_last: AtomicU64,
}

impl Default for LiveSearchStreamEmitter {
    fn default() -> Self {
        Self {
            config: SearchStreamConfig::default(),
            queue: Mutex::new(VecDeque::new()),
            next_sequence: AtomicU64::new(0),
            emitted_total: AtomicU64::new(0),
            dropped_total: AtomicU64::new(0),
            backpressure_rejections: AtomicU64::new(0),
            pending_dropped_since_last: AtomicU64::new(0),
        }
    }
}

impl LiveSearchStreamEmitter {
    /// Create a stream emitter with explicit configuration.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] if `config` is invalid.
    pub fn new(config: SearchStreamConfig) -> SearchResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            ..Self::default()
        })
    }

    fn lock_queue(&self) -> MutexGuard<'_, VecDeque<LiveSearchFrame>> {
        match self.queue.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// Publish a canonical `search` event into the stream.
    ///
    /// In lossy mode, if full, the oldest queued frame is dropped.
    /// In non-lossy mode, if full, returns [`SearchError::QueueFull`].
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when `event` is not a `search` event.
    /// Returns [`SearchError::QueueFull`] when mode is non-lossy and queue is full.
    pub fn publish_search(
        &self,
        event: TelemetryEnvelope,
    ) -> SearchResult<SearchStreamPublishOutcome> {
        if event.v != TELEMETRY_SCHEMA_VERSION {
            return Err(SearchError::InvalidConfig {
                field: "search_stream.event.schema_version".to_owned(),
                value: event.v.to_string(),
                reason: format!(
                    "live search stream expects schema_version {TELEMETRY_SCHEMA_VERSION}"
                ),
            });
        }

        if !matches!(event.event, TelemetryEvent::Search { .. }) {
            return Err(SearchError::InvalidConfig {
                field: "search_stream.event.type".to_owned(),
                value: "non-search".to_owned(),
                reason: "live search stream only accepts search telemetry events".to_owned(),
            });
        }

        let mut queue = self.lock_queue();
        if queue.len() >= self.config.capacity {
            match self.config.mode {
                SearchStreamMode::Lossy => {
                    let _ = queue.pop_front();
                    self.dropped_total.fetch_add(1, Ordering::Relaxed);
                    self.pending_dropped_since_last
                        .fetch_add(1, Ordering::Relaxed);
                }
                SearchStreamMode::NonLossy => {
                    self.backpressure_rejections.fetch_add(1, Ordering::Relaxed);
                    return Err(SearchError::QueueFull {
                        pending: queue.len(),
                        capacity: self.config.capacity,
                    });
                }
            }
        }

        let sequence = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        let cursor = format!("search-{sequence:020}");
        let dropped_since_last = self.pending_dropped_since_last.swap(0, Ordering::Relaxed);
        queue.push_back(LiveSearchFrame {
            cursor: cursor.clone(),
            sequence,
            dropped_since_last,
            event,
        });
        self.emitted_total.fetch_add(1, Ordering::Relaxed);

        Ok(SearchStreamPublishOutcome {
            cursor,
            sequence,
            buffered: queue.len(),
            dropped_since_last,
        })
    }

    /// Drain up to `max_items` frames from oldest to newest.
    #[must_use]
    pub fn drain(&self, max_items: usize) -> Vec<LiveSearchFrame> {
        if max_items == 0 {
            return Vec::new();
        }

        let mut queue = self.lock_queue();
        let take = max_items.min(queue.len());
        let mut drained = Vec::with_capacity(take);
        for _ in 0..take {
            if let Some(frame) = queue.pop_front() {
                drained.push(frame);
            }
        }
        drained
    }

    /// Snapshot stream health counters for timeline/live-feed diagnostics.
    #[must_use]
    pub fn health(&self) -> SearchStreamHealth {
        let buffered = self.lock_queue().len();
        SearchStreamHealth {
            mode: self.config.mode,
            capacity: self.config.capacity,
            buffered,
            emitted_total: self.emitted_total.load(Ordering::Relaxed),
            dropped_total: self.dropped_total.load(Ordering::Relaxed),
            backpressure_rejections: self.backpressure_rejections.load(Ordering::Relaxed),
        }
    }
}
