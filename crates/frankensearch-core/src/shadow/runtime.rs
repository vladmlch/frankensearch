use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll, Wake, Waker};

use tracing::{info, warn};

use super::SHADOW_DEGRADATION_SCHEMA_VERSION;
use super::artifacts::ShadowArtifactSink;
use super::types::{
    ShadowDegradationKind, ShadowDegradationRecord, ShadowDocument, ShadowLexicalConfig,
    ShadowLoadProbe, ShadowShedReason, ShadowStatus, compute_shadow_corpus_hash,
};
use crate::{Cx, IndexableDocument, ScoredResult, SearchError, SearchFuture, SearchResult};

#[derive(Debug, Clone, Copy)]
pub(super) enum ShadowSearchPath {
    Search,
    FusionCandidates,
}

impl ShadowSearchPath {
    pub(super) const fn fusion_candidates(self) -> bool {
        matches!(self, Self::FusionCandidates)
    }
}

pub(super) struct ShadowState {
    pub(super) enabled: AtomicBool,
    pub(super) allow_inline_shadow: bool,
    pub(super) sample_rate_basis_points: u16,
    max_in_flight: usize,
    pub(super) score_epsilon: f32,
    pub(super) sample_ordinal: AtomicU64,
    in_flight: AtomicUsize,
    pub(super) sampled: AtomicU64,
    pub(super) sampled_out: AtomicU64,
    shed: AtomicU64,
    pub(super) completed: AtomicU64,
    pub(super) divergences: AtomicU64,
    degradations: AtomicU64,
    pub(super) manifest_generation: AtomicU64,
    pub(super) corpus_revision: AtomicU64,
    pub(super) shadow_ready: AtomicBool,
    pub(super) corpus: Mutex<BTreeMap<String, ShadowDocument>>,
    pub(super) sink: ShadowArtifactSink,
    pub(super) load_probe: Arc<dyn ShadowLoadProbe>,
}

impl ShadowState {
    pub(super) fn corpus_snapshot(&self) -> (String, Vec<ShadowDocument>) {
        let documents = self
            .corpus
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .cloned()
            .collect::<Vec<_>>();
        (compute_shadow_corpus_hash(&documents), documents)
    }

    pub(super) fn record_shed(&self, reason: ShadowShedReason, query: &str, generation: u64) {
        self.shed.fetch_add(1, Ordering::Relaxed);
        info!(
            event = "shadow_query_shed",
            query,
            manifest_generation = generation,
            shed_reason = reason.label(),
            "shadow lexical query was shed without affecting serving"
        );
    }

    pub(super) fn record_degradation(
        &self,
        generation: u64,
        kind: ShadowDegradationKind,
        detail: impl Into<String>,
        persist: bool,
    ) {
        let detail = detail.into();
        self.degradations.fetch_add(1, Ordering::Relaxed);
        warn!(
            event = "shadow_degraded",
            manifest_generation = generation,
            degradation_kind = ?kind,
            detail,
            "shadow lexical failure was isolated from serving"
        );
        if persist {
            let record = ShadowDegradationRecord {
                schema_version: SHADOW_DEGRADATION_SCHEMA_VERSION,
                manifest_generation: generation,
                kind,
                detail,
            };
            if let Err(error) = self.sink.append_degradation(&record) {
                warn!(
                    event = "shadow_artifact_write_failed",
                    error = %error,
                    "shadow degradation record could not be persisted"
                );
            }
        }
    }

    pub(super) fn try_admit(self: &Arc<Self>) -> Option<InFlightGuard> {
        let mut observed = self.in_flight.load(Ordering::Acquire);
        loop {
            if observed >= self.max_in_flight {
                return None;
            }
            match self.in_flight.compare_exchange_weak(
                observed,
                observed.saturating_add(1),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Some(InFlightGuard {
                        state: Arc::clone(self),
                    });
                }
                Err(actual) => observed = actual,
            }
        }
    }
}

pub(super) struct InFlightGuard {
    state: Arc<ShadowState>,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.state.in_flight.fetch_sub(1, Ordering::AcqRel);
    }
}

struct ShadowWake(asupersync::channel::mpsc::Sender<()>);

impl Wake for ShadowWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        // One pending wake is enough: coalesce repeated wakes without allowing
        // a backend to allocate an unbounded notification queue.
        let _ = self.0.try_send(());
    }
}

/// Keep synchronous work inside an async backend's poll off the serving
/// scheduler. Each poll belongs to the same region and blocking pool as the
/// shadow task; waiting for the next wake uses a cancel-aware runtime channel.
/// Pool-less contexts may run inline only when the caller explicitly opts in.
pub(super) async fn poll_shadow_on_pool(
    cx: &Cx,
    mut future: SearchFuture<'static, Vec<ScoredResult>>,
    allow_inline_shadow: bool,
) -> SearchResult<Vec<ScoredResult>> {
    let Some(pool) = cx.blocking_pool_handle() else {
        if !allow_inline_shadow {
            return Err(SearchError::SubsystemError {
                subsystem: "shadow",
                source: "shadow search requires a caller-owned blocking pool".into(),
            });
        }
        cx.checkpoint().map_err(|error| SearchError::Cancelled {
            phase: "shadow.search".to_owned(),
            reason: error.to_string(),
        })?;
        let result = future.await;
        cx.checkpoint().map_err(|error| SearchError::Cancelled {
            phase: "shadow.search".to_owned(),
            reason: error.to_string(),
        })?;
        return result;
    };
    let (wake_tx, mut wake_rx) = asupersync::channel::mpsc::channel(1);
    let wake = Arc::new(ShadowWake(wake_tx));
    loop {
        cx.checkpoint().map_err(|error| SearchError::Cancelled {
            phase: "shadow.search".to_owned(),
            reason: error.to_string(),
        })?;
        if pool.is_shutdown() {
            return Err(SearchError::SubsystemError {
                subsystem: "shadow",
                source: "caller-owned blocking pool is shut down".into(),
            });
        }
        let waker = Waker::from(Arc::clone(&wake));
        let mut worker = cx
            .spawn_blocking(move |child| {
                // A pool shutdown can race admission. Asupersync may recover
                // rejected work inline; do not let that run backend code on
                // the serving scheduler.
                if Cx::is_active() {
                    return Err(SearchError::SubsystemError {
                        subsystem: "shadow",
                        source: "blocking pool dispatch fell back to an async executor".into(),
                    });
                }
                child.checkpoint().map_err(|error| SearchError::Cancelled {
                    phase: "shadow.search".to_owned(),
                    reason: error.to_string(),
                })?;
                let outcome = future.as_mut().poll(&mut Context::from_waker(&waker));
                Ok((future, outcome))
            })
            .map_err(|error| SearchError::SubsystemError {
                subsystem: "shadow",
                source: error.into(),
            })?;
        let (next, outcome) = worker.join(cx).await.map_err(|error| match error {
            asupersync::runtime::JoinError::Cancelled(reason) => SearchError::Cancelled {
                phase: "shadow.search".to_owned(),
                reason: reason.to_string(),
            },
            error => SearchError::SubsystemError {
                subsystem: "shadow",
                source: error.into(),
            },
        })??;
        future = next;
        // join itself does not check request cancellation. Never publish a
        // result after cancellation or exhaustion during a blocking poll.
        cx.checkpoint().map_err(|error| SearchError::Cancelled {
            phase: "shadow.search".to_owned(),
            reason: error.to_string(),
        })?;
        if let Poll::Ready(result) = outcome {
            return result;
        }
        wake_rx
            .recv(cx)
            .await
            .map_err(|error| SearchError::Cancelled {
                phase: "shadow.search".to_owned(),
                reason: error.to_string(),
            })?;
    }
}

pub(super) fn build_shadow_state(
    config: ShadowLexicalConfig,
    load_probe: Arc<dyn ShadowLoadProbe>,
) -> SearchResult<Arc<ShadowState>> {
    config.validate()?;
    Ok(Arc::new(ShadowState {
        enabled: AtomicBool::new(config.enabled),
        allow_inline_shadow: config.allow_inline_shadow,
        sample_rate_basis_points: config.sample_rate_basis_points,
        max_in_flight: config.max_in_flight,
        score_epsilon: config.score_epsilon,
        sample_ordinal: AtomicU64::new(0),
        in_flight: AtomicUsize::new(0),
        sampled: AtomicU64::new(0),
        sampled_out: AtomicU64::new(0),
        shed: AtomicU64::new(0),
        completed: AtomicU64::new(0),
        divergences: AtomicU64::new(0),
        degradations: AtomicU64::new(0),
        manifest_generation: AtomicU64::new(config.initial_generation),
        corpus_revision: AtomicU64::new(0),
        shadow_ready: AtomicBool::new(true),
        corpus: Mutex::new(BTreeMap::new()),
        sink: ShadowArtifactSink::new(config.artifact_directory),
        load_probe,
    }))
}

pub(super) fn seed_shadow_corpus(
    state: &Arc<ShadowState>,
    documents: &[IndexableDocument],
    manifest_generation: u64,
) {
    let mut corpus = state.corpus.lock().unwrap_or_else(PoisonError::into_inner);
    corpus.clear();
    corpus.extend(
        documents
            .iter()
            .map(|document| (document.id.clone(), ShadowDocument::from(document))),
    );
    drop(corpus);
    state
        .manifest_generation
        .store(manifest_generation, Ordering::Release);
    state.corpus_revision.fetch_add(1, Ordering::AcqRel);
}

pub(super) fn shadow_status(state: &Arc<ShadowState>) -> ShadowStatus {
    let (corpus_hash, _) = state.corpus_snapshot();
    ShadowStatus {
        enabled: state.enabled.load(Ordering::Acquire),
        manifest_generation: state.manifest_generation.load(Ordering::Acquire),
        corpus_hash,
        in_flight: state.in_flight.load(Ordering::Acquire),
        sampled: state.sampled.load(Ordering::Relaxed),
        sampled_out: state.sampled_out.load(Ordering::Relaxed),
        shed: state.shed.load(Ordering::Relaxed),
        completed: state.completed.load(Ordering::Relaxed),
        divergences: state.divergences.load(Ordering::Relaxed),
        degradations: state.degradations.load(Ordering::Relaxed),
    }
}
