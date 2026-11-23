use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(target_os = "linux")]
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use asupersync::Cx;
#[cfg(not(any(feature = "lexical", feature = "quill")))]
use asupersync::types::CancelKind;

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::{EmbeddingIdentityBundleV1, QuantizationFormat};
use frankensearch_core::traits::{Embedder, MetricsExporter, ModelCategory, SearchFuture};
use frankensearch_core::types::{EmbeddingMetrics, IndexMetrics, SearchMetrics};
use frankensearch_embed::auto_detect::EmbedderStack;

mod build_and_identity;
mod lexical;
mod stats_and_hybrid_open;

#[cfg(target_os = "linux")]
fn owned_admitted_v2_sync_dir() -> std::path::PathBuf {
    static NONCE: AtomicU64 = AtomicU64::new(0);
    let parent = std::env::temp_dir().join("frankensearch_admitted_v2_sync_tests");
    std::fs::create_dir_all(&parent).expect("create durable test parent");
    for _ in 0..1024 {
        let nonce = NONCE.fetch_add(1, AtomicOrdering::Relaxed);
        let dir = parent.join(format!(
            "{}-{}-{nonce}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        match std::fs::create_dir(&dir) {
            Ok(()) => return dir,
            // A name this process already minted: fall through to the next
            // nonce. The match is the loop body's final expression, so an
            // empty arm retries exactly as an explicit `continue` did.
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => panic!("create unique admitted-v2 sync directory: {error}"),
        }
    }
    panic!("exhausted admitted-v2 sync test directory names")
}

struct StubEmbedder {
    id: &'static str,
    dim: usize,
}

impl Embedder for StubEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        let dim = self.dim;
        Box::pin(async move {
            let mut vec = vec![0.0; dim];
            // Simple deterministic embedding from text length.
            vec[text.len() % dim] = 1.0;
            Ok(vec)
        })
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn id(&self) -> &str {
        self.id
    }

    fn model_name(&self) -> &str {
        self.id
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::StaticEmbedder
    }
}

struct SelectiveFailEmbedder;

impl Embedder for SelectiveFailEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            if text.contains("fail-fast-embedding") {
                return Err(SearchError::EmbeddingFailed {
                    model: "selective-fail".to_owned(),
                    source: Box::new(std::io::Error::other("intentional test failure")),
                });
            }
            Ok(vec![1.0, 0.0, 0.0, 0.0])
        })
    }

    fn dimension(&self) -> usize {
        4
    }

    fn id(&self) -> &'static str {
        "selective-fail"
    }

    fn model_name(&self) -> &'static str {
        "selective-fail"
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::StaticEmbedder
    }
}

#[cfg(not(any(feature = "lexical", feature = "quill")))]
struct CancelOnFirstEmbedder {
    id: &'static str,
    calls: Arc<AtomicUsize>,
    return_typed_error: bool,
}

#[cfg(not(any(feature = "lexical", feature = "quill")))]
impl Embedder for CancelOnFirstEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        let id = self.id;
        let calls = Arc::clone(&self.calls);
        let return_typed_error = self.return_typed_error;
        Box::pin(async move {
            let call = calls.fetch_add(1, Ordering::SeqCst);
            assert_eq!(call, 0, "{id} was called after cancelling the build");
            cx.cancel_with(CancelKind::User, Some("cancel-on-first embedder"));
            if return_typed_error {
                Err(SearchError::Cancelled {
                    phase: format!("{id} embedding"),
                    reason: "cancel-on-first embedder".to_owned(),
                })
            } else {
                Ok(vec![1.0, 0.0, 0.0, 0.0])
            }
        })
    }

    fn dimension(&self) -> usize {
        4
    }

    fn id(&self) -> &str {
        self.id
    }

    fn model_name(&self) -> &str {
        self.id
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::StaticEmbedder
    }
}

#[derive(Debug, Default)]
struct RecordingExporter {
    search: Mutex<Vec<SearchMetrics>>,
    embedding: Mutex<Vec<EmbeddingMetrics>>,
    index: Mutex<Vec<IndexMetrics>>,
    errors: Mutex<Vec<String>>,
}

impl MetricsExporter for RecordingExporter {
    fn on_search_completed(&self, metrics: &SearchMetrics) {
        self.search
            .lock()
            .expect("search metrics lock")
            .push(metrics.clone());
    }

    fn on_embedding_completed(&self, metrics: &EmbeddingMetrics) {
        self.embedding
            .lock()
            .expect("embedding metrics lock")
            .push(metrics.clone());
    }

    fn on_index_updated(&self, metrics: &IndexMetrics) {
        self.index
            .lock()
            .expect("index metrics lock")
            .push(metrics.clone());
    }

    fn on_error(&self, error: &SearchError) {
        self.errors
            .lock()
            .expect("errors lock")
            .push(error.to_string());
    }
}

fn stub_stack() -> EmbedderStack {
    let fast = Arc::new(StubEmbedder {
        id: "stub-fast",
        dim: 4,
    });
    let quality = Arc::new(StubEmbedder {
        id: "stub-quality",
        dim: 4,
    });
    EmbedderStack::from_parts(fast, Some(quality))
}

#[cfg(not(any(feature = "lexical", feature = "quill")))]
fn cancel_on_first_stack(cancelled_tier: &'static str, calls: Arc<AtomicUsize>) -> EmbedderStack {
    let cancelling: Arc<dyn Embedder> = Arc::new(CancelOnFirstEmbedder {
        id: cancelled_tier,
        calls,
        return_typed_error: cancelled_tier != "fast-canceller",
    });
    if cancelled_tier == "fast-canceller" {
        EmbedderStack::from_parts(cancelling, None)
    } else {
        let fast: Arc<dyn Embedder> = Arc::new(StubEmbedder {
            id: "stub-fast",
            dim: 4,
        });
        EmbedderStack::from_parts(fast, Some(cancelling))
    }
}

/// Identity-aware stub (bd-9xuj T2-C2): same deterministic vectors as
/// [`StubEmbedder`], but it supplies a complete identity bundle the way
/// production embedders do, so builds through it must bind the typed
/// producing identity — not just the id string.
struct IdentityStubEmbedder {
    id: &'static str,
    dim: usize,
    identity: frankensearch_core::generation::EmbeddingIdentityBundleV1,
}

impl IdentityStubEmbedder {
    fn new(id: &'static str, dim: usize) -> Self {
        Self {
            id,
            dim,
            identity:
                frankensearch_core::generation::EmbeddingIdentityBundleV1::explicit_test_model(
                    id,
                    u32::try_from(dim).expect("test dimension fits u32"),
                ),
        }
    }
}

impl Embedder for IdentityStubEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        let dim = self.dim;
        Box::pin(async move {
            let mut vec = vec![0.0; dim];
            if let Some(slot) = vec.get_mut(text.len() % dim) {
                *slot = 1.0;
            }
            Ok(vec)
        })
    }

    fn identity(&self) -> SearchResult<&frankensearch_core::generation::EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn id(&self) -> &str {
        self.id
    }

    fn model_name(&self) -> &str {
        self.id
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::StaticEmbedder
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoundBuildFault {
    None,
    ForeignIdentity,
    MalformedForeignIdentity,
    ChangedProducer,
    NonFinite,
    WrongDimension,
    CancelWithError,
    RejectIdentity,
    PersistedOutputContract,
}

struct BoundBuildEmbedder {
    inner: IdentityStubEmbedder,
    fault: BoundBuildFault,
    changed_identity: EmbeddingIdentityBundleV1,
    changed: AtomicBool,
    raw_calls: AtomicUsize,
    bound_calls: AtomicUsize,
}

impl BoundBuildEmbedder {
    fn new(id: &'static str, dim: usize, fault: BoundBuildFault) -> Self {
        let mut inner = IdentityStubEmbedder::new(id, dim);
        if fault == BoundBuildFault::PersistedOutputContract {
            "fsvi-v2".clone_into(&mut inner.identity.storage.format);
            inner.identity.storage.quantization = QuantizationFormat::F16;
            "little-endian".clone_into(&mut inner.identity.storage.endianness);
            inner.identity.validate().unwrap();
        }
        let mut changed_identity = inner.identity.clone();
        changed_identity
            .producer
            .implementation_revision
            .push_str("-changed");
        Self {
            inner,
            fault,
            changed_identity,
            changed: AtomicBool::new(false),
            raw_calls: AtomicUsize::new(0),
            bound_calls: AtomicUsize::new(0),
        }
    }
}

impl Embedder for BoundBuildEmbedder {
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        if self.fault == BoundBuildFault::RejectIdentity {
            return Err(SearchError::UnverifiableRemoteSpace {
                producer: self.id().to_owned(),
                reason: "injected producer trust refusal".to_owned(),
            });
        }
        if self.changed.load(Ordering::SeqCst) {
            Ok(&self.changed_identity)
        } else {
            self.inner.identity()
        }
    }

    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        self.raw_calls.fetch_add(1, Ordering::SeqCst);
        // Raw inference deliberately succeeds: only callers of the real
        // bound operation can observe the injected provider failures.
        self.inner.embed(cx, text)
    }

    fn embed_bound<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, frankensearch_core::IdentityBoundEmbedding> {
        Box::pin(async move {
            self.bound_calls.fetch_add(1, Ordering::SeqCst);
            let mut bound = self.inner.embed_bound(cx, text).await?;
            if text == "fault-trigger" {
                match self.fault {
                    BoundBuildFault::ForeignIdentity => {
                        bound.identity = self.changed_identity.clone();
                    }
                    BoundBuildFault::MalformedForeignIdentity => {
                        bound.identity.space.dimension = 0;
                    }
                    BoundBuildFault::ChangedProducer => {
                        self.changed.store(true, Ordering::SeqCst);
                    }
                    BoundBuildFault::NonFinite => bound.values[0] = f32::NAN,
                    BoundBuildFault::WrongDimension => {
                        bound.values.pop();
                    }
                    BoundBuildFault::CancelWithError => {
                        cx.set_cancel_requested(true);
                        return Err(SearchError::EmbeddingFailed {
                            model: self.id().to_owned(),
                            source: "cancelled provider failure".into(),
                        });
                    }
                    BoundBuildFault::None
                    | BoundBuildFault::RejectIdentity
                    | BoundBuildFault::PersistedOutputContract => {}
                }
            }
            Ok(bound)
        })
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
    fn id(&self) -> &str {
        self.inner.id()
    }
    fn model_name(&self) -> &str {
        self.inner.model_name()
    }
    fn is_semantic(&self) -> bool {
        true
    }
    fn category(&self) -> ModelCategory {
        self.inner.category()
    }
}

fn bound_build_stack(
    fault_tier: &str,
    fault: BoundBuildFault,
) -> (
    EmbedderStack,
    Arc<BoundBuildEmbedder>,
    Arc<BoundBuildEmbedder>,
) {
    let fast = Arc::new(BoundBuildEmbedder::new(
        "bound-fast",
        4,
        if fault_tier == "fast" {
            fault
        } else {
            BoundBuildFault::None
        },
    ));
    let quality = Arc::new(BoundBuildEmbedder::new(
        "bound-quality",
        6,
        if fault_tier == "quality" {
            fault
        } else {
            BoundBuildFault::None
        },
    ));
    let stack = EmbedderStack::from_parts(
        Arc::clone(&fast) as Arc<dyn Embedder>,
        Some(Arc::clone(&quality) as Arc<dyn Embedder>),
    );
    (stack, fast, quality)
}

fn fast_only_stack() -> EmbedderStack {
    let fast = Arc::new(StubEmbedder {
        id: "stub-fast",
        dim: 4,
    });
    EmbedderStack::from_parts(fast, None)
}
