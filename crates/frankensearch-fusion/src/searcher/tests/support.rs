use super::*;

pub(super) fn is_valid_ulid_like(candidate: &str) -> bool {
    let ulid = if let Some((_, suffix)) = candidate.rsplit_once('_') {
        suffix
    } else {
        candidate
    };
    if ulid.len() != 26 {
        return false;
    }
    ulid.bytes().all(|byte| {
        matches!(
            byte,
            b'0'..=b'9'
                | b'A'..=b'H'
                | b'J'..=b'K'
                | b'M'..=b'N'
                | b'P'..=b'T'
                | b'V'..=b'Z'
        )
    })
}

// ─── Stub Embedder ──────────────────────────────────────────────────

pub(super) struct StubEmbedder {
    pub(super) id: &'static str,
    pub(super) dimension: usize,
}

impl StubEmbedder {
    pub(super) const fn new(id: &'static str, dimension: usize) -> Self {
        Self { id, dimension }
    }
}

impl Embedder for StubEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        let dim = self.dimension;
        Box::pin(async move {
            let mut vec = vec![0.0; dim];
            if !vec.is_empty() {
                vec[0] = 1.0; // Simple deterministic embedding
            }
            Ok(vec)
        })
    }

    fn dimension(&self) -> usize {
        self.dimension
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

pub(super) struct NonSemanticEmbedder {
    pub(super) id: &'static str,
    pub(super) dimension: usize,
}

impl NonSemanticEmbedder {
    pub(super) const fn new(id: &'static str, dimension: usize) -> Self {
        Self { id, dimension }
    }
}

impl Embedder for NonSemanticEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        let dim = self.dimension;
        Box::pin(async move {
            let mut vec = vec![0.0; dim];
            if !vec.is_empty() {
                vec[0] = 1.0;
            }
            Ok(vec)
        })
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn id(&self) -> &str {
        self.id
    }

    fn model_name(&self) -> &str {
        self.id
    }

    fn is_semantic(&self) -> bool {
        false
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
}

// ─── Failing Embedder ───────────────────────────────────────────────

pub(super) struct FailingEmbedder;

impl Embedder for FailingEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async {
            Err(SearchError::EmbeddingFailed {
                model: "failing-embedder".into(),
                source: Box::new(std::io::Error::other("intentional test failure")),
            })
        })
    }

    fn dimension(&self) -> usize {
        4
    }

    fn id(&self) -> &str {
        "failing-embedder"
    }

    fn model_name(&self) -> &str {
        "failing-embedder"
    }

    fn is_semantic(&self) -> bool {
        false
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
}

pub(super) struct CancelledEmbedder;

impl Embedder for CancelledEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async {
            Err(SearchError::Cancelled {
                phase: "embed".to_owned(),
                reason: "test cancellation".to_owned(),
            })
        })
    }

    fn dimension(&self) -> usize {
        4
    }

    fn id(&self) -> &str {
        "cancelled-embedder"
    }

    fn model_name(&self) -> &str {
        "cancelled-embedder"
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::StaticEmbedder
    }
}

pub(super) struct PendingEmbedder {
    pub(super) id: &'static str,
    pub(super) dimension: usize,
}

impl PendingEmbedder {
    pub(super) const fn new(id: &'static str, dimension: usize) -> Self {
        Self { id, dimension }
    }
}

impl Embedder for PendingEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(std::future::pending())
    }

    fn dimension(&self) -> usize {
        self.dimension
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

// ─── Stub Lexical Search ────────────────────────────────────────────

pub(super) struct StubLexical;

impl LexicalRead for StubLexical {
    fn search<'a>(
        &'a self,
        _cx: &'a Cx,
        _query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            Ok((0..limit.min(3))
                .map(|i| ScoredResult {
                    doc_id: format!("lex-doc-{i}").into(),
                    score: (3 - i) as f32,
                    source: ScoreSource::Lexical,
                    index: None,
                    fast_score: None,
                    quality_score: None,
                    lexical_score: Some((3 - i) as f32),
                    rerank_score: None,
                    explanation: None,
                    metadata: None,
                })
                .collect())
        })
    }

    fn doc_count(&self) -> SearchResult<usize> {
        Ok(3)
    }
}

pub(super) struct DeferredMetadataLexical;

impl DeferredMetadataLexical {
    pub(super) fn results(limit: usize, metadata_path: Option<&str>) -> Vec<ScoredResult> {
        (0..limit.min(3))
            .map(|i| ScoredResult {
                doc_id: format!("doc-{i}").into(),
                score: (3 - i) as f32,
                source: ScoreSource::Lexical,
                index: None,
                fast_score: None,
                quality_score: None,
                lexical_score: Some((3 - i) as f32),
                rerank_score: None,
                explanation: None,
                metadata: metadata_path.map(|path| Arc::new(serde_json::json!({ "path": path }))),
            })
            .collect()
    }
}

impl LexicalRead for DeferredMetadataLexical {
    fn search<'a>(
        &'a self,
        _cx: &'a Cx,
        _query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move { Ok(Self::results(limit, Some("direct"))) })
    }

    fn search_candidates<'a>(
        &'a self,
        _cx: &'a Cx,
        _query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, frankensearch_core::LexicalCandidateBatch> {
        Box::pin(async move {
            Ok(frankensearch_core::LexicalCandidateBatch::deferred(
                Self::results(limit, None),
                LexicalHydrationContext::new("stub-deferred", Box::new(())),
            ))
        })
    }

    fn hydrate_candidates<'a>(
        &'a self,
        _cx: &'a Cx,
        context: Option<&'a LexicalHydrationContext>,
        results: &'a mut [ScoredResult],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            assert!(
                context.is_some_and(|context| context.backend() == "stub-deferred"),
                "hydration must receive the batch's own context"
            );
            for result in results
                .iter_mut()
                .filter(|result| result.lexical_score.is_some())
            {
                result.metadata = Some(Arc::new(serde_json::json!({ "path": "hydrated" })));
            }
            Ok(())
        })
    }

    fn doc_count(&self) -> SearchResult<usize> {
        Ok(3)
    }
}

pub(super) struct CancelledLexical;

impl LexicalRead for CancelledLexical {
    fn search<'a>(
        &'a self,
        _cx: &'a Cx,
        _query: &'a str,
        _limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async {
            Err(SearchError::Cancelled {
                phase: "lexical_search".to_owned(),
                reason: "test cancellation".to_owned(),
            })
        })
    }

    fn doc_count(&self) -> SearchResult<usize> {
        Ok(0)
    }
}

/// Index crafted so the quality tier disagrees with the fast tier:
/// `doc-c` is weak on cosine-fast (0.6) but exact on quality (1.0), so a
/// full-pool refinement must promote it past the Initial page boundary.
pub(super) fn build_promotion_index() -> Arc<TwoTierIndex> {
    let dir = std::env::temp_dir().join(format!(
        "frankensearch-searcher-promo-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("create index");
    builder.set_fast_embedder_id("stub-fast");
    builder.set_quality_embedder_id("stub-quality");
    let docs: [(&str, [f32; 4], [f32; 4]); 4] = [
        ("doc-a", [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]),
        ("doc-b", [0.95, 0.05, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]),
        ("doc-c", [0.6, 0.8, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
        ("doc-d", [0.5, 0.866, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]),
    ];
    for (id, fast, quality) in &docs {
        builder
            .add_fast_record(id.to_string(), fast)
            .expect("add fast record");
        builder
            .add_quality_record(id.to_string(), quality)
            .expect("add quality record");
    }
    Arc::new(builder.finish().expect("finish index"))
}

#[derive(Debug)]
pub(super) struct RecordingHostAdapter {
    pub(super) identity: AdapterIdentity,
    pub(super) telemetry: Mutex<Vec<TelemetryEnvelope>>,
    pub(super) lifecycle: Mutex<Vec<AdapterLifecycleEvent>>,
}

impl RecordingHostAdapter {
    pub(super) fn new(host_project: &str) -> Self {
        Self {
            identity: AdapterIdentity {
                adapter_id: "test-host-adapter".to_owned(),
                adapter_version: "1.0.0".to_owned(),
                host_project: host_project.to_owned(),
                runtime_role: Some("query".to_owned()),
                instance_uuid: Some("test-instance-uuid".to_owned()),
                telemetry_schema_version: 1,
                redaction_policy_version: "v1".to_owned(),
            },
            telemetry: Mutex::new(Vec::new()),
            lifecycle: Mutex::new(Vec::new()),
        }
    }

    pub(super) fn telemetry_events(&self) -> Vec<TelemetryEnvelope> {
        self.telemetry.lock().expect("telemetry lock").clone()
    }

    pub(super) fn lifecycle_events(&self) -> Vec<AdapterLifecycleEvent> {
        self.lifecycle.lock().expect("lifecycle lock").clone()
    }
}

impl HostAdapter for RecordingHostAdapter {
    fn identity(&self) -> AdapterIdentity {
        self.identity.clone()
    }

    fn emit_telemetry(&self, envelope: &TelemetryEnvelope) -> SearchResult<()> {
        self.telemetry
            .lock()
            .expect("telemetry lock")
            .push(envelope.clone());
        Ok(())
    }

    fn on_lifecycle_event(&self, event: &AdapterLifecycleEvent) -> SearchResult<()> {
        self.lifecycle
            .lock()
            .expect("lifecycle lock")
            .push(event.clone());
        Ok(())
    }
}

pub(super) fn assert_single_cancelled_session_stop(adapter: &RecordingHostAdapter, phase: &str) {
    let lifecycle_hooks = adapter.lifecycle_events();
    assert_eq!(
        lifecycle_hooks
            .iter()
            .filter(|event| matches!(event, AdapterLifecycleEvent::SessionStart { .. }))
            .count(),
        1,
        "the search must start exactly one host session"
    );
    assert_eq!(
        lifecycle_hooks
            .iter()
            .filter(|event| matches!(event, AdapterLifecycleEvent::SessionStop { .. }))
            .count(),
        1,
        "cancellation must terminalize the host session exactly once"
    );

    let lifecycle_events = adapter.telemetry_events();
    assert!(
        !lifecycle_events.iter().any(|envelope| {
            matches!(
                &envelope.event,
                TelemetryEvent::Lifecycle {
                    state: LifecycleState::Stopped,
                    ..
                }
            )
        }),
        "cancelled search must not also emit a normal stopped lifecycle event"
    );
    assert_eq!(
        lifecycle_events
            .iter()
            .filter(|envelope| {
                matches!(
                    &envelope.event,
                    TelemetryEvent::Lifecycle {
                        state: LifecycleState::Degraded,
                        severity: LifecycleSeverity::Warn,
                        reason: Some(reason),
                        ..
                    } if reason.starts_with(&format!("cancelled:{phase}:"))
                )
            })
            .count(),
        1,
        "cancellation must emit one typed terminal lifecycle event"
    );
}

#[derive(Debug, Default)]
pub(super) struct RecordingExporter {
    pub(super) search: Mutex<Vec<SearchMetrics>>,
    pub(super) embedding: Mutex<Vec<EmbeddingMetrics>>,
    pub(super) index: Mutex<Vec<IndexMetrics>>,
    pub(super) errors: Mutex<Vec<String>>,
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
            .expect("error metrics lock")
            .push(error.to_string());
    }
}

pub(super) fn build_test_index(dimension: usize) -> Arc<TwoTierIndex> {
    let dir = std::env::temp_dir().join(format!(
        "frankensearch-searcher-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("create index");
    builder.set_fast_embedder_id("stub-fast");
    for i in 0..10 {
        let mut vec = vec![0.0; dimension];
        vec[i % dimension] = 1.0;
        builder
            .add_fast_record(format!("doc-{i}"), &vec)
            .expect("add record");
    }
    Arc::new(builder.finish().expect("finish index"))
}

/// Build a test index with both fast and quality vectors.
pub(super) fn build_test_index_with_quality(dimension: usize) -> Arc<TwoTierIndex> {
    let dir = std::env::temp_dir().join(format!(
        "frankensearch-searcher-test-qual-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("create index");
    builder.set_fast_embedder_id("stub-fast");
    builder.set_quality_embedder_id("stub-quality");
    for i in 0..10 {
        let mut vec = vec![0.0; dimension];
        vec[i % dimension] = 1.0;
        builder
            .add_fast_record(format!("doc-{i}"), &vec)
            .expect("add fast record");
        builder
            .add_quality_record(format!("doc-{i}"), &vec)
            .expect("add quality record");
    }
    Arc::new(builder.finish().expect("finish index"))
}

// ─── Owner-backed activation fixtures (bd-ctzo) ─────────────────────

#[derive(Clone, Copy)]
pub(super) enum BoundResponseAction {
    Fail,
    Cancel,
    CancelAndFail,
    CancelAndReturnTypedError,
}

/// An embedder that declares a complete immutable identity AND counts
/// every inference it is asked for.
///
/// The count is the instrument: the whole claim of C1 is that a query in
/// the wrong space is refused BEFORE any work, and "before" is only
/// observable if the work is countable. A refusal that still embedded
/// would leave this at 1.
pub(super) struct IdentityCountingEmbedder {
    pub(super) id: &'static str,
    pub(super) identity: EmbeddingIdentityBundleV1,
    pub(super) vector: Vec<f32>,
    pub(super) embeds: Arc<AtomicU64>,
    pub(super) bound_embeds: AtomicU64,
    pub(super) bound_response_identity: Option<EmbeddingIdentityBundleV1>,
    pub(super) bound_response_values: Option<Vec<f32>>,
    pub(super) async_bound_response: bool,
    pub(super) identity_calls: Arc<AtomicU64>,
    pub(super) cancel_on_identity_call: Option<(Cx, u64)>,
    pub(super) replacement_identity: Option<EmbeddingIdentityBundleV1>,
    pub(super) identity_changed: std::sync::atomic::AtomicBool,
    pub(super) reported_dimension: Option<usize>,
    pub(super) identity_refusal: Option<&'static str>,
    pub(super) bound_action: Option<BoundResponseAction>,
}

impl IdentityCountingEmbedder {
    pub(super) fn new(
        id: &'static str,
        identity: EmbeddingIdentityBundleV1,
        vector: Vec<f32>,
    ) -> Self {
        Self {
            id,
            identity,
            vector,
            embeds: Arc::new(AtomicU64::new(0)),
            bound_embeds: AtomicU64::new(0),
            bound_response_identity: None,
            bound_response_values: None,
            async_bound_response: false,
            identity_calls: Arc::new(AtomicU64::new(0)),
            cancel_on_identity_call: None,
            replacement_identity: None,
            identity_changed: std::sync::atomic::AtomicBool::new(false),
            reported_dimension: None,
            identity_refusal: None,
            bound_action: None,
        }
    }

    pub(super) fn embed_count(&self) -> u64 {
        self.embeds.load(Ordering::Relaxed)
    }

    pub(super) fn with_bound_response_identity(
        mut self,
        identity: EmbeddingIdentityBundleV1,
    ) -> Self {
        self.bound_response_identity = Some(identity);
        self
    }

    pub(super) fn with_bound_response_values(mut self, values: Vec<f32>) -> Self {
        self.bound_response_values = Some(values);
        self
    }

    pub(super) fn with_async_bound_response(mut self) -> Self {
        self.async_bound_response = true;
        self
    }

    pub(super) fn cancel_on_identity_call(mut self, cx: Cx, call: u64) -> Self {
        self.cancel_on_identity_call = Some((cx, call));
        self
    }

    pub(super) fn identity_count(&self) -> u64 {
        self.identity_calls.load(Ordering::Relaxed)
    }
}

impl Embedder for IdentityCountingEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        self.embeds.fetch_add(1, Ordering::Relaxed);
        let vector = self.vector.clone();
        Box::pin(async move { Ok(vector) })
    }

    fn embed_bound<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, IdentityBoundEmbedding> {
        self.bound_embeds.fetch_add(1, Ordering::Relaxed);
        Box::pin(async move {
            if self.async_bound_response {
                let mut yielded = false;
                std::future::poll_fn(|task| {
                    if yielded {
                        std::task::Poll::Ready(())
                    } else {
                        yielded = true;
                        task.waker().wake_by_ref();
                        std::task::Poll::Pending
                    }
                })
                .await;
            }
            if let Some(action) = self.bound_action {
                if !matches!(action, BoundResponseAction::Fail) {
                    cx.cancel_with(
                        asupersync::CancelKind::User,
                        Some("bound response cancelled"),
                    );
                }
                if matches!(action, BoundResponseAction::CancelAndReturnTypedError) {
                    return Err(SearchError::UnverifiableRemoteSpace {
                        producer: "query-test".to_owned(),
                        reason: "provider refusal after cancellation".to_owned(),
                    });
                }
                if matches!(
                    action,
                    BoundResponseAction::Fail | BoundResponseAction::CancelAndFail
                ) {
                    return Err(SearchError::EmbeddingFailed {
                        model: self.id.to_owned(),
                        source: "provider failure".into(),
                    });
                }
            }
            let values = self.embed(cx, text).await?;
            let values = self.bound_response_values.clone().unwrap_or(values);
            let identity = self
                .bound_response_identity
                .as_ref()
                .map_or_else(|| self.identity().cloned(), |identity| Ok(identity.clone()))?;
            // Deliberately return provider output unvalidated: the consumer
            // must enforce its boundary even for custom trait overrides.
            Ok(IdentityBoundEmbedding { values, identity })
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        let call = self.identity_calls.fetch_add(1, Ordering::Relaxed) + 1;
        if let Some((cx, cancellation_call)) = &self.cancel_on_identity_call
            && call == *cancellation_call
        {
            cx.cancel_with(
                asupersync::CancelKind::User,
                Some("cancel after quality identity"),
            );
        }
        if let Some(field) = self.identity_refusal {
            return Err(SearchError::InvalidConfig {
                field: field.to_owned(),
                value: "unavailable".to_owned(),
                reason: "explicit test identity refusal".to_owned(),
            });
        }
        if (self.bound_embeds.load(Ordering::Relaxed) > 0
            || self.identity_changed.load(Ordering::Relaxed))
            && let Some(identity) = &self.replacement_identity
        {
            return Ok(identity);
        }
        Ok(&self.identity)
    }

    fn dimension(&self) -> usize {
        self.reported_dimension.unwrap_or(self.vector.len())
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
        if self.async_bound_response {
            ModelCategory::ApiEmbedder
        } else {
            ModelCategory::StaticEmbedder
        }
    }
}

/// The in-process f32 storage variant an embedder must declare.
pub(super) fn in_memory_identity(model_id: &str, dimension: u32) -> EmbeddingIdentityBundleV1 {
    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(model_id, dimension);
    "in-memory-f32".clone_into(&mut identity.storage.format);
    identity.storage.quantization = QuantizationFormat::F32;
    "native-f32-values".clone_into(&mut identity.storage.endianness);
    identity
}

/// The persisted fsvi-v2 variant of the SAME mathematical space. Query and
/// artifact bundles can never be byte-equal — one is in-process f32, the
/// other little-endian f16 on disk — which is exactly why the join is on
/// the space and the producer, not on the whole bundle.
pub(super) fn artifact_binding(
    model_id: &str,
    dimension: u32,
    sequence: u64,
) -> FsviV2IdentityBinding {
    let mut artifact = EmbeddingIdentityBundleV1::explicit_test_model(model_id, dimension);
    "fsvi-v2".clone_into(&mut artifact.storage.format);
    artifact.storage.quantization = QuantizationFormat::F16;
    "little-endian".clone_into(&mut artifact.storage.endianness);
    FsviV2IdentityBinding::new(
        ArtifactGenerationIdentityV1::new(sequence, [0x5c; 16]).expect("test generation"),
        artifact.freeze().expect("freeze artifact identity"),
    )
    .expect("valid FSVI v2 binding")
}

pub(super) fn owner_backed_dir(label: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "frankensearch-ctzo-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

pub(super) fn write_v2_tier(
    path: &std::path::Path,
    binding: &FsviV2IdentityBinding,
    rows: &[(&str, &[f32])],
) {
    let mut writer = frankensearch_index::VectorIndex::create_v2(path, binding.clone())
        .expect("create_v2 fixture");
    for (doc_id, vector) in rows {
        writer.write_record(doc_id, vector).expect("write v2 row");
    }
    writer.finish().expect("finish v2 fixture");
}

/// A fast-only index opened through exact FSVI v2 admission, so it retains
/// both the sealed owner and the binding it was admitted under.
pub(super) fn owner_backed_index(
    dir: &std::path::Path,
    binding: &FsviV2IdentityBinding,
) -> Arc<TwoTierIndex> {
    let fast_path = dir.join("vector.fast.idx");
    write_v2_tier(
        &fast_path,
        binding,
        &[
            ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
            ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
        ],
    );
    Arc::new(
        TwoTierIndex::open_admitted_v2_with_paths(
            &frankensearch_index::TwoTierIndexPaths::new(&fast_path),
            TwoTierConfig::default(),
            binding,
            None,
        )
        .expect("admit the fast tier"),
    )
}

/// A two-tier index opened through exact FSVI v2 admission where the
/// QUALITY tier holds a document the FAST tier does not contain at all.
pub(super) fn owner_backed_two_tier_index(
    dir: &std::path::Path,
    fast_binding: &FsviV2IdentityBinding,
    quality_binding: &FsviV2IdentityBinding,
) -> Arc<TwoTierIndex> {
    let fast_path = dir.join("vector.fast.idx");
    let quality_path = dir.join("vector.quality.idx");
    write_v2_tier(
        &fast_path,
        fast_binding,
        &[
            ("doc-near", &[1.0, 0.0, 0.0, 0.0]),
            ("doc-far", &[0.0, 0.0, 1.0, 0.0]),
        ],
    );
    write_v2_tier(
        &quality_path,
        quality_binding,
        &[
            ("doc-near", &[0.0, 0.0, 1.0, 0.0]),
            ("doc-far", &[0.0, 0.0, 0.5, 0.0]),
            ("doc-quality-only", &[0.0, 1.0, 0.0, 0.0]),
        ],
    );
    Arc::new(
        TwoTierIndex::open_admitted_v2_with_paths(
            &frankensearch_index::TwoTierIndexPaths::new(&fast_path)
                .with_quality_index(&quality_path),
            TwoTierConfig::default(),
            fast_binding,
            Some(quality_binding),
        )
        .expect("admit both tiers"),
    )
}

/// Reopen each production layout with independent 4D fast and 6D quality
/// spaces. The quality-only document cannot be discovered by fast rescoring.
pub(super) fn query_admission_index(layout: &str) -> Arc<TwoTierIndex> {
    let dir = owner_backed_dir(layout);
    let fast_rows: &[(&str, &[f32])] = &[
        ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
        ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
    ];
    let quality_rows: &[(&str, &[f32])] = &[
        ("doc-a", &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ("doc-b", &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        ("doc-quality-only", &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    ];
    if layout == "stamped-v1" {
        let mut writer = TwoTierIndex::create(&dir, TwoTierConfig::default()).unwrap();
        writer
            .set_fast_identity(&in_memory_identity("response-fast", 4))
            .unwrap();
        writer
            .set_quality_identity(&in_memory_identity("response-quality", 6))
            .unwrap();
        for (id, vector) in fast_rows {
            writer.add_fast_record(*id, vector).unwrap();
        }
        for (id, vector) in quality_rows {
            writer.add_quality_record(*id, vector).unwrap();
        }
        drop(writer.finish().unwrap());
        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).unwrap();
        assert!(index.fast_admitted_binding().is_none());
        assert!(!index.fast_embedder_revision().is_empty());
        return Arc::new(index);
    }
    let fast_binding = artifact_binding("response-fast", 4, 79);
    let quality_binding = artifact_binding("response-quality", 6, 79);
    let fast_path = dir.join("vector.fast.idx");
    let quality_path = dir.join("vector.quality.idx");
    write_v2_tier(&fast_path, &fast_binding, fast_rows);
    write_v2_tier(&quality_path, &quality_binding, quality_rows);
    let mut index = TwoTierIndex::open_admitted_v2_with_paths(
        &frankensearch_index::TwoTierIndexPaths::new(&fast_path).with_quality_index(&quality_path),
        TwoTierConfig::default(),
        &fast_binding,
        Some(&quality_binding),
    )
    .unwrap();
    if layout == "native-v2" {
        let params = frankensearch_index::native_hnsw::HnswParams::default();
        index.enable_native_fast_hnsw(params, 43).unwrap();
        index.enable_native_quality_hnsw(params, 47).unwrap();
    } else {
        assert_eq!(layout, "exact-v2");
    }
    Arc::new(index)
}

pub(super) fn query_admission_embedder(tier: &str) -> IdentityCountingEmbedder {
    if tier == "fast" {
        IdentityCountingEmbedder::new(
            "fast",
            in_memory_identity("response-fast", 4),
            vec![1.0, 0.0, 0.0, 0.0],
        )
    } else {
        IdentityCountingEmbedder::new(
            "quality",
            in_memory_identity("response-quality", 6),
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        )
    }
}

/// An owner-backed partial-quality fixture whose shared document has
/// deliberately different coordinates in the two embedding spaces.
pub(super) fn owner_backed_partial_quality_index(
    dir: &std::path::Path,
    fast_binding: &FsviV2IdentityBinding,
    quality_binding: &FsviV2IdentityBinding,
) -> Arc<TwoTierIndex> {
    let fast_path = dir.join("vector.fast.idx");
    let quality_path = dir.join("vector.quality.idx");
    write_v2_tier(
        &fast_path,
        fast_binding,
        &[
            ("doc-shared", &[1.0, 0.0, 0.0, 0.0]),
            ("doc-fast-only", &[0.0, 1.0, 0.0, 0.0]),
        ],
    );
    write_v2_tier(
        &quality_path,
        quality_binding,
        &[
            ("doc-shared", &[0.0, 1.0, 0.0, 0.0]),
            ("doc-quality-query", &[1.0, 0.0, 0.0, 0.0]),
            ("doc-quality-wrong", &[0.6, 0.8, 0.0, 0.0]),
        ],
    );
    Arc::new(
        TwoTierIndex::open_admitted_v2_with_paths(
            &frankensearch_index::TwoTierIndexPaths::new(&fast_path)
                .with_quality_index(&quality_path),
            TwoTierConfig::default(),
            fast_binding,
            Some(quality_binding),
        )
        .expect("admit partial-quality tiers"),
    )
}

/// Convert a `ParsedQuery` to `NormalizedExclusions` for testing.
pub(super) fn to_exclusions(parsed: &ParsedQuery) -> NormalizedExclusions {
    NormalizedExclusions {
        terms: parsed
            .negative_terms
            .iter()
            .map(|t| normalize_for_negation_match(t))
            .collect(),
        phrases: parsed
            .negative_phrases
            .iter()
            .map(|p| normalize_for_negation_match(p))
            .collect(),
    }
}

// ─── Counting Embedder (for cache-wiring tests) ────────────────────

/// Embedder that counts inner `embed()` invocations via an external counter.
pub(super) struct CountingEmbedder {
    pub(super) id: &'static str,
    pub(super) dimension: usize,
    pub(super) calls: Arc<std::sync::atomic::AtomicUsize>,
}

impl CountingEmbedder {
    pub(super) fn new(
        id: &'static str,
        dimension: usize,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    ) -> Self {
        Self {
            id,
            dimension,
            calls,
        }
    }
}

impl Embedder for CountingEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dim = self.dimension;
        Box::pin(async move {
            let mut vec = vec![0.0; dim];
            if !vec.is_empty() {
                vec[0] = 1.0;
            }
            Ok(vec)
        })
    }

    fn dimension(&self) -> usize {
        self.dimension
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

/// Successful quality embedder that cancels its real invocation context
/// immediately before returning a controlled vector.
pub(super) struct CancellingEmbedder {
    pub(super) id: &'static str,
    pub(super) dimension: usize,
    pub(super) vector: Vec<f32>,
    pub(super) calls: Arc<std::sync::atomic::AtomicUsize>,
}

impl CancellingEmbedder {
    pub(super) fn new(
        id: &'static str,
        dimension: usize,
        vector: Vec<f32>,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    ) -> Self {
        Self {
            id,
            dimension,
            vector,
            calls,
        }
    }
}

impl Embedder for CancellingEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let vector = self.vector.clone();
        Box::pin(async move {
            cx.cancel_with(
                asupersync::CancelKind::User,
                Some("cancel after quality embed"),
            );
            Ok(vector)
        })
    }

    fn dimension(&self) -> usize {
        self.dimension
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

#[cfg(feature = "rerank")]
pub(super) struct CountingReranker {
    pub(super) calls: Arc<std::sync::atomic::AtomicUsize>,
    pub(super) cancel_after_scores: bool,
}

#[cfg(feature = "rerank")]
impl CountingReranker {
    pub(super) fn new(calls: Arc<std::sync::atomic::AtomicUsize>) -> Self {
        Self {
            calls,
            cancel_after_scores: false,
        }
    }

    pub(super) fn cancelling(calls: Arc<std::sync::atomic::AtomicUsize>) -> Self {
        Self {
            calls,
            cancel_after_scores: true,
        }
    }
}

#[cfg(feature = "rerank")]
impl Reranker for CountingReranker {
    fn rerank<'a>(
        &'a self,
        cx: &'a Cx,
        _query: &'a str,
        docs: &'a [frankensearch_core::traits::RerankDocument],
    ) -> SearchFuture<'a, Vec<frankensearch_core::traits::RerankScore>> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let cancel_after_scores = self.cancel_after_scores;
        let scores = docs
            .iter()
            .enumerate()
            .map(
                |(original_rank, doc)| frankensearch_core::traits::RerankScore {
                    doc_id: doc.doc_id.clone(),
                    score: if cancel_after_scores {
                        // Ascending scores would reverse the reranked
                        // window if mutation proceeded after cancellation.
                        original_rank as f32
                    } else {
                        1.0 - original_rank as f32 * 0.1
                    },
                    original_rank,
                    raw_logit: None,
                },
            )
            .collect();
        Box::pin(async move {
            if cancel_after_scores {
                cx.cancel_with(
                    asupersync::CancelKind::User,
                    Some("cancel after reranker scores"),
                );
            }
            Ok(scores)
        })
    }

    fn id(&self) -> &str {
        "counting-reranker"
    }

    fn model_name(&self) -> &str {
        "counting-reranker"
    }
}
