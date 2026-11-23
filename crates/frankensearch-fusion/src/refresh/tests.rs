use std::collections::HashMap;
pub(super) use std::path::{Path, PathBuf};
pub(super) use std::sync::Arc;
pub(super) use std::sync::atomic::Ordering;
pub(super) use std::time::{Duration, Instant};

pub(super) use asupersync::Cx;

pub(super) use frankensearch_core::canonicalize::DefaultCanonicalizer;
pub(super) use frankensearch_core::config::TwoTierConfig;
pub(super) use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::ArtifactGenerationIdentityV1;
pub(super) use frankensearch_core::generation::{EmbeddingIdentityBundleV1, QuantizationFormat};
pub(super) use frankensearch_core::traits::{
    Embedder, IdentityBoundEmbedding, ModelCategory, SearchFuture,
};
pub(super) use frankensearch_core::{BoundQueryEmbedding, TieredQueryEmbeddings};
use frankensearch_index::VECTOR_INDEX_FALLBACK_FILENAME;
pub(super) use frankensearch_index::two_tier::{FsviTierObservation, observe_tier};
pub(super) use frankensearch_index::{
    FsviV2IdentityBinding, TwoTierIndex, TwoTierIndexPaths, VECTOR_INDEX_FAST_FILENAME,
    VECTOR_INDEX_QUALITY_FILENAME, ValidatedFsviBytes, VectorIndex,
};

pub(super) use crate::cache::{IndexCache, SentinelFileDetector};
pub(super) use crate::queue::{
    EmbeddingJob, EmbeddingQueue, EmbeddingQueueConfig, EmbeddingRequest, JobOutcome,
};

pub(super) use super::admission::{STAGED_V2_DIR_NAME, fingerprint_hex};
pub(super) use super::{
    RefreshMetrics, RefreshMetricsSnapshot, RefreshWorker, RefreshWorkerConfig,
};

mod classification_tests;
mod cycle_tests;
mod lease_tests;
mod staging_tests;

// -- Stub embedders for tests ----------------------------------------------

pub(super) struct StubEmbedder {
    id: &'static str,
    dimension: usize,
    space_offset: f32,
    identity: EmbeddingIdentityBundleV1,
    /// Direct spy counters (r3 audit item 3, #8366/#8367): every
    /// `embed` / `embed_batch` invocation is counted at call time, so
    /// refusal tests can assert ZERO embed-class work directly instead
    /// of inferring it from worker metrics.
    embed_calls: std::sync::atomic::AtomicUsize,
    embed_batch_calls: std::sync::atomic::AtomicUsize,
}

impl StubEmbedder {
    pub(super) fn new(id: &'static str, dimension: usize) -> Self {
        Self::in_space(id, dimension, 0.0)
    }

    /// Total direct `embed*`/`embed_batch*` invocations observed.
    ///
    /// The trait's `embed_bound` / `embed_batch_bound` defaults
    /// delegate into `embed` / `embed_batch`, so a ZERO here proves no
    /// embed-class method performed (or even started) any inference.
    /// Counting happens synchronously at invocation, before the
    /// returned future is polled, so an un-awaited call still counts.
    pub(super) fn embed_invocations(&self) -> usize {
        self.embed_calls.load(Ordering::SeqCst) + self.embed_batch_calls.load(Ordering::SeqCst)
    }

    /// Same display id, different `space_offset` = a genuinely different
    /// vector space AND a different typed space identity: the identity
    /// model key embeds the offset, mirroring how a real model change
    /// alters the immutable space identity even when the display id
    /// stays the same.
    pub(super) fn in_space(id: &'static str, dimension: usize, space_offset: f32) -> Self {
        let identity = EmbeddingIdentityBundleV1::explicit_test_model(
            &format!("{id}#space-{space_offset}"),
            u32::try_from(dimension).expect("test dimension fits u32"),
        );
        Self {
            id,
            dimension,
            space_offset,
            identity,
            embed_calls: std::sync::atomic::AtomicUsize::new(0),
            embed_batch_calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Stub with an explicit identity bundle (foreign-producer fixtures).
    pub(super) fn with_identity(
        id: &'static str,
        dimension: usize,
        identity: EmbeddingIdentityBundleV1,
    ) -> Self {
        Self {
            id,
            dimension,
            space_offset: 0.0,
            identity,
            embed_calls: std::sync::atomic::AtomicUsize::new(0),
            embed_batch_calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub(super) fn identity_bundle(&self) -> &EmbeddingIdentityBundleV1 {
        &self.identity
    }
}

impl Embedder for StubEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        self.embed_calls.fetch_add(1, Ordering::SeqCst);
        let dim = self.dimension;
        let seed = text.len() as f32 + self.space_offset;
        Box::pin(async move { Ok((0..dim).map(|i| (seed + i as f32).sin()).collect()) })
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        self.embed_batch_calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            let mut out = Vec::with_capacity(texts.len());
            for text in texts {
                out.push(self.embed(cx, text).await?);
            }
            Ok(out)
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn id(&self) -> &str {
        self.id
    }

    fn model_name(&self) -> &str {
        self.id
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn is_semantic(&self) -> bool {
        false
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
}

pub(super) struct FailingEmbedder;

impl Embedder for FailingEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async {
            Err(SearchError::EmbeddingFailed {
                model: "failing-embedder".into(),
                source: Box::new(std::io::Error::other("intentional failure")),
            })
        })
    }

    fn id(&self) -> &'static str {
        "failing-embedder"
    }

    fn model_name(&self) -> &'static str {
        "failing-embedder"
    }

    fn dimension(&self) -> usize {
        256
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn is_semantic(&self) -> bool {
        false
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
}

/// An embedder whose bound outputs carry a DIFFERENT (but
/// conformance-certified sibling) producer than its own `identity()` —
/// the hostile shape the per-record seam check exists for.
pub(super) struct TwoFacedEmbedder {
    pub(super) public_identity: EmbeddingIdentityBundleV1,
    pub(super) bound_identity: EmbeddingIdentityBundleV1,
    pub(super) dimension: usize,
}

impl Embedder for TwoFacedEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        let dim = self.dimension;
        let seed = text.len() as f32;
        Box::pin(async move { Ok((0..dim).map(|i| (seed + i as f32).sin()).collect()) })
    }

    fn embed_batch_bound<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            let vectors = self.embed_batch(cx, texts).await?;
            vectors
                .into_iter()
                .map(|values| {
                    let bound = IdentityBoundEmbedding {
                        values,
                        identity: self.bound_identity.clone(),
                    };
                    bound.validate()?;
                    Ok(bound)
                })
                .collect()
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.public_identity)
    }

    fn id(&self) -> &'static str {
        "two-faced-embedder"
    }

    fn model_name(&self) -> &'static str {
        "two-faced-embedder"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn is_semantic(&self) -> bool {
        false
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
}

// -- Test helpers ----------------------------------------------------------

pub(super) fn make_queue(capacity: usize) -> Arc<EmbeddingQueue> {
    Arc::new(EmbeddingQueue::new(
        EmbeddingQueueConfig {
            capacity,
            batch_size: 100,
            max_retries: 3,
        },
        Box::new(DefaultCanonicalizer::default()),
    ))
}

pub(super) fn submit(queue: &EmbeddingQueue, doc_id: &str, text: &str) {
    queue
        .submit(EmbeddingRequest {
            doc_id: doc_id.into(),
            text: text.to_owned(),
            metadata: None,
            submitted_at: Instant::now(),
        })
        .unwrap();
}

/// Create a temporary directory with a unique name.
pub(super) fn temp_index_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "frankensearch-refresh-test-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Seed an initial index on disk (required for `IndexCache::open`).
fn seed_index(dir: &Path, dimension: usize) -> TwoTierIndex {
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let writer = VectorIndex::create(&fast_path, "stub-fast", dimension).unwrap();
    writer.finish().unwrap();
    TwoTierIndex::open(dir, TwoTierConfig::default()).unwrap()
}

/// Seed a fast+quality index pair so tests can exercise quality-tier
/// refresh without tripping the post-bd-8utj topology-change refusal.
fn seed_two_tier_index(
    dir: &Path,
    fast_dimension: usize,
    quality_dimension: usize,
) -> TwoTierIndex {
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    VectorIndex::create(&fast_path, "stub-fast", fast_dimension)
        .unwrap()
        .finish()
        .unwrap();
    let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
    VectorIndex::create(&quality_path, "stub-quality", quality_dimension)
        .unwrap()
        .finish()
        .unwrap();
    TwoTierIndex::open(dir, TwoTierConfig::default()).unwrap()
}

pub(super) fn make_cache(dir: &Path, dimension: usize) -> Arc<IndexCache> {
    seed_index(dir, dimension);
    let detector = Box::new(SentinelFileDetector::new());
    Arc::new(IndexCache::open(dir, TwoTierConfig::default(), detector).unwrap())
}

pub(super) fn make_cache_with_quality(
    dir: &Path,
    fast_dimension: usize,
    quality_dimension: usize,
) -> Arc<IndexCache> {
    seed_two_tier_index(dir, fast_dimension, quality_dimension);
    let detector = Box::new(SentinelFileDetector::new());
    Arc::new(IndexCache::open(dir, TwoTierConfig::default(), detector).unwrap())
}

/// Seed the cache from an empty v1 FALLBACK artifact (`vector.idx`), so
/// a v2 canonical fast artifact (`vector.fast.idx`) can be installed
/// afterwards without breaking the (v1-only) cache open.
pub(super) fn make_cache_with_fallback_seed(dir: &Path, dimension: usize) -> Arc<IndexCache> {
    let seed_path = dir.join(VECTOR_INDEX_FALLBACK_FILENAME);
    VectorIndex::create(&seed_path, "stub-fast", dimension)
        .unwrap()
        .finish()
        .unwrap();
    let detector = Box::new(SentinelFileDetector::new());
    Arc::new(IndexCache::open(dir, TwoTierConfig::default(), detector).unwrap())
}

pub(super) fn make_worker(
    queue: Arc<EmbeddingQueue>,
    dir: &Path,
    dimension: usize,
) -> (RefreshWorker, Arc<IndexCache>, Arc<StubEmbedder>) {
    let cache = make_cache(dir, dimension);
    let config = RefreshWorkerConfig::new(dir).with_poll_interval(Duration::from_millis(10));
    let fast = Arc::new(StubEmbedder::new("stub-fast", dimension));
    let worker = RefreshWorker::new(config, queue, fast.clone(), cache.clone());
    (worker, cache, fast)
}

/// r3 audit item 3 (#8366/#8367): direct proof that a refusing call
/// performed zero `embed*`/`embed_batch*` work — the counting spy on
/// [`StubEmbedder`] replaces inference from worker metrics.
#[track_caller]
pub(super) fn assert_no_embed_invocations_since(embedder: &StubEmbedder, baseline: usize) {
    assert_eq!(
        embedder.embed_invocations(),
        baseline,
        "a refusal path must not invoke embed*/embed_batch* on the embedder"
    );
}

/// Pairwise certified-conformance shape (same space, byte-identical
/// pinned golden-vector certificate): the predicate the frozen r3 chain
/// exposed as `EmbeddingIdentityBundleV1::is_conformance_compatible_with`,
/// inlined here because the trunk's generation API superseded the
/// pairwise method with the witness/certificate flow.
pub(super) fn conformance_compatible(
    a: &EmbeddingIdentityBundleV1,
    b: &EmbeddingIdentityBundleV1,
) -> bool {
    a.validate().is_ok()
        && b.validate().is_ok()
        && a.space.fingerprint() == b.space.fingerprint()
        && a.producer.golden_vectors == b.producer.golden_vectors
}

pub(super) fn normalized(dim: usize, seed: f32) -> Vec<f32> {
    let raw: Vec<f32> = (0..dim).map(|i| seed + 1.0 + i as f32).collect();
    let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    raw.iter().map(|x| x / norm).collect()
}

/// The fsvi-v2 artifact-storage variant of an in-memory identity bundle
/// (what `artifact_identity_for` derives for an embedder).
fn artifact_variant(bundle: &EmbeddingIdentityBundleV1) -> EmbeddingIdentityBundleV1 {
    let mut artifact = bundle.clone();
    "fsvi-v2".clone_into(&mut artifact.storage.format);
    artifact.storage.quantization = QuantizationFormat::F16;
    "little-endian".clone_into(&mut artifact.storage.endianness);
    artifact
}

pub(super) fn v2_binding(
    bundle_in_memory: &EmbeddingIdentityBundleV1,
    sequence: u64,
) -> FsviV2IdentityBinding {
    let artifact = artifact_variant(bundle_in_memory);
    let generation =
        ArtifactGenerationIdentityV1::new(sequence, [0x3b; 16]).expect("test generation");
    FsviV2IdentityBinding::new(generation, artifact.freeze().expect("freeze artifact"))
        .expect("valid binding")
}

pub(super) fn write_v2_tier(
    path: &Path,
    binding: &FsviV2IdentityBinding,
    rows: &[(&str, Vec<f32>)],
) {
    write_v2_tier_with_nonce(path, binding, rows, 0);
}

pub(super) fn write_v2_tier_with_nonce(
    path: &Path,
    binding: &FsviV2IdentityBinding,
    rows: &[(&str, Vec<f32>)],
    nonce: u16,
) {
    let mut writer = VectorIndex::create_v2(path, binding.clone())
        .expect("create_v2 fixture")
        .with_publication_nonce(nonce);
    for (doc_id, vector) in rows {
        writer.write_record(doc_id, vector).expect("write v2 row");
    }
    writer.finish().expect("finish v2 fixture");
}

pub(super) fn admitted_vectors_by_doc(owner: &ValidatedFsviBytes) -> HashMap<String, Vec<f32>> {
    let mut out = HashMap::new();
    for i in 0..owner.record_count() {
        let row = owner.row(i).expect("row");
        if !row.flags().is_live() {
            continue;
        }
        out.insert(
            row.doc_id().to_owned(),
            owner.vector_at_f32(i).expect("vector"),
        );
    }
    out
}

/// Assert a typed `InvalidConfig` refusal with the exact field and value.
#[track_caller]
pub(super) fn assert_invalid_config(
    error: &SearchError,
    expected_field: &str,
    expected_value: &str,
) {
    assert!(
        matches!(
            error,
            SearchError::InvalidConfig { field, value, .. }
                if field == expected_field && value == expected_value
        ),
        "expected InvalidConfig {{ field: {expected_field:?}, value: {expected_value:?} }}, \
         got {error:?}"
    );
}

/// Reason string of a typed `InvalidConfig` refusal (empty for other
/// variants; call after [`assert_invalid_config`]).
pub(super) fn invalid_config_reason(error: &SearchError) -> &str {
    match error {
        SearchError::InvalidConfig { reason, .. } => reason.as_str(),
        _ => "",
    }
}

// ─── bd-9xuj T2 C4-write red proofs ───────────────────────────────

pub(super) const V2_DIM: usize = 8;

/// Fixture: cache seeded from an empty v1 FALLBACK artifact, then a live
/// attested v2 canonical fast generation written under `bundle`.
pub(super) fn v2_canonical_fixture(
    label: &str,
    bundle: &EmbeddingIdentityBundleV1,
    rows: &[(&str, Vec<f32>)],
    sequence: u64,
) -> (PathBuf, Arc<IndexCache>, FsviV2IdentityBinding, PathBuf) {
    let dir = temp_index_dir(label);
    let cache = make_cache_with_fallback_seed(&dir, V2_DIM);
    let binding = v2_binding(bundle, sequence);
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_v2_tier(&fast_path, &binding, rows);
    (dir, cache, binding, fast_path)
}
