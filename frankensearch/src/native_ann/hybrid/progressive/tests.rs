use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1,
    EmbeddingSpaceKindV1, QuantizationFormat,
};
use frankensearch_core::traits::{IdentityBoundEmbedding, ModelCategory, Reranker, SearchFuture};
use frankensearch_core::{LexicalCandidateBatch, LexicalHydrationContext};
use frankensearch_index::native_hnsw::HnswParams;
use frankensearch_index::{FsviV2IdentityBinding, ValidatedFsviBytes, VectorIndex};

use crate::native_ann::{NativeAnnIndex, invalid};
use crate::{Cx, Embedder, LexicalRead, ScoreSource, ScoredResult, SearchError, SearchResult};

mod refinement;
mod rerank_stage;

#[derive(Clone, Copy)]
enum Reply {
    Correct,
    Failed,
    Cancelled,
    Pending,
    ForeignProducer,
}

struct Provider {
    identity: EmbeddingIdentityBundleV1,
    foreign_identity: EmbeddingIdentityBundleV1,
    advertise_foreign: AtomicBool,
    calls: AtomicUsize,
    drops: AtomicUsize,
    reply: Reply,
}

impl Provider {
    fn new(name: &str, dimension: u32) -> Self {
        // Synthetic semantic-shaped fixture, not evidence of a real model.
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(name, dimension);
        identity.space.kind = EmbeddingSpaceKindV1::Semantic;
        identity.space.hash_control = None;
        identity.space.artifact_manifest_fingerprint = "a".repeat(64);
        identity.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
            role: "weights".to_owned(),
            sha256: "b".repeat(64),
            size: 1,
        }];
        identity.producer.space_fingerprint = identity.space.fingerprint();
        identity.validate().unwrap();
        let mut foreign_identity = identity.clone();
        foreign_identity.producer.backend = "foreign-backend".to_owned();
        Self {
            identity,
            foreign_identity,
            advertise_foreign: AtomicBool::new(false),
            calls: AtomicUsize::new(0),
            drops: AtomicUsize::new(0),
            reply: Reply::Correct,
        }
    }
}

struct DropCount<'a>(&'a AtomicUsize);

impl Drop for DropCount<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl Embedder for Provider {
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            let mut values = vec![0.0; self.dimension()];
            values[0] = 1.0;
            Ok(values)
        })
    }

    fn embed_bound<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, IdentityBoundEmbedding> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let _guard = DropCount(&self.drops);
            let identity = match self.reply {
                Reply::Correct => self.identity.clone(),
                Reply::ForeignProducer => self.foreign_identity.clone(),
                Reply::Failed => {
                    return Err(invalid("test.provider", "failed", "provider failed"));
                }
                Reply::Cancelled => {
                    return Err(SearchError::Cancelled {
                        phase: "test.provider".to_owned(),
                        reason: "provider cancelled".to_owned(),
                    });
                }
                Reply::Pending => return std::future::pending().await,
            };
            Ok(IdentityBoundEmbedding {
                values: self.embed(cx, text).await?,
                identity,
            })
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(if self.advertise_foreign.load(Ordering::SeqCst) {
            &self.foreign_identity
        } else {
            &self.identity
        })
    }

    fn id(&self) -> &str {
        &self.identity.space.logical_model_id
    }

    fn model_name(&self) -> &str {
        self.id()
    }

    fn dimension(&self) -> usize {
        usize::try_from(self.identity.space.dimension).unwrap()
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::TransformerEmbedder
    }
}

struct Lexical {
    ids: Vec<&'static str>,
    calls: AtomicUsize,
    hydration_calls: AtomicUsize,
    hydration_drops: AtomicUsize,
    snapshot: Arc<usize>,
    current_generation: AtomicUsize,
    pending_search: bool,
    refinement_hydration: Reply,
}

impl Lexical {
    fn new(ids: &[&'static str]) -> Self {
        Self {
            ids: ids.to_vec(),
            calls: AtomicUsize::new(0),
            hydration_calls: AtomicUsize::new(0),
            hydration_drops: AtomicUsize::new(0),
            snapshot: Arc::new(1),
            current_generation: AtomicUsize::new(1),
            pending_search: false,
            refinement_hydration: Reply::Correct,
        }
    }
}

impl LexicalRead for Lexical {
    fn search<'a>(
        &'a self,
        _cx: &'a Cx,
        _text: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.pending_search {
                std::future::pending::<()>().await;
            }
            Ok(self
                .ids
                .iter()
                .take(limit)
                .map(|id| ScoredResult {
                    doc_id: (*id).into(),
                    score: 10.0,
                    source: ScoreSource::Lexical,
                    index: None,
                    fast_score: None,
                    quality_score: None,
                    lexical_score: Some(10.0),
                    rerank_score: None,
                    explanation: None,
                    metadata: None,
                })
                .collect())
        })
    }

    fn search_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, LexicalCandidateBatch> {
        Box::pin(async move {
            let results = self.search(cx, text, limit).await?;
            self.current_generation.store(2, Ordering::SeqCst);
            Ok(LexicalCandidateBatch::deferred(
                results,
                LexicalHydrationContext::new(
                    "native-progressive-test",
                    Box::new(Arc::clone(&self.snapshot)),
                ),
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
            let call = self.hydration_calls.fetch_add(1, Ordering::SeqCst);
            let _guard = DropCount(&self.hydration_drops);
            let pin = context
                .and_then(LexicalHydrationContext::downcast_ref::<Arc<usize>>)
                .filter(|pin| Arc::ptr_eq(pin, &self.snapshot))
                .ok_or_else(|| invalid("test.snapshot", "foreign", "wrong snapshot"))?;
            for result in results {
                if result.lexical_score.is_some() {
                    result.metadata = Some(Arc::new(serde_json::json!({"generation": **pin})));
                }
            }
            if call > 0 {
                match self.refinement_hydration {
                    Reply::Pending => return std::future::pending().await,
                    Reply::Failed => {
                        return Err(invalid("test.hydration", "failed", "partial hydration"));
                    }
                    Reply::Cancelled => {
                        return Err(SearchError::Cancelled {
                            phase: "test.hydration".to_owned(),
                            reason: "cancelled after writing partial metadata".to_owned(),
                        });
                    }
                    Reply::Correct | Reply::ForeignProducer => {}
                }
            }
            Ok(())
        })
    }

    fn doc_count(&self) -> SearchResult<usize> {
        Ok(self.ids.len())
    }
}

fn generation() -> ArtifactGenerationIdentityV1 {
    ArtifactGenerationIdentityV1::new(1, [0x61; 16]).unwrap()
}

fn native_index(
    cx: &Cx,
    provider: &Provider,
    generation: ArtifactGenerationIdentityV1,
    rows: &[(&str, &[f32])],
) -> NativeAnnIndex {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tier.fsvi");
    let mut bundle = provider.identity.clone();
    bundle.storage.format = "fsvi-v2".to_owned();
    bundle.storage.quantization = QuantizationFormat::F32;
    bundle.storage.endianness = "little-endian".to_owned();
    let binding = FsviV2IdentityBinding::new(generation, bundle.freeze().unwrap()).unwrap();
    let mut writer = VectorIndex::create_v2(&path, binding.clone()).unwrap();
    for &(id, values) in rows {
        writer.write_record(id, values).unwrap();
    }
    writer.finish().unwrap();
    let bytes: Arc<[u8]> = std::fs::read(path).unwrap().into();
    let owner = Arc::new(ValidatedFsviBytes::from_arc(bytes, &binding).unwrap());
    NativeAnnIndex::build(cx, owner, HnswParams::default(), 7).unwrap()
}

fn fast_index(cx: &Cx, provider: &Provider) -> NativeAnnIndex {
    native_index(
        cx,
        provider,
        generation(),
        &[
            ("z-fast", &[1.0, 0.0]),
            ("middle", &[0.8, 0.6]),
            ("near", &[0.6, 0.8]),
            ("a-quality", &[0.0, 1.0]),
        ],
    )
}

fn quality_index(cx: &Cx, provider: &Provider) -> NativeAnnIndex {
    native_index(
        cx,
        provider,
        generation(),
        &[
            ("a-quality", &[1.0, 0.0, 0.0]),
            ("near", &[0.0, 1.0, 0.0]),
            ("middle", &[0.0, 0.0, 1.0]),
            ("z-fast", &[-1.0, 0.0, 0.0]),
        ],
    )
}

#[derive(Clone, Copy)]
enum RerankReply {
    Winner,
    Failed,
    Pending,
    Cancelled,
    CancelWithScores,
    DuplicateRank,
    ForeignId,
    OutOfRange,
    MissingScore,
    NonFinite,
    NonFiniteLogit,
    Unavailable,
}

struct CrossEncoder {
    reply: RerankReply,
    calls: AtomicUsize,
    drops: AtomicUsize,
}

impl CrossEncoder {
    fn new(reply: RerankReply) -> Self {
        Self {
            reply,
            calls: AtomicUsize::new(0),
            drops: AtomicUsize::new(0),
        }
    }
}

impl Reranker for CrossEncoder {
    fn rerank<'a>(
        &'a self,
        cx: &'a Cx,
        _query: &'a str,
        documents: &'a [frankensearch_core::traits::RerankDocument],
    ) -> SearchFuture<'a, Vec<frankensearch_core::traits::RerankScore>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let _guard = DropCount(&self.drops);
            let mut scores: Vec<_> = documents
                .iter()
                .enumerate()
                .map(|(rank, doc)| {
                    assert_eq!(doc.text, format!("version-one:{}", doc.doc_id));
                    frankensearch_core::traits::RerankScore {
                        doc_id: doc.doc_id.clone(),
                        score: if doc.doc_id == "near" { 0.9 } else { 0.1 },
                        original_rank: rank,
                        raw_logit: None,
                    }
                })
                .collect();
            match self.reply {
                RerankReply::Winner => {}
                RerankReply::Pending => return std::future::pending().await,
                RerankReply::Failed => {
                    return Err(invalid("test.reranker", "failed", "model failed"));
                }
                RerankReply::Cancelled => {
                    return Err(SearchError::Cancelled {
                        phase: "test.reranker".to_owned(),
                        reason: "model cancelled".to_owned(),
                    });
                }
                RerankReply::CancelWithScores => {
                    cx.cancel_with(
                        asupersync::CancelKind::User,
                        Some("native rerank finished after cancellation"),
                    );
                }
                RerankReply::DuplicateRank => scores[1] = scores[0].clone(),
                RerankReply::ForeignId => scores[0].doc_id = "foreign".to_owned(),
                RerankReply::OutOfRange => scores[0].original_rank = documents.len(),
                RerankReply::MissingScore => {
                    let _ = scores.pop();
                }
                RerankReply::NonFinite => scores[0].score = f32::INFINITY,
                RerankReply::NonFiniteLogit => scores[0].raw_logit = Some(f32::NAN),
                RerankReply::Unavailable => panic!("unavailable model must not run"),
            }
            // The admission boundary must use original_rank, not output order.
            scores.reverse();
            Ok(scores)
        })
    }

    // `Reranker::id` declares `-> &str`; a trait impl cannot narrow that to
    // `&'static str`, which is what clippy's suggestion would do.
    #[allow(clippy::unnecessary_literal_bound)]
    fn id(&self) -> &str {
        "native-test-cross-encoder"
    }
    fn model_name(&self) -> &str {
        self.id()
    }
    fn is_available(&self) -> bool {
        !matches!(self.reply, RerankReply::Unavailable)
    }
}

// Fed to with_reranker as `&dyn Fn(&str) -> Option<String>`, so the Option
// is required by the callback type rather than incidental.
#[allow(clippy::unnecessary_wraps)]
fn source_text(id: &str) -> Option<String> {
    Some(format!("version-one:{id}"))
}
