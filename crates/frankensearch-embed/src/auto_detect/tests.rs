#[cfg(all(
    feature = "hash",
    any(feature = "bundled-default-models", feature = "model2vec")
))]
use std::fs;
#[cfg(all(
    feature = "hash",
    any(feature = "bundled-default-models", feature = "model2vec")
))]
use std::path::Path;
#[cfg(all(feature = "bundled-default-models", feature = "hash"))]
use std::path::PathBuf;
use std::sync::Arc;

use asupersync::Cx;
#[cfg(all(feature = "download", feature = "model2vec"))]
use asupersync::test_utils::run_test_with_cx;

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::traits::{Embedder, ModelCategory, SearchFuture};

#[cfg(all(feature = "bundled-default-models", feature = "hash"))]
use crate::fastembed_embedder::DEFAULT_MODEL_NAME as MINILM_MODEL_NAME;
#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
use crate::model_download::DownloadProgress;
#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
use crate::model_manifest::{ConsentSource, DownloadConsent};

#[cfg(all(feature = "model2vec", feature = "hash"))]
use super::discovery::POTION_MODEL_NAME;
#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
use super::discovery::parse_bool_flag;
#[cfg(all(feature = "download", feature = "model2vec"))]
use super::lazy_download::LazyModel2VecEmbedder;
#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
use super::lazy_download::{
    DownloadPolicy, PROGRESS_BAR_WIDTH, format_bytes, format_eta, format_speed,
    progress_percent_x100, render_progress_bar,
};
use super::*;

#[cfg(all(feature = "bundled-default-models", feature = "hash"))]
#[derive(Debug, PartialEq, Eq)]
struct ObservedModelEntry {
    file_kind: u8,
    len: u64,
    modified: std::time::SystemTime,
    created: Option<std::time::SystemTime>,
    changed: Option<(i64, i64)>,
    permission_key: u64,
    bytes: Option<Vec<u8>>,
}

#[cfg(all(feature = "bundled-default-models", feature = "hash"))]
fn snapshot_model_tree(root: &Path) -> Vec<(PathBuf, ObservedModelEntry)> {
    fn visit(root: &Path, path: &Path, entries: &mut Vec<(PathBuf, ObservedModelEntry)>) {
        let mut children = fs::read_dir(path)
            .expect("read model tree")
            .map(|entry| entry.expect("read model tree entry").path())
            .collect::<Vec<_>>();
        children.sort();
        for child in children {
            let relative = child
                .strip_prefix(root)
                .expect("model tree entry under root")
                .to_path_buf();
            let metadata = fs::symlink_metadata(&child).expect("read model tree metadata");
            let file_type = metadata.file_type();
            let file_kind = if file_type.is_dir() {
                1
            } else if file_type.is_file() {
                2
            } else if file_type.is_symlink() {
                3
            } else {
                4
            };
            #[cfg(unix)]
            let (permission_key, changed) = {
                use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
                (
                    u64::from(metadata.permissions().mode()),
                    Some((metadata.ctime(), metadata.ctime_nsec())),
                )
            };
            #[cfg(not(unix))]
            let permission_key = u64::from(metadata.permissions().readonly());
            #[cfg(not(unix))]
            let changed = None;
            entries.push((
                relative,
                ObservedModelEntry {
                    file_kind,
                    len: metadata.len(),
                    modified: metadata.modified().expect("read model tree mtime"),
                    created: metadata.created().ok(),
                    changed,
                    permission_key,
                    bytes: file_type
                        .is_file()
                        .then(|| fs::read(&child).expect("read model tree file")),
                },
            ));
            if file_type.is_dir() {
                visit(root, &child, entries);
            }
        }
    }

    if !root.exists() {
        return Vec::new();
    }
    let mut entries = Vec::new();
    visit(root, root, &mut entries);
    entries
}

#[cfg(all(feature = "bundled-default-models", feature = "hash"))]
fn detect_observationally(model_root: &Path) -> SearchResult<EmbedderStack> {
    #[cfg(all(
        feature = "download",
        any(feature = "model2vec", feature = "fastembed")
    ))]
    {
        EmbedderStack::auto_detect_with_policy(
            Some(model_root),
            DownloadPolicy::for_tests(
                DownloadConsent::denied(Some(ConsentSource::Programmatic)),
                false,
                false,
            ),
            None,
        )
    }
    #[cfg(not(all(
        feature = "download",
        any(feature = "model2vec", feature = "fastembed")
    )))]
    {
        EmbedderStack::auto_detect_with(Some(model_root))
    }
}

#[cfg(all(feature = "bundled-default-models", feature = "hash"))]
#[test]
fn auto_detect_does_not_materialize_an_absent_bundled_cache() {
    let temp = tempfile::tempdir().expect("tempdir");
    let model_root = temp.path().join("absent-model-root");

    let stack = detect_observationally(&model_root).expect("observational detection");

    assert_eq!(stack.availability(), TwoTierAvailability::HashOnly);
    assert!(
        !model_root.exists(),
        "auto-detection must not create an absent bundled-model cache"
    );
}

#[cfg(all(feature = "bundled-default-models", feature = "hash"))]
#[test]
fn auto_detect_leaves_corrupt_bundled_bytes_and_receipts_unchanged() {
    let temp = tempfile::tempdir().expect("tempdir");
    let model_root = temp.path().join("models");
    let fast_dir = model_root.join(POTION_MODEL_NAME);
    let quality_dir = model_root.join(MINILM_MODEL_NAME);
    fs::create_dir_all(&fast_dir).expect("create corrupt fast model dir");
    fs::create_dir_all(&quality_dir).expect("create corrupt quality model dir");
    fs::write(fast_dir.join("tokenizer.json"), b"corrupt-fast-tokenizer")
        .expect("write corrupt fast bytes");
    fs::write(fast_dir.join(".verified"), b"stale-fast-receipt").expect("write stale fast receipt");
    fs::write(quality_dir.join("model.onnx"), b"corrupt-quality-model")
        .expect("write corrupt quality bytes");
    fs::write(quality_dir.join(".verified"), b"stale-quality-receipt")
        .expect("write stale quality receipt");
    let before = snapshot_model_tree(&model_root);

    let stack = detect_observationally(&model_root).expect("observational detection");

    assert_eq!(stack.availability(), TwoTierAvailability::HashOnly);
    assert_eq!(
        snapshot_model_tree(&model_root),
        before,
        "auto-detection must not repair model bytes or mint receipts"
    );
}

struct MrlFixtureEmbedder {
    identity: EmbeddingIdentityBundleV1,
}

impl MrlFixtureEmbedder {
    fn semantic_nomic() -> Self {
        let identity = crate::model_manifest::ModelArtifactManifestV1::nomic_fastembed()
            .expect("registered Nomic manifest")
            .declared_identity_bundle(
                frankensearch_core::generation::QuantizationFormat::F32,
                "in-memory-f32-v1",
            )
            .expect("derive semantic MRL fixture identity");
        Self { identity }
    }
}

impl Embedder for MrlFixtureEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            Err(SearchError::EmbeddingFailed {
                model: "mrl-structural-fixture".to_owned(),
                source: "identity-only test fixture does not run inference".into(),
            })
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn dimension(&self) -> usize {
        usize::try_from(self.identity.space.dimension).expect("fixture dimension must fit usize")
    }

    fn id(&self) -> &'static str {
        "mrl-structural-fixture"
    }

    fn model_name(&self) -> &'static str {
        "MRL structural fixture"
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::TransformerEmbedder
    }

    fn supports_mrl(&self) -> bool {
        true
    }
}

#[cfg(all(feature = "hash", not(feature = "bundled-default-models")))]
#[test]
fn fast_only_detection_refuses_hash_control_and_never_carries_a_quality_tier() {
    // Planted negative: with no model bytes on disk and an explicit offline
    // policy, the fast-only detector must fail closed exactly like the
    // two-tier semantic detector (hash control is not a search engine).
    let temp = tempfile::tempdir().unwrap();
    let options = DetectOptions {
        offline: Some(true),
    };
    let refused =
        EmbedderStack::auto_detect_fast_semantic_with_options(Some(temp.path()), &options);
    assert!(
        matches!(refused, Err(SearchError::EmbedderUnavailable { .. })),
        "an empty model root must not yield a usable fast-only stack: {refused:?}"
    );

    // Positive shape: a fast-only stack built from the same parts the
    // detector produces never reports a quality tier, so a host that
    // resolves quality separately cannot be handed a half-loaded pair.
    let fast: Arc<dyn Embedder> = Arc::new(SemanticTestEmbedder::new("test-fast", 4));
    let stack = EmbedderStack::from_parts(fast, None);
    assert_eq!(stack.availability(), TwoTierAvailability::FastOnly);
    assert!(stack.quality().is_none());
    assert!(stack.require_semantic().is_ok());
}

#[cfg(feature = "hash")]
#[test]
fn auto_detect_hash_only_when_no_models_present() {
    let temp = tempfile::tempdir().unwrap();
    #[cfg(all(
        feature = "download",
        any(feature = "model2vec", feature = "fastembed")
    ))]
    let stack = EmbedderStack::auto_detect_with_policy(
        Some(temp.path()),
        DownloadPolicy::for_tests(
            DownloadConsent::denied(Some(ConsentSource::Programmatic)),
            false,
            false,
        ),
        None,
    )
    .unwrap();
    #[cfg(not(all(
        feature = "download",
        any(feature = "model2vec", feature = "fastembed")
    )))]
    let stack = EmbedderStack::auto_detect_with(Some(temp.path())).unwrap();
    assert_eq!(stack.availability(), TwoTierAvailability::HashOnly);
    assert_eq!(stack.fast().category(), ModelCategory::HashEmbedder);
    assert!(stack.quality().is_none());
    assert!(matches!(
        stack.require_semantic(),
        Err(SearchError::EmbedderUnavailable { .. })
    ));
    assert!(matches!(
        EmbedderStack::auto_detect_semantic_with(Some(temp.path())),
        Err(SearchError::EmbedderUnavailable { .. })
    ));
    assert!(matches!(
        EmbedderStack::auto_detect_semantic_with_options(
            Some(temp.path()),
            &DetectOptions::default()
        ),
        Err(SearchError::EmbedderUnavailable { .. })
    ));
}

#[cfg(all(
    feature = "model2vec",
    feature = "hash",
    not(feature = "bundled-default-models")
))]
#[test]
fn auto_detect_rejects_unverified_model2vec_layout() {
    let temp = tempfile::tempdir().unwrap();
    let model_dir = temp.path().join(POTION_MODEL_NAME);
    fs::create_dir_all(&model_dir).unwrap();
    create_test_model2vec_layout(&model_dir, 16, 8);

    #[cfg(all(
        feature = "download",
        any(feature = "model2vec", feature = "fastembed")
    ))]
    let stack = EmbedderStack::auto_detect_with_policy(
        Some(temp.path()),
        DownloadPolicy::for_tests(
            DownloadConsent::denied(Some(ConsentSource::Programmatic)),
            false,
            false,
        ),
        None,
    )
    .unwrap();
    #[cfg(not(all(
        feature = "download",
        any(feature = "model2vec", feature = "fastembed")
    )))]
    let stack = EmbedderStack::auto_detect_with(Some(temp.path())).unwrap();
    assert_eq!(stack.availability(), TwoTierAvailability::HashOnly);
    assert_eq!(stack.fast().category(), ModelCategory::HashEmbedder);
    assert!(stack.quality().is_none());
}

#[cfg(all(
    feature = "model2vec",
    feature = "hash",
    not(feature = "bundled-default-models")
))]
#[test]
fn corrupted_model2vec_falls_back_to_hash() {
    let temp = tempfile::tempdir().unwrap();
    let model_dir = temp.path().join(POTION_MODEL_NAME);
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(model_dir.join("tokenizer.json"), "{}").unwrap();
    fs::write(model_dir.join("model.safetensors"), b"not-safetensors").unwrap();

    #[cfg(all(
        feature = "download",
        any(feature = "model2vec", feature = "fastembed")
    ))]
    let stack = EmbedderStack::auto_detect_with_policy(
        Some(temp.path()),
        DownloadPolicy::for_tests(
            DownloadConsent::denied(Some(ConsentSource::Programmatic)),
            false,
            false,
        ),
        None,
    )
    .unwrap();
    #[cfg(not(all(
        feature = "download",
        any(feature = "model2vec", feature = "fastembed")
    )))]
    let stack = EmbedderStack::auto_detect_with(Some(temp.path())).unwrap();
    assert_eq!(stack.availability(), TwoTierAvailability::HashOnly);
    assert_eq!(stack.fast().category(), ModelCategory::HashEmbedder);
}

#[cfg(all(feature = "download", feature = "model2vec", feature = "hash"))]
#[test]
fn auto_detect_prefers_lazy_fast_embedder_when_download_enabled() {
    let temp = tempfile::tempdir().unwrap();
    let stack = EmbedderStack::auto_detect_with_policy(
        Some(temp.path()),
        DownloadPolicy::for_tests(
            DownloadConsent::granted(ConsentSource::Programmatic),
            false,
            false,
        ),
        None,
    )
    .unwrap();
    assert_eq!(stack.fast().id(), POTION_MODEL_NAME);
    assert_eq!(stack.fast().category(), ModelCategory::StaticEmbedder);
}

#[cfg(all(feature = "download", feature = "model2vec"))]
#[test]
fn lazy_model2vec_returns_unavailable_when_download_is_denied() {
    let temp = tempfile::tempdir().unwrap();
    let lazy = LazyModel2VecEmbedder::new(
        Some(temp.path().to_path_buf()),
        DownloadPolicy::for_tests(
            DownloadConsent::denied(Some(ConsentSource::Programmatic)),
            false,
            false,
        ),
    )
    .unwrap();

    run_test_with_cx(|cx| async move {
        let err = lazy
            .embed(&cx, "hello world")
            .await
            .expect_err("download-denied lazy model should error");
        assert!(matches!(err, SearchError::EmbedderUnavailable { .. }));
    });
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
#[test]
fn parse_bool_flag_supports_common_values() {
    assert_eq!(parse_bool_flag("1"), Some(true));
    assert_eq!(parse_bool_flag("true"), Some(true));
    assert_eq!(parse_bool_flag("YES"), Some(true));
    assert_eq!(parse_bool_flag("0"), Some(false));
    assert_eq!(parse_bool_flag("false"), Some(false));
    assert_eq!(parse_bool_flag("off"), Some(false));
    assert_eq!(parse_bool_flag("invalid"), None);
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
#[test]
fn progress_percent_accounts_for_current_file_fraction() {
    let progress = DownloadProgress {
        file_name: "model.onnx".to_owned(),
        bytes_downloaded: 50,
        total_bytes: Some(100),
        files_completed: 1,
        files_total: 4,
        speed_bytes_per_sec: 1.0,
        eta_seconds: Some(2.0),
    };
    // 1 full file + 50% of one file over 4 total files = 37.5%
    assert_eq!(progress_percent_x100(&progress), 3_750);
}

#[cfg(all(
    feature = "model2vec",
    feature = "hash",
    not(feature = "bundled-default-models")
))]
fn create_test_model2vec_layout(dir: &Path, vocab_size: usize, dimensions: usize) {
    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [{
            "id": 0,
            "content": "[UNK]",
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true
        }],
        "normalizer": { "type": "Lowercase" },
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": create_test_vocab(vocab_size),
            "unk_token": "[UNK]"
        }
    });
    fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string_pretty(&tokenizer_json).unwrap(),
    )
    .unwrap();
    create_test_safetensors(dir, vocab_size, dimensions);
}

#[cfg(all(
    feature = "model2vec",
    feature = "hash",
    not(feature = "bundled-default-models")
))]
fn create_test_vocab(vocab_size: usize) -> serde_json::Value {
    let mut vocab = serde_json::Map::new();
    vocab.insert("[UNK]".to_owned(), serde_json::Value::from(0));
    for idx in 1..vocab_size {
        vocab.insert(format!("token{idx}"), serde_json::Value::from(idx));
    }
    serde_json::Value::Object(vocab)
}

#[cfg(feature = "hash")]
struct SemanticTestEmbedder {
    id: &'static str,
    dimension: usize,
}

#[cfg(feature = "hash")]
impl SemanticTestEmbedder {
    const fn new(id: &'static str, dimension: usize) -> Self {
        Self { id, dimension }
    }
}

#[cfg(feature = "hash")]
impl Embedder for SemanticTestEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        let dim = self.dimension;
        Box::pin(async move { Ok(vec![0.0; dim]) })
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

#[cfg(feature = "hash")]
#[test]
fn from_parts_hash_only_availability() {
    let hash = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
    let stack = EmbedderStack::from_parts(hash, None);
    assert_eq!(stack.availability(), TwoTierAvailability::HashOnly);
    assert!(stack.quality().is_none());
    assert!(stack.quality_arc().is_none());
    assert_eq!(stack.fast().category(), ModelCategory::HashEmbedder);
    assert_eq!(stack.fast_embedder().id(), stack.fast().id());
}

#[cfg(feature = "hash")]
#[test]
fn from_parts_two_hash_embedders_is_hash_only() {
    let fast: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
    let quality: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_384());
    let stack = EmbedderStack::from_parts(fast, Some(quality));
    assert_eq!(stack.availability(), TwoTierAvailability::HashOnly);
    assert!(
        stack.require_semantic().is_err(),
        "two hash embedders must not pass require_semantic"
    );
}

#[cfg(feature = "hash")]
#[test]
fn from_parts_semantic_pair_is_full() {
    let fast: Arc<dyn Embedder> = Arc::new(SemanticTestEmbedder::new("test-fast", 4));
    let quality: Arc<dyn Embedder> = Arc::new(SemanticTestEmbedder::new("test-quality", 4));
    let stack = EmbedderStack::from_parts(fast, Some(quality));
    assert_eq!(stack.availability(), TwoTierAvailability::Full);
    assert!(stack.quality().is_some());
    assert!(stack.quality_arc().is_some());
    assert_eq!(
        stack.quality_embedder().unwrap().id(),
        stack.quality().unwrap().id()
    );
}

#[cfg(feature = "hash")]
#[test]
fn from_parts_semantic_plus_hash_quality_is_fast_only() {
    let fast: Arc<dyn Embedder> = Arc::new(SemanticTestEmbedder::new("test-fast", 4));
    let quality: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_384());
    let stack = EmbedderStack::from_parts(fast, Some(quality));
    assert_eq!(stack.availability(), TwoTierAvailability::FastOnly);
    assert!(stack.require_semantic().is_ok());
}

#[cfg(feature = "hash")]
#[test]
fn dim_reduce_rejects_zero_target_dim() {
    let inner: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
    let err = DimReduceEmbedder::new(inner, 0).expect_err("should reject target_dim=0");
    assert!(matches!(err, SearchError::InvalidConfig { .. }));
}

#[cfg(feature = "hash")]
#[test]
fn dim_reduce_rejects_target_exceeding_inner_dim() {
    let inner: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
    let err = DimReduceEmbedder::new(inner, 512).expect_err("should reject target_dim > inner dim");
    assert!(matches!(err, SearchError::InvalidConfig { .. }));
}

#[cfg(feature = "hash")]
#[test]
fn dim_reduce_rejects_non_mrl_embedder() {
    let inner: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
    assert!(!inner.supports_mrl());
    let err = DimReduceEmbedder::new(inner, 64).expect_err("should reject non-MRL embedder");
    assert!(matches!(err, SearchError::InvalidConfig { .. }));
}

#[test]
fn dim_reduce_derives_structural_identity() {
    let inner: Arc<dyn Embedder> = Arc::new(MrlFixtureEmbedder::semantic_nomic());
    assert!(inner.is_semantic());
    assert!(inner.supports_mrl());
    let parent = inner.identity().unwrap().clone();
    let reduced = DimReduceEmbedder::new(inner, 4).unwrap();
    let child = reduced.identity().unwrap();
    child.validate().unwrap();
    let projection = child.space.projection.as_ref().unwrap();
    assert_eq!(
        projection.parent_space_fingerprint,
        parent.space.fingerprint()
    );
    assert_eq!(projection.source_dimension, 768);
    assert_eq!(projection.output_dimension, 4);
    assert_eq!(child.storage.dimension, 4);
}

#[cfg(feature = "hash")]
#[test]
fn with_mrl_target_dim_zero_is_rejected() {
    let fast: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
    let stack = EmbedderStack::from_parts(fast, None);
    let err = stack
        .with_mrl_target_dim(0)
        .expect_err("should reject target_dim=0");
    assert!(matches!(err, SearchError::InvalidConfig { .. }));
}

#[cfg(feature = "hash")]
#[test]
fn with_mrl_passthrough_when_non_mrl() {
    let fast: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
    let stack = EmbedderStack::from_parts(fast, None);
    // hash embedder doesn't support MRL, so with_mrl_target_dim should
    // pass through without wrapping (target_dim < dimension but !supports_mrl)
    let stack = stack.with_mrl_target_dim(64).unwrap();
    assert_eq!(stack.availability(), TwoTierAvailability::HashOnly);
    // dimension should be unchanged since wrapping was skipped
    assert_eq!(stack.fast().dimension(), 256);
}

#[cfg(feature = "hash")]
#[test]
fn embedder_stack_debug_format() {
    let fast: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
    let stack = EmbedderStack::from_parts(fast, None);
    let debug = format!("{stack:?}");
    assert!(debug.contains("EmbedderStack"));
    assert!(debug.contains("HashOnly"));
}

// ─── Offline Fallback Diagnostic Tests ───────────────────────────────────

#[cfg(feature = "hash")]
#[test]
fn availability_is_degraded_for_hash_only() {
    assert!(TwoTierAvailability::HashOnly.is_degraded());
    assert!(TwoTierAvailability::FastOnly.is_degraded());
    assert!(!TwoTierAvailability::Full.is_degraded());
}

#[cfg(feature = "hash")]
#[test]
fn availability_display_format() {
    let full = format!("{}", TwoTierAvailability::Full);
    assert!(full.contains("full"));
    let fast = format!("{}", TwoTierAvailability::FastOnly);
    assert!(fast.contains("degraded"));
    let hash = format!("{}", TwoTierAvailability::HashOnly);
    assert!(hash.contains("minimal"));
}

#[cfg(feature = "hash")]
#[test]
fn degradation_summary_none_for_full() {
    assert!(TwoTierAvailability::Full.degradation_summary().is_none());
}

#[cfg(feature = "hash")]
#[test]
fn degradation_summary_present_for_degraded() {
    assert!(
        TwoTierAvailability::FastOnly
            .degradation_summary()
            .unwrap()
            .contains("Quality model")
    );
    let hash_summary = TwoTierAvailability::HashOnly.degradation_summary().unwrap();
    assert!(
        hash_summary.contains("not semantic search") && !hash_summary.contains("reduced relevance"),
        "hash-only must not be framed as weaker semantic search: {hash_summary}"
    );
}

#[cfg(feature = "hash")]
#[test]
fn diagnose_hash_only_provides_suggestions() {
    let fast: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
    let stack = EmbedderStack::from_parts(fast, None);
    let diag = stack.diagnose();
    assert_eq!(diag.availability, TwoTierAvailability::HashOnly);
    assert!(matches!(diag.fast_status, ModelStatus::HashFallback));
    assert!(!diag.suggestions.is_empty());
    assert!(
        diag.suggestions
            .iter()
            .any(|s| s.contains("FRANKENSEARCH_MODEL_DIR"))
    );
    #[cfg(not(feature = "bundled-default-models"))]
    assert!(diag.suggestions.iter().any(|s| s.contains("air-gapped")));
    #[cfg(feature = "bundled-default-models")]
    assert!(
        diag.suggestions
            .iter()
            .any(|s| s.contains("download-models"))
    );
}

#[cfg(feature = "hash")]
#[test]
fn diagnose_full_has_no_suggestions() {
    let fast: Arc<dyn Embedder> = Arc::new(SemanticTestEmbedder::new("test-fast", 4));
    let quality: Arc<dyn Embedder> = Arc::new(SemanticTestEmbedder::new("test-quality", 4));
    let stack = EmbedderStack::from_parts(fast, Some(quality));
    let diag = stack.diagnose();
    assert_eq!(diag.availability, TwoTierAvailability::Full);
    assert!(diag.suggestions.is_empty());
}

#[cfg(feature = "hash")]
#[test]
fn degradation_message_none_when_full() {
    let fast: Arc<dyn Embedder> = Arc::new(SemanticTestEmbedder::new("test-fast", 4));
    let quality: Arc<dyn Embedder> = Arc::new(SemanticTestEmbedder::new("test-quality", 4));
    let stack = EmbedderStack::from_parts(fast, Some(quality));
    assert!(stack.degradation_message().is_none());
}

#[cfg(feature = "hash")]
#[test]
fn degradation_message_present_when_degraded() {
    let fast: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
    let stack = EmbedderStack::from_parts(fast, None);
    let msg = stack
        .degradation_message()
        .expect("should be present for hash-only");
    assert!(msg.contains("Model cache:"));
    assert!(msg.contains("FRANKENSEARCH_MODEL_DIR"));
    assert!(
        msg.contains("not semantic search")
            && msg.contains("cannot repair a hash-written index")
            && !msg.contains("reduced relevance")
            && !msg.contains("To improve search quality"),
        "hash-only message must not sound like a quality downgrade: {msg}"
    );
}

#[cfg(feature = "hash")]
#[test]
fn model_status_display_ready() {
    let status = ModelStatus::Ready {
        id: "test-model".to_owned(),
    };
    let display = format!("{status}");
    assert!(display.contains("ready"));
    assert!(display.contains("test-model"));
}

#[cfg(feature = "hash")]
#[test]
fn model_status_display_not_found() {
    let status = ModelStatus::NotFound {
        model_name: "test-model".to_owned(),
        hf_repo_url: "https://huggingface.co/test/model".to_owned(),
        searched_paths: vec![],
    };
    let display = format!("{status}");
    assert!(display.contains("NOT FOUND"));
    assert!(display.contains("https://huggingface.co/test/model"));
}

#[cfg(feature = "hash")]
#[test]
fn model_status_display_download_blocked() {
    let status = ModelStatus::DownloadBlocked {
        model_name: "test-model".to_owned(),
        reason: "offline mode".to_owned(),
    };
    let display = format!("{status}");
    assert!(display.contains("BLOCKED"));
    assert!(display.contains("offline mode"));
}

#[cfg(feature = "hash")]
#[test]
fn model_status_display_feature_disabled() {
    let status = ModelStatus::FeatureDisabled {
        feature_flag: "fastembed".to_owned(),
    };
    let display = format!("{status}");
    assert!(display.contains("DISABLED"));
    assert!(display.contains("fastembed"));
}

#[cfg(feature = "hash")]
#[test]
fn diagnostic_display_includes_all_sections() {
    let diag = ModelAvailabilityDiagnostic {
        availability: TwoTierAvailability::HashOnly,
        cache_dir: std::path::PathBuf::from("/tmp/test-cache"),
        offline: false,
        fast_status: ModelStatus::HashFallback,
        quality_status: ModelStatus::FeatureDisabled {
            feature_flag: "fastembed".to_owned(),
        },
        suggestions: vec!["Fix something".to_owned()],
    };
    let display = format!("{diag}");
    assert!(display.contains("minimal"));
    assert!(display.contains("/tmp/test-cache"));
    assert!(display.contains("hash control"));
    assert!(display.contains("Fix something"));
}

#[cfg(feature = "hash")]
#[test]
fn diagnostic_display_offline_mode_indicator() {
    let diag = ModelAvailabilityDiagnostic {
        availability: TwoTierAvailability::HashOnly,
        cache_dir: std::path::PathBuf::from("/tmp/test-cache"),
        offline: true,
        fast_status: ModelStatus::HashFallback,
        quality_status: ModelStatus::DownloadBlocked {
            model_name: "test".to_owned(),
            reason: "offline".to_owned(),
        },
        suggestions: vec![],
    };
    let display = format!("{diag}");
    assert!(display.contains("OFFLINE"));
}

// ─── bd-1il3 tests begin ───

#[test]
fn two_tier_availability_clone_copy_eq() {
    let a = TwoTierAvailability::Full;
    let b = a;
    assert_eq!(a, b);
    #[allow(clippy::clone_on_copy)]
    let c = a.clone();
    assert_eq!(a, c);
    assert_ne!(TwoTierAvailability::Full, TwoTierAvailability::HashOnly);
    assert_ne!(TwoTierAvailability::FastOnly, TwoTierAvailability::HashOnly);
}

#[test]
fn two_tier_availability_debug() {
    let debug = format!("{:?}", TwoTierAvailability::Full);
    assert_eq!(debug, "Full");
    let debug = format!("{:?}", TwoTierAvailability::FastOnly);
    assert_eq!(debug, "FastOnly");
    let debug = format!("{:?}", TwoTierAvailability::HashOnly);
    assert_eq!(debug, "HashOnly");
}

#[test]
fn model_status_debug_all_variants() {
    let ready = ModelStatus::Ready {
        id: "test".to_owned(),
    };
    let debug = format!("{ready:?}");
    assert!(debug.contains("Ready"));
    assert!(debug.contains("test"));

    let not_found = ModelStatus::NotFound {
        model_name: "m".to_owned(),
        hf_repo_url: "url".to_owned(),
        searched_paths: vec![],
    };
    let debug = format!("{not_found:?}");
    assert!(debug.contains("NotFound"));

    let blocked = ModelStatus::DownloadBlocked {
        model_name: "m".to_owned(),
        reason: "r".to_owned(),
    };
    let debug = format!("{blocked:?}");
    assert!(debug.contains("DownloadBlocked"));

    let disabled = ModelStatus::FeatureDisabled {
        feature_flag: "f".to_owned(),
    };
    let debug = format!("{disabled:?}");
    assert!(debug.contains("FeatureDisabled"));

    let hash = ModelStatus::HashFallback;
    let debug = format!("{hash:?}");
    assert!(debug.contains("HashFallback"));
}

#[test]
fn model_status_clone_all_variants() {
    // Verify Clone works for each variant by cloning into a Vec and formatting
    fn clone_and_format(status: &ModelStatus) -> String {
        format!("{}", status.clone())
    }

    assert!(
        clone_and_format(&ModelStatus::Ready {
            id: "test".to_owned(),
        })
        .contains("test")
    );

    assert!(
        clone_and_format(&ModelStatus::NotFound {
            model_name: "m".to_owned(),
            hf_repo_url: "u".to_owned(),
            searched_paths: vec![std::path::PathBuf::from("/tmp/p")],
        })
        .contains("NOT FOUND")
    );

    assert!(
        clone_and_format(&ModelStatus::DownloadBlocked {
            model_name: "m".to_owned(),
            reason: "r".to_owned(),
        })
        .contains("BLOCKED")
    );

    assert!(
        clone_and_format(&ModelStatus::FeatureDisabled {
            feature_flag: "f".to_owned(),
        })
        .contains("DISABLED")
    );

    assert!(clone_and_format(&ModelStatus::HashFallback).contains("hash control"));
}

#[test]
fn model_status_hash_fallback_display() {
    let status = ModelStatus::HashFallback;
    let display = format!("{status}");
    assert_eq!(display, "hash control (not a semantic model)");
}

#[test]
fn model_availability_diagnostic_clone_debug() {
    let diag = ModelAvailabilityDiagnostic {
        availability: TwoTierAvailability::Full,
        cache_dir: std::path::PathBuf::from("/tmp/cache"),
        offline: false,
        fast_status: ModelStatus::Ready {
            id: "fast".to_owned(),
        },
        quality_status: ModelStatus::Ready {
            id: "quality".to_owned(),
        },
        suggestions: vec![],
    };
    let cloned = diag.clone();
    assert_eq!(cloned.availability, TwoTierAvailability::Full);
    assert!(!cloned.offline);
    assert!(cloned.suggestions.is_empty());

    let debug = format!("{diag:?}");
    assert!(debug.contains("ModelAvailabilityDiagnostic"));
    assert!(debug.contains("Full"));
}

#[test]
fn diagnostic_display_no_suggestions_skips_resolve_section() {
    let diag = ModelAvailabilityDiagnostic {
        availability: TwoTierAvailability::Full,
        cache_dir: std::path::PathBuf::from("/tmp/cache"),
        offline: false,
        fast_status: ModelStatus::Ready {
            id: "fast".to_owned(),
        },
        quality_status: ModelStatus::Ready {
            id: "quality".to_owned(),
        },
        suggestions: vec![],
    };
    let display = format!("{diag}");
    assert!(!display.contains("To resolve:"));
}

#[cfg(feature = "hash")]
#[test]
fn embedder_stack_clone() {
    let fast: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
    let stack = EmbedderStack::from_parts(fast, None);
    let cloned = stack.clone();
    assert_eq!(cloned.availability(), stack.availability());
    assert_eq!(cloned.fast().id(), stack.fast().id());
}

#[cfg(feature = "hash")]
#[test]
fn embedder_stack_fast_arc_returns_same_id() {
    let fast: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
    let stack = EmbedderStack::from_parts(fast, None);
    let arc = stack.fast_arc();
    assert_eq!(arc.id(), stack.fast().id());
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
#[test]
fn format_bytes_edge_cases() {
    assert_eq!(format_bytes(0), "0 B");
    assert_eq!(format_bytes(1), "1 B");
    assert_eq!(format_bytes(1023), "1023 B");
    assert_eq!(format_bytes(1024), "1.0 KB");
    assert_eq!(format_bytes(1536), "1.5 KB");
    assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
    assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
    assert_eq!(
        format_bytes(1024 * 1024 * 1024 + 512 * 1024 * 1024),
        "1.5 GB"
    );
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
#[test]
fn format_speed_edge_cases() {
    assert_eq!(format_speed(0.0), "0 B/s");
    assert_eq!(format_speed(-1.0), "0 B/s");
    assert_eq!(format_speed(f64::NAN), "0 B/s");
    assert_eq!(format_speed(f64::INFINITY), "0 B/s");
    assert_eq!(format_speed(f64::NEG_INFINITY), "0 B/s");
    assert_eq!(format_speed(500.0), "500 B/s");
    assert!(format_speed(2048.0).contains("KB/s"));
    assert!(format_speed(2.0 * 1024.0 * 1024.0).contains("MB/s"));
    assert!(format_speed(2.0 * 1024.0 * 1024.0 * 1024.0).contains("GB/s"));
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
#[test]
fn format_eta_edge_cases() {
    assert_eq!(format_eta(None), "?");
    assert_eq!(format_eta(Some(f64::NAN)), "?");
    assert_eq!(format_eta(Some(f64::INFINITY)), "?");
    assert_eq!(format_eta(Some(-1.0)), "?");
    assert_eq!(format_eta(Some(0.0)), "0.0s");
    assert_eq!(format_eta(Some(5.5)), "5.5s");
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
#[test]
fn render_progress_bar_edge_cases() {
    let empty_bar = render_progress_bar(0);
    assert_eq!(empty_bar.len(), PROGRESS_BAR_WIDTH);
    assert!(empty_bar.chars().all(|c| c == ' '));

    let full_bar = render_progress_bar(10_000);
    assert_eq!(full_bar.len(), PROGRESS_BAR_WIDTH);
    assert!(full_bar.chars().all(|c| c == '='));

    let half_bar = render_progress_bar(5_000);
    assert_eq!(half_bar.len(), PROGRESS_BAR_WIDTH);
    let filled = half_bar.chars().filter(|&c| c == '=').count();
    assert_eq!(filled, PROGRESS_BAR_WIDTH / 2);

    // Over 100% should clamp
    let bar_over = render_progress_bar(20_000);
    assert_eq!(bar_over.len(), PROGRESS_BAR_WIDTH);
    assert!(bar_over.chars().all(|c| c == '='));
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
#[test]
fn parse_bool_flag_whitespace_and_on() {
    assert_eq!(parse_bool_flag(" 1 "), Some(true));
    assert_eq!(parse_bool_flag("  true  "), Some(true));
    assert_eq!(parse_bool_flag("on"), Some(true));
    assert_eq!(parse_bool_flag("ON"), Some(true));
    assert_eq!(parse_bool_flag("On"), Some(true));
    assert_eq!(parse_bool_flag("no"), Some(false));
    assert_eq!(parse_bool_flag("NO"), Some(false));
    assert_eq!(parse_bool_flag("No"), Some(false));
    assert_eq!(parse_bool_flag(""), None);
    assert_eq!(parse_bool_flag("   "), None);
    assert_eq!(parse_bool_flag("maybe"), None);
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
#[test]
fn progress_percent_zero_files_completed() {
    let progress = DownloadProgress {
        file_name: "model.onnx".to_owned(),
        bytes_downloaded: 0,
        total_bytes: Some(100),
        files_completed: 0,
        files_total: 2,
        speed_bytes_per_sec: 0.0,
        eta_seconds: None,
    };
    assert_eq!(progress_percent_x100(&progress), 0);
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
#[test]
fn progress_percent_all_files_completed() {
    let progress = DownloadProgress {
        file_name: "last.onnx".to_owned(),
        bytes_downloaded: 100,
        total_bytes: Some(100),
        files_completed: 3,
        files_total: 4,
        speed_bytes_per_sec: 1000.0,
        eta_seconds: Some(0.0),
    };
    // 3 complete files + 100% of current file / 4 total = 100%
    assert_eq!(progress_percent_x100(&progress), 10_000);
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
#[test]
fn progress_percent_no_total_bytes() {
    let progress = DownloadProgress {
        file_name: "model.onnx".to_owned(),
        bytes_downloaded: 50,
        total_bytes: None,
        files_completed: 1,
        files_total: 2,
        speed_bytes_per_sec: 100.0,
        eta_seconds: None,
    };
    // No total_bytes means current file contributes 0%
    // 1 complete / 2 total = 50% = 5000
    assert_eq!(progress_percent_x100(&progress), 5_000);
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
#[test]
fn download_policy_can_download_logic() {
    let allowed = DownloadPolicy::for_tests(
        DownloadConsent::granted(ConsentSource::Programmatic),
        false,
        false,
    );
    assert!(allowed.can_download());

    let offline = DownloadPolicy::for_tests(
        DownloadConsent::granted(ConsentSource::Programmatic),
        true,
        false,
    );
    assert!(!offline.can_download());

    let denied = DownloadPolicy::for_tests(
        DownloadConsent::denied(Some(ConsentSource::Programmatic)),
        false,
        false,
    );
    assert!(!denied.can_download());
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
#[test]
fn download_policy_blocked_reason_offline() {
    let policy = DownloadPolicy::for_tests(
        DownloadConsent::granted(ConsentSource::Programmatic),
        true,
        false,
    );
    let reason = policy.blocked_reason();
    assert!(reason.contains("OFFLINE"));
}

#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
#[test]
fn download_policy_blocked_reason_consent_denied() {
    let policy = DownloadPolicy::for_tests(
        DownloadConsent::denied(Some(ConsentSource::Environment)),
        false,
        false,
    );
    let reason = policy.blocked_reason();
    assert!(reason.contains("consent denied"));
}

// ─── bd-1il3 tests end ───

#[cfg(all(
    feature = "model2vec",
    feature = "hash",
    not(feature = "bundled-default-models")
))]
fn create_test_safetensors(dir: &Path, vocab_size: usize, dimensions: usize) {
    use std::collections::HashMap;

    let mut data = Vec::with_capacity(vocab_size * dimensions * 4);
    for row in 0..vocab_size {
        for col in 0..dimensions {
            #[allow(clippy::cast_precision_loss)]
            let value = (row as f32).mul_add(0.01, (col as f32) * 0.001);
            data.extend_from_slice(&value.to_le_bytes());
        }
    }

    let mut tensors = HashMap::new();
    tensors.insert(
        "embeddings".to_owned(),
        safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            vec![vocab_size, dimensions],
            &data,
        )
        .unwrap(),
    );
    let encoded = safetensors::tensor::serialize(&tensors, None).unwrap();
    fs::write(dir.join("model.safetensors"), encoded).unwrap();
}
