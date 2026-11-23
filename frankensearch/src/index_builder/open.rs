use std::path::Path;
use std::sync::Arc;

use asupersync::Cx;

use frankensearch_core::config::TwoTierConfig;
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_fusion::SyncTwoTierSearcher;
use frankensearch_index::{FsviV2IdentityBinding, TwoTierIndex, TwoTierIndexPaths};

use super::lexical::open_lexical_reader;
use super::types::HybridIndexParts;

fn is_hash_generation_id(embedder_id: &str) -> bool {
    frankensearch_core::is_hash_generation_id(embedder_id)
}

/// Open every arm of an index directory produced by [`super::IndexBuilder`].
///
/// The advertised hybrid flow used to build lexical data and then construct
/// a searcher without attaching it; this helper makes the correct wiring the
/// ergonomic default:
///
/// ```rust,ignore
/// let parts = open_hybrid(&cx, "./my_index", TwoTierConfig::default()).await?;
/// let mut searcher = TwoTierSearcher::new(parts.vectors, fast_embedder, config);
/// if let Some(lexical) = parts.lexical {
///     searcher = searcher.with_lexical(lexical);
/// }
/// ```
///
/// # Errors
///
/// Returns an error when the vector index cannot be opened, when the fast
/// tier carries a hash-embedder identity (that generation is not a hybrid
/// semantic product), or when a lexical directory exists but its index
/// fails to open (a corrupt lexical arm is reported, never silently
/// dropped). A missing lexical directory
/// yields `lexical: None` and `lexical_backend: None`, as does a build without
/// a lexical backend compiled in. When the arm is present,
/// `lexical_backend` records whether it is the default Quill reader or an
/// explicit Tantivy oracle/rollback reader.
pub async fn open_hybrid(
    cx: &Cx,
    data_dir: impl AsRef<Path>,
    config: TwoTierConfig,
) -> SearchResult<HybridIndexParts> {
    let data_dir = data_dir.as_ref();
    let vectors = Arc::new(TwoTierIndex::open(data_dir, config)?);
    if is_hash_generation_id(vectors.fast_embedder_id()) {
        return Err(SearchError::EmbedderUnavailable {
            model: vectors.fast_embedder_id().to_owned(),
            reason: "open_hybrid refuses hash-identity generations; rebuild with a semantic \
                     embedder, or open TwoTierIndex directly for explicit hash control"
                .to_owned(),
        });
    }

    let lexical_dir = data_dir.join("lexical");
    let (lexical, lexical_backend) = if lexical_dir.is_dir() {
        open_lexical_reader(cx, &lexical_dir).await?
    } else {
        (None, None)
    };

    Ok(HybridIndexParts {
        vectors,
        lexical,
        lexical_backend,
    })
}

/// Open the default synchronous product for exactly admitted FSVI v2 tiers.
///
/// Unlike [`open_hybrid`], this opener requires the caller's v2 identity
/// bindings because a v2 artifact has no legitimate path-only open. It
/// performs the same admitted-product open used by the shipping index API,
/// then routes the retained owners through the in-memory synchronous product
/// and its optional generation-keyed residual cache. A missing, corrupt, or
/// unavailable cache leaves the exact flat scan selected; it does not select a
/// different retrieval algorithm.
///
/// `residual_cache_dir` is shared safely by both tiers because cache artifact
/// names are generation-keyed. It need not exist: an unavailable optional cache
/// is a flat-exact fallback rather than an opening error.
///
/// # Errors
///
/// Returns exact v2 admission errors, including a missing quality binding for a
/// configured quality path, plus source-vector loading errors from the
/// synchronous product.
pub fn open_admitted_v2_sync_with_residual_sidecar_cache(
    paths: &TwoTierIndexPaths,
    fast_binding: &FsviV2IdentityBinding,
    quality_binding: Option<&FsviV2IdentityBinding>,
    residual_cache_dir: impl AsRef<Path>,
    config: TwoTierConfig,
) -> SearchResult<SyncTwoTierSearcher> {
    // This opener consumes only retained admitted FSVI owners to construct the
    // in-memory exact product. Strip optional ANN paths before `TwoTierIndex`
    // assembly: otherwise its `ann` path can build and persist HNSW sidecars
    // that are immediately discarded with this temporary owner container.
    // Fast/quality index paths remain intact, so exact tier admission,
    // cross-tier publication validation, and identity checks are unchanged.
    let mut owner_paths = TwoTierIndexPaths::new(paths.fast_index().to_path_buf());
    if let Some(quality_path) = paths.quality_index() {
        owner_paths = owner_paths.with_quality_index(quality_path.to_path_buf());
    }
    let admitted = TwoTierIndex::open_admitted_v2_with_paths(
        &owner_paths,
        config.clone(),
        fast_binding,
        quality_binding,
    )?;
    let fast_source = admitted
        .fast_admitted_owner()
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "admitted_v2.fast_owner".to_owned(),
            value: paths.fast_index().display().to_string(),
            reason: "exact v2 product open did not retain its fast owner".to_owned(),
        })?;
    let cache_dir = residual_cache_dir.as_ref();
    let quality_source = admitted
        .quality_admitted_owner()
        .map(|source| (source, cache_dir));
    SyncTwoTierSearcher::from_admitted_v2_with_residual_sidecar_cache(
        fast_source,
        cache_dir,
        quality_source,
        config,
    )
}
