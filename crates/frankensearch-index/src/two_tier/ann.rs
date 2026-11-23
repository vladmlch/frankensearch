use std::path::Path;

#[cfg(test)]
use frankensearch_core::SearchResult;
use frankensearch_core::TwoTierConfig;
use tracing::{debug, warn};

use crate::hnsw::HnswLoadDisposition;
use crate::{HNSW_DEFAULT_MAX_LAYER, HnswConfig, HnswIndex, VectorIndex};

use super::paths::{TwoTierIndexPaths, validate_ann_persistence_paths};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AnnPersistenceReason {
    FreshBuild,
    RebuiltFallback,
}

#[derive(Debug)]
pub(super) struct AnnOpenPlan {
    pub(super) index: HnswIndex,
    persistence: Option<AnnPersistenceReason>,
}

impl AnnOpenPlan {
    pub(super) const fn needs_persistence(&self) -> bool {
        self.persistence.is_some()
    }
}

pub(super) fn plan_load_or_build_ann(
    vector_index: &VectorIndex,
    ann_path: &Path,
    threshold: usize,
    config: &TwoTierConfig,
    tier: &str,
) -> Option<AnnOpenPlan> {
    if vector_index.record_count() < threshold {
        return None;
    }

    let ann_config = HnswConfig {
        m: config.hnsw_m,
        ef_construction: config.hnsw_ef_construction,
        ef_search: config.hnsw_ef_search,
        max_layer: HNSW_DEFAULT_MAX_LAYER,
    };

    if ann_path.exists() {
        match HnswIndex::load_with_disposition(ann_path, vector_index) {
            Ok((ann, load_disposition)) => match ann.matches_vector_index(vector_index) {
                Ok(true) => {
                    let loaded_config = ann.config();
                    if loaded_config == ann_config {
                        return Some(AnnOpenPlan {
                            index: ann,
                            persistence: (load_disposition == HnswLoadDisposition::Rebuilt)
                                .then_some(AnnPersistenceReason::RebuiltFallback),
                        });
                    }
                    warn!(
                        tier,
                        ann_path = %ann_path.display(),
                        ?loaded_config,
                        ?ann_config,
                        "ANN sidecar config differs from requested config; rebuilding"
                    );
                }
                Ok(false) => {
                    warn!(
                        tier,
                        ann_path = %ann_path.display(),
                        "ANN sidecar exists but does not match vector index; rebuilding"
                    );
                }
                Err(error) => {
                    warn!(
                        tier,
                        ann_path = %ann_path.display(),
                        ?error,
                        "failed to validate ANN sidecar; rebuilding"
                    );
                }
            },
            Err(error) => {
                warn!(
                    tier,
                    ann_path = %ann_path.display(),
                    ?error,
                    "failed to load ANN sidecar; rebuilding"
                );
            }
        }
    }

    let ann = match HnswIndex::build_from_vector_index(vector_index, ann_config) {
        Ok(ann) => ann,
        Err(error) => {
            warn!(
                tier,
                ?error,
                "failed to build ANN index; using brute-force fallback"
            );
            return None;
        }
    };

    Some(AnnOpenPlan {
        index: ann,
        persistence: Some(AnnPersistenceReason::FreshBuild),
    })
}

pub(super) fn persist_ann_plan(
    plan: &AnnOpenPlan,
    ann_path: &Path,
    tier: &str,
    paths: &TwoTierIndexPaths,
    fast_ann_enabled: bool,
    quality_ann_enabled: bool,
) {
    let Some(reason) = plan.persistence else {
        return;
    };
    if let Err(error) = validate_ann_persistence_paths(paths, fast_ann_enabled, quality_ann_enabled)
        .and_then(|()| HnswIndex::save(&plan.index, ann_path))
    {
        warn!(
            tier,
            ann_path = %ann_path.display(),
            ?reason,
            ?error,
            "failed to persist ANN sidecar; ANN stays in-memory for this process, persistence \
             durability was not confirmed, and the next startup may rebuild it again; check path \
             permissions and free space"
        );
    } else {
        debug!(
            tier,
            ann_path = %ann_path.display(),
            ?reason,
            "persisted ANN sidecar"
        );
    }
}

#[cfg(test)]
pub(super) fn maybe_load_or_build_ann_with_save<Validate, Save>(
    vector_index: &VectorIndex,
    ann_path: &Path,
    threshold: usize,
    config: &TwoTierConfig,
    tier: &str,
    validate_before_save: Validate,
    save_ann: Save,
) -> Option<HnswIndex>
where
    Validate: Fn() -> SearchResult<()>,
    Save: Fn(&HnswIndex, &Path) -> SearchResult<()>,
{
    let plan = plan_load_or_build_ann(vector_index, ann_path, threshold, config, tier)?;
    if let Some(reason) = plan.persistence {
        if let Err(error) = validate_before_save().and_then(|()| save_ann(&plan.index, ann_path)) {
            warn!(
                tier,
                ann_path = %ann_path.display(),
                ?reason,
                ?error,
                "failed to persist ANN sidecar; ANN stays in-memory for this process"
            );
        }
    }
    Some(plan.index)
}
