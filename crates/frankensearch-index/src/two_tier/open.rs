use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicU8;
#[cfg(feature = "ann")]
use std::sync::atomic::AtomicU64;

use frankensearch_core::{SearchError, SearchResult, TwoTierConfig};
use tracing::{debug, info, warn};

use crate::native_hnsw::{
    HnswParams, NativeHnswGenerationReceiptV2, ValidatedNativeHnsw,
    native_hnsw_generation_receipt_path,
};
use crate::{FsviV2IdentityBinding, ValidatedFsviBytes, VectorIndex};

use super::activation::{admit_v2_tier, warn_if_wal_rows_replayed};
#[cfg(feature = "ann")]
use super::ann::{AnnOpenPlan, persist_ann_plan, plan_load_or_build_ann};
use super::builder::TwoTierIndexBuilder;
use super::observe::resolve_fast_path;
#[cfg(feature = "ann")]
use super::paths::validate_ann_persistence_paths;
use super::paths::{TwoTierIndexPaths, paths_alias, validate_index_paths};
use super::{
    NativeTierHnsw, QualityAlignment, TierSource, TwoTierIndex, VECTOR_INDEX_QUALITY_FILENAME,
    ZERO_SIGNAL_NONE,
};
#[cfg(feature = "ann")]
use super::{VECTOR_ANN_FAST_FILENAME, VECTOR_ANN_QUALITY_FILENAME};

impl TwoTierIndex {
    /// Open a two-tier index from a directory.
    ///
    /// Fast index lookup order:
    /// 1. `{dir}/vector.fast.idx`
    /// 2. `{dir}/vector.idx` (fallback)
    ///
    /// Quality index (optional):
    /// - `{dir}/vector.quality.idx`
    ///
    /// # Errors
    ///
    /// Returns `SearchError::IndexCandidatesNotFound` if neither fast-tier
    /// candidate exists, and propagates fast-tier parse/corruption errors from
    /// `VectorIndex::open`. A discovered quality-tier file is optional: an
    /// unavailable or corrupt one degrades this constructor to fast-only.
    pub fn open(dir: &Path, config: TwoTierConfig) -> SearchResult<Self> {
        let fast_path = resolve_fast_path(dir)?;
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
        let mut paths = TwoTierIndexPaths::new(fast_path);
        if quality_path.exists() {
            paths = paths.with_quality_index(quality_path);
        }
        #[cfg(feature = "ann")]
        {
            paths = paths.with_fast_ann(dir.join(VECTOR_ANN_FAST_FILENAME));
            if paths.quality_index().is_some() {
                paths = paths.with_quality_ann(dir.join(VECTOR_ANN_QUALITY_FILENAME));
            }
        }
        Self::open_with_paths_inner(&paths, config, true)
    }

    /// Open a two-tier index from explicit consumer-owned paths.
    ///
    /// Unlike [`Self::open`], this constructor performs no filename discovery.
    /// A supplied quality path is required to exist; omit it to open a
    /// fast-only index. ANN sidecars are used only when explicitly configured
    /// on [`TwoTierIndexPaths`].
    ///
    /// # Errors
    ///
    /// Returns `SearchError::IndexNotFound` when an explicitly supplied index
    /// path is missing, `SearchError::InvalidConfig` when configured artifact
    /// roles alias, and propagates parse/corruption errors from
    /// `VectorIndex::open`. Relative paths are frozen against the current
    /// directory before validation and opening.
    pub fn open_with_paths(paths: &TwoTierIndexPaths, config: TwoTierConfig) -> SearchResult<Self> {
        Self::open_with_paths_inner(paths, config, false)
    }

    #[allow(clippy::too_many_lines)]
    fn open_with_paths_inner(
        paths: &TwoTierIndexPaths,
        config: TwoTierConfig,
        degrade_discovered_quality_errors: bool,
    ) -> SearchResult<Self> {
        let paths = paths.clone().into_absolute()?;
        validate_index_paths(&paths)?;
        let fast_index = VectorIndex::open_read_only(paths.fast_index())?;
        warn_if_wal_rows_replayed("fast", paths.fast_index(), &fast_index);
        let fast_source = TierSource::PathOpened(fast_index);
        let quality_source = match paths.quality_index() {
            Some(quality_path) => match VectorIndex::open_read_only(quality_path) {
                Ok(quality_index) => {
                    warn_if_wal_rows_replayed("quality", quality_path, &quality_index);
                    Some(TierSource::PathOpened(quality_index))
                }
                Err(error) if degrade_discovered_quality_errors => {
                    warn!(
                        path = %quality_path.display(),
                        ?error,
                        "discovered optional quality index is unavailable; degrading to fast-only"
                    );
                    None
                }
                Err(error) => return Err(error),
            },
            None => None,
        };
        Self::assemble_opened(fast_source, quality_source, &paths, config)
    }

    /// Open a two-tier generation whose tiers are identity-complete FSVI v2
    /// artifacts, through exact admission (bd-9xuj T2-C4-write).
    ///
    /// [`VectorIndex::open`] is strictly v1 — it rejects v2 bytes with
    /// `IndexVersionMismatch` — so v2 tiers can never be plain-opened. Each
    /// supplied tier is admitted via [`VectorIndex::open_admitted_v2`]
    /// against its caller-held [`FsviV2IdentityBinding`], and the sealed
    /// [`ValidatedFsviBytes`] owner is RETAINED IN FULL (r2 repair of the
    /// C4-write NO-GO): the `Arc`'d byte image, complete witness, and
    /// publication state stay reachable via
    /// [`Self::fast_admitted_owner`] / [`Self::quality_admitted_owner`], and
    /// the tier is served by reference from inside the owner. The resulting
    /// index reports header-ATTESTED identity:
    /// [`Self::fast_identity_is_attested`] (and the quality counterpart when
    /// a quality tier is supplied) return `true`, and the per-tier space
    /// fingerprints come from each artifact's own validated header bytes.
    ///
    /// A quality path and its binding must be supplied together: a v2 tier
    /// without a binding has no legitimate open path.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when a quality path/binding is
    /// supplied without its counterpart, typed admission failures (mapped
    /// from [`FsviAdmissionError`](crate::FsviAdmissionError), naming the tier), and the same path
    /// validation errors as [`Self::open_with_paths`].
    pub fn open_admitted_v2_with_paths(
        paths: &TwoTierIndexPaths,
        config: TwoTierConfig,
        fast_binding: &FsviV2IdentityBinding,
        quality_binding: Option<&FsviV2IdentityBinding>,
    ) -> SearchResult<Self> {
        let paths = paths.clone().into_absolute()?;
        validate_index_paths(&paths)?;
        let fast_source = TierSource::AdmittedV2 {
            owner: Arc::new(admit_v2_tier(paths.fast_index(), fast_binding, "fast")?),
            binding: fast_binding.clone(),
            source_path: paths.fast_index().to_path_buf(),
        };
        let quality_source = match (paths.quality_index(), quality_binding) {
            (Some(path), Some(binding)) => Some(TierSource::AdmittedV2 {
                owner: Arc::new(admit_v2_tier(path, binding, "quality")?),
                binding: binding.clone(),
                source_path: path.to_path_buf(),
            }),
            (None, None) => None,
            (Some(path), None) => {
                return Err(SearchError::InvalidConfig {
                    field: "two_tier.quality_v2_admission".to_owned(),
                    value: path.display().to_string(),
                    reason: "a quality index path was supplied without its identity binding; \
                             v2 tiers are only opened through exact admission"
                        .to_owned(),
                });
            }
            (None, Some(_)) => {
                return Err(SearchError::InvalidConfig {
                    field: "two_tier.quality_v2_admission".to_owned(),
                    value: "<no-quality-path>".to_owned(),
                    reason: "a quality identity binding was supplied without a quality index path"
                        .to_owned(),
                });
            }
        };
        Self::assemble_opened(fast_source, quality_source, &paths, config)
    }

    /// Shared assembly over tiers that were already opened — by the plain v1
    /// [`VectorIndex::open`] path or by exact FSVI v2 admission
    /// (bd-9xuj T2-C4-write refactor). Computes quality alignment, plans ANN
    /// sidecars, and retains per-tier header identity. Behavior for the v1
    /// path is unchanged; the identity retention comment below applies to
    /// both sources. Admitted v2 sources arrive as sealed retained owners
    /// ([`TierSource::AdmittedV2`]) and are only ever borrowed here.
    #[allow(clippy::too_many_lines)]
    fn assemble_opened(
        fast_source: TierSource,
        quality_source: Option<TierSource>,
        paths: &TwoTierIndexPaths,
        config: TwoTierConfig,
    ) -> SearchResult<Self> {
        let fast_index = fast_source.index();
        let mut quality_alignment = QualityAlignment::None;

        // A fast/quality pair is only blendable when both tiers came from the
        // SAME publication: a crash between the two per-tier installs leaves
        // a mixed-generation pair whose nonces disagree, and blending it
        // silently launders vectors from different corpus states (bd-miio8).
        // Legacy pre-nonce pairs read 0 == 0 and remain accepted. The
        // degraded pair drops the quality SOURCE (not just its borrow), so a
        // mixed-generation quality tier can never serve.
        let quality_source = quality_source.filter(|source| {
            let quality_nonce = source.index().publication_nonce();
            if quality_nonce == fast_index.publication_nonce() {
                true
            } else {
                warn!(
                    fast_nonce = fast_index.publication_nonce(),
                    quality_nonce,
                    "fast/quality publication identities disagree (mixed-generation pair, \
                     likely a crash between tier installs); degrading to fast-only"
                );
                false
            }
        });

        let quality_index = if let Some(quality) = quality_source.as_ref().map(TierSource::index) {
            if quality.record_count() != fast_index.record_count() {
                warn!(
                    fast_records = fast_index.record_count(),
                    quality_records = quality.record_count(),
                    "fast and quality index record counts differ; using doc-id alignment"
                );
            }

            quality_alignment = QualityAlignment::Aligned;
            let mut f_idx = 0;
            let mut q_idx = 0;
            let f_count = fast_index.record_count();
            let q_count = quality.record_count();
            let mut unmatched_quality_docs = 0;

            // Switch `quality_alignment` from `Aligned` to `Mapping` if not already.
            let ensure_mapping = |quality_alignment: &mut QualityAlignment,
                                  current_f_idx: usize| {
                if matches!(quality_alignment, QualityAlignment::Aligned) {
                    let map = (0..current_f_idx).map(Some).collect();
                    *quality_alignment = QualityAlignment::Mapping(map);
                }
            };

            while f_idx < f_count && q_idx < q_count {
                let f_rec = fast_index.record_at(f_idx)?;
                let q_rec = quality.record_at(q_idx)?;

                if crate::is_tombstoned_flags(f_rec.flags) {
                    ensure_mapping(&mut quality_alignment, f_idx);
                    if let QualityAlignment::Mapping(vec) = &mut quality_alignment {
                        vec.push(None);
                    }
                    f_idx += 1;
                    continue;
                }
                if crate::is_tombstoned_flags(q_rec.flags) {
                    q_idx += 1;
                    continue;
                }

                // If indices diverged, we must be in mapping mode
                if matches!(quality_alignment, QualityAlignment::Aligned) && f_idx != q_idx {
                    ensure_mapping(&mut quality_alignment, f_idx);
                }

                match f_rec.doc_id_hash.cmp(&q_rec.doc_id_hash) {
                    std::cmp::Ordering::Less => {
                        // Fast has doc, Quality missing
                        ensure_mapping(&mut quality_alignment, f_idx);
                        if let QualityAlignment::Mapping(vec) = &mut quality_alignment {
                            vec.push(None);
                        }
                        f_idx += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        unmatched_quality_docs += 1;
                        q_idx += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        let f_id = fast_index.doc_id_at(f_idx)?;
                        let q_id = quality.doc_id_at(q_idx)?;

                        match f_id.cmp(q_id) {
                            std::cmp::Ordering::Equal => {
                                if let QualityAlignment::Mapping(vec) = &mut quality_alignment {
                                    vec.push(Some(q_idx));
                                }
                                f_idx += 1;
                                q_idx += 1;
                            }
                            std::cmp::Ordering::Less => {
                                ensure_mapping(&mut quality_alignment, f_idx);
                                if let QualityAlignment::Mapping(vec) = &mut quality_alignment {
                                    vec.push(None);
                                }
                                f_idx += 1;
                            }
                            std::cmp::Ordering::Greater => {
                                unmatched_quality_docs += 1;
                                q_idx += 1;
                            }
                        }
                    }
                }
            }

            // Handle trailing fast docs
            if f_idx < f_count {
                ensure_mapping(&mut quality_alignment, f_idx);
                if let QualityAlignment::Mapping(vec) = &mut quality_alignment {
                    while vec.len() < f_count {
                        vec.push(None);
                    }
                }
            }

            while q_idx < q_count {
                let q_rec = quality.record_at(q_idx)?;
                if !crate::is_tombstoned_flags(q_rec.flags) {
                    unmatched_quality_docs += 1;
                }
                q_idx += 1;
            }

            if unmatched_quality_docs > 0 {
                warn!(
                    unmatched_quality_docs,
                    "quality index contains doc_ids that are not present in fast index"
                );
            }

            Some(quality)
        } else {
            None
        };

        #[cfg(feature = "ann")]
        let fast_ann_plan = paths.fast_ann.as_deref().and_then(|fast_ann_path| {
            plan_load_or_build_ann(
                fast_index,
                fast_ann_path,
                config.hnsw_threshold,
                &config,
                "fast",
            )
        });

        #[cfg(feature = "ann")]
        let quality_ann_plan = quality_index.and_then(|quality_index| {
            paths.quality_ann.as_deref().and_then(|quality_ann_path| {
                plan_load_or_build_ann(
                    quality_index,
                    quality_ann_path,
                    config.hnsw_threshold,
                    &config,
                    "quality",
                )
            })
        });

        #[cfg(feature = "ann")]
        {
            let fast_needs_persistence = fast_ann_plan
                .as_ref()
                .is_some_and(AnnOpenPlan::needs_persistence);
            let quality_needs_persistence = quality_ann_plan
                .as_ref()
                .is_some_and(AnnOpenPlan::needs_persistence);
            let persistence_prepared = if fast_needs_persistence || quality_needs_persistence {
                match validate_ann_persistence_paths(
                    paths,
                    fast_needs_persistence,
                    quality_needs_persistence,
                ) {
                    Ok(()) => true,
                    Err(error @ SearchError::InvalidConfig { .. }) => return Err(error),
                    Err(error) => {
                        warn!(
                            ?error,
                            fast_needs_persistence,
                            quality_needs_persistence,
                            "failed to prepare ANN persistence identities; all rebuilt ANN tiers \
                             stay in-memory for this process and the next startup may rebuild them"
                        );
                        false
                    }
                }
            } else {
                true
            };
            if persistence_prepared {
                if let (Some(plan), Some(path)) = (fast_ann_plan.as_ref(), paths.fast_ann()) {
                    persist_ann_plan(
                        plan,
                        path,
                        "fast",
                        paths,
                        fast_needs_persistence,
                        quality_needs_persistence,
                    );
                }
                if let (Some(plan), Some(path)) = (quality_ann_plan.as_ref(), paths.quality_ann()) {
                    persist_ann_plan(
                        plan,
                        path,
                        "quality",
                        paths,
                        fast_needs_persistence,
                        quality_needs_persistence,
                    );
                }
            }
        }

        #[cfg(feature = "ann")]
        let fast_ann = fast_ann_plan.map(|plan| plan.index);

        #[cfg(feature = "ann")]
        let quality_ann = quality_ann_plan.map(|plan| plan.index);

        #[cfg(feature = "ann")]
        debug!(
            fast_path = %paths.fast_index().display(),
            quality_path = ?paths.quality_index(),
            quality_available = quality_index.is_some(),
            fast_ann = fast_ann.is_some(),
            quality_ann = quality_ann.is_some(),
            doc_count = fast_index.record_count(),
            "opened two-tier index"
        );

        #[cfg(not(feature = "ann"))]
        debug!(
            fast_path = %paths.fast_index().display(),
            quality_path = ?paths.quality_index(),
            quality_available = quality_index.is_some(),
            doc_count = fast_index.record_count(),
            "opened two-tier index"
        );

        // Retain each tier's embedding-space identity from its artifact
        // header when the artifact carries one (bd-9xuj T2-C2). Legacy v1
        // artifacts have no identity header — `identity_v2()` is
        // structurally `None` there — and that absence is kept as the typed
        // legacy-unidentified state, never synthesized from id strings.
        let fast_space_fingerprint_hex = fast_index
            .identity_v2()
            .map(|identity| crate::fingerprint_hex(&identity.space_fingerprint));
        let quality_space_fingerprint_hex = quality_index
            .and_then(VectorIndex::identity_v2)
            .map(|identity| crate::fingerprint_hex(&identity.space_fingerprint));

        Ok(Self {
            fast_source,
            quality_source,
            native_fast_hnsw: None,
            native_quality_hnsw: None,
            #[cfg(feature = "ann")]
            fast_ann,
            #[cfg(feature = "ann")]
            quality_ann,
            #[cfg(feature = "ann")]
            ann_fallback_count: AtomicU64::new(0),
            last_zero_signal: AtomicU8::new(ZERO_SIGNAL_NONE),
            quality_alignment,
            config,
            fast_space_fingerprint_hex,
            quality_space_fingerprint_hex,
            fast_declared_identity: None,
            quality_declared_identity: None,
        })
    }

    /// Create a builder for a new two-tier index directory.
    ///
    /// The builder buffers added vectors and writes FSVI files on `finish()`.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Io` if the directory cannot be created.
    pub fn create(dir: &Path, config: TwoTierConfig) -> SearchResult<TwoTierIndexBuilder> {
        fs::create_dir_all(dir)?;
        Ok(TwoTierIndexBuilder::new(dir.to_path_buf(), config))
    }

    /// Borrow the fast tier's index for read paths.
    pub(super) fn fast_tier(&self) -> &VectorIndex {
        self.fast_source.index()
    }

    /// Borrow the quality tier's index for read paths, when loaded.
    pub(super) fn quality_tier(&self) -> Option<&VectorIndex> {
        self.quality_source.as_ref().map(TierSource::index)
    }

    /// Test-only mutable access to a PATH-OPENED fast tier (WAL injection).
    ///
    /// `None` for an admitted v2 tier: sealed owners are never mutably
    /// exposed, in tests or otherwise.
    #[cfg(test)]
    pub(super) fn fast_tier_mut_for_test(&mut self) -> Option<&mut VectorIndex> {
        match &mut self.fast_source {
            TierSource::PathOpened(index) => Some(index),
            TierSource::AdmittedV2 { .. } => None,
        }
    }

    /// The retained sealed admission owner of the fast tier, when this index
    /// was opened through exact FSVI v2 admission
    /// ([`Self::open_admitted_v2_with_paths`]).
    ///
    /// The owner carries the `Arc`'d validated byte image, the complete
    /// [`crate::FsviV2Witness`], and the publication state
    /// ([`ValidatedFsviBytes::published_wal_absent`]). Replacing, renaming,
    /// or mutating the source pathname after admission cannot alter what the
    /// retained owner serves. `None` for plain v1 path-opened tiers.
    ///
    /// # Drop order
    ///
    /// The returned borrow cannot be held across a refresh. bd-3t52d C2 left
    /// the "borrowed views cannot outlive a `&mut self` install" argument
    /// asserted but unpinned; this is the pin (bd-r6lwt). Installing a
    /// successor replaces the `TierSource` the borrow points into, so the
    /// borrow checker must reject holding one across
    /// [`Self::try_replace_admitted_v2`]. The error code is pinned too, so a
    /// snippet that stops compiling for an unrelated reason -- a renamed method
    /// or a moved re-export -- fails this doctest instead of passing it.
    ///
    /// ```compile_fail,E0502
    /// use frankensearch_index::{FsviV2IdentityBinding, TwoTierIndex, TwoTierIndexPaths};
    ///
    /// fn hold_owner_across_install(
    ///     index: &mut TwoTierIndex,
    ///     paths: &TwoTierIndexPaths,
    ///     binding: &FsviV2IdentityBinding,
    /// ) {
    ///     let owner = index.fast_admitted_owner();
    ///     let _ = index.try_replace_admitted_v2(paths, binding, None);
    ///     let _ = owner;
    /// }
    /// ```
    ///
    /// The complementary runtime half -- that field drop order *inside* the
    /// owner is not load-bearing, because the serving index shares the owner's
    /// allocation rather than borrowing it -- is pinned by
    /// `owner_drop_order_is_not_load_bearing_because_the_image_is_shared_not_borrowed`.
    #[must_use]
    pub fn fast_admitted_owner(&self) -> Option<&ValidatedFsviBytes> {
        self.fast_source.admitted_owner()
    }

    /// The exact identity binding the fast tier was admitted under, when this
    /// index was opened through exact FSVI v2 admission (bd-ctzo).
    ///
    /// This is the artifact's own identity, not a caller's claim about it:
    /// admission refused the open unless the persisted canonical identity
    /// bytes, every component fingerprint, the storage contract and the
    /// full-width generation were byte-equal to it. It is what a query-side
    /// seam joins against, because the owner retains identity only as
    /// one-way canonical bytes.
    #[must_use]
    pub const fn fast_admitted_binding(&self) -> Option<&FsviV2IdentityBinding> {
        self.fast_source.admitted_binding()
    }

    /// Quality-tier counterpart of [`Self::fast_admitted_binding`].
    #[must_use]
    pub fn quality_admitted_binding(&self) -> Option<&FsviV2IdentityBinding> {
        self.quality_source
            .as_ref()
            .and_then(TierSource::admitted_binding)
    }

    /// Opt into native HNSW candidate retrieval for the admitted fast tier.
    ///
    /// This routes [`ActivatedTierSearch::search_fast`](super::ActivatedTierSearch::search_fast) and the fast arm of
    /// [`ActivatedTierSearch::search_union`](super::ActivatedTierSearch::search_union) through the native graph. The
    /// progressive searcher binds the actual fast embedding response before
    /// traversal. Quality retrieval remains independently configured, and
    /// [`ActivatedTier::search_top_k`](super::ActivatedTier::search_top_k) remains an explicit exact scan.
    ///
    /// Construction is synchronous CPU work on the caller's lane and shares
    /// the exact admitted owner. No sidecar is read or written. Graph memory
    /// scales with physical rows and degree; tombstone routing can widen query
    /// candidates to the full physical row count. Smaller beams are approximate
    /// without a certified recall claim. Build and query errors propagate
    /// without an exact retry. A failed build preserves the previous policy.
    ///
    /// [`Self::try_replace_admitted_v2`] builds the successor's graph before
    /// swapping either tier. Fresh pathname opens require explicit opt-in again,
    /// either by construction or [`Self::load_native_fast_hnsw`].
    ///
    /// # Errors
    /// Rejects invalid graph parameters, legacy tiers, or disagreement between
    /// the complete generation, document-ID contract or semantic/control kind
    /// of admitted tiers, before graph construction.
    pub fn enable_native_fast_hnsw(&mut self, params: HnswParams, seed: u64) -> SearchResult<()> {
        params.validate()?;
        self.validate_native_hnsw_tiers("fast")?;
        let TierSource::AdmittedV2 { owner, .. } = &self.fast_source else {
            unreachable!("native tier validation requires an admitted fast owner");
        };
        let graph = ValidatedNativeHnsw::build(Arc::clone(owner), params, seed)?;
        self.native_fast_hnsw = Some(NativeTierHnsw {
            params,
            seed,
            graph: Some(graph),
        });
        Ok(())
    }

    /// Whether admitted fast retrieval currently uses a native graph.
    ///
    /// This is independent of quality retrieval and the legacy `ann` feature.
    #[must_use]
    pub fn has_native_fast_hnsw(&self) -> bool {
        self.native_fast_hnsw
            .as_ref()
            .is_some_and(|policy| policy.graph.is_some())
    }

    /// Opt into native HNSW candidate retrieval for the admitted quality tier.
    ///
    /// This routes [`ActivatedTierSearch::search_quality`](super::ActivatedTierSearch::search_quality) and its quality
    /// contribution to [`ActivatedTierSearch::search_union`](super::ActivatedTierSearch::search_union) through the native
    /// graph. The progressive searcher therefore retrieves quality candidates
    /// independently of its fast pool. Query producer admission still precedes
    /// traversal; [`ActivatedTier::search_top_k`](super::ActivatedTier::search_top_k) remains an explicit exact scan.
    /// Fast retrieval is independently configured; the legacy ANN adapter is
    /// unaffected.
    ///
    /// Construction is synchronous CPU work on the caller's lane. It retains
    /// the quality tier's exact admitted owner without copying its vector slab,
    /// and installs the graph only after successful structural admission. No
    /// sidecar is opened, written, repaired or selected. Graph memory scales
    /// with physical rows and configured degree; tombstones remain routing
    /// vertices and may widen query candidate memory to the full physical row
    /// count. Smaller beams are approximate, without a certified recall claim.
    ///
    /// The policy is reapplied transactionally by
    /// [`Self::try_replace_admitted_v2`]. A successor without quality keeps the
    /// policy with no graph; a later quality-bearing successor builds its own
    /// graph. A fresh pathname reopen starts in exact mode until explicitly
    /// opted in again, either by construction or [`Self::load_native_quality_hnsw`].
    /// Build and query failures propagate without exact retry.
    ///
    /// # Errors
    /// Rejects invalid graph parameters, legacy tiers, or disagreement between
    /// the complete generation, document-ID contract or semantic/control kind
    /// of the admitted tiers. A failed build leaves the prior policy unchanged.
    pub fn enable_native_quality_hnsw(
        &mut self,
        params: HnswParams,
        seed: u64,
    ) -> SearchResult<()> {
        params.validate()?;
        self.validate_native_hnsw_tiers("quality")?;
        let graph = match self.quality_source.as_ref() {
            Some(TierSource::AdmittedV2 { owner, .. }) => {
                Some(ValidatedNativeHnsw::build(Arc::clone(owner), params, seed)?)
            }
            Some(TierSource::PathOpened(_)) => {
                unreachable!("native tier validation refuses an unadmitted quality owner");
            }
            None => None,
        };
        self.native_quality_hnsw = Some(NativeTierHnsw {
            params,
            seed,
            graph,
        });
        Ok(())
    }

    /// Persist the installed fast graph and its exact-owner receipt.
    ///
    /// The destination must use the `.fshnsw` extension. Its adjacent
    /// `.fshnsw.receipt` binds the graph to the complete retained FSVI bytes,
    /// generation, producer, input, storage, live document set, parameters and
    /// seed. Neither output may alias either vector artifact.
    ///
    /// This is an explicit staging operation. The caller owns a trusted,
    /// generation-specific destination and serializes its writers. Publish
    /// the containing generation only after both files are durable: the two
    /// file replacements are not one transaction. This never selects a new
    /// generation or changes the installed retrieval policy.
    ///
    /// # Errors
    /// Returns an error if no fast graph is installed, an output aliases a
    /// vector artifact, or graph/receipt validation or persistence fails.
    pub fn save_native_fast_hnsw(
        &self,
        graph_path: &Path,
    ) -> SearchResult<NativeHnswGenerationReceiptV2> {
        self.save_native_hnsw_policy(self.native_fast_hnsw.as_ref(), "fast", graph_path)
    }

    /// Persist the installed quality graph and its exact-owner receipt.
    ///
    /// Follows the explicit staging and durability contract of
    /// [`Self::save_native_fast_hnsw`]. A fast-only generation cannot persist
    /// an absent quality graph, even when its quality policy is retained.
    ///
    /// # Errors
    /// Returns an error if no quality graph is installed, an output aliases a
    /// vector artifact, or graph/receipt validation or persistence fails.
    pub fn save_native_quality_hnsw(
        &self,
        graph_path: &Path,
    ) -> SearchResult<NativeHnswGenerationReceiptV2> {
        self.save_native_hnsw_policy(self.native_quality_hnsw.as_ref(), "quality", graph_path)
    }

    /// Load and install a persisted fast graph for this exact retained tier.
    ///
    /// Receipt admission checks the complete owner identity and bytes before
    /// graph loading. The graph's stored parameters and seed must also equal
    /// the requested policy. Loading performs no writes and never rebuilds or
    /// silently falls back to exact retrieval. Any failure preserves both
    /// tiers and their existing retrieval policies. After success, queries
    /// retain the admitted graph in memory; later pathname changes cannot
    /// retarget it. Generation replacement rebuilds for the successor owner.
    ///
    /// # Errors
    /// Rejects legacy/mixed tiers, invalid or mismatched parameters/seed, and
    /// missing, stale, corrupt or incompatible graph/receipt artifacts.
    pub fn load_native_fast_hnsw(
        &mut self,
        graph_path: &Path,
        params: HnswParams,
        seed: u64,
    ) -> SearchResult<()> {
        let policy = self.load_native_hnsw_policy(
            Some(&self.fast_source),
            "fast",
            graph_path,
            params,
            seed,
        )?;
        self.native_fast_hnsw = Some(policy);
        Ok(())
    }

    /// Load and install a persisted quality graph for this exact retained tier.
    ///
    /// Follows [`Self::load_native_fast_hnsw`]'s read-only, validate-before-swap
    /// contract. A missing quality tier is an error; a sidecar cannot supply
    /// missing vector coverage. Quality candidates remain independent of the
    /// fast candidate pool after loading.
    ///
    /// # Errors
    /// Rejects absent/legacy/mixed tiers, invalid or mismatched parameters/seed,
    /// and missing, stale, corrupt or incompatible graph/receipt artifacts.
    pub fn load_native_quality_hnsw(
        &mut self,
        graph_path: &Path,
        params: HnswParams,
        seed: u64,
    ) -> SearchResult<()> {
        let policy = self.load_native_hnsw_policy(
            self.quality_source.as_ref(),
            "quality",
            graph_path,
            params,
            seed,
        )?;
        self.native_quality_hnsw = Some(policy);
        Ok(())
    }

    fn save_native_hnsw_policy(
        &self,
        policy: Option<&NativeTierHnsw>,
        tier: &str,
        graph_path: &Path,
    ) -> SearchResult<NativeHnswGenerationReceiptV2> {
        let graph = policy
            .and_then(|policy| policy.graph.as_ref())
            .ok_or_else(|| SearchError::InvalidConfig {
                field: format!("two_tier.native_{tier}_hnsw.graph"),
                value: "absent".to_owned(),
                reason: "build or load this tier's graph before persisting it".to_owned(),
            })?;
        let receipt_path = native_hnsw_generation_receipt_path(graph_path)?;
        let outputs = [graph_path, receipt_path.as_path()];
        for output in outputs {
            for vector in [
                Some(self.fast_source.artifact_path()),
                self.quality_source.as_ref().map(TierSource::artifact_path),
            ]
            .into_iter()
            .flatten()
            {
                if paths_alias(output, vector)? {
                    return Err(SearchError::InvalidConfig {
                        field: format!("two_tier.native_{tier}_hnsw.path"),
                        value: "vector-alias".to_owned(),
                        reason:
                            "graph and receipt outputs must be distinct from both vector artifacts"
                                .to_owned(),
                    });
                }
            }
        }
        if paths_alias(graph_path, &receipt_path)? {
            return Err(SearchError::InvalidConfig {
                field: format!("two_tier.native_{tier}_hnsw.path"),
                value: "receipt-alias".to_owned(),
                reason: "graph and receipt outputs must be distinct artifacts".to_owned(),
            });
        }
        graph.save(graph_path)
    }

    fn load_native_hnsw_policy(
        &self,
        source: Option<&TierSource>,
        tier: &str,
        graph_path: &Path,
        params: HnswParams,
        seed: u64,
    ) -> SearchResult<NativeTierHnsw> {
        params.validate()?;
        self.validate_native_hnsw_tiers(tier)?;
        let Some(TierSource::AdmittedV2 { owner, .. }) = source else {
            return Err(SearchError::InvalidConfig {
                field: format!("two_tier.native_{tier}_hnsw.owner"),
                value: "absent".to_owned(),
                reason: "a persisted graph requires its admitted vector tier".to_owned(),
            });
        };
        let (graph, receipt) = ValidatedNativeHnsw::load(Arc::clone(owner), graph_path)?;
        let parameters_match = [
            (receipt.params.m, params.m),
            (receipt.params.m0, params.m0),
            (receipt.params.ef_construction, params.ef_construction),
            (receipt.params.ef_search, params.ef_search),
        ]
        .into_iter()
        .all(|(loaded, requested)| usize::try_from(loaded) == Ok(requested));
        if !parameters_match || receipt.seed != seed {
            return Err(SearchError::InvalidConfig {
                field: format!("two_tier.native_{tier}_hnsw.policy"),
                value: "mismatch".to_owned(),
                reason: "persisted graph parameters and seed must equal the requested policy"
                    .to_owned(),
            });
        }
        Ok(NativeTierHnsw {
            params,
            seed,
            graph: Some(graph),
        })
    }

    /// Join the complete retained generation and cross-tier contracts before
    /// any graph reads vectors. Legacy publication nonces are insufficient.
    fn validate_native_hnsw_tiers(&self, requested_tier: &str) -> SearchResult<()> {
        let (fast_owner, fast_binding) =
            self.fast_source
                .admitted_pair()
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: format!("two_tier.native_{requested_tier}_hnsw.owner"),
                    value: "legacy-fast".to_owned(),
                    reason: "native retrieval requires an admitted FSVI v2 index".to_owned(),
                })?;
        match self.quality_source.as_ref() {
            Some(TierSource::AdmittedV2 { owner, binding, .. }) => {
                // Legacy publication nonces are not the full v2 generation.
                // Join the exact retained witnesses before graph construction.
                if fast_owner.witness().generation != owner.witness().generation {
                    return Err(SearchError::InvalidConfig {
                        field: format!("two_tier.native_{requested_tier}_hnsw.generation"),
                        value: "mismatch".to_owned(),
                        reason:
                            "native retrieval tiers must share one complete artifact generation"
                                .to_owned(),
                    });
                }
                let fast_identity = &fast_binding.frozen_identity().identity;
                let quality_identity = &binding.frozen_identity().identity;
                if fast_identity.input.doc_id_semantics != quality_identity.input.doc_id_semantics
                    || fast_identity.space.kind != quality_identity.space.kind
                {
                    return Err(SearchError::InvalidConfig {
                        field: format!("two_tier.native_{requested_tier}_hnsw.tier_contract"),
                        value: "mismatch".to_owned(),
                        reason: "native retrieval tiers require one document-ID contract and semantic/control kind"
                            .to_owned(),
                    });
                }
            }
            Some(TierSource::PathOpened(_)) => {
                return Err(SearchError::InvalidConfig {
                    field: format!("two_tier.native_{requested_tier}_hnsw.owner"),
                    value: "legacy-quality".to_owned(),
                    reason: "native retrieval cannot adopt an unadmitted quality tier".to_owned(),
                });
            }
            None => {}
        }
        Ok(())
    }

    /// Whether admitted quality retrieval currently uses a native graph.
    ///
    /// A requested policy on a fast-only generation returns false; it does not
    /// invent quality coverage. This is independent of the legacy `ann` feature.
    #[must_use]
    pub fn has_native_quality_hnsw(&self) -> bool {
        self.native_quality_hnsw
            .as_ref()
            .is_some_and(|policy| policy.graph.is_some())
    }

    /// Admit a candidate generation and install it only if every tier of it
    /// is admitted (bd-typed-fsvi-owner-retention C5).
    ///
    /// The candidate is opened in full — path validation, per-tier exact
    /// admission against its own binding, alignment planning and ANN
    /// planning — BEFORE anything of `self` is disturbed. Any failure
    /// returns the typed error with the previously retained owners still
    /// installed and still serving byte-identically: there is no window in
    /// which this index is neither the old generation nor the new one, and a
    /// rejected candidate can neither modify nor evict the incumbent.
    ///
    /// A reader that opened its own [`TwoTierIndex`] over the same artifacts
    /// is unaffected either way — each index retains its own admitted byte
    /// image, so an installed successor never invalidates a live predecessor.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::open_admitted_v2_with_paths`] and of any
    /// requested native graph build, before any state change.
    pub fn try_replace_admitted_v2(
        &mut self,
        paths: &TwoTierIndexPaths,
        fast_binding: &FsviV2IdentityBinding,
        quality_binding: Option<&FsviV2IdentityBinding>,
    ) -> SearchResult<()> {
        let mut candidate = match Self::open_admitted_v2_with_paths(
            paths,
            self.config.clone(),
            fast_binding,
            quality_binding,
        ) {
            Ok(candidate) => candidate,
            Err(error) => {
                // Bounded digests only: generation sequence and truncated
                // fingerprint prefixes of the RETAINED owner, plus the typed
                // reason. Never vectors, queries, document text, or paths.
                warn!(
                    incumbent_generation = self.admitted_generation_sequence(),
                    incumbent_space = self.bounded_fast_space_digest(),
                    reason = %error,
                    "candidate generation was refused; the retained owner keeps serving"
                );
                return Err(error);
            }
        };
        if let Some(policy) = &self.native_fast_hnsw {
            candidate.enable_native_fast_hnsw(policy.params, policy.seed)?;
        }
        if let Some(policy) = &self.native_quality_hnsw {
            // Build against the candidate's own retained allocation before
            // replacing any part of the incumbent. A failure leaves both its
            // previous vector owner and graph installed.
            candidate.enable_native_quality_hnsw(policy.params, policy.seed)?;
        }
        info!(
            previous_generation = self.admitted_generation_sequence(),
            installed_generation = candidate.admitted_generation_sequence(),
            installed_space = candidate.bounded_fast_space_digest(),
            installed_records = candidate.doc_count(),
            "installed an admitted successor generation"
        );
        *self = candidate;
        Ok(())
    }

    /// Generation sequence of the retained fast owner, or `None` for a tier
    /// that was not admitted as FSVI v2.
    fn admitted_generation_sequence(&self) -> Option<u64> {
        self.fast_admitted_owner()
            .map(|owner| owner.witness().generation.sequence)
    }

    /// First 16 hex characters of the fast tier's space fingerprint — enough
    /// to correlate an event, bounded so a log line can never carry a full
    /// identity image.
    fn bounded_fast_space_digest(&self) -> Option<String> {
        self.fast_space_fingerprint_hex
            .as_ref()
            .map(|hex| hex.chars().take(16).collect())
    }
}
