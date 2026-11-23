//! Two-tier index wrapper for fast and quality vector indices.
//!
//! `TwoTierIndex` provides a single object that coordinates:
//! - fast-tier retrieval from `vector.fast.idx` (or `vector.idx` fallback)
//! - optional quality-tier rescoring from `vector.quality.idx`
//! - doc-id alignment between both tiers

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicU8;
#[cfg(feature = "ann")]
use std::sync::atomic::AtomicU64;

use frankensearch_core::TwoTierConfig;
use frankensearch_core::config::ZeroSignalReason;
use frankensearch_core::generation::EmbeddingIdentityBundleV1;

#[cfg(feature = "ann")]
use crate::HnswIndex;
use crate::native_hnsw::{HnswParams, ValidatedNativeHnsw};
use crate::{FsviV2IdentityBinding, ValidatedFsviBytes, VectorIndex};

mod activation;
#[cfg(feature = "ann")]
mod ann;
mod builder;
mod observe;
mod open;
mod paths;
mod search;
#[cfg(test)]
mod tests;

pub use activation::{ActivatedTier, ActivatedTierSearch};
pub use builder::{SemanticVectorTier, TwoTierIndexBuilder};
pub use observe::{FsviTierObservation, FsviV1Observation, observe_tier};
pub use paths::TwoTierIndexPaths;

/// Preferred fast-tier index filename.
pub const VECTOR_INDEX_FAST_FILENAME: &str = "vector.fast.idx";
/// Optional quality-tier index filename.
pub const VECTOR_INDEX_QUALITY_FILENAME: &str = "vector.quality.idx";
/// Fallback single-tier index filename used as the fast tier when no dedicated fast file exists.
pub const VECTOR_INDEX_FALLBACK_FILENAME: &str = "vector.idx";
/// Serialized fast-tier ANN sidecar.
#[cfg(feature = "ann")]
pub const VECTOR_ANN_FAST_FILENAME: &str = "vector.fast.hnsw";
/// Serialized quality-tier ANN sidecar.
#[cfg(feature = "ann")]
pub const VECTOR_ANN_QUALITY_FILENAME: &str = "vector.quality.hnsw";

#[derive(Debug)]
enum QualityAlignment {
    None,
    Aligned,
    Mapping(Vec<Option<usize>>),
}

/// One opened tier: either a plain v1 path-opened index or a fully retained
/// sealed FSVI v2 admission owner (bd-9xuj C4-write r2).
///
/// The `AdmittedV2` variant retains the complete [`ValidatedFsviBytes`]
/// owner — the `Arc`'d byte image, the complete admission witness, and the
/// publication state — honoring the owner contract on [`ValidatedFsviBytes`]:
/// the owner is never converted into a mutable/path-opened [`VectorIndex`];
/// the tier's index is only ever borrowed from inside the retained owner.
///
/// It also retains the exact [`FsviV2IdentityBinding`] the tier was admitted
/// under (bd-ctzo). That binding is the ONLY materialized
/// `EmbeddingIdentityBundleV1` describing this artifact — the owner carries
/// the identity as one-way canonical bytes, and
/// [`EmbeddingIdentityBundleV1`] has no decoder — so query-side producer
/// conformance would otherwise have to take an expected bundle from whoever
/// happens to call `search`. Retaining it here makes owner and binding a
/// single indivisible admission product: they are installed together by
/// [`TwoTierIndex::open_admitted_v2_with_paths`] and replaced together by
/// [`TwoTierIndex::try_replace_admitted_v2`], so a binding that does not
/// describe this artifact is not a refusal a caller can trigger — it is
/// unrepresentable.
// The variants retain different metadata, but exactly one `TierSource`
// exists per tier per opened index. The admitted owner can also be shared
// with a native graph without duplicating its validated vector allocation.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum TierSource {
    /// Plain v1 [`VectorIndex::open`] tier (mutable mapping, WAL-bearing).
    PathOpened(VectorIndex),
    /// Sealed FSVI v2 admission owner, retained in full together with the
    /// binding it was admitted under.
    AdmittedV2 {
        /// The sealed byte owner, witness and publication state.
        owner: Arc<ValidatedFsviBytes>,
        /// The exact identity this artifact was admitted against.
        /// [`validate_expected_v2_binding`] proved, before this value
        /// existed, that the persisted canonical identity bytes, every
        /// component fingerprint, the storage contract and the full-width
        /// generation are byte-equal to it.
        ///
        /// [`validate_expected_v2_binding`]: crate::VectorIndex::open_admitted_v2
        binding: FsviV2IdentityBinding,
        /// Absolute path the artifact was admitted from. The owner keeps only
        /// bytes (its index path is the synthetic `<owned-fsvi-v2>`), so
        /// output-alias checks compare against this path instead.
        source_path: PathBuf,
    },
}

impl TierSource {
    /// Borrow the tier's index for read-only serving.
    ///
    /// For `AdmittedV2` this borrows the validated index INSIDE the retained
    /// owner (crate-internal field access); the owner itself stays sealed and
    /// is never moved out of, so its byte/witness/publication capabilities
    /// remain intact for the lifetime of the tier.
    fn index(&self) -> &VectorIndex {
        match self {
            Self::PathOpened(index) => index,
            Self::AdmittedV2 { owner, .. } => &owner.index,
        }
    }

    /// The retained sealed admission owner, when this tier came from exact
    /// FSVI v2 admission.
    fn admitted_owner(&self) -> Option<&ValidatedFsviBytes> {
        match self {
            Self::PathOpened(_) => None,
            Self::AdmittedV2 { owner, .. } => Some(owner),
        }
    }

    /// The identity binding this tier was admitted under, when it came from
    /// exact FSVI v2 admission.
    const fn admitted_binding(&self) -> Option<&FsviV2IdentityBinding> {
        match self {
            Self::PathOpened(_) => None,
            Self::AdmittedV2 { binding, .. } => Some(binding),
        }
    }

    /// Owner and binding together, as the single admission product they are.
    ///
    /// Activation consumes this rather than two independent `Option`s so the
    /// half-present state — an owner paired with someone else's binding, or a
    /// binding with no artifact behind it — has no representation to check
    /// for.
    fn admitted_pair(&self) -> Option<(&ValidatedFsviBytes, &FsviV2IdentityBinding)> {
        match self {
            Self::PathOpened(_) => None,
            Self::AdmittedV2 { owner, binding, .. } => Some((owner, binding)),
        }
    }

    /// The on-disk artifact this tier serves.
    fn artifact_path(&self) -> &Path {
        match self {
            Self::PathOpened(index) => &index.path,
            Self::AdmittedV2 { source_path, .. } => source_path,
        }
    }
}

/// Explicit tier retrieval policy and its current owner's built or loaded graph.
/// A fast-only successor retains the policy with no graph until quality returns.
#[derive(Debug)]
struct NativeTierHnsw {
    params: HnswParams,
    seed: u64,
    graph: Option<ValidatedNativeHnsw>,
}

/// Dual-index container used by progressive search orchestration.
#[derive(Debug)]
pub struct TwoTierIndex {
    fast_source: TierSource,
    quality_source: Option<TierSource>,
    native_fast_hnsw: Option<NativeTierHnsw>,
    native_quality_hnsw: Option<NativeTierHnsw>,
    #[cfg(feature = "ann")]
    fast_ann: Option<HnswIndex>,
    #[cfg(feature = "ann")]
    quality_ann: Option<HnswIndex>,
    #[cfg(feature = "ann")]
    ann_fallback_count: AtomicU64,
    /// Last state-scoped [`ZeroSignalReason`] observed on the fast tier,
    /// encoded via [`zero_signal_code`]; [`ZERO_SIGNAL_NONE`] when the last
    /// search produced hits. Availability transitions log once per state
    /// change, never per query (bd-tqhc no-warn-storm policy).
    last_zero_signal: AtomicU8,
    quality_alignment: QualityAlignment,
    config: TwoTierConfig,
    /// Lowercase hex SHA-256 fingerprint of the fast tier's embedding space,
    /// when known (bd-9xuj T2-C2). Filled from the artifact's validated FSVI
    /// v2 identity header at open, or from the builder-declared producing
    /// identity when [`TwoTierIndexBuilder::finish`] wrote this tier in the
    /// current process. `None` is the typed legacy-unidentified state (v1
    /// artifacts) — never fabricated from the header's id/revision strings.
    fast_space_fingerprint_hex: Option<String>,
    /// Quality-tier counterpart of [`Self::fast_space_fingerprint_hex`].
    quality_space_fingerprint_hex: Option<String>,
    /// Complete identity bundle of the embedder that produced the fast
    /// tier's vectors, when the builder that wrote this tier declared it
    /// (bd-9xuj T2-C2). Retained so bundle-holding seams can apply the full
    /// admission law
    /// ([`frankensearch_core::BoundQueryEmbedding::verify_producer_conformance`])
    /// rather than the fingerprint-only join. Its storage component
    /// describes the PRODUCER's output contract, not this index's persisted
    /// encoding. Process-local for v1 artifacts: a reopen from disk has no
    /// bundle to retain and stays `None`.
    fast_declared_identity: Option<EmbeddingIdentityBundleV1>,
    /// Quality-tier counterpart of [`Self::fast_declared_identity`].
    quality_declared_identity: Option<EmbeddingIdentityBundleV1>,
}

/// Sentinel for "the last fast-tier search produced hits".
const ZERO_SIGNAL_NONE: u8 = u8::MAX;

/// Stable per-variant code for the transition state machine.
const fn zero_signal_code(reason: ZeroSignalReason) -> u8 {
    match reason {
        ZeroSignalReason::CallerRequestedZeroK => 0,
        ZeroSignalReason::FilterEliminatedAll => 1,
        ZeroSignalReason::NonFiniteQuery => 2,
        ZeroSignalReason::ZeroNormQuery => 3,
        ZeroSignalReason::NewlyCreatedEmpty => 4,
        ZeroSignalReason::AllTombstoned => 5,
        ZeroSignalReason::WalOnlyNoLiveRecords => 6,
        ZeroSignalReason::NoUsableVectors => 7,
        ZeroSignalReason::AnnReturnedEmptyDespiteUsableVectors => 8,
    }
}
