//! Canonical generation inspection, identity admission, and typed refusals.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use tracing::warn;

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::{EmbeddingIdentityBundleV1, QuantizationFormat};
use frankensearch_core::traits::Embedder;
use frankensearch_core::{BoundQueryEmbedding, SpaceIdentityAdmission};
use frankensearch_index::two_tier::{FsviTierObservation, observe_tier};
use frankensearch_index::{
    FsviAdmissionError, FsviV2IdentityBinding, FsviV2IdentityMetadata,
    VECTOR_INDEX_FALLBACK_FILENAME, VECTOR_INDEX_FAST_FILENAME, VECTOR_INDEX_QUALITY_FILENAME,
    ValidatedFsviBytes, VectorIndex, VectorMetadata,
};

use super::RefreshWorker;

// ---------------------------------------------------------------------------
// Canonical generation inspection + typed refusals
// ---------------------------------------------------------------------------

/// Subdirectory of the index dir holding staged (non-canonical) identity-bound
/// v2 replacement generations. Never discovered by `TwoTierIndex::open`.
pub(super) const STAGED_V2_DIR_NAME: &str = "v2-staged";

/// Typed state of one canonical tier artifact on disk.
pub(super) enum TierState {
    /// No artifact at the tier's path.
    Missing,
    /// Recognized legacy FSVI v1 bytes, established by READ-ONLY observation
    /// only (r2 repair: classification never runs the mutable
    /// [`VectorIndex::open`], which deletes stale WAL sidecars and truncates
    /// corrupt WAL trailers). `retains_content` is
    /// [`frankensearch_index::two_tier::FsviV1Observation::retains_content`]:
    /// conservative — the main slab's `record_count` counts tombstoned rows
    /// too, so an all-tombstoned v1 tier fails CLOSED here instead of
    /// classifying as bootstrap-replaceable. Flag-level precision returns
    /// when a read-only record-table inspector lands in the index crate root
    /// (the observational-open train).
    LegacyV1 { retains_content: bool },
    /// Identity-complete FSVI v2 header (attested identity available from
    /// the artifact's own bytes). Header recognition only: content admission
    /// (digest recomputation, sealed-owner retention) is performed by
    /// [`RefreshWorker::admit_existing_generation`] through
    /// [`VectorIndex::open_admitted_v2`] before any admissibility claim is
    /// made.
    V2 { metadata: Box<VectorMetadata> },
}

/// Inspect one tier artifact without mutating anything — including its WAL
/// sidecar and its timestamps, to the extent the platform allows.
///
/// All recognition goes through the read-only
/// [`frankensearch_index::two_tier::observe_tier`]: v2 because plain
/// [`VectorIndex::open`] is strictly v1 (any refresh branch keyed on
/// `identity_v2()` from a plain open would be unreachable for on-disk v2),
/// and v1 because the mutable open's WAL side effects
/// (stale-sidecar deletion, corrupt-trailer truncation) must never fire
/// during classification (NO-GO item 2, 868c0801 refresh.rs:244-250).
fn inspect_tier(path: &Path) -> SearchResult<TierState> {
    if !path.exists() {
        return Ok(TierState::Missing);
    }
    match observe_tier(path)? {
        FsviTierObservation::V2IdentityComplete(metadata) => Ok(TierState::V2 { metadata }),
        FsviTierObservation::V1(observation) => Ok(TierState::LegacyV1 {
            retains_content: observation.retains_content(),
        }),
        FsviTierObservation::UpgradeRequired(upgrade) => Err(SearchError::InvalidConfig {
            field: "refresh.index_format".to_owned(),
            value: format!("fsvi-v{}", upgrade.found_version),
            reason: format!(
                "existing generation uses a newer FSVI schema than this reader supports \
                 (through v{}); a reader upgrade is required before any refresh",
                upgrade.supported_version
            ),
        }),
    }
}

/// The landed generation-containment refusal for identityless legacy tiers
/// (origin 5386b39e): equal display ids, revisions, and dimensions cannot
/// prove that two embeddings inhabit the same vector space, so a live v1
/// tier fails closed and requires a full identity-bound rebuild.
fn identityless_refusal(tier: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("refresh.{tier}_index_identity"),
        value: "identityless-fsvi-v1".to_owned(),
        reason: "refusing incremental vector merge because the existing generation has no \
                 complete immutable embedding identity; a full identity-bound rebuild is required"
            .to_owned(),
    }
}

/// An attested quality tier retains content but this worker has no quality
/// embedder, so no identity-bound replacement of that tier can be produced.
fn quality_republication_unavailable() -> SearchError {
    SearchError::InvalidConfig {
        field: "refresh.index_publication".to_owned(),
        value: "identity-bound-republication-unavailable".to_owned(),
        reason: "the attested quality tier retains content but this worker has no quality \
                 embedder to produce its identity-bound replacement; configure a quality \
                 embedder attesting the same producer or run a full rebuild"
            .to_owned(),
    }
}

/// The bd-xomn composite-generation-authority gate (C4-write disposition,
/// r2 wording).
///
/// This refusal is only reachable AFTER the existing generation was FULLY
/// ADMITTED ([`RefreshWorker::admit_existing_generation`]): header identity
/// gates joined, content and docset digests recomputed byte-for-byte via
/// [`VectorIndex::open_admitted_v2`], and the sealed owners retained through
/// the check. What remains unavailable is only canonical INSTALLATION:
/// installing a fast/quality replacement over the canonical filenames is a
/// split two-rename publication with no atomic pair authority. This slice
/// deliberately does not invent one; canonical publication reopens when the
/// composite generation-authority primitive (bd-xomn.1/.3) lands. The
/// refusal fires BEFORE any queue drain so the (currently permanent)
/// condition never consumes retry budget or drops queued documents.
pub(super) fn composite_authority_refusal(index_dir: &Path) -> SearchError {
    SearchError::InvalidConfig {
        field: "refresh.canonical_publication".to_owned(),
        value: "composite-generation-authority-unavailable".to_owned(),
        reason: format!(
            "the existing generation for {} was fully admitted (header identity gates plus \
             recomputed content/docset digests via exact v2 admission), and an identity-bound \
             replacement can be staged and proven via stage_identity_bound_generation, but \
             canonical installation of a split fast/quality generation pair is refused until \
             the composite generation-authority primitive lands (bd-xomn.1/.3); no per-tier \
             rename sequence can make the pair atomic",
            index_dir.display()
        ),
    }
}

/// Guard 7 at its strictest: only `SameProducer` is admitted in this slice.
///
/// A producer that differs from the expected/attested producer is refused
/// with this typed reason even when it carries a byte-identical golden-vector
/// certificate (i.e. even when the pairing would classify as
/// [`SpaceIdentityAdmission::ConformanceCompatibleProducer`]): certificate
/// equality attests what some implementation once produced, not what the
/// implementation executing THIS merge produces. The conformance-compatible
/// lane reopens when executing-producer attestation (attestation of the
/// running implementation, in code) lands; until then this is a deliberate
/// narrowing, and the copied-certificate attack shape (same space, foreign
/// producer, cloned certificate) is rejected by construction.
fn attestation_unavailable_refusal(
    tier: &str,
    query_producer_fingerprint: &str,
    expected_producer_fingerprint: &str,
) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("refresh.{tier}_producer_conformance"),
        value: "executing-producer-attestation-unavailable".to_owned(),
        reason: format!(
            "producer fingerprint {query_producer_fingerprint} shares the {tier} tier's \
             embedding space but is not the attested producer \
             {expected_producer_fingerprint}; this slice admits SameProducer only — a \
             golden-vector certificate alone cannot attest the implementation executing \
             this merge, so the conformance-compatible lane stays closed until \
             executing-producer attestation lands"
        ),
    }
}

/// Cross-space refusal at the artifact level (pre-drain): the executing
/// embedder's space does not join the attested space of the existing tier.
fn space_identity_refusal(
    tier: &str,
    executing_space_fingerprint: &str,
    attested_space_fingerprint: &str,
) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("refresh.{tier}_space_identity"),
        value: executing_space_fingerprint.to_owned(),
        reason: format!(
            "the executing {tier} embedder produces vectors in a different embedding space \
             than the attested {tier} generation (attested space fingerprint \
             {attested_space_fingerprint}); refusing the identity-bound merge — reindex \
             into a new generation instead"
        ),
    }
}

/// Map a typed [`FsviAdmissionError`] into the refresh error surface, naming
/// the tier. I/O and corruption pass through unchanged.
fn admission_error_to_refresh_error(
    error: FsviAdmissionError,
    tier: &str,
    path: &Path,
) -> SearchError {
    match error {
        FsviAdmissionError::Index(error) => error,
        other => SearchError::InvalidConfig {
            field: format!("refresh.{tier}_v2_admission"),
            value: path.display().to_string(),
            reason: other.to_string(),
        },
    }
}

/// Lowercase hex encoding of raw fingerprint bytes.
pub(super) fn fingerprint_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        })
}

/// The parsed v2 identity block of an inspected tier, or a typed error when
/// inspection and metadata disagree (never fabricated).
fn v2_identity_of<'metadata>(
    metadata: &'metadata VectorMetadata,
    tier: &str,
) -> SearchResult<&'metadata FsviV2IdentityMetadata> {
    metadata
        .identity_v2
        .as_ref()
        .ok_or_else(|| SearchError::InvalidConfig {
            field: format!("refresh.{tier}_index_identity"),
            value: "missing-v2-identity-metadata".to_owned(),
            reason: "inspection classified the artifact as FSVI v2 but its parsed metadata \
                     carries no identity block"
                .to_owned(),
        })
}

/// Derive the FSVI v2 artifact identity bundle for an embedder: the same
/// space, producer, and input contracts, with the storage component rewritten
/// to the canonical persisted contract (`fsvi-v2`, F16, little-endian) that
/// [`VectorIndex::create_v2`] requires. The result is re-validated: storage
/// is physical, so the space and producer fingerprints are unchanged.
pub(super) fn artifact_identity_for(
    embedder: &dyn Embedder,
) -> SearchResult<EmbeddingIdentityBundleV1> {
    let mut bundle = embedder.identity()?.clone();
    "fsvi-v2".clone_into(&mut bundle.storage.format);
    bundle.storage.quantization = QuantizationFormat::F16;
    "little-endian".clone_into(&mut bundle.storage.endianness);
    bundle.validate()?;
    Ok(bundle)
}

/// Reconstruct the exact identity binding an existing attested tier was
/// published under, from the executing embedder's identity plus the
/// artifact's own header metadata.
///
/// The bundle canonical bytes in a v2 header are encode-only (no structured
/// decoder exists, by design), so exact admission needs the caller to hold
/// the structured bundle. The executing embedder holds the space/producer/
/// input components (already gated equal to the attested fingerprints); only
/// the physical storage quantization can legitimately vary, so both F16 and
/// F32 candidates are tried against the header's full-bundle fingerprint.
fn reconstruct_admission_binding(
    metadata: &VectorMetadata,
    embedder_identity: &EmbeddingIdentityBundleV1,
    tier: &str,
) -> SearchResult<FsviV2IdentityBinding> {
    let identity = v2_identity_of(metadata, tier)?;
    let attested_bundle_hex = fingerprint_hex(&identity.identity_bundle_fingerprint);
    for quantization in [QuantizationFormat::F16, QuantizationFormat::F32] {
        let mut candidate = embedder_identity.clone();
        "fsvi-v2".clone_into(&mut candidate.storage.format);
        candidate.storage.quantization = quantization;
        "little-endian".clone_into(&mut candidate.storage.endianness);
        if candidate.validate().is_err() {
            continue;
        }
        if candidate.fingerprint() == attested_bundle_hex {
            return FsviV2IdentityBinding::new(identity.generation, candidate.freeze()?);
        }
    }
    Err(SearchError::InvalidConfig {
        field: format!("refresh.{tier}_storage_identity"),
        value: fingerprint_hex(&identity.storage_fingerprint),
        reason: format!(
            "the attested {tier} generation's identity bundle ({attested_bundle_hex}) cannot \
             be reconstructed from the executing embedder's identity under any supported \
             storage contract; identity-bound republication is unavailable for this artifact"
        ),
    })
}

/// Unique non-zero nonce material for one staged generation build attempt.
///
/// The generation nonce is uniqueness material, never a credential (see
/// [`frankensearch_core::generation::ArtifactGenerationIdentityV1`]): it exists so two build attempts of the
/// same sequence are distinguishable. Uniqueness comes from OS-seeded
/// [`std::collections::hash_map::RandomState`] draws (fresh keys per
/// process from OS randomness, distinct per instantiation), domain-mixed
/// with the deterministic build context.
pub(super) fn generation_nonce(index_dir: &Path, tier: &str, sequence: u64) -> [u8; 16] {
    use std::hash::BuildHasher as _;

    let mut hasher = Sha256::new();
    hasher.update(index_dir.as_os_str().as_encoded_bytes());
    hasher.update(tier.as_bytes());
    hasher.update(sequence.to_le_bytes());
    for draw in 0_u64..4 {
        let os_seeded = std::collections::hash_map::RandomState::new();
        hasher.update(os_seeded.hash_one(draw).to_le_bytes());
    }
    let digest = hasher.finalize();
    let mut nonce = [0_u8; 16];
    nonce.copy_from_slice(&digest[..16]);
    nonce
}

/// Guard 7 seam check on one bound embedding: apply the complete C1r2
/// admission law against `expected` and admit ONLY `SameProducer`.
/// A `ConformanceCompatibleProducer` outcome is telemetry-logged and refused
/// (see [`attestation_unavailable_refusal`] for the deliberate narrowing and
/// its reopening condition).
pub(super) fn require_same_producer(
    bound: &BoundQueryEmbedding,
    expected: &EmbeddingIdentityBundleV1,
    tier: &str,
) -> SearchResult<()> {
    match bound.verify_producer_conformance(expected, tier)? {
        SpaceIdentityAdmission::SameProducer => Ok(()),
        SpaceIdentityAdmission::ConformanceCompatibleProducer {
            query_producer_fingerprint,
            expected_producer_fingerprint,
        } => {
            warn!(
                target: "frankensearch.refresh",
                tier,
                %query_producer_fingerprint,
                %expected_producer_fingerprint,
                "conformance-compatible producer refused: executing-producer attestation \
                 unavailable (SameProducer-only slice)"
            );
            Err(attestation_unavailable_refusal(
                tier,
                &query_producer_fingerprint,
                &expected_producer_fingerprint,
            ))
        }
        // `SpaceIdentityAdmission` is #[non_exhaustive]: a future admission
        // basis this seam does not understand must fail CLOSED, never admit.
        other => {
            warn!(
                target: "frankensearch.refresh",
                tier,
                admission = other.code(),
                "unrecognized producer-conformance admission outcome refused (fail closed)"
            );
            Err(SearchError::InvalidConfig {
                field: format!("refresh.{tier}_producer_conformance"),
                value: other.code().to_owned(),
                reason: "unrecognized producer-conformance admission outcome; this \
                         SameProducer-only seam fails closed on admission bases it cannot \
                         verify"
                    .to_owned(),
            })
        }
    }
}

/// Gate one attested v2 tier against the executing embedder's artifact
/// bundle: space join first (typed cross-space refusal), then the
/// SameProducer-only producer law against the header's attested producer
/// fingerprint. Both expected values come from the artifact's OWN header —
/// never from a caller-supplied fingerprint.
fn admit_attested_tier(
    tier: &str,
    metadata: &VectorMetadata,
    executing_bundle: &EmbeddingIdentityBundleV1,
) -> SearchResult<()> {
    let identity = v2_identity_of(metadata, tier)?;
    let attested_space = fingerprint_hex(&identity.space_fingerprint);
    let executing_space = executing_bundle.space.fingerprint();
    if executing_space != attested_space {
        return Err(space_identity_refusal(
            tier,
            &executing_space,
            &attested_space,
        ));
    }
    let attested_producer = fingerprint_hex(&identity.producer_fingerprint);
    let executing_producer = executing_bundle.producer.fingerprint();
    if executing_producer != attested_producer {
        return Err(attestation_unavailable_refusal(
            tier,
            &executing_producer,
            &attested_producer,
        ));
    }
    Ok(())
}

/// One existing canonical tier admitted IN FULL at classification time:
/// binding reconstructed from the artifact's own header, content and docset
/// digests recomputed via [`VectorIndex::open_admitted_v2`], and the sealed
/// owner retained (never peeled).
pub(super) struct AdmittedCanonicalTier {
    /// The retained sealed admission owner. Its `Arc`'d bytes — not the
    /// canonical pathname — are the authority for every subsequent read.
    pub(super) owner: ValidatedFsviBytes,
    /// Attested space fingerprint (lowercase hex) read from the artifact's
    /// own validated header.
    pub(super) attested_space_hex: String,
}

/// Admit one existing attested canonical tier exactly and retain the sealed
/// owner plus the attested space fingerprint (lowercase hex) read from the
/// artifact's own header.
fn admit_existing_tier(
    path: &Path,
    metadata: &VectorMetadata,
    embedder: &dyn Embedder,
    tier: &str,
) -> SearchResult<AdmittedCanonicalTier> {
    let attested_space_hex = fingerprint_hex(&v2_identity_of(metadata, tier)?.space_fingerprint);
    let binding = reconstruct_admission_binding(metadata, embedder.identity()?, tier)?;
    let owner = VectorIndex::open_admitted_v2(path, &binding)
        .map_err(|error| admission_error_to_refresh_error(error, tier, path))?;
    Ok(AdmittedCanonicalTier {
        owner,
        attested_space_hex,
    })
}

/// Next generation sequence for a staged replacement: the attested prior
/// generation's sequence plus one, or 1 for a fresh lineage.
pub(super) fn next_generation_sequence(prior: Option<&ValidatedFsviBytes>) -> SearchResult<u64> {
    prior.map_or_else(
        || Ok(1),
        |owner| {
            owner
                .identity_v2()
                .generation
                .sequence
                .checked_add(1)
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "refresh.generation_sequence".to_owned(),
                    value: u64::MAX.to_string(),
                    reason: "generation sequence overflow".to_owned(),
                })
        },
    )
}

/// Classification of the existing canonical generation after all identity
/// gates AND full content admission have been applied
/// ([`RefreshWorker::admit_existing_generation`]).
// One value exists per classification and it lives for one cycle step; the
// variant-size asymmetry (retained sealed owners vs. no data) is the point
// of the r2 repair, not an allocation concern worth a Box indirection.
#[allow(clippy::large_enum_variant)]
pub(super) enum ExistingGenerationClass {
    /// Nothing is retained on disk (missing artifacts or empty legacy v1
    /// seeds): a bootstrap replacement cannot mix vector spaces.
    BootstrapReplaceable,
    /// At least one tier is an attested FSVI v2 artifact, every gate (space
    /// join, `SameProducer`, quality republication capability) passed, and
    /// every attested tier was FULLY ADMITTED through
    /// [`VectorIndex::open_admitted_v2`] — content and docset digests
    /// recomputed — with the sealed owners RETAINED here. A header-valid but
    /// content-corrupt artifact can never reach this variant (it fails
    /// admission with its own typed corruption error instead).
    AttestedV2 {
        /// Retained admission owner of the canonical fast tier, when it is
        /// an attested v2 artifact.
        fast: Option<AdmittedCanonicalTier>,
        /// Retained admission owner of the canonical quality tier, when it
        /// is an attested v2 artifact.
        quality: Option<AdmittedCanonicalTier>,
    },
}

impl RefreshWorker {
    /// Resolve the existing canonical fast-tier artifact path, if any.
    fn resolve_existing_fast_path(&self) -> Option<PathBuf> {
        let fast_path = self.config.index_dir.join(VECTOR_INDEX_FAST_FILENAME);
        if fast_path.exists() {
            return Some(fast_path);
        }
        let fallback_path = self.config.index_dir.join(VECTOR_INDEX_FALLBACK_FILENAME);
        if fallback_path.exists() {
            return Some(fallback_path);
        }
        None
    }

    /// Inspect both canonical tiers.
    pub(super) fn canonical_tier_states(&self) -> SearchResult<(TierState, TierState)> {
        let fast_state = match self.resolve_existing_fast_path() {
            Some(path) => inspect_tier(&path)?,
            None => TierState::Missing,
        };
        let quality_path = self.config.index_dir.join(VECTOR_INDEX_QUALITY_FILENAME);
        let quality_state = inspect_tier(&quality_path)?;
        Ok((fast_state, quality_state))
    }

    /// Apply the full identity admission law to the existing canonical
    /// generation (guards 2, 7, 8) and PERFORM the admission it claims
    /// (r2 repair of NO-GO item 1: the r1 revision stopped at header
    /// inspection, so a header-valid/content-corrupt v2 artifact sailed past
    /// to the wrong refusal).
    ///
    /// - content-retaining legacy v1 tiers keep the landed containment
    ///   refusal, the fast tier named first (matching origin 5386b39e);
    ///   liveness is the conservative READ-ONLY observation (see
    ///   [`TierState::LegacyV1`]) — classification never mutably opens a v1
    ///   tier;
    /// - attested v2 tiers are gated per tier: space join against the
    ///   artifact's OWN header fingerprints, then SameProducer-only producer
    ///   conformance (cheap, typed refusals first); an attested quality tier
    ///   additionally requires a quality embedder capable of republishing
    ///   it; then each attested tier is FULLY ADMITTED via
    ///   [`VectorIndex::open_admitted_v2`] — exact binding reconstruction,
    ///   content/docset digest recomputation — and the sealed owner is
    ///   RETAINED in the returned classification;
    /// - only a content-free generation (missing artifacts or empty v1
    ///   seeds) classifies as bootstrap-replaceable.
    pub(super) fn admit_existing_generation(
        &self,
        fast_state: &TierState,
        quality_state: &TierState,
    ) -> SearchResult<ExistingGenerationClass> {
        let fast_v1_retains = matches!(
            fast_state,
            TierState::LegacyV1 {
                retains_content: true
            }
        );
        let quality_v1_retains = matches!(
            quality_state,
            TierState::LegacyV1 {
                retains_content: true
            }
        );
        if fast_v1_retains || quality_v1_retains {
            // Name the fast tier whenever it is identityless, else quality —
            // the same attribution the landed containment refusal used.
            let tier = if matches!(fast_state, TierState::LegacyV1 { .. }) {
                "fast"
            } else {
                "quality"
            };
            return Err(identityless_refusal(tier));
        }

        let fast_is_v2 = matches!(fast_state, TierState::V2 { .. });
        let quality_is_v2 = matches!(quality_state, TierState::V2 { .. });
        if !fast_is_v2 && !quality_is_v2 {
            return Ok(ExistingGenerationClass::BootstrapReplaceable);
        }

        let fast = if let TierState::V2 { metadata } = fast_state {
            let bundle = artifact_identity_for(self.fast_embedder.as_ref())?;
            admit_attested_tier("fast", metadata, &bundle)?;
            let path =
                self.resolve_existing_fast_path()
                    .ok_or_else(|| SearchError::IndexNotFound {
                        path: self.config.index_dir.join(VECTOR_INDEX_FAST_FILENAME),
                    })?;
            Some(admit_existing_tier(
                &path,
                metadata,
                self.fast_embedder.as_ref(),
                "fast",
            )?)
        } else {
            None
        };
        let quality = if let TierState::V2 { metadata } = quality_state {
            let Some(quality_embedder) = &self.quality_embedder else {
                return Err(quality_republication_unavailable());
            };
            let bundle = artifact_identity_for(quality_embedder.as_ref())?;
            admit_attested_tier("quality", metadata, &bundle)?;
            let path = self.config.index_dir.join(VECTOR_INDEX_QUALITY_FILENAME);
            Some(admit_existing_tier(
                &path,
                metadata,
                quality_embedder.as_ref(),
                "quality",
            )?)
        } else {
            None
        };
        // A canonical pair whose tiers came from DIFFERENT publications must
        // never be admitted for a merge: carrying both forward would stamp
        // the staged generation with one fresh shared nonce and erase the
        // split origin that `assemble_opened` degrades on (bd-miio8).
        // Legacy pre-nonce pairs read 0 == 0 and remain admissible.
        if let (Some(fast_tier), Some(quality_tier)) = (&fast, &quality) {
            let fast_nonce = fast_tier.owner.metadata().publication_nonce;
            let quality_nonce = quality_tier.owner.metadata().publication_nonce;
            if fast_nonce != quality_nonce {
                return Err(SearchError::InvalidConfig {
                    field: "refresh.canonical_pair_publication_nonce".to_owned(),
                    value: format!("fast={fast_nonce} quality={quality_nonce}"),
                    reason: "canonical fast/quality tiers carry disagreeing publication \
                             nonces (mixed-generation pair, likely a crash between tier \
                             installs); refusing to launder the pair into one staged \
                             generation — rebuild from source instead"
                        .to_owned(),
                });
            }
        }
        Ok(ExistingGenerationClass::AttestedV2 { fast, quality })
    }

    /// Pre-drain admission for the canonical cycle: only the bootstrap lane
    /// may proceed; a FULLY ADMITTED attested v2 generation (content digests
    /// recomputed, owners retained through the check) refuses with the
    /// composite generation-authority reason (see the module docs and
    /// [`composite_authority_refusal`]). The retained owners are dropped
    /// here because the refusal aborts the cycle before any consumer exists;
    /// [`Self::stage_identity_bound_generation`] is the path that carries
    /// them forward.
    pub(super) fn ensure_canonical_cycle_admissible(&self) -> SearchResult<()> {
        let (fast_state, quality_state) = self.canonical_tier_states()?;
        match self.admit_existing_generation(&fast_state, &quality_state)? {
            ExistingGenerationClass::BootstrapReplaceable => Ok(()),
            ExistingGenerationClass::AttestedV2 { .. } => {
                Err(composite_authority_refusal(&self.config.index_dir))
            }
        }
    }
}
