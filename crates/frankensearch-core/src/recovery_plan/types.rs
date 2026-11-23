use serde::{Deserialize, Serialize};

use super::authorization::ModelAcquisitionAuthorization;
use super::wire_support::{WireField, recovery_model_tier_wire};
use crate::{config::ZeroSignalReason, traits::ModelTier, types::RetrievalTopology};

/// Wire discriminator for [`super::RecoveryPlan`].
///
/// Using a closed enum rather than an arbitrary string makes serde reject
/// older or unknown schemas before a caller can accidentally execute their
/// actions with v4 semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryPlanSchemaVersion {
    #[serde(rename = "frankensearch.recovery_plan.v4")]
    V4,
}

/// Wire discriminator for [`super::ModelAcquisitionAuthorization`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelAcquisitionAuthorizationSchemaVersion {
    #[serde(rename = "frankensearch.model_acquisition_authorization.v3")]
    V3,
}

/// Verified producer provenance for a semantic-ready lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerifiedSemanticProvenance {
    /// Frozen local artifacts passed manifest verification and load self-test.
    Local,
    /// A remote producer passed the pinned response-attestation contract.
    Remote,
    /// A daemon producer passed the pinned daemon-attestation contract.
    Daemon,
}

/// Trust classification exposed with every recovery plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticProvenance {
    VerifiedLocal,
    VerifiedRemote,
    VerifiedDaemon,
    /// Explicit remote intent exists, but its producer space is not attested.
    UnverifiedRemote,
    /// Explicit non-semantic hash test/control lane.
    HashControl,
    /// No producer is currently admissible.
    Unavailable,
}

impl From<VerifiedSemanticProvenance> for SemanticProvenance {
    fn from(value: VerifiedSemanticProvenance) -> Self {
        match value {
            VerifiedSemanticProvenance::Local => Self::VerifiedLocal,
            VerifiedSemanticProvenance::Remote => Self::VerifiedRemote,
            VerifiedSemanticProvenance::Daemon => Self::VerifiedDaemon,
        }
    }
}

/// Why the caller's request cannot be served semantically right now.
///
/// This is the planner's input state, produced by readiness probes
/// (model manifest checks, index census, generation binding). Variants
/// mirror the states bd-vmv7 enumerates; [`ZeroSignalReason`] carries the
/// finer classification for empty-index states so the two vocabularies
/// stay aligned rather than diverging.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "state", content = "detail")]
pub enum SemanticReadiness {
    /// The semantic lane is fully usable: verified model, loadable, and a
    /// compatible index with usable live vectors.
    Ready {
        provenance: VerifiedSemanticProvenance,
    },
    /// No usable model artifact exists in the configured cache for this exact
    /// progressive tier.
    ModelMissing {
        #[serde(with = "recovery_model_tier_wire")]
        tier: ModelTier,
    },
    /// A model artifact exists for this exact progressive tier but failed
    /// verification or load self-test.
    ModelUnloadable {
        #[serde(with = "recovery_model_tier_wire")]
        tier: ModelTier,
    },
    /// A verified, loadable model exists but no vector index does.
    IndexAbsent,
    /// The index exists but its embedding-space identity does not match the
    /// configured model (legacy generation or intentional identity change).
    IdentityMismatch,
    /// A daemon serves a different embedding space than the local
    /// configuration expects.
    DaemonMismatch,
    /// The index exists and is readable but produced zero signal; the typed
    /// reason distinguishes benign emptiness from availability failures.
    IndexEmpty(ZeroSignalReason),
    /// The index or model manifest is corrupt or fails safety validation
    /// and must not be trusted.
    ManifestUnsafe,
    /// The ANN sidecar belongs to an older generation than the vector
    /// index; exact search works but ANN must not serve.
    AnnStale,
    /// An index generation was interrupted before publication.
    GenerationIncomplete,
    /// Fast-tier search works but some records lack quality-tier
    /// embeddings, so refinement coverage is partial.
    PartialQualityCoverage {
        provenance: VerifiedSemanticProvenance,
        /// Fraction of live documents with quality-tier embeddings.
        coverage_ppm: u32,
    },
    /// Explicit remote intent exists, but the producer cannot be attested.
    /// This state is durable, non-indexable, and never becomes local/hash
    /// fallback inside the planner.
    RemoteUnverified,
    /// Explicit non-semantic hash test/control lane.
    HashControl,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadyDetailWire {
    provenance: VerifiedSemanticProvenance,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelTierDetailWire {
    #[serde(with = "recovery_model_tier_wire")]
    tier: ModelTier,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PartialQualityCoverageDetailWire {
    provenance: VerifiedSemanticProvenance,
    coverage_ppm: u32,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum SemanticReadinessTagWire {
    Ready,
    ModelMissing,
    ModelUnloadable,
    IndexAbsent,
    IdentityMismatch,
    DaemonMismatch,
    IndexEmpty,
    ManifestUnsafe,
    AnnStale,
    GenerationIncomplete,
    PartialQualityCoverage,
    RemoteUnverified,
    HashControl,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum SemanticReadinessDetailWire {
    Ready(ReadyDetailWire),
    ModelTier(ModelTierDetailWire),
    PartialQualityCoverage(PartialQualityCoverageDetailWire),
    ZeroSignal(ZeroSignalReason),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SemanticReadinessWire {
    state: SemanticReadinessTagWire,
    #[serde(default)]
    detail: WireField<SemanticReadinessDetailWire>,
}

impl<'de> Deserialize<'de> for SemanticReadiness {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = SemanticReadinessWire::deserialize(deserializer)?;
        match (wire.state, wire.detail) {
            (
                SemanticReadinessTagWire::Ready,
                WireField::Present(SemanticReadinessDetailWire::Ready(ReadyDetailWire {
                    provenance,
                })),
            ) => Ok(Self::Ready { provenance }),
            (
                SemanticReadinessTagWire::ModelMissing,
                WireField::Present(SemanticReadinessDetailWire::ModelTier(ModelTierDetailWire {
                    tier,
                })),
            ) => Ok(Self::ModelMissing { tier }),
            (
                SemanticReadinessTagWire::ModelUnloadable,
                WireField::Present(SemanticReadinessDetailWire::ModelTier(ModelTierDetailWire {
                    tier,
                })),
            ) => Ok(Self::ModelUnloadable { tier }),
            (SemanticReadinessTagWire::IndexAbsent, WireField::Missing) => Ok(Self::IndexAbsent),
            (SemanticReadinessTagWire::IdentityMismatch, WireField::Missing) => {
                Ok(Self::IdentityMismatch)
            }
            (SemanticReadinessTagWire::DaemonMismatch, WireField::Missing) => {
                Ok(Self::DaemonMismatch)
            }
            (
                SemanticReadinessTagWire::IndexEmpty,
                WireField::Present(SemanticReadinessDetailWire::ZeroSignal(reason)),
            ) => Ok(Self::IndexEmpty(reason)),
            (SemanticReadinessTagWire::ManifestUnsafe, WireField::Missing) => {
                Ok(Self::ManifestUnsafe)
            }
            (SemanticReadinessTagWire::AnnStale, WireField::Missing) => Ok(Self::AnnStale),
            (SemanticReadinessTagWire::GenerationIncomplete, WireField::Missing) => {
                Ok(Self::GenerationIncomplete)
            }
            (
                SemanticReadinessTagWire::PartialQualityCoverage,
                WireField::Present(SemanticReadinessDetailWire::PartialQualityCoverage(
                    PartialQualityCoverageDetailWire {
                        provenance,
                        coverage_ppm,
                    },
                )),
            ) => Ok(Self::PartialQualityCoverage {
                provenance,
                coverage_ppm,
            }),
            (SemanticReadinessTagWire::RemoteUnverified, WireField::Missing) => {
                Ok(Self::RemoteUnverified)
            }
            (SemanticReadinessTagWire::HashControl, WireField::Missing) => Ok(Self::HashControl),
            _ => Err(serde::de::Error::custom(
                "readiness detail is missing, forbidden, or inconsistent with state",
            )),
        }
    }
}

impl SemanticReadiness {
    /// Stable three-segment state code.
    #[must_use]
    pub const fn state_code(&self) -> &'static str {
        match self {
            Self::Ready { .. } => "recovery.state.ready",
            Self::ModelMissing { .. } => "recovery.state.model_missing",
            Self::ModelUnloadable { .. } => "recovery.state.model_unloadable",
            Self::IndexAbsent => "recovery.state.index_absent",
            Self::IdentityMismatch => "recovery.state.identity_mismatch",
            Self::DaemonMismatch => "recovery.state.daemon_mismatch",
            Self::IndexEmpty(_) => "recovery.state.index_empty",
            Self::ManifestUnsafe => "recovery.state.manifest_unsafe",
            Self::AnnStale => "recovery.state.ann_stale",
            Self::GenerationIncomplete => "recovery.state.generation_incomplete",
            Self::PartialQualityCoverage { .. } => "recovery.state.partial_quality_coverage",
            Self::RemoteUnverified => "recovery.state.remote_unverified",
            Self::HashControl => "recovery.state.hash_control",
        }
    }

    /// True when semantic results can be served (possibly with reduced
    /// refinement quality). Only [`Self::Ready`] and
    /// [`Self::PartialQualityCoverage`] qualify: partial coverage degrades
    /// refinement, not availability.
    #[must_use]
    pub const fn semantic_available(&self) -> bool {
        matches!(
            self,
            Self::Ready { .. } | Self::PartialQualityCoverage { .. }
        )
    }

    /// Producer trust associated with the current readiness state.
    #[must_use]
    pub const fn provenance(&self) -> SemanticProvenance {
        match self {
            Self::Ready { provenance } | Self::PartialQualityCoverage { provenance, .. } => {
                match provenance {
                    VerifiedSemanticProvenance::Local => SemanticProvenance::VerifiedLocal,
                    VerifiedSemanticProvenance::Remote => SemanticProvenance::VerifiedRemote,
                    VerifiedSemanticProvenance::Daemon => SemanticProvenance::VerifiedDaemon,
                }
            }
            Self::RemoteUnverified => SemanticProvenance::UnverifiedRemote,
            Self::HashControl => SemanticProvenance::HashControl,
            Self::ModelMissing { .. }
            | Self::ModelUnloadable { .. }
            | Self::IndexAbsent
            | Self::IdentityMismatch
            | Self::DaemonMismatch
            | Self::IndexEmpty(_)
            | Self::ManifestUnsafe
            | Self::AnnStale
            | Self::GenerationIncomplete => SemanticProvenance::Unavailable,
        }
    }
}

/// What the caller asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestMode {
    /// The caller explicitly requires semantic results: fail closed when
    /// the lane is unavailable.
    ExplicitSemantic,
    /// The caller accepts hybrid results: lexical fallback is permitted,
    /// but only with explicit degradation metadata.
    Hybrid,
    /// Explicit non-semantic hash test/control request.
    HashControl,
}

/// Requested operation, including the exact retrieval topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct RecoveryRequest {
    pub mode: RequestMode,
    pub requested_topology: RetrievalTopology,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecoveryRequestWire {
    mode: RequestMode,
    requested_topology: RetrievalTopology,
}

impl<'de> Deserialize<'de> for RecoveryRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = RecoveryRequestWire::deserialize(deserializer)?;
        Self {
            mode: wire.mode,
            requested_topology: wire.requested_topology,
        }
        .validate()
        .map_err(serde::de::Error::custom)
    }
}

impl RecoveryRequest {
    /// Validate that mode and topology describe one unambiguous request.
    ///
    /// `PartialQuality` and `LexicalOnly` are realized topologies, never
    /// semantic request targets. Hash is legal only through the explicit
    /// `HashControl` mode.
    ///
    /// # Errors
    ///
    /// Returns [`RecoveryContractError::InvalidRequestTopology`] for an
    /// ambiguous or silently degrading combination.
    pub fn validate(self) -> Result<Self, RecoveryContractError> {
        let valid = match self.mode {
            RequestMode::ExplicitSemantic | RequestMode::Hybrid => matches!(
                self.requested_topology,
                RetrievalTopology::FastOnly
                    | RetrievalTopology::QualityOnly
                    | RetrievalTopology::FullProgressive
            ),
            RequestMode::HashControl => {
                matches!(self.requested_topology, RetrievalTopology::HashControl)
            }
        };
        if valid {
            Ok(self)
        } else {
            Err(RecoveryContractError::InvalidRequestTopology {
                mode: self.mode,
                topology: self.requested_topology,
            })
        }
    }
}

/// Whether a human can be asked for consent right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionPolicy {
    Interactive,
    NonInteractive,
}

/// Whether network access is permitted for recovery actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NetworkPolicy {
    Allowed,
    Offline,
}

/// The caller's environment policy, combined.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RecoveryPolicy {
    pub interaction: InteractionPolicy,
    pub network: NetworkPolicy,
    /// Exact non-TTY/programmatic model-acquisition authorization, when
    /// already granted. It satisfies consent only when byte-for-byte equal
    /// to the action's required authorization.
    pub acquisition_authorization: Option<ModelAcquisitionAuthorization>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecoveryPolicyWire {
    interaction: InteractionPolicy,
    network: NetworkPolicy,
    #[serde(default)]
    acquisition_authorization: WireField<Option<ModelAcquisitionAuthorization>>,
}

impl<'de> Deserialize<'de> for RecoveryPolicy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = RecoveryPolicyWire::deserialize(deserializer)?;
        let WireField::Present(acquisition_authorization) = wire.acquisition_authorization else {
            return Err(serde::de::Error::missing_field("acquisition_authorization"));
        };
        Ok(Self {
            interaction: wire.interaction,
            network: wire.network,
            acquisition_authorization,
        })
    }
}

/// Typed validation failure for a recovery request or response contract.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RecoveryContractError {
    #[error("request mode {mode:?} cannot request topology {topology:?}")]
    InvalidRequestTopology {
        mode: RequestMode,
        topology: RetrievalTopology,
    },
    #[error("hash-control request mode and readiness must be selected together")]
    HashControlModeReadinessMismatch,
    #[error("acquisition scope field `{field}` is empty or contains control characters")]
    InvalidAcquisitionScopeField { field: &'static str },
    #[error("model acquisition space identity is invalid: {reason}")]
    InvalidAcquisitionSpaceIdentity { reason: String },
    #[error("model acquisition requires a semantic embedding space")]
    NonSemanticAcquisitionSpace,
    #[error("acquisition identity field `{field}` conflicts with the complete space identity")]
    InconsistentAcquisitionIdentity { field: &'static str },
    #[error("readiness tier {tier} cannot satisfy requested topology {requested_topology:?}")]
    UnavailableTierTopologyMismatch {
        tier: ModelTier,
        requested_topology: RetrievalTopology,
    },
    #[error("acquisition target tier {target_tier} does not match readiness tier {readiness_tier}")]
    AcquisitionTargetTierMismatch {
        readiness_tier: ModelTier,
        target_tier: ModelTier,
    },
    #[error("model acquisition byte budget must be non-zero")]
    ZeroAcquisitionByteBudget,
    #[error(
        "model acquisition authorization window is invalid: issued_at={issued_at_unix_seconds}, \
         expires_at={expires_at_unix_seconds}"
    )]
    InvalidAcquisitionAuthorizationWindow {
        issued_at_unix_seconds: u64,
        expires_at_unix_seconds: u64,
    },
    #[error(
        "model acquisition authorization lifetime {lifetime_seconds}s exceeds the \
         {max_lifetime_seconds}s maximum"
    )]
    AcquisitionAuthorizationLifetimeExceeded {
        lifetime_seconds: u64,
        max_lifetime_seconds: u64,
    },
    #[error(
        "model acquisition authorization nonce must be a nonzero 128-bit value encoded as \
         exactly 32 lowercase hexadecimal characters"
    )]
    InvalidAcquisitionAuthorizationNonce,
    #[error(
        "model acquisition authorization is not yet valid: issued_at={issued_at_unix_seconds}, \
         evaluated_at={evaluation_time_unix_seconds}"
    )]
    AcquisitionAuthorizationNotYetValid {
        issued_at_unix_seconds: u64,
        evaluation_time_unix_seconds: u64,
    },
    #[error(
        "model acquisition authorization expired: expires_at={expires_at_unix_seconds}, \
         evaluated_at={evaluation_time_unix_seconds}"
    )]
    AcquisitionAuthorizationExpired {
        expires_at_unix_seconds: u64,
        evaluation_time_unix_seconds: u64,
    },
    #[error("model acquisition requires an exact supplied authorization before execution")]
    MissingAcquisitionAuthorization,
    #[error("model acquisition authorization was supplied when no exact acquisition requires it")]
    SurplusAcquisitionAuthorization,
    #[error("model acquisition authorization field `{field}` does not match the required scope")]
    MismatchedAcquisitionAuthorization { field: &'static str },
    #[error("network model acquisition requires at least one credential-free source host")]
    MissingNetworkSourceHosts,
    #[error(
        "network model acquisition source host must be a credential-free DNS name, IPv4 address, \
         or bracketed IPv6 address with an optional non-zero port"
    )]
    InvalidNetworkSourceHost,
    #[error("coverage_ppm {coverage_ppm} is outside the valid range for {topology:?}")]
    InvalidCoverage {
        topology: RetrievalTopology,
        coverage_ppm: u32,
    },
    #[error("requested topology {requested:?} cannot realize as {realized:?}")]
    IncompatibleResponseTopology {
        requested: RetrievalTopology,
        realized: RetrievalTopology,
    },
    #[error("non-semantic topology {topology:?} cannot admit {admitted} semantic scores")]
    NonSemanticScoresAdmitted {
        topology: RetrievalTopology,
        admitted: u64,
    },
    #[error("lexical-only response requires exactly one degradation reason code")]
    MissingDegradationReason,
    #[error("non-lexical response cannot carry a degradation reason code")]
    UnexpectedDegradationReason,
    #[error("recovery plan field `{field}` is inconsistent with the typed decision")]
    InconsistentRecoveryPlan { field: &'static str },
}
