use serde::{Deserialize, Serialize};

use super::MAX_MODEL_ACQUISITION_AUTHORIZATION_LIFETIME_SECONDS;
use super::types::{
    ModelAcquisitionAuthorizationSchemaVersion, NetworkPolicy, RecoveryContractError,
};
use super::wire_support::{WireField, recovery_model_tier_wire};
use crate::{
    generation::{EmbeddingSpaceIdentityV1, EmbeddingSpaceKindV1},
    traits::ModelTier,
};

/// Path-free class of destination bound by model-acquisition consent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelDestinationClass {
    /// Product-managed model cache.
    ManagedCache,
    /// Caller-selected model directory outside the managed cache.
    ExplicitDirectory,
}

/// Machine-distinguishable byte source authorized for acquisition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ModelAcquisitionSource {
    /// Immutable HTTPS sources named by credential-free host.
    Network { source_hosts: Vec<String> },
    /// Complete operator-supplied artifact tree.
    LocalBundle,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum ModelAcquisitionSourceTagWire {
    Network,
    LocalBundle,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelAcquisitionSourceWire {
    kind: ModelAcquisitionSourceTagWire,
    #[serde(default)]
    source_hosts: WireField<Vec<String>>,
}

impl<'de> Deserialize<'de> for ModelAcquisitionSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = ModelAcquisitionSourceWire::deserialize(deserializer)?;
        match (wire.kind, wire.source_hosts) {
            (ModelAcquisitionSourceTagWire::Network, WireField::Present(source_hosts)) => {
                Ok(Self::Network { source_hosts })
            }
            (ModelAcquisitionSourceTagWire::Network, WireField::Missing) => {
                Err(serde::de::Error::missing_field("source_hosts"))
            }
            (ModelAcquisitionSourceTagWire::LocalBundle, WireField::Missing) => {
                Ok(Self::LocalBundle)
            }
            (ModelAcquisitionSourceTagWire::LocalBundle, WireField::Present(_)) => Err(
                serde::de::Error::custom("source_hosts is forbidden for local_bundle source"),
            ),
        }
    }
}

/// Exact, path-free authorization required before model bytes are acquired.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelAcquisitionAuthorization {
    pub schema_version: ModelAcquisitionAuthorizationSchemaVersion,
    /// Stable logical ID passed to the exact-model acquisition command.
    pub model_id: String,
    /// Progressive tier this artifact will serve.
    #[serde(with = "recovery_model_tier_wire")]
    pub model_tier: ModelTier,
    /// Complete mathematical identity of the vectors this model produces.
    ///
    /// A model name, revision, or dimension alone never establishes space
    /// compatibility.
    pub embedding_space: EmbeddingSpaceIdentityV1,
    pub manifest_fingerprint: String,
    pub upstream_revision: String,
    pub license_spdx: String,
    pub source: ModelAcquisitionSource,
    pub byte_budget: u64,
    pub destination_class: ModelDestinationClass,
    /// Bounded hash of the canonical destination, never the raw path.
    pub destination_fingerprint: String,
    /// Exact corpus size shown when consent is requested.
    pub document_count: u64,
    /// Caller-supplied estimate of the reindex wall-clock cost.
    ///
    /// The unit is explicit so renderers never guess. The pure planner does
    /// not derive this value from ambient telemetry or filesystem state.
    pub estimated_reindex_duration_ms: u64,
    /// Caller-frozen issuance time for this exact authorization.
    pub issued_at_unix_seconds: u64,
    /// Caller-frozen exclusive expiry time for this exact authorization.
    pub expires_at_unix_seconds: u64,
    /// Caller-generated 128-bit nonce encoded as exactly 32 lowercase
    /// hexadecimal characters.
    pub nonce: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelAcquisitionAuthorizationWire {
    schema_version: ModelAcquisitionAuthorizationSchemaVersion,
    model_id: String,
    #[serde(with = "recovery_model_tier_wire")]
    model_tier: ModelTier,
    embedding_space: EmbeddingSpaceIdentityV1,
    manifest_fingerprint: String,
    upstream_revision: String,
    license_spdx: String,
    source: ModelAcquisitionSource,
    byte_budget: u64,
    destination_class: ModelDestinationClass,
    destination_fingerprint: String,
    document_count: u64,
    estimated_reindex_duration_ms: u64,
    issued_at_unix_seconds: u64,
    expires_at_unix_seconds: u64,
    nonce: String,
}

impl<'de> Deserialize<'de> for ModelAcquisitionAuthorization {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = ModelAcquisitionAuthorizationWire::deserialize(deserializer)?;
        let authorization = Self {
            schema_version: wire.schema_version,
            model_id: wire.model_id,
            model_tier: wire.model_tier,
            embedding_space: wire.embedding_space,
            manifest_fingerprint: wire.manifest_fingerprint,
            upstream_revision: wire.upstream_revision,
            license_spdx: wire.license_spdx,
            source: wire.source,
            byte_budget: wire.byte_budget,
            destination_class: wire.destination_class,
            destination_fingerprint: wire.destination_fingerprint,
            document_count: wire.document_count,
            estimated_reindex_duration_ms: wire.estimated_reindex_duration_ms,
            issued_at_unix_seconds: wire.issued_at_unix_seconds,
            expires_at_unix_seconds: wire.expires_at_unix_seconds,
            nonce: wire.nonce,
        };
        authorization.validate().map_err(serde::de::Error::custom)?;
        Ok(authorization)
    }
}

impl ModelAcquisitionAuthorization {
    /// Validate that every exact authorization scope is present and usable.
    ///
    /// # Errors
    ///
    /// Rejects blank or control-bearing scope fields, zero byte budgets,
    /// network acquisition without at least one source host, and source hosts
    /// that are not credential-free DNS names, IPv4 addresses, or bracketed
    /// IPv6 addresses (each optionally followed by an explicit non-zero port).
    pub fn validate(&self) -> Result<(), RecoveryContractError> {
        validate_scope_text("model_id", &self.model_id)?;
        self.embedding_space.validate().map_err(|error| {
            RecoveryContractError::InvalidAcquisitionSpaceIdentity {
                reason: error.to_string(),
            }
        })?;
        if self.embedding_space.kind != EmbeddingSpaceKindV1::Semantic {
            return Err(RecoveryContractError::NonSemanticAcquisitionSpace);
        }
        // ubs:ignore — model IDs are public embedding-space identities, not authenticators.
        if self.model_id != self.embedding_space.logical_model_id {
            return Err(RecoveryContractError::InconsistentAcquisitionIdentity {
                field: "model_id",
            });
        }
        validate_scope_text("manifest_fingerprint", &self.manifest_fingerprint)?;
        validate_scope_text("upstream_revision", &self.upstream_revision)?;
        if self.upstream_revision != self.embedding_space.immutable_revision {
            return Err(RecoveryContractError::InconsistentAcquisitionIdentity {
                field: "upstream_revision",
            });
        }
        validate_scope_text("license_spdx", &self.license_spdx)?;
        validate_scope_text("destination_fingerprint", &self.destination_fingerprint)?;
        if self.byte_budget == 0 {
            return Err(RecoveryContractError::ZeroAcquisitionByteBudget);
        }
        if let ModelAcquisitionSource::Network { source_hosts } = &self.source {
            if source_hosts.is_empty() {
                return Err(RecoveryContractError::MissingNetworkSourceHosts);
            }
            for host in source_hosts {
                validate_network_source_host(host)?;
            }
        }
        validate_acquisition_authorization_window(
            self.issued_at_unix_seconds,
            self.expires_at_unix_seconds,
        )?;
        validate_acquisition_authorization_nonce(&self.nonce)?;
        Ok(())
    }

    /// Revalidate this authorization against caller-supplied trusted time.
    ///
    /// Executors must call this immediately before the first acquisition side
    /// effect. A prior planning or wire-promotion check does not keep an
    /// authorization valid after its exclusive expiry boundary.
    ///
    /// # Errors
    ///
    /// Returns the same structural validation errors as [`Self::validate`],
    /// [`RecoveryContractError::AcquisitionAuthorizationNotYetValid`] before
    /// issuance, or [`RecoveryContractError::AcquisitionAuthorizationExpired`]
    /// at and after expiry.
    pub fn validate_at(
        &self,
        evaluation_time_unix_seconds: u64,
    ) -> Result<(), RecoveryContractError> {
        self.validate()?;
        if evaluation_time_unix_seconds < self.issued_at_unix_seconds {
            return Err(RecoveryContractError::AcquisitionAuthorizationNotYetValid {
                issued_at_unix_seconds: self.issued_at_unix_seconds,
                evaluation_time_unix_seconds,
            });
        }
        if evaluation_time_unix_seconds >= self.expires_at_unix_seconds {
            return Err(RecoveryContractError::AcquisitionAuthorizationExpired {
                expires_at_unix_seconds: self.expires_at_unix_seconds,
                evaluation_time_unix_seconds,
            });
        }
        Ok(())
    }
}

/// Source-independent target from which the planner derives the exact
/// network or local-bundle authorization required by policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelAcquisitionTarget {
    /// Stable logical ID selected by the caller's frozen manifest.
    pub model_id: String,
    /// Progressive tier selected by the caller's requested topology.
    pub model_tier: ModelTier,
    /// Complete mathematical identity selected by the caller's frozen
    /// manifest and readiness probe.
    pub embedding_space: EmbeddingSpaceIdentityV1,
    pub manifest_fingerprint: String,
    pub upstream_revision: String,
    pub license_spdx: String,
    pub network_source_hosts: Vec<String>,
    pub byte_budget: u64,
    pub destination_class: ModelDestinationClass,
    pub destination_fingerprint: String,
    /// Corpus census supplied by the caller for consent presentation.
    pub document_count: u64,
    /// Caller-computed reindex estimate, explicitly expressed in
    /// milliseconds. The planner never derives or adjusts it.
    pub estimated_reindex_duration_ms: u64,
    /// Caller-frozen issuance time copied into the exact authorization.
    pub issued_at_unix_seconds: u64,
    /// Caller-frozen exclusive expiry time copied into the exact
    /// authorization.
    pub expires_at_unix_seconds: u64,
    /// Caller-generated 128-bit lowercase-hex nonce copied into the exact
    /// authorization.
    pub nonce: String,
}

impl ModelAcquisitionTarget {
    pub(super) fn authorization_for(
        &self,
        network: NetworkPolicy,
    ) -> Result<ModelAcquisitionAuthorization, RecoveryContractError> {
        let authorization = ModelAcquisitionAuthorization {
            schema_version: ModelAcquisitionAuthorizationSchemaVersion::V3,
            model_id: self.model_id.clone(),
            model_tier: self.model_tier,
            embedding_space: self.embedding_space.clone(),
            manifest_fingerprint: self.manifest_fingerprint.clone(),
            upstream_revision: self.upstream_revision.clone(),
            license_spdx: self.license_spdx.clone(),
            source: match network {
                NetworkPolicy::Allowed => ModelAcquisitionSource::Network {
                    source_hosts: self.network_source_hosts.clone(),
                },
                NetworkPolicy::Offline => ModelAcquisitionSource::LocalBundle,
            },
            byte_budget: self.byte_budget,
            destination_class: self.destination_class,
            destination_fingerprint: self.destination_fingerprint.clone(),
            document_count: self.document_count,
            estimated_reindex_duration_ms: self.estimated_reindex_duration_ms,
            issued_at_unix_seconds: self.issued_at_unix_seconds,
            expires_at_unix_seconds: self.expires_at_unix_seconds,
            nonce: self.nonce.clone(),
        };
        authorization.validate()?;
        Ok(authorization)
    }
}

fn validate_acquisition_authorization_window(
    issued_at_unix_seconds: u64,
    expires_at_unix_seconds: u64,
) -> Result<(), RecoveryContractError> {
    if expires_at_unix_seconds <= issued_at_unix_seconds {
        return Err(
            RecoveryContractError::InvalidAcquisitionAuthorizationWindow {
                issued_at_unix_seconds,
                expires_at_unix_seconds,
            },
        );
    }
    let lifetime_seconds = expires_at_unix_seconds - issued_at_unix_seconds;
    if lifetime_seconds > MAX_MODEL_ACQUISITION_AUTHORIZATION_LIFETIME_SECONDS {
        return Err(
            RecoveryContractError::AcquisitionAuthorizationLifetimeExceeded {
                lifetime_seconds,
                max_lifetime_seconds: MAX_MODEL_ACQUISITION_AUTHORIZATION_LIFETIME_SECONDS,
            },
        );
    }
    Ok(())
}

fn validate_acquisition_authorization_nonce(nonce: &str) -> Result<(), RecoveryContractError> {
    let valid_shape = nonce.len() == 32
        && nonce
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    // ubs:ignore — this persisted acquisition nonce is public uniqueness evidence, not a bearer token.
    let nonzero = nonce.bytes().any(|byte| byte != b'0');
    if !valid_shape || !nonzero {
        return Err(RecoveryContractError::InvalidAcquisitionAuthorizationNonce);
    }
    Ok(())
}

fn validate_scope_text(field: &'static str, value: &str) -> Result<(), RecoveryContractError> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(RecoveryContractError::InvalidAcquisitionScopeField { field });
    }
    Ok(())
}

fn validate_network_source_host(host: &str) -> Result<(), RecoveryContractError> {
    let invalid = || RecoveryContractError::InvalidNetworkSourceHost;
    if host.is_empty()
        || host.chars().any(|character| {
            character.is_whitespace()
                || character.is_control()
                || matches!(character, '@' | '/' | '?' | '#' | '\\')
        })
        || host.contains("://")
    {
        return Err(invalid());
    }

    if let Some(bracketed) = host.strip_prefix('[') {
        let Some(closing_bracket) = bracketed.find(']') else {
            return Err(invalid());
        };
        let address = &bracketed[..closing_bracket];
        let suffix = &bracketed[closing_bracket + 1..];
        address
            .parse::<std::net::Ipv6Addr>()
            .map_err(|_| invalid())?;
        if !suffix.is_empty() {
            let port = suffix.strip_prefix(':').ok_or_else(invalid)?;
            validate_network_source_port(port)?;
        }
        return Ok(());
    }

    if host.contains(['[', ']']) || host.bytes().filter(|byte| *byte == b':').count() > 1 {
        return Err(invalid());
    }
    let (address, port) = match host.rsplit_once(':') {
        Some((address, port)) => (address, Some(port)),
        None => (host, None),
    };
    if let Some(port) = port {
        validate_network_source_port(port)?;
    }
    if address.parse::<std::net::Ipv4Addr>().is_ok() {
        return Ok(());
    }
    if address
        .bytes()
        .all(|byte| byte.is_ascii_digit() || byte == b'.')
        || address.len() > 253
        || address.ends_with('.')
    {
        return Err(invalid());
    }
    for label in address.split('.') {
        if label.is_empty()
            || label.len() > 63
            || !label
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
            || !label
                .as_bytes()
                .first()
                .is_some_and(u8::is_ascii_alphanumeric)
            || !label
                .as_bytes()
                .last()
                .is_some_and(u8::is_ascii_alphanumeric)
        {
            return Err(invalid());
        }
    }
    Ok(())
}

fn validate_network_source_port(port: &str) -> Result<(), RecoveryContractError> {
    if port.is_empty() || !port.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(RecoveryContractError::InvalidNetworkSourceHost);
    }
    let parsed = port
        .parse::<u16>()
        .map_err(|_| RecoveryContractError::InvalidNetworkSourceHost)?;
    if parsed == 0 {
        return Err(RecoveryContractError::InvalidNetworkSourceHost);
    }
    Ok(())
}

fn mismatched_authorization_field(
    required: &ModelAcquisitionAuthorization,
    supplied: &ModelAcquisitionAuthorization,
) -> Option<&'static str> {
    // ubs:ignore — model IDs are public frozen acquisition facts, not credentials.
    if required.model_id != supplied.model_id {
        Some("model_id")
    // ubs:ignore — model tiers are public frozen acquisition facts.
    } else if required.model_tier != supplied.model_tier {
        Some("model_tier")
    // ubs:ignore — upstream revisions are public provenance, not secrets.
    } else if required.upstream_revision != supplied.upstream_revision {
        Some("upstream_revision")
    // ubs:ignore — embedding-space identities are public compatibility facts.
    } else if required.embedding_space != supplied.embedding_space {
        Some("embedding_space")
    // ubs:ignore — manifest fingerprints are public integrity evidence, not secrets.
    } else if required.manifest_fingerprint != supplied.manifest_fingerprint {
        Some("manifest_fingerprint")
    // ubs:ignore — license identifiers are public provenance facts.
    } else if required.license_spdx != supplied.license_spdx {
        Some("license_spdx")
    // ubs:ignore — model sources are public provenance facts, not credentials.
    } else if required.source != supplied.source {
        Some("source")
    // ubs:ignore — byte budgets are public acquisition-policy facts.
    } else if required.byte_budget != supplied.byte_budget {
        Some("byte_budget")
    // ubs:ignore — destination classes are public storage-policy facts.
    } else if required.destination_class != supplied.destination_class {
        Some("destination_class")
    // ubs:ignore — destination fingerprints are public integrity evidence, not secrets.
    } else if required.destination_fingerprint != supplied.destination_fingerprint {
        Some("destination_fingerprint")
    // ubs:ignore — document counts are public corpus-binding facts.
    } else if required.document_count != supplied.document_count {
        Some("document_count")
    // ubs:ignore — duration estimates are public planning facts.
    } else if required.estimated_reindex_duration_ms != supplied.estimated_reindex_duration_ms {
        Some("estimated_reindex_duration_ms")
    // ubs:ignore — issue times are public authorization-contract facts.
    } else if required.issued_at_unix_seconds != supplied.issued_at_unix_seconds {
        Some("issued_at_unix_seconds")
    // ubs:ignore — expiry times are public authorization-contract facts.
    } else if required.expires_at_unix_seconds != supplied.expires_at_unix_seconds {
        Some("expires_at_unix_seconds")
    // ubs:ignore — this persisted nonce is public replay-binding evidence, not a secret.
    } else if required.nonce != supplied.nonce {
        Some("nonce")
    } else {
        None
    }
}

pub(super) fn validate_authorization_binding(
    required: Option<&ModelAcquisitionAuthorization>,
    supplied: Option<&ModelAcquisitionAuthorization>,
    evaluation_time_unix_seconds: u64,
) -> Result<(), RecoveryContractError> {
    match (required, supplied) {
        (None, None) => Ok(()),
        (None, Some(_)) => Err(RecoveryContractError::SurplusAcquisitionAuthorization),
        (Some(required), None) => required.validate_at(evaluation_time_unix_seconds),
        (Some(required), Some(supplied)) => {
            required.validate_at(evaluation_time_unix_seconds)?;
            supplied.validate_at(evaluation_time_unix_seconds)?;
            if let Some(field) = mismatched_authorization_field(required, supplied) {
                return Err(RecoveryContractError::MismatchedAcquisitionAuthorization { field });
            }
            Ok(())
        }
    }
}

pub(super) fn validate_execution_authorization_binding(
    required: Option<&ModelAcquisitionAuthorization>,
    supplied: Option<&ModelAcquisitionAuthorization>,
    evaluation_time_unix_seconds: u64,
) -> Result<(), RecoveryContractError> {
    match (required, supplied) {
        (None, None) => Ok(()),
        (None, Some(_)) => Err(RecoveryContractError::SurplusAcquisitionAuthorization),
        (Some(_), None) => Err(RecoveryContractError::MissingAcquisitionAuthorization),
        (Some(required), Some(supplied)) => {
            required.validate_at(evaluation_time_unix_seconds)?;
            supplied.validate_at(evaluation_time_unix_seconds)?;
            if let Some(field) = mismatched_authorization_field(required, supplied) {
                return Err(RecoveryContractError::MismatchedAcquisitionAuthorization { field });
            }
            Ok(())
        }
    }
}
