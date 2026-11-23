use serde::{Deserialize, Serialize};

use super::COMPLETE_COVERAGE_PPM;
use super::authorization::{
    ModelAcquisitionAuthorization, ModelAcquisitionTarget, validate_execution_authorization_binding,
};
use super::planner::plan;
use super::types::{
    RecoveryContractError, RecoveryPlanSchemaVersion, RecoveryPolicy, RecoveryRequest, RequestMode,
    SemanticProvenance, SemanticReadiness,
};
use super::wire_support::WireField;
use crate::types::{RetrievalTopology, retrieval_topology_fits_request};

/// Whether retrying the original request can succeed, and under what
/// condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Retryability {
    /// The lane is usable; no recovery action is needed.
    NotNeeded,
    /// Retry after executing the recommended action.
    AfterAction,
    /// The current request itself produced no signal; retry only after
    /// changing its zero-k, filter, or vector input.
    AfterRequestChange,
    /// The recommended action cannot run under the current policy; the
    /// listed prerequisites must be granted first.
    BlockedByPolicy,
    /// The recommended action has no executable implementation in the
    /// current runtime. A listed capability must land before retrying.
    BlockedByCapability,
}

/// One truthful next action.
///
/// The four booleans are independent schema-mandated facts about the
/// action (bd-vmv7's field list), not an encoded state machine, so a
/// bitflag or enum representation would obscure the serialized contract.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RecoveryAction {
    /// Stable action code, append-only within a schema version.
    pub(super) code: String,
    /// Human-readable explanation of what the action does and why.
    pub(super) explanation: String,
    /// Command as an argv array. Placeholder tokens ([`super::ARG_INDEX_DIR`],
    /// [`super::ARG_SOURCE_DIR`]) are substituted by integrators; the pure
    /// planner never handles real paths.
    pub(super) argv: Vec<String>,
    /// The action needs network access (model downloads).
    pub(super) network_required: bool,
    /// The action needs explicit human consent (it replaces existing
    /// artifacts).
    pub(super) consent_required: bool,
    /// The pre-existing data this action is intended to repair survives. For
    /// index recovery this means user documents and index contents; for model
    /// recovery it includes the cached model artifact itself. Reacquisition
    /// is therefore `false` even though corpus and index data remain intact.
    pub(super) preserves_old_data: bool,
    /// The action replaces or rewrites existing artifacts.
    pub(super) potentially_destructive: bool,
    /// Stable codes of conditions that must hold before the action can
    /// run (policy grants).
    pub(super) prerequisites: Vec<String>,
    /// Stable code of the state expected after the action succeeds. Never
    /// `recovery.state.ready` for acquisition actions: readiness
    /// additionally requires the load self-test and a compatible
    /// non-empty index.
    pub(super) expected_postcondition: String,
    /// Exact acquisition authorization this action requires. `None` for
    /// non-acquisition actions and when the caller failed to bind a frozen
    /// model target (which blocks the plan through a prerequisite).
    pub(super) required_authorization: Option<ModelAcquisitionAuthorization>,
}

impl RecoveryAction {
    /// Stable action code.
    #[must_use]
    pub fn code(&self) -> &str {
        &self.code
    }

    /// Human-readable reason and effect.
    #[must_use]
    pub fn explanation(&self) -> &str {
        &self.explanation
    }

    /// Parser-executable argv, or an empty slice when prerequisites make the
    /// action unavailable.
    #[must_use]
    pub fn argv(&self) -> &[String] {
        &self.argv
    }

    /// Whether execution requires network access.
    #[must_use]
    pub const fn network_required(&self) -> bool {
        self.network_required
    }

    /// Whether execution requires explicit consent.
    #[must_use]
    pub const fn consent_required(&self) -> bool {
        self.consent_required
    }

    /// Whether the pre-existing data this action is intended to repair
    /// survives.
    #[must_use]
    pub const fn preserves_old_data(&self) -> bool {
        self.preserves_old_data
    }

    /// Whether the action rewrites or replaces artifacts.
    #[must_use]
    pub const fn potentially_destructive(&self) -> bool {
        self.potentially_destructive
    }

    /// Stable prerequisite codes that currently block execution.
    #[must_use]
    pub fn prerequisites(&self) -> &[String] {
        &self.prerequisites
    }

    /// Stable postcondition code expected after successful execution.
    #[must_use]
    pub fn expected_postcondition(&self) -> &str {
        &self.expected_postcondition
    }

    /// Exact scoped authorization required for acquisition.
    #[must_use]
    pub const fn required_authorization(&self) -> Option<&ModelAcquisitionAuthorization> {
        self.required_authorization.as_ref()
    }

    /// Render the argv for a POSIX shell, quoting every argument that
    /// contains characters beyond `[A-Za-z0-9_./:=-]` and the placeholder
    /// tokens (which are documentation, not shell input).
    #[must_use]
    pub fn shell_command(&self) -> String {
        self.argv
            .iter()
            .map(|arg| shell_quote(arg))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

// This raw shape mirrors the same independent schema-mandated facts as
// RecoveryAction; collapsing them would make wire validation less explicit.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecoveryActionWire {
    code: String,
    explanation: String,
    argv: Vec<String>,
    network_required: bool,
    consent_required: bool,
    preserves_old_data: bool,
    potentially_destructive: bool,
    prerequisites: Vec<String>,
    expected_postcondition: String,
    #[serde(default)]
    required_authorization: WireField<Option<ModelAcquisitionAuthorization>>,
}

fn shell_quote(arg: &str) -> String {
    let safe = !arg.is_empty()
        && arg.chars().all(|c| {
            c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '/' | ':' | '=' | '-' | '<' | '>')
        });
    if safe {
        arg.to_owned()
    } else {
        // POSIX single-quote escaping: close, escaped quote, reopen.
        format!("'{}'", arg.replace('\'', "'\\''"))
    }
}

/// Truthful semantic contribution metadata for one response.
///
/// This contract is reusable by product output schemas after query
/// execution. The planner initializes `admitted_semantic_scores` to zero;
/// a search path must replace it with the actual admitted count before
/// emitting a completed semantic response. Non-semantic realized topologies
/// are permanently constrained to zero. The trusted type is Serialize-only:
/// wire data inside an [`UntrustedRecoveryPlan`] is decoded into a private raw
/// shape and compared field-for-field with a freshly planned contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SemanticResponseContract {
    pub(super) requested_topology: RetrievalTopology,
    pub(super) realized_topology: RetrievalTopology,
    pub(super) coverage_ppm: u32,
    pub(super) admitted_semantic_scores: u64,
    /// Present exactly for a lexical-only hybrid degradation.
    pub(super) degradation_reason_code: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SemanticResponseContractWire {
    requested_topology: RetrievalTopology,
    realized_topology: RetrievalTopology,
    coverage_ppm: u32,
    admitted_semantic_scores: u64,
    #[serde(default)]
    degradation_reason_code: WireField<Option<String>>,
}

impl SemanticResponseContract {
    /// Construct and validate one response contract.
    ///
    /// # Errors
    ///
    /// Rejects impossible requested/realized topology pairs, out-of-range
    /// coverage, semantic-score admission by lexical/hash lanes, and
    /// missing or spurious degradation reasons.
    pub fn new(
        requested_topology: RetrievalTopology,
        realized_topology: RetrievalTopology,
        coverage_ppm: u32,
        admitted_semantic_scores: u64,
        degradation_reason_code: Option<String>,
    ) -> Result<Self, RecoveryContractError> {
        let contract = Self {
            requested_topology,
            realized_topology,
            coverage_ppm,
            admitted_semantic_scores,
            degradation_reason_code,
        };
        contract.validate()?;
        Ok(contract)
    }

    /// Retrieval topology the caller requested.
    #[must_use]
    pub const fn requested_topology(&self) -> RetrievalTopology {
        self.requested_topology
    }

    /// Retrieval topology the response actually realizes.
    #[must_use]
    pub const fn realized_topology(&self) -> RetrievalTopology {
        self.realized_topology
    }

    /// Semantic document coverage in parts per million.
    #[must_use]
    pub const fn coverage_ppm(&self) -> u32 {
        self.coverage_ppm
    }

    /// Number of semantic scores admitted into the response.
    #[must_use]
    pub const fn admitted_semantic_scores(&self) -> u64 {
        self.admitted_semantic_scores
    }

    /// Typed degradation reason for lexical-only hybrid fallback.
    #[must_use]
    pub fn degradation_reason_code(&self) -> Option<&str> {
        self.degradation_reason_code.as_deref()
    }

    /// Replace the planning-boundary zero with the count admitted by a
    /// completed response, re-validating non-semantic invariants.
    ///
    /// # Errors
    ///
    /// Returns [`RecoveryContractError::NonSemanticScoresAdmitted`] if a
    /// lexical/hash response attempts to claim semantic contribution.
    pub fn with_admitted_semantic_scores(
        mut self,
        admitted_semantic_scores: u64,
    ) -> Result<Self, RecoveryContractError> {
        self.admitted_semantic_scores = admitted_semantic_scores;
        self.validate()?;
        Ok(self)
    }

    fn validate(&self) -> Result<(), RecoveryContractError> {
        let topology_compatible =
            retrieval_topology_fits_request(self.requested_topology, self.realized_topology);
        if !topology_compatible {
            return Err(RecoveryContractError::IncompatibleResponseTopology {
                requested: self.requested_topology,
                realized: self.realized_topology,
            });
        }

        let coverage_valid = match self.realized_topology {
            RetrievalTopology::LexicalOnly | RetrievalTopology::HashControl => {
                self.coverage_ppm == 0
            }
            RetrievalTopology::FastOnly
            | RetrievalTopology::QualityOnly
            // ubs:ignore — coverage is a public retrieval-plan fact, not security material.
            | RetrievalTopology::FullProgressive => self.coverage_ppm == COMPLETE_COVERAGE_PPM,
            RetrievalTopology::PartialQuality { coverage_ppm } => {
                coverage_ppm == self.coverage_ppm
                    && (1..COMPLETE_COVERAGE_PPM).contains(&coverage_ppm)
            }
        };
        if !coverage_valid {
            return Err(RecoveryContractError::InvalidCoverage {
                topology: self.realized_topology,
                coverage_ppm: self.coverage_ppm,
            });
        }

        if !self.realized_topology.is_semantic() && self.admitted_semantic_scores != 0 {
            return Err(RecoveryContractError::NonSemanticScoresAdmitted {
                topology: self.realized_topology,
                admitted: self.admitted_semantic_scores,
            });
        }

        match (
            self.requested_topology,
            self.realized_topology,
            self.degradation_reason_code.as_deref(),
        ) {
            (RetrievalTopology::LexicalOnly, RetrievalTopology::LexicalOnly, None) => Ok(()),
            (RetrievalTopology::LexicalOnly, RetrievalTopology::LexicalOnly, Some(_)) => {
                Err(RecoveryContractError::UnexpectedDegradationReason)
            }
            (_, RetrievalTopology::LexicalOnly, Some(code)) if !code.trim().is_empty() => Ok(()),
            (_, RetrievalTopology::LexicalOnly, _) => {
                Err(RecoveryContractError::MissingDegradationReason)
            }
            (_, _, None) => Ok(()),
            (_, _, Some(_)) => Err(RecoveryContractError::UnexpectedDegradationReason),
        }
    }
}

/// The full plan: current state, verdict for the requested mode, and the
/// truthful next action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RecoveryPlan {
    /// [`super::RECOVERY_PLAN_SCHEMA_VERSION`].
    pub(super) schema_version: RecoveryPlanSchemaVersion,
    /// The readiness state the plan was computed from.
    pub(super) state: SemanticReadiness,
    /// Stable code for `state` (denormalized for consumers that do not
    /// decode the enum).
    pub(super) state_code: String,
    /// Producer trust classification derived from `state`.
    pub(super) provenance: SemanticProvenance,
    /// The mode the caller requested.
    pub(super) mode: RequestMode,
    /// Exact retrieval topology the caller requested.
    pub(super) requested_topology: RetrievalTopology,
    /// The policy the plan was computed under.
    pub(super) policy: RecoveryPolicy,
    /// Whether semantic results can be served right now.
    pub(super) semantic_available: bool,
    /// Whether retrying can succeed, and under what condition.
    pub(super) retryability: Retryability,
    /// The truthful next action; `None` when the state needs none (ready,
    /// or the emptiness was request-scoped).
    pub(super) action: Option<RecoveryAction>,
    /// Response shape allowed by this decision. `None` means the explicit
    /// request fails closed and no response may be emitted.
    pub(super) response_contract: Option<SemanticResponseContract>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecoveryPlanWire {
    schema_version: RecoveryPlanSchemaVersion,
    state: SemanticReadiness,
    state_code: String,
    provenance: SemanticProvenance,
    mode: RequestMode,
    requested_topology: RetrievalTopology,
    policy: RecoveryPolicy,
    semantic_available: bool,
    retryability: Retryability,
    #[serde(default)]
    action: WireField<Option<RecoveryActionWire>>,
    #[serde(default)]
    response_contract: WireField<Option<SemanticResponseContractWire>>,
}

impl RecoveryPlanWire {
    fn validate_required_option_presence(&self) -> Result<(), &'static str> {
        let action = self
            .action
            .required_option_ref()
            .map_err(|()| "missing required field action")?;
        let response = self
            .response_contract
            .required_option_ref()
            .map_err(|()| "missing required field response_contract")?;
        if let Some(action) = action {
            action
                .required_authorization
                .required_option_ref()
                .map_err(|()| "missing required field action.required_authorization")?;
        }
        if let Some(response) = response {
            response
                .degradation_reason_code
                .required_option_ref()
                .map_err(|()| "missing required field response_contract.degradation_reason_code")?;
        }
        Ok(())
    }
}

/// Opaque, non-executable recovery-plan payload decoded from an untrusted
/// transport.
///
/// No action or argv accessor exists on this type. The only promotion path is
/// [`Self::validate_against`], which compares every payload field with a new
/// plan derived exclusively from a [`TrustedRecoveryContext`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UntrustedRecoveryPlan {
    wire: RecoveryPlanWire,
}

impl<'de> Deserialize<'de> for UntrustedRecoveryPlan {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = RecoveryPlanWire::deserialize(deserializer)?;
        wire.validate_required_option_presence()
            .map_err(serde::de::Error::custom)?;
        Ok(Self { wire })
    }
}

/// Independently sourced inputs that may promote one untrusted payload to a
/// trusted, executable [`RecoveryPlan`].
///
/// Every value must come from the current readiness probe, caller request,
/// environment policy, or frozen manifest. Never populate this context from
/// the payload being validated.
#[derive(Debug, Clone, Copy)]
pub struct TrustedRecoveryContext<'a> {
    pub(super) state: &'a SemanticReadiness,
    pub(super) request: RecoveryRequest,
    pub(super) policy: &'a RecoveryPolicy,
    pub(super) acquisition_target: Option<&'a ModelAcquisitionTarget>,
    pub(super) evaluation_time_unix_seconds: u64,
}

impl<'a> TrustedRecoveryContext<'a> {
    /// Bind authoritative runtime inputs for untrusted-plan validation.
    #[must_use]
    pub const fn new(
        state: &'a SemanticReadiness,
        request: RecoveryRequest,
        policy: &'a RecoveryPolicy,
        acquisition_target: Option<&'a ModelAcquisitionTarget>,
        evaluation_time_unix_seconds: u64,
    ) -> Self {
        Self {
            state,
            request,
            policy,
            acquisition_target,
            evaluation_time_unix_seconds,
        }
    }
}

impl UntrustedRecoveryPlan {
    /// Compare every payload field with a fresh plan derived exclusively from
    /// trusted inputs.
    ///
    /// # Errors
    ///
    /// Returns [`RecoveryContractError::InconsistentRecoveryPlan`] for any
    /// payload substitution, including a coherent multi-field forgery.
    /// Trusted-context planning and target-validation errors are preserved.
    pub fn validate_against(
        self,
        trusted: TrustedRecoveryContext<'_>,
    ) -> Result<RecoveryPlan, RecoveryContractError> {
        let canonical = plan(trusted)?;
        validate_wire_against_canonical(&self.wire, &canonical)?;
        Ok(canonical)
    }
}

impl RecoveryPlan {
    /// Wire schema version.
    #[must_use]
    pub const fn schema_version(&self) -> RecoveryPlanSchemaVersion {
        self.schema_version
    }

    /// Readiness state used by the planner.
    #[must_use]
    pub const fn state(&self) -> &SemanticReadiness {
        &self.state
    }

    /// Stable code derived from [`Self::state`].
    #[must_use]
    pub fn state_code(&self) -> &str {
        &self.state_code
    }

    /// Producer provenance derived from readiness.
    #[must_use]
    pub const fn provenance(&self) -> SemanticProvenance {
        self.provenance
    }

    /// Requested mode.
    #[must_use]
    pub const fn mode(&self) -> RequestMode {
        self.mode
    }

    /// Requested retrieval topology.
    #[must_use]
    pub const fn requested_topology(&self) -> RetrievalTopology {
        self.requested_topology
    }

    /// Trusted policy used to construct the plan.
    #[must_use]
    pub const fn policy(&self) -> &RecoveryPolicy {
        &self.policy
    }

    /// Whether semantic results are currently admissible.
    #[must_use]
    pub const fn semantic_available(&self) -> bool {
        self.semantic_available
    }

    /// Retry verdict derived by the planner.
    #[must_use]
    pub const fn retryability(&self) -> Retryability {
        self.retryability
    }

    /// Canonical next action, when one exists.
    #[must_use]
    pub const fn action(&self) -> Option<&RecoveryAction> {
        self.action.as_ref()
    }

    /// Canonical response-admission contract, when a response is allowed.
    #[must_use]
    pub const fn response_contract(&self) -> Option<&SemanticResponseContract> {
        self.response_contract.as_ref()
    }

    /// Revalidate acquisition authorization immediately before execution.
    ///
    /// Planning and untrusted-wire promotion validate against the trusted time
    /// supplied for those operations. They do not mint a timeless capability:
    /// an executor must call this method again at the acquisition boundary
    /// using independently trusted current time. An interactive plan created
    /// before consent cannot execute: the caller must re-plan with the freshly
    /// granted exact authorization in [`RecoveryPolicy::acquisition_authorization`].
    ///
    /// # Errors
    ///
    /// Returns an authorization structural, scope-binding, not-yet-valid, or
    /// expiry error when the action is no longer executable at
    /// `evaluation_time_unix_seconds`.
    pub fn validate_for_execution_at(
        &self,
        evaluation_time_unix_seconds: u64,
    ) -> Result<(), RecoveryContractError> {
        validate_execution_authorization_binding(
            self.action
                .as_ref()
                .and_then(RecoveryAction::required_authorization),
            self.policy.acquisition_authorization.as_ref(),
            evaluation_time_unix_seconds,
        )
    }
}

fn inconsistent(field: &'static str) -> RecoveryContractError {
    RecoveryContractError::InconsistentRecoveryPlan { field }
}

fn validate_wire_against_canonical(
    wire: &RecoveryPlanWire,
    canonical: &RecoveryPlan,
) -> Result<(), RecoveryContractError> {
    // ubs:ignore — schema versions are public recovery-wire facts, not authenticators.
    if wire.schema_version != canonical.schema_version {
        return Err(inconsistent("schema_version"));
    }
    // ubs:ignore — readiness state is public recovery-wire data, not secret material.
    if wire.state != canonical.state {
        return Err(inconsistent("state"));
    }
    // ubs:ignore — state codes are public recovery-wire data, not authenticators.
    if wire.state_code != canonical.state_code {
        return Err(inconsistent("state_code"));
    }
    // ubs:ignore — provenance is public recovery-wire evidence, not an authenticator.
    if wire.provenance != canonical.provenance {
        return Err(inconsistent("provenance"));
    }
    // ubs:ignore — mode is a public recovery-wire fact, not security material.
    if wire.mode != canonical.mode {
        return Err(inconsistent("mode"));
    }
    // ubs:ignore — topology is public recovery-wire data, not an authenticator.
    if wire.requested_topology != canonical.requested_topology {
        return Err(inconsistent("requested_topology"));
    }
    // ubs:ignore — policy is public recovery-wire configuration, not an authenticator.
    if wire.policy != canonical.policy {
        return Err(inconsistent("policy"));
    }
    // ubs:ignore — availability is a public recovery-wire fact, not secret material.
    if wire.semantic_available != canonical.semantic_available {
        return Err(inconsistent("semantic_available"));
    }
    // ubs:ignore — retryability is a public recovery-wire fact, not an authenticator.
    if wire.retryability != canonical.retryability {
        return Err(inconsistent("retryability"));
    }
    let wire_action = wire
        .action
        .required_option_ref()
        .map_err(|()| inconsistent("action"))?;
    validate_action_wire(wire_action, canonical.action.as_ref())?;
    let wire_response = wire
        .response_contract
        .required_option_ref()
        .map_err(|()| inconsistent("response_contract"))?;
    validate_response_wire(wire_response, canonical.response_contract.as_ref())?;
    Ok(())
}

fn validate_action_wire(
    wire: Option<&RecoveryActionWire>,
    canonical: Option<&RecoveryAction>,
) -> Result<(), RecoveryContractError> {
    let (Some(wire), Some(canonical)) = (wire, canonical) else {
        return if wire.is_none() && canonical.is_none() {
            Ok(())
        } else {
            Err(inconsistent("action"))
        };
    };
    // ubs:ignore — action codes are public recovery-wire facts, not authenticators.
    if wire.code != canonical.code {
        return Err(inconsistent("action.code"));
    }
    // ubs:ignore — explanations are public operator guidance, not secret material.
    if wire.explanation != canonical.explanation {
        return Err(inconsistent("action.explanation"));
    }
    // ubs:ignore — argv is public recovery-action evidence, not an authenticator.
    if wire.argv != canonical.argv {
        return Err(inconsistent("action.argv"));
    }
    // ubs:ignore — network requirement is public recovery-action metadata.
    if wire.network_required != canonical.network_required {
        return Err(inconsistent("action.network_required"));
    }
    // ubs:ignore — consent requirement is public recovery-action metadata.
    if wire.consent_required != canonical.consent_required {
        return Err(inconsistent("action.consent_required"));
    }
    // ubs:ignore — preservation is public recovery-action metadata, not an authenticator.
    if wire.preserves_old_data != canonical.preserves_old_data {
        return Err(inconsistent("action.preserves_old_data"));
    }
    // ubs:ignore — destructive intent is public recovery-action metadata, not a secret.
    if wire.potentially_destructive != canonical.potentially_destructive {
        return Err(inconsistent("action.potentially_destructive"));
    }
    // ubs:ignore — prerequisites are public recovery-action facts, not authenticators.
    if wire.prerequisites != canonical.prerequisites {
        return Err(inconsistent("action.prerequisites"));
    }
    // ubs:ignore — postconditions are public recovery-action facts, not authenticators.
    if wire.expected_postcondition != canonical.expected_postcondition {
        return Err(inconsistent("action.expected_postcondition"));
    }
    let wire_authorization = wire
        .required_authorization
        .required_option_ref()
        .map_err(|()| inconsistent("action.required_authorization"))?;
    // ubs:ignore — this is a public frozen authorization contract, not a credential.
    if wire_authorization != canonical.required_authorization.as_ref() {
        return Err(inconsistent("action.required_authorization"));
    }
    Ok(())
}

fn validate_response_wire(
    wire: Option<&SemanticResponseContractWire>,
    canonical: Option<&SemanticResponseContract>,
) -> Result<(), RecoveryContractError> {
    let (Some(wire), Some(canonical)) = (wire, canonical) else {
        return if wire.is_none() && canonical.is_none() {
            Ok(())
        } else {
            Err(inconsistent("response_contract"))
        };
    };
    // ubs:ignore — requested topology is public response-contract metadata.
    if wire.requested_topology != canonical.requested_topology {
        return Err(inconsistent("response_contract.requested_topology"));
    }
    // ubs:ignore — realized topology is public response-contract metadata.
    if wire.realized_topology != canonical.realized_topology {
        return Err(inconsistent("response_contract.realized_topology"));
    }
    // ubs:ignore — coverage is public response-contract evidence, not a secret.
    if wire.coverage_ppm != canonical.coverage_ppm {
        return Err(inconsistent("response_contract.coverage_ppm"));
    }
    // ubs:ignore — score admission is public response-contract evidence.
    if wire.admitted_semantic_scores != canonical.admitted_semantic_scores {
        return Err(inconsistent("response_contract.admitted_semantic_scores"));
    }
    let wire_degradation_reason = wire
        .degradation_reason_code
        .required_option_ref()
        .map_err(|()| inconsistent("response_contract.degradation_reason_code"))?;
    // ubs:ignore — degradation reason codes are public operator evidence.
    if wire_degradation_reason != canonical.degradation_reason_code.as_ref() {
        return Err(inconsistent("response_contract.degradation_reason_code"));
    }
    Ok(())
}
