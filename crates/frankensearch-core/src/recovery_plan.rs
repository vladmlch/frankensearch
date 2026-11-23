//! Central typed semantic [`RecoveryPlan`] and truthful readiness planner
//! (bd-vmv7).
//!
//! One versioned, machine-readable contract that maps the current semantic
//! readiness state plus the requested mode and policy to a truthful next
//! action. Products (facade, fsfs, CASS) consume this shared type instead of
//! parsing error strings; the terminal integration bead wires state
//! producers and executors around it.
//!
//! # Truthfulness invariants
//!
//! - Installing a model never claims semantic-searchable: the acquire
//!   action's postcondition is model-acquired-unverified, and readiness
//!   additionally requires a load self-test and a compatible non-empty
//!   index. This is enforced by test, not convention.
//! - Explicit semantic requests fail closed for every unavailable state.
//!   Typed partial-quality coverage remains semantically available at its
//!   measured coverage; hybrid requests may otherwise proceed lexical-only,
//!   but only with a [`SemanticResponseContract`] that names the requested
//!   and realized topologies, reports zero coverage, and admits zero semantic
//!   scores.
//! - The planner is pure and exhaustive: every enumerated state is matched
//!   without wildcards, so a new readiness state fails compilation until it
//!   is planned for.
//! - Model acquisition is always scoped to one logical model, tier, complete
//!   mathematical embedding-space identity, frozen manifest, revision,
//!   license assertion, source, byte budget, path-free destination identity,
//!   document census, and caller-computed reindex estimate. An authorization
//!   for any other scope is not interchangeable.
//! - Serialized plans are deliberately untrusted and non-executable. A caller
//!   must validate one against independently obtained readiness, request,
//!   policy, and frozen-target inputs; success returns a newly planned
//!   [`RecoveryPlan`] rather than promoting payload fields.
//! - A recovery action exposes argv only when the current product command
//!   enforces the complete authorization and semantic identity contract.
//!   Model acquisition, daemon/ANN repair, and all semantic index mutations
//!   are capability-blocked in this core-only tranche because current generic
//!   commands do not provide that binding.
//!
//! # Why schema v4
//!
//! The original v1 foundation represented offline recovery as a blocked
//! network download, represented request mode without retrieval topology,
//! and had no producer provenance, response-admission contract, or scoped
//! acquisition authorization. Correcting those facts changes required wire
//! fields and reverses the meaning of the offline transition, so decoding the
//! v2 contract as v1 would be unsafe. V3 additionally bound acquisition
//! consent to the exact model ID, tier, and mathematical space plus the
//! caller-supplied document count and estimated reindex duration, and refuses
//! to emit acquisition argv until an executor can consume that entire scope.
//! It also separated untrusted wire payloads from executable plans and pinned
//! recovery-local tier values to lowercase `fast` / `quality` spellings
//! instead of inheriting the Rust enum's incidental serde representation.
//! V4 makes every acquisition authorization short-lived, nonce-bound, and
//! evaluated only against caller-supplied trusted time. A v3 client cannot
//! safely present or validate those required anti-replay facts, so v4
//! deliberately fails closed on older payloads instead of installing a
//! compatibility shim. Every v1/v2/v3 stable
//! state/action/postcondition/policy code remains unchanged.
//!
//! # Stable codes
//!
//! Every state, action, postcondition, and policy-prerequisite code is a
//! three-segment lowercase dotted identifier (the same format the
//! observability lint enforces for reason codes) and is append-only within
//! a schema version. All codes live in this module so the full table is
//! auditable in one place; a test validates format and uniqueness against
//! [`crate::decision_plane::ReasonCode`] rules.

mod authorization;
mod contract;
mod planner;
mod types;
mod wire_support;

pub use authorization::{
    ModelAcquisitionAuthorization, ModelAcquisitionSource, ModelAcquisitionTarget,
    ModelDestinationClass,
};
pub use contract::{
    RecoveryAction, RecoveryPlan, Retryability, SemanticResponseContract, TrustedRecoveryContext,
    UntrustedRecoveryPlan,
};
pub use planner::plan;
pub use types::{
    InteractionPolicy, ModelAcquisitionAuthorizationSchemaVersion, NetworkPolicy,
    RecoveryContractError, RecoveryPlanSchemaVersion, RecoveryPolicy, RecoveryRequest, RequestMode,
    SemanticProvenance, SemanticReadiness, VerifiedSemanticProvenance,
};

/// Schema version for serialized [`RecoveryPlan`] payloads.
pub const RECOVERY_PLAN_SCHEMA_VERSION: &str = "frankensearch.recovery_plan.v4";

/// Schema version for a scoped [`ModelAcquisitionAuthorization`].
pub const MODEL_ACQUISITION_AUTHORIZATION_SCHEMA_VERSION: &str =
    "frankensearch.model_acquisition_authorization.v3";

/// Maximum lifetime of one exact model-acquisition authorization.
///
/// The planner is clock-free: callers freeze issuance, expiry, and trusted
/// evaluation time. This bound limits replay exposure but does not claim
/// single-use semantics; an executor promising single use must additionally
/// consume nonces atomically.
pub const MAX_MODEL_ACQUISITION_AUTHORIZATION_LIFETIME_SECONDS: u64 = 15 * 60;

/// One million parts per million: complete semantic document coverage.
pub const COMPLETE_COVERAGE_PPM: u32 = 1_000_000;

/// Placeholder token integrators substitute with the resolved index
/// directory.
///
/// The pure planner never sees real user paths (they are redacted from
/// telemetry); rendering a runnable command is the integrator's job.
pub const ARG_INDEX_DIR: &str = "<index-dir>";

/// Placeholder token integrators substitute with the corpus source
/// directory to (re-)ingest.
pub const ARG_SOURCE_DIR: &str = "<source-dir>";

/// Reserved placeholder for a future operator-supplied local model bundle.
///
/// The current fsfs parser does not implement offline model import, so the
/// planner never emits this token in executable argv. It remains a public
/// schema vocabulary item for the future capability rather than pretending a
/// fictional command is runnable today.
pub const ARG_MODEL_BUNDLE: &str = "ARG_MODEL_BUNDLE";

#[cfg(test)]
mod tests;
