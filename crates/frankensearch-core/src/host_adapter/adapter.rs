use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::identity::{AdapterIdentity, CanonicalHostProject};
use crate::collectors::{TELEMETRY_SCHEMA_VERSION, TelemetryEnvelope};
use crate::error::SearchResult;

/// Lifecycle hooks exposed to host adapters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AdapterLifecycleEvent {
    /// Adapter session startup.
    SessionStart { ts: String },
    /// Adapter session shutdown.
    SessionStop { ts: String },
    /// Periodic health tick.
    HealthTick { ts: String },
}

/// Host adapter interface used by integrations.
pub trait HostAdapter: fmt::Debug + Send + Sync {
    /// Identity handshake called during initialization and conformance checks.
    fn identity(&self) -> AdapterIdentity;

    /// Emit one canonical telemetry envelope.
    ///
    /// Implementations must be non-blocking and should preserve envelope fidelity.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when emission fails due to adapter transport,
    /// serialization, or sink-level validation problems.
    fn emit_telemetry(&self, envelope: &TelemetryEnvelope) -> SearchResult<()>;

    /// Lifecycle hook invoked by orchestration and conformance harnesses.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when lifecycle processing fails.
    fn on_lifecycle_event(&self, event: &AdapterLifecycleEvent) -> SearchResult<()>;
}

/// Non-blocking sink abstraction used by concrete host adapters.
pub trait AdapterSink: Send + Sync {
    /// Emit a canonical telemetry envelope.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when sink delivery fails.
    fn emit(&self, envelope: &TelemetryEnvelope) -> SearchResult<()>;

    /// Process one lifecycle event.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when lifecycle handling fails.
    fn on_lifecycle_event(&self, event: &AdapterLifecycleEvent) -> SearchResult<()>;
}

/// No-op sink used by default when callers only need conformance-safe identities.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopAdapterSink;

impl AdapterSink for NoopAdapterSink {
    fn emit(&self, _: &TelemetryEnvelope) -> SearchResult<()> {
        Ok(())
    }

    fn on_lifecycle_event(&self, _: &AdapterLifecycleEvent) -> SearchResult<()> {
        Ok(())
    }
}

/// Reusable host adapter implementation that forwards to an injected sink.
///
/// This gives host projects one deterministic adapter path:
/// - canonical identity handshake
/// - non-blocking sink forwarding
/// - shared conformance harness compatibility
pub struct ForwardingHostAdapter {
    identity: AdapterIdentity,
    sink: Arc<dyn AdapterSink>,
}

impl fmt::Debug for ForwardingHostAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ForwardingHostAdapter")
            .field("identity", &self.identity)
            .finish_non_exhaustive()
    }
}

impl ForwardingHostAdapter {
    /// Construct an adapter with an explicit identity and no-op sink.
    #[must_use]
    pub fn new(identity: AdapterIdentity) -> Self {
        Self {
            identity,
            sink: Arc::new(NoopAdapterSink),
        }
    }

    /// Construct a canonical host adapter profile with shared defaults.
    #[must_use]
    pub fn for_host(
        host: CanonicalHostProject,
        adapter_version: impl Into<String>,
        runtime_role: Option<String>,
        instance_uuid: Option<String>,
    ) -> Self {
        let runtime_role = runtime_role.or_else(|| Some(host.default_runtime_role().to_owned()));
        Self::new(AdapterIdentity {
            adapter_id: host.adapter_id().to_owned(),
            adapter_version: adapter_version.into(),
            host_project: host.host_project_key().to_owned(),
            runtime_role,
            instance_uuid,
            telemetry_schema_version: TELEMETRY_SCHEMA_VERSION,
            redaction_policy_version: "v1".to_owned(),
        })
    }

    /// Canonical profile for `coding_agent_session_search` (cass).
    #[must_use]
    pub fn for_cass(adapter_version: impl Into<String>, instance_uuid: Option<String>) -> Self {
        Self::for_host(
            CanonicalHostProject::CodingAgentSessionSearch,
            adapter_version,
            None,
            instance_uuid,
        )
    }

    /// Canonical profile for `xf`.
    #[must_use]
    pub fn for_xf(adapter_version: impl Into<String>, instance_uuid: Option<String>) -> Self {
        Self::for_host(
            CanonicalHostProject::Xf,
            adapter_version,
            None,
            instance_uuid,
        )
    }

    /// Canonical profile for `mcp_agent_mail_rust`.
    #[must_use]
    pub fn for_mcp_agent_mail(
        adapter_version: impl Into<String>,
        instance_uuid: Option<String>,
    ) -> Self {
        Self::for_host(
            CanonicalHostProject::McpAgentMailRust,
            adapter_version,
            None,
            instance_uuid,
        )
    }

    /// Canonical profile for `frankenterm`.
    #[must_use]
    pub fn for_frankenterm(
        adapter_version: impl Into<String>,
        instance_uuid: Option<String>,
    ) -> Self {
        Self::for_host(
            CanonicalHostProject::Frankenterm,
            adapter_version,
            None,
            instance_uuid,
        )
    }

    /// Replace the sink implementation used for forwarding.
    #[must_use]
    pub fn with_sink(mut self, sink: Arc<dyn AdapterSink>) -> Self {
        self.sink = sink;
        self
    }

    /// Borrow this adapter's identity.
    #[must_use]
    pub const fn identity_ref(&self) -> &AdapterIdentity {
        &self.identity
    }
}

impl HostAdapter for ForwardingHostAdapter {
    fn identity(&self) -> AdapterIdentity {
        self.identity.clone()
    }

    fn emit_telemetry(&self, envelope: &TelemetryEnvelope) -> SearchResult<()> {
        self.sink.emit(envelope)
    }

    fn on_lifecycle_event(&self, event: &AdapterLifecycleEvent) -> SearchResult<()> {
        self.sink.on_lifecycle_event(event)
    }
}
