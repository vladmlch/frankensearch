use serde::{Deserialize, Serialize};

/// Adapter identity handshake payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterIdentity {
    /// Stable adapter identifier (e.g. `xf-host-adapter`).
    pub adapter_id: String,
    /// Adapter implementation version.
    pub adapter_version: String,
    /// Host project identifier (required).
    pub host_project: String,
    /// Optional runtime role label (e.g. `indexer`, `query`, `control-plane`).
    pub runtime_role: Option<String>,
    /// Optional stable instance UUID advertised by the host integration.
    pub instance_uuid: Option<String>,
    /// Telemetry schema version emitted by this adapter.
    pub telemetry_schema_version: u8,
    /// Redaction policy version enforced by this adapter.
    pub redaction_policy_version: String,
}

/// Canonical host projects with first-class adapter support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalHostProject {
    CodingAgentSessionSearch,
    Xf,
    McpAgentMailRust,
    Frankenterm,
}

impl CanonicalHostProject {
    /// Deterministic host iteration order used by tests and harnesses.
    pub const ALL: [Self; 4] = [
        Self::CodingAgentSessionSearch,
        Self::Xf,
        Self::McpAgentMailRust,
        Self::Frankenterm,
    ];

    /// Canonical host project key.
    #[must_use]
    pub const fn host_project_key(self) -> &'static str {
        match self {
            Self::CodingAgentSessionSearch => "coding_agent_session_search",
            Self::Xf => "xf",
            Self::McpAgentMailRust => "mcp_agent_mail_rust",
            Self::Frankenterm => "frankenterm",
        }
    }

    /// Stable adapter identifier for this host.
    #[must_use]
    pub const fn adapter_id(self) -> &'static str {
        match self {
            Self::CodingAgentSessionSearch => "cass-host-adapter",
            Self::Xf => "xf-host-adapter",
            Self::McpAgentMailRust => "mcp-agent-mail-host-adapter",
            Self::Frankenterm => "frankenterm-host-adapter",
        }
    }

    /// Default runtime role label for this host adapter.
    #[must_use]
    pub const fn default_runtime_role(self) -> &'static str {
        match self {
            Self::McpAgentMailRust => "control-plane",
            Self::CodingAgentSessionSearch | Self::Xf | Self::Frankenterm => "query",
        }
    }
}
