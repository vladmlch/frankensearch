//! Host adapter SDK and conformance harness for telemetry integrations.
//!
//! This module provides:
//! - A stable adapter interface for host identity handshake, telemetry emission,
//!   and lifecycle hooks.
//! - Shared conformance validators for schema-version and redaction compliance.
//! - A harness that runs fixture-based contract checks with actionable diagnostics.

mod adapter;
mod attribution;
mod conformance;
mod identity;

pub use adapter::{
    AdapterLifecycleEvent, AdapterSink, ForwardingHostAdapter, HostAdapter, NoopAdapterSink,
};
pub use attribution::{HostProjectAttribution, resolve_host_project_attribution};
pub use conformance::{
    ConformanceConfig, ConformanceHarness, ConformanceReport, ConformanceViolation,
};
pub use identity::{AdapterIdentity, CanonicalHostProject};

/// Default forbidden substrings used by redaction conformance checks.
///
/// These patterns intentionally target high-signal secret formats to minimize false positives
/// on normal telemetry payloads.
pub const DEFAULT_REDACTION_FORBIDDEN_PATTERNS: &[&str] = &[
    "BEGIN PRIVATE KEY",
    "AWS_SECRET_ACCESS_KEY=",
    "ghp_",
    "xoxb-",
];

#[cfg(test)]
mod tests;
