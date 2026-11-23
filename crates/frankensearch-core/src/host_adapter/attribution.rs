use serde::{Deserialize, Serialize};

/// Canonical host-project attribution result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostProjectAttribution {
    /// Canonical project key or `unknown` when unresolved.
    pub resolved_project_key: String,
    /// Confidence score in `[0, 100]`.
    pub confidence_score: u8,
    /// Machine-stable reason code for diagnostics.
    pub reason_code: String,
    /// Whether multiple conflicting project candidates were observed.
    pub collision: bool,
}

impl HostProjectAttribution {
    #[must_use]
    pub fn unknown(reason_code: impl Into<String>) -> Self {
        Self {
            resolved_project_key: "unknown".to_owned(),
            confidence_score: 20,
            reason_code: reason_code.into(),
            collision: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    project: &'static str,
    weight: u8,
    reason: &'static str,
}

/// Resolve canonical host-project attribution from adapter identity + telemetry hints.
///
/// The algorithm is deterministic with source-precedence tie breaks:
/// adapter identity > telemetry project key > host name hint.
#[must_use]
pub fn resolve_host_project_attribution(
    identity_host_project: Option<&str>,
    telemetry_project_key: Option<&str>,
    host_name_hint: Option<&str>,
) -> HostProjectAttribution {
    let mut seen_project_mask = 0_u8;
    let mut winner = None;

    observe_project_candidates(
        identity_host_project,
        4,
        "adapter_identity",
        &mut seen_project_mask,
        &mut winner,
    );
    observe_project_candidates(
        telemetry_project_key,
        3,
        "telemetry_project_key",
        &mut seen_project_mask,
        &mut winner,
    );
    observe_project_candidates(
        host_name_hint,
        1,
        "host_name",
        &mut seen_project_mask,
        &mut winner,
    );

    let Some(winner) = winner else {
        return HostProjectAttribution::unknown("attribution.unknown");
    };
    let collision = seen_project_mask.count_ones() > 1;

    let mut confidence_score: u8 = match winner.weight {
        4 => 95,
        3 => 85,
        1 => 60,
        _ => 50,
    };
    if collision {
        confidence_score = confidence_score.saturating_sub(25);
    }

    let reason_code = if collision {
        "attribution.collision".to_owned()
    } else {
        format!("attribution.{}", winner.reason)
    };

    HostProjectAttribution {
        resolved_project_key: winner.project.to_owned(),
        confidence_score,
        reason_code,
        collision,
    }
}

const PROJECT_CODING_AGENT_SESSION_SEARCH: u8 = 1 << 0;
const PROJECT_FRANKENTERM: u8 = 1 << 1;
const PROJECT_MCP_AGENT_MAIL_RUST: u8 = 1 << 2;
const PROJECT_XF: u8 = 1 << 3;

const PROJECTS_BY_TIE_ORDER: &[(u8, &str)] = &[
    (
        PROJECT_CODING_AGENT_SESSION_SEARCH,
        "coding_agent_session_search",
    ),
    (PROJECT_FRANKENTERM, "frankenterm"),
    (PROJECT_MCP_AGENT_MAIL_RUST, "mcp_agent_mail_rust"),
    (PROJECT_XF, "xf"),
];

fn observe_project_candidates(
    hint: Option<&str>,
    weight: u8,
    reason: &'static str,
    seen_project_mask: &mut u8,
    winner: &mut Option<Candidate>,
) {
    let Some(hint) = hint else {
        return;
    };
    let project_mask = canonical_project_mask_from_hint(hint);
    if project_mask == 0 {
        return;
    }

    *seen_project_mask |= project_mask;
    for (project_bit, project) in PROJECTS_BY_TIE_ORDER {
        if project_mask & project_bit == 0 {
            continue;
        }
        let candidate = Candidate {
            project,
            weight,
            reason,
        };
        let replace = winner
            .as_ref()
            .is_none_or(|current| candidate_beats(candidate, *current));
        if replace {
            *winner = Some(candidate);
        }
    }
}

fn candidate_beats(left: Candidate, right: Candidate) -> bool {
    left.weight > right.weight
        || (left.weight == right.weight
            && (left.project < right.project
                || (left.project == right.project && left.reason < right.reason)))
}

fn canonical_project_mask_from_hint(hint: &str) -> u8 {
    let normalized = normalize_project_hint(hint);
    if normalized.is_empty() {
        return 0;
    }

    project_mask_from_normalized_hint(&normalized)
}

fn project_mask_from_normalized_hint(normalized: &str) -> u8 {
    let mut project_mask = 0_u8;
    let mut prev3 = "";
    let mut prev2 = "";
    let mut prev1 = "";

    for part in normalized.split('_') {
        project_mask |= project_mask_from_single_part(part);
        if prev1 == "agent" && part == "mail" {
            project_mask |= PROJECT_MCP_AGENT_MAIL_RUST;
        }
        if prev3 == "coding" && prev2 == "agent" && prev1 == "session" && part == "search" {
            project_mask |= PROJECT_CODING_AGENT_SESSION_SEARCH;
        }
        if prev3 == "mcp" && prev2 == "agent" && prev1 == "mail" && part == "rust" {
            project_mask |= PROJECT_MCP_AGENT_MAIL_RUST;
        }
        prev3 = prev2;
        prev2 = prev1;
        prev1 = part;
    }

    project_mask
}

fn project_mask_from_single_part(part: &str) -> u8 {
    match part {
        "cass" | "codingagentsessionsearch" => PROJECT_CODING_AGENT_SESSION_SEARCH,
        "xf" => PROJECT_XF,
        "frankenterm" => PROJECT_FRANKENTERM,
        "mcpagentmailrust" | "mcpagentmail" | "agentmail" | "amail" => PROJECT_MCP_AGENT_MAIL_RUST,
        _ => 0,
    }
}

fn normalize_project_hint(hint: &str) -> String {
    let mut normalized = String::with_capacity(hint.len());
    let mut pending_separator = false;

    for ch in hint.chars() {
        if ch.is_ascii_alphanumeric() {
            if pending_separator && !normalized.is_empty() {
                normalized.push('_');
            }
            normalized.push(ch.to_ascii_lowercase());
            pending_separator = false;
        } else {
            pending_separator = true;
        }
    }

    normalized
}
