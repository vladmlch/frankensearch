use std::sync::Mutex;
use std::time::{Duration, Instant};

use asupersync::Cx;
use asupersync::http::h1::{ClientError, HttpClient, HttpClientConfig, RedirectPolicy};

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::{
    EmbeddingIdentityBundleV1, EmbeddingSpaceKindV1, FrozenEmbeddingIdentityBundleV1,
    QuantizationFormat,
};

use crate::api_provider::ApiProvider;

use super::types::ApiEmbedderConfig;

pub(super) const API_OUTPUT_NORMALIZATION_V1: &str = "l2-f32-zero-on-degenerate-v1";
pub(super) const API_STORAGE_FORMAT_V1: &str = "in-memory-f32-v1";
pub(super) const API_STORAGE_ENDIANNESS_V1: &str = "native-f32-values";

// ─── Rate Limiter ───────────────────────────────────────────────────────────

/// Simple token-bucket rate limiter for API calls.
#[derive(Debug)]
pub(super) struct RateLimiter {
    state: Mutex<RateLimiterState>,
    requests_per_minute: u32,
}

#[derive(Debug)]
struct RateLimiterState {
    available_permits: f64,
    last_refill: Instant,
}

impl RateLimiter {
    pub(super) fn new(requests_per_minute: u32) -> Self {
        Self {
            state: Mutex::new(RateLimiterState {
                available_permits: f64::from(requests_per_minute),
                last_refill: Instant::now(),
            }),
            requests_per_minute,
        }
    }

    /// Returns the duration to wait before making a request, or `None` if
    /// a token is available immediately.
    pub(super) fn acquire(&self) -> Option<Duration> {
        if self.requests_per_minute == 0 {
            return None;
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_refill).as_secs_f64();
        let requests_per_minute = f64::from(self.requests_per_minute);
        let refill = elapsed * (requests_per_minute / 60.0);
        state.available_permits = (state.available_permits + refill).min(requests_per_minute);
        state.last_refill = now;

        if state.available_permits >= 1.0 {
            state.available_permits -= 1.0;
            drop(state);
            None
        } else {
            let deficit = 1.0 - state.available_permits;
            state.available_permits = 0.0; // consume all partial permits
            drop(state);
            let wait_secs = deficit / (requests_per_minute / 60.0);
            Some(Duration::from_secs_f64(wait_secs))
        }
    }
}

pub(super) fn validate_remote_identity(
    provider: &dyn ApiProvider,
    immutable_space_epoch: Option<FrozenEmbeddingIdentityBundleV1>,
) -> SearchResult<EmbeddingIdentityBundleV1> {
    if provider.max_batch_size() == 0 {
        return Err(unverifiable_remote_space(
            provider.provider_name(),
            "provider maximum batch size must be greater than zero",
        ));
    }
    let frozen_identity = immutable_space_epoch.ok_or_else(|| {
        unverifiable_remote_space(
            provider.provider_name(),
            "no explicit immutable space epoch was supplied",
        )
    })?;
    frozen_identity.validate().map_err(|_error| {
        unverifiable_remote_space(
            provider.provider_name(),
            "explicit immutable space epoch failed canonical validation",
        )
    })?;
    let identity = frozen_identity.identity;
    if identity.space.kind != EmbeddingSpaceKindV1::Semantic {
        return Err(unverifiable_remote_space(
            provider.provider_name(),
            "remote API embedders require a semantic space identity",
        ));
    }
    if identity.space.logical_model_id != provider.api_model_id() {
        return Err(unverifiable_remote_space(
            provider.provider_name(),
            "explicit space epoch model disagrees with the provider contract",
        ));
    }
    if identity.producer.backend != provider.identity_backend() {
        return Err(unverifiable_remote_space(
            provider.provider_name(),
            "explicit producer backend disagrees with the provider contract",
        ));
    }
    if identity.producer.protocol_revision != provider.identity_protocol_revision() {
        return Err(unverifiable_remote_space(
            provider.provider_name(),
            "explicit producer protocol disagrees with the provider contract",
        ));
    }
    if identity.space.output_normalization != API_OUTPUT_NORMALIZATION_V1 {
        return Err(unverifiable_remote_space(
            provider.provider_name(),
            "explicit space epoch normalization disagrees with the API embedder contract",
        ));
    }
    if identity.storage.quantization != QuantizationFormat::F32
        || identity.storage.format != API_STORAGE_FORMAT_V1
        || identity.storage.endianness != API_STORAGE_ENDIANNESS_V1
    {
        return Err(unverifiable_remote_space(
            provider.provider_name(),
            "explicit storage epoch disagrees with the API embedder's in-memory f32 output contract",
        ));
    }
    let provider_dimension = u32::try_from(provider.dimension()).map_err(|_| {
        unverifiable_remote_space(
            provider.provider_name(),
            "provider dimension does not fit the identity schema",
        )
    })?;
    if identity.space.dimension != provider_dimension {
        return Err(unverifiable_remote_space(
            provider.provider_name(),
            "explicit space epoch dimension disagrees with the provider contract",
        ));
    }
    Ok(identity)
}

pub(super) fn validate_http_config(config: &ApiEmbedderConfig) -> SearchResult<()> {
    if config.request_timeout.is_zero() {
        return Err(SearchError::InvalidConfig {
            field: "api.request_timeout".to_owned(),
            value: "0ms".to_owned(),
            reason: "must be greater than zero".to_owned(),
        });
    }
    if config.max_response_bytes == 0 {
        return Err(SearchError::InvalidConfig {
            field: "api.max_response_bytes".to_owned(),
            value: "0".to_owned(),
            reason: "must be greater than zero".to_owned(),
        });
    }
    Ok(())
}

pub(super) fn verify_response_vectors(
    provider: &dyn ApiProvider,
    embeddings: &[Vec<f32>],
    expected_count: usize,
    expected_dimension: usize,
) -> SearchResult<()> {
    if embeddings.len() != expected_count {
        return Err(bounded_embedding_failure(
            provider,
            "remote response vector count disagrees with the request",
        ));
    }
    if embeddings
        .iter()
        .any(|vector| vector.len() != expected_dimension)
    {
        return Err(bounded_embedding_failure(
            provider,
            "remote response vector dimension disagrees with the expected dimension",
        ));
    }
    Ok(())
}

pub(super) fn build_http_client(config: &ApiEmbedderConfig) -> HttpClient {
    let mut client_config = HttpClientConfig::default();
    client_config.redirect_policy = RedirectPolicy::Limited(5);
    client_config.user_agent = Some(format!(
        "frankensearch/{} (api-embedder)",
        env!("CARGO_PKG_VERSION")
    ));
    client_config.max_body_size = Some(config.max_response_bytes);
    client_config.request_timeout = Some(config.request_timeout);
    HttpClient::with_config(client_config)
}

pub(super) fn bounded_embedding_failure(provider: &dyn ApiProvider, reason: &str) -> SearchError {
    let reason = if reason.len() <= 256 && !reason.chars().any(char::is_control) {
        reason
    } else {
        "remote embedding operation failed"
    };
    SearchError::EmbeddingFailed {
        model: bounded_remote_identifier(provider.embedder_id()),
        source: reason.to_owned().into(),
    }
}

pub(super) fn api_client_error_to_search(
    error: &ClientError,
    provider: &dyn ApiProvider,
) -> SearchError {
    if matches!(error, ClientError::Cancelled) {
        return SearchError::Cancelled {
            phase: "remote-api-embedding-request".to_owned(),
            reason: "HTTP client observed structured cancellation".to_owned(),
        };
    }
    let reason = match error {
        ClientError::InvalidUrl(_) => "remote transport invalid URL",
        ClientError::DnsError(_) => "remote transport DNS failure",
        ClientError::ConnectError(_) | ClientError::Io(_) => "remote transport connection failure",
        ClientError::TlsError(_) => "remote transport TLS failure",
        ClientError::HttpError(_) => "remote HTTP protocol failure",
        ClientError::TooManyRedirects { .. } => "remote transport redirect limit exceeded",
        ClientError::DeadlineExceeded => "remote request deadline exceeded",
        ClientError::ConnectTunnelRefused { .. } => "remote proxy tunnel refused",
        ClientError::InvalidConnectInput(_) => "remote proxy configuration invalid",
        ClientError::ProxyError(_) => "remote proxy failure",
        ClientError::PoolExhausted { .. } => "remote connection pool exhausted",
        ClientError::Cancelled => unreachable!("handled above"),
    };
    bounded_embedding_failure(provider, reason)
}

pub(super) fn bounded_remote_identifier(value: &str) -> String {
    if value.len() <= 128
        && !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'/'))
    {
        value.to_owned()
    } else {
        "<redacted-remote-identifier>".to_owned()
    }
}

pub(super) fn bounded_remote_producer_label(provider: &str) -> String {
    if provider.len() <= 128
        && !provider.is_empty()
        && provider
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        provider.to_owned()
    } else {
        "<redacted-remote-producer>".to_owned()
    }
}

pub(super) fn unverifiable_remote_space(provider: &str, reason: &str) -> SearchError {
    let producer = bounded_remote_producer_label(provider);
    let reason = if reason.len() <= 512 && !reason.chars().any(char::is_control) {
        reason.to_owned()
    } else {
        "remote identity validation failed".to_owned()
    };
    SearchError::UnverifiableRemoteSpace { producer, reason }
}

pub(super) fn encode_lower_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

/// Refuse cooperatively before starting work when the caller has cancelled.
pub(super) fn embed_checkpoint(cx: &Cx, phase: &'static str) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: phase.to_owned(),
        reason: cx
            .cancel_reason()
            .map_or_else(|| error.to_string(), |reason| reason.to_string()),
    })
}
