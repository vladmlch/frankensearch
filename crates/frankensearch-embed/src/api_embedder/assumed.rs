use std::fmt;
use std::future::poll_fn;
use std::pin::Pin;

use asupersync::Cx;
use asupersync::bytes::Buf;
use asupersync::http::body::{Body, Frame};
use asupersync::http::h1::{HttpClient, Method};
use tracing::{debug, warn};

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::{EmbeddingIdentityBundleV1, FrozenEmbeddingIdentityBundleV1};
use frankensearch_core::traits::{SearchFuture, l2_normalize_in_place};

use crate::api_provider::ApiProvider;

use super::transport::{
    RateLimiter, api_client_error_to_search, bounded_embedding_failure,
    bounded_remote_producer_label, build_http_client, validate_http_config,
    validate_remote_identity, verify_response_vectors,
};
use super::types::{ApiEmbedderConfig, AssumedRemoteEmbeddingBatchV1, RemoteApiTrustLevelV1};

/// Explicit transient-only client for an unattested remote provider.
///
/// This type intentionally does not implement [`Embedder`](frankensearch_core::traits::Embedder), does not expose a
/// cache constructor, and returns [`AssumedRemoteEmbeddingBatchV1`] rather than
/// identity-bound vectors. It is suitable only for visibly labeled
/// exploration; persistent indexing and verified semantic readiness require
/// [`ApiEmbedder`](super::ApiEmbedder).
pub struct AssumedRemoteApi {
    provider: Box<dyn ApiProvider>,
    client: HttpClient,
    rate_limiter: RateLimiter,
    config: ApiEmbedderConfig,
    pub(super) identity: EmbeddingIdentityBundleV1,
}

impl fmt::Debug for AssumedRemoteApi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AssumedRemoteApi")
            .field(
                "provider",
                &bounded_remote_producer_label(self.provider.provider_name()),
            )
            .field("config", &self.config)
            .field("trust", &RemoteApiTrustLevelV1::AssumedRemote)
            .finish_non_exhaustive()
    }
}

impl AssumedRemoteApi {
    /// Construct the explicit transient-only path.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::UnverifiableRemoteSpace`] if the caller-authored
    /// expected identity is malformed or disagrees with the provider contract.
    pub fn new(
        provider: Box<dyn ApiProvider>,
        config: ApiEmbedderConfig,
        assumed_space_epoch: Option<FrozenEmbeddingIdentityBundleV1>,
    ) -> SearchResult<Self> {
        validate_http_config(&config)?;
        let identity = validate_remote_identity(provider.as_ref(), assumed_space_epoch)?;
        let client = build_http_client(&config);
        let rate_limiter = RateLimiter::new(config.requests_per_minute);
        Ok(Self {
            provider,
            client,
            rate_limiter,
            config,
            identity,
        })
    }

    /// Construct the transient-only path with default HTTP bounds.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::UnverifiableRemoteSpace`] for an invalid expected
    /// identity.
    pub fn with_defaults(
        provider: Box<dyn ApiProvider>,
        assumed_space_epoch: Option<FrozenEmbeddingIdentityBundleV1>,
    ) -> SearchResult<Self> {
        Self::new(provider, ApiEmbedderConfig::default(), assumed_space_epoch)
    }

    /// This path is always visibly unverified.
    #[must_use]
    pub const fn remote_trust_level(&self) -> RemoteApiTrustLevelV1 {
        RemoteApiTrustLevelV1::AssumedRemote
    }

    /// Embed one text for immediate exploratory use.
    pub fn embed_transient<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, AssumedRemoteEmbeddingBatchV1> {
        Box::pin(async move {
            let texts = [text];
            self.embed_batch_transient(cx, &texts).await
        })
    }

    /// Embed a batch for immediate exploratory use.
    ///
    /// The result remains wrapped in an explicit `AssumedRemote` type and is
    /// not accepted by verified cache/index constructors.
    pub fn embed_batch_transient<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, AssumedRemoteEmbeddingBatchV1> {
        Box::pin(async move {
            if texts.is_empty() {
                return Ok(AssumedRemoteEmbeddingBatchV1 {
                    vectors: Vec::new(),
                    identity_fingerprint: self.identity.fingerprint(),
                });
            }

            let mut vectors = Vec::with_capacity(texts.len());
            for chunk in texts.chunks(self.provider.max_batch_size()) {
                let mut batch = self.request_batch_unattested(cx, chunk).await?;
                for vector in &mut batch {
                    l2_normalize_in_place(vector);
                }
                vectors.extend(batch);
            }
            Ok(AssumedRemoteEmbeddingBatchV1 {
                vectors,
                identity_fingerprint: self.identity.fingerprint(),
            })
        })
    }

    async fn request_batch_unattested(
        &self,
        cx: &Cx,
        texts: &[&str],
    ) -> SearchResult<Vec<Vec<f32>>> {
        let headers = self.provider.request_headers();
        let url = self.provider.request_url();
        let safe_provider = bounded_remote_producer_label(self.provider.provider_name());
        let body = self.provider.serialize_request(texts).map_err(|_error| {
            bounded_embedding_failure(
                self.provider.as_ref(),
                "remote request serialization failed",
            )
        })?;

        let mut last_err = None;
        'retry: for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let multiplier = 1_u32.checked_shl((attempt - 1).min(31)).unwrap_or(u32::MAX);
                let backoff = self.config.retry_base_delay.saturating_mul(multiplier);
                debug!(
                    provider = safe_provider.as_str(),
                    attempt,
                    backoff_ms = backoff.as_millis(),
                    trust = "assumed-remote",
                    "retrying transient API request"
                );
                asupersync::time::sleep(cx.now(), backoff).await;
            }

            if let Some(wait) = self.rate_limiter.acquire() {
                asupersync::time::sleep(cx.now(), wait).await;
            }

            let response = self
                .client
                .request_streaming(
                    cx,
                    Method::Post,
                    &url,
                    headers
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect(),
                    body.clone(),
                )
                .await;
            let mut response = match response {
                Ok(response) => response,
                Err(error) => {
                    let error = api_client_error_to_search(&error, self.provider.as_ref());
                    if matches!(error, SearchError::Cancelled { .. }) {
                        return Err(error);
                    }
                    last_err = Some(error);
                    continue;
                }
            };

            let status = response.head.status;
            let mut response_body = Vec::new();
            while let Some(frame) = poll_fn(|cx| Pin::new(&mut response.body).poll_frame(cx)).await
            {
                match frame {
                    Ok(Frame::Data(mut chunk)) => {
                        while chunk.has_remaining() {
                            let bytes = chunk.chunk();
                            if bytes.is_empty() {
                                break;
                            }
                            if response_body.len().saturating_add(bytes.len())
                                > self.config.max_response_bytes
                            {
                                return Err(bounded_embedding_failure(
                                    self.provider.as_ref(),
                                    "remote response exceeded the configured byte budget",
                                ));
                            }
                            response_body.extend_from_slice(bytes);
                            chunk.advance(bytes.len());
                        }
                    }
                    Ok(Frame::Trailers(_)) => {}
                    Err(_error) => {
                        last_err = Some(bounded_embedding_failure(
                            self.provider.as_ref(),
                            "remote HTTP response body failed",
                        ));
                        continue 'retry;
                    }
                }
            }

            if (200..300).contains(&status) {
                let embeddings =
                    self.provider
                        .deserialize_response(&response_body)
                        .map_err(|_error| {
                            bounded_embedding_failure(
                                self.provider.as_ref(),
                                "remote response decoding failed",
                            )
                        })?;
                verify_response_vectors(
                    self.provider.as_ref(),
                    &embeddings,
                    texts.len(),
                    self.provider.dimension(),
                )?;
                return Ok(embeddings);
            }

            if status == 429 || status >= 500 {
                warn!(
                    provider = safe_provider.as_str(),
                    status,
                    attempt,
                    trust = "assumed-remote",
                    "transient API error"
                );
                last_err = Some(bounded_embedding_failure(
                    self.provider.as_ref(),
                    &format!("HTTP {status} remote provider error"),
                ));
                continue;
            }
            return Err(bounded_embedding_failure(
                self.provider.as_ref(),
                &format!("HTTP {status} remote provider error"),
            ));
        }

        Err(last_err.unwrap_or_else(|| {
            bounded_embedding_failure(self.provider.as_ref(), "all retries exhausted")
        }))
    }
}
