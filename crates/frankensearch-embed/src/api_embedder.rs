//! Cloud API embedder implementing the `Embedder` trait.
//!
//! Wraps any [`super::api_provider::ApiProvider`] with HTTP transport, retry
//! logic, rate limiting, and L2 normalization. Gated behind the `api` feature.

use std::fmt;
use std::future::poll_fn;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use asupersync::Cx;
use asupersync::bytes::Buf;
use asupersync::http::body::{Body, Frame};
use asupersync::http::h1::{HttpClient, Method};
use tracing::{debug, warn};

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::{EmbeddingIdentityBundleV1, FrozenEmbeddingIdentityBundleV1};
use frankensearch_core::traits::{Embedder, ModelCategory, SearchFuture, l2_normalize_in_place};

use crate::api_provider::{
    ApiProvider, RemoteEmbeddingAttestationV1, RemoteEmbeddingChallengeV1,
    remote_embedding_payload_sha256, remote_endpoint_fingerprint, remote_ordered_request_sha256,
};
use crate::cached_embedder::CachedEmbedder;

mod assumed;
mod transport;
mod types;

pub use assumed::AssumedRemoteApi;
pub use types::{
    ApiEmbedderConfig, AssumedRemoteEmbeddingBatchV1, PinnedRemoteAttesterV1, RemoteApiTrustLevelV1,
};

use transport::{
    RateLimiter, api_client_error_to_search, bounded_embedding_failure, bounded_remote_identifier,
    bounded_remote_producer_label, build_http_client, embed_checkpoint, unverifiable_remote_space,
    validate_http_config, validate_remote_identity,
};

// ─── ApiEmbedder ────────────────────────────────────────────────────────────

#[derive(Debug)]
struct VerifiedProviderContractV1 {
    provider_name: String,
    model: String,
    embedder_id: String,
    endpoint_fingerprint: String,
    dimension: usize,
    max_batch_size: usize,
    supports_mrl: bool,
}

impl VerifiedProviderContractV1 {
    fn capture(provider: &dyn ApiProvider) -> SearchResult<Self> {
        let provider_name = provider.provider_name().to_owned();
        if bounded_remote_producer_label(&provider_name) != provider_name {
            return Err(unverifiable_remote_space(
                provider.provider_name(),
                "provider name is not a bounded attestation identifier",
            ));
        }
        let model = provider.api_model_id().to_owned();
        if bounded_remote_identifier(&model) != model {
            return Err(unverifiable_remote_space(
                provider.provider_name(),
                "provider model is not a bounded attestation identifier",
            ));
        }
        let embedder_id = provider.embedder_id().to_owned();
        if bounded_remote_identifier(&embedder_id) != embedder_id {
            return Err(unverifiable_remote_space(
                provider.provider_name(),
                "provider embedder ID is not a bounded diagnostic identifier",
            ));
        }
        Ok(Self {
            provider_name,
            model,
            embedder_id,
            endpoint_fingerprint: remote_endpoint_fingerprint(provider.endpoint_url()),
            dimension: provider.dimension(),
            max_batch_size: provider.max_batch_size(),
            supports_mrl: provider.supports_mrl(),
        })
    }

    fn verify_unchanged(&self, provider: &dyn ApiProvider) -> SearchResult<()> {
        let unchanged = provider.provider_name() == self.provider_name
            && provider.api_model_id() == self.model
            && provider.embedder_id() == self.embedder_id
            && remote_endpoint_fingerprint(provider.endpoint_url()) == self.endpoint_fingerprint
            && provider.dimension() == self.dimension
            && provider.max_batch_size() == self.max_batch_size
            && provider.supports_mrl() == self.supports_mrl;
        if unchanged {
            return Ok(());
        }
        Err(unverifiable_remote_space(
            &self.provider_name,
            "remote provider contract drifted after verified construction",
        ))
    }
}

/// Cloud API embedder wrapping any [`ApiProvider`].
///
/// Handles HTTP transport, retry with exponential backoff, rate limiting,
/// batch chunking, and L2 normalization. Every successful response must carry
/// a space/producer attestation matching the construction-time epoch; an epoch
/// supplied by the caller does not authenticate the responding service.
pub struct ApiEmbedder {
    provider: Box<dyn ApiProvider>,
    client: HttpClient,
    rate_limiter: RateLimiter,
    config: ApiEmbedderConfig,
    identity: EmbeddingIdentityBundleV1,
    provider_contract: VerifiedProviderContractV1,
    attester: PinnedRemoteAttesterV1,
    request_sequence: AtomicU64,
}

impl fmt::Debug for ApiEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApiEmbedder")
            .field("provider", &self.provider_contract.provider_name)
            .field("config", &self.config)
            .field("trust", &RemoteApiTrustLevelV1::VerifiedRemote)
            .field("attester", &self.attester)
            .finish_non_exhaustive()
    }
}

impl ApiEmbedder {
    /// Reject the legacy caller-asserted construction path.
    ///
    /// A frozen identity supplied by the caller is expected state, not producer
    /// proof. Use [`Self::new_verified`] with a pinned gateway authority.
    ///
    /// # Errors
    ///
    /// Always returns [`SearchError::UnverifiableRemoteSpace`]. The identity is
    /// still validated first so malformed explicit configuration retains its
    /// precise fail-closed boundary.
    pub fn new(
        provider: Box<dyn ApiProvider>,
        _config: ApiEmbedderConfig,
        immutable_space_epoch: Option<FrozenEmbeddingIdentityBundleV1>,
    ) -> SearchResult<Self> {
        let _identity = validate_remote_identity(provider.as_ref(), immutable_space_epoch)?;
        let provider_name = provider.provider_name().to_owned();
        drop(provider);
        Err(unverifiable_remote_space(
            &provider_name,
            "verified remote construction requires a pinned producer authentication key",
        ))
    }

    /// Create an API embedder whose every response must authenticate against a
    /// pinned gateway key and immutable generation.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::UnverifiableRemoteSpace`] if the epoch is absent,
    /// malformed, non-semantic, incompatible with the provider/output
    /// contract, or lacks a valid pinned producer authority.
    pub fn new_verified(
        provider: Box<dyn ApiProvider>,
        config: ApiEmbedderConfig,
        immutable_space_epoch: Option<FrozenEmbeddingIdentityBundleV1>,
        attester: PinnedRemoteAttesterV1,
    ) -> SearchResult<Self> {
        validate_http_config(&config)?;
        let identity = validate_remote_identity(provider.as_ref(), immutable_space_epoch)?;
        let provider_contract = VerifiedProviderContractV1::capture(provider.as_ref())?;
        let client = build_http_client(&config);
        let rate_limiter = RateLimiter::new(config.requests_per_minute);
        Ok(Self {
            provider,
            client,
            rate_limiter,
            config,
            identity,
            provider_contract,
            attester,
            request_sequence: AtomicU64::new(0),
        })
    }

    /// Create a verified embedder with default HTTP bounds.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::UnverifiableRemoteSpace`] for an invalid epoch or
    /// pinned authority.
    pub fn with_verified_defaults(
        provider: Box<dyn ApiProvider>,
        immutable_space_epoch: Option<FrozenEmbeddingIdentityBundleV1>,
        attester: PinnedRemoteAttesterV1,
    ) -> SearchResult<Self> {
        Self::new_verified(
            provider,
            ApiEmbedderConfig::default(),
            immutable_space_epoch,
            attester,
        )
    }

    /// Trust classification of every vector this type can return.
    #[must_use]
    pub const fn remote_trust_level(&self) -> RemoteApiTrustLevelV1 {
        RemoteApiTrustLevelV1::VerifiedRemote
    }

    /// Pinned gateway key fingerprint for credential-safe telemetry.
    #[must_use]
    pub fn attestation_key_fingerprint(&self) -> String {
        self.attester.key_fingerprint()
    }

    fn build_challenge_with_entropy(
        &self,
        texts: &[&str],
        entropy: &[u8; 32],
    ) -> SearchResult<RemoteEmbeddingChallengeV1> {
        self.provider_contract
            .verify_unchanged(self.provider.as_ref())?;
        let input_count = u32::try_from(texts.len()).map_err(|_| {
            unverifiable_remote_space(
                &self.provider_contract.provider_name,
                "remote request batch exceeds the attestation schema",
            )
        })?;
        let sequence = self.request_sequence.fetch_add(1, Ordering::Relaxed);
        let challenge = RemoteEmbeddingChallengeV1 {
            schema_version: crate::api_provider::REMOTE_EMBEDDING_CHALLENGE_SCHEMA_V1,
            request_nonce: self.attester.fresh_nonce(sequence, entropy),
            ordered_request_sha256: remote_ordered_request_sha256(texts),
            input_count,
            endpoint_fingerprint: self.provider_contract.endpoint_fingerprint.clone(),
            identity_fingerprint: self.identity.fingerprint(),
            space_fingerprint: self.identity.space.fingerprint(),
            producer_fingerprint: self.identity.producer.fingerprint(),
        };
        challenge.validate()?;
        Ok(challenge)
    }

    #[cfg(test)]
    fn build_challenge(&self, texts: &[&str]) -> SearchResult<RemoteEmbeddingChallengeV1> {
        self.build_challenge_with_entropy(texts, &[0x42; 32])
    }

    /// Create with default HTTP configuration but without producer authority.
    ///
    /// This legacy entry point is retained only to fail explicit direct
    /// OpenAI/Gemini construction with a typed non-retryable error.
    ///
    /// # Errors
    ///
    /// Always returns [`SearchError::UnverifiableRemoteSpace`] after validating
    /// the caller-supplied epoch.
    pub fn with_defaults(
        provider: Box<dyn ApiProvider>,
        immutable_space_epoch: Option<FrozenEmbeddingIdentityBundleV1>,
    ) -> SearchResult<Self> {
        Self::new(
            provider,
            ApiEmbedderConfig::default(),
            immutable_space_epoch,
        )
    }

    /// Wrap this verified embedder with a cache (convenience).
    #[must_use]
    pub fn cached(self, capacity: usize) -> CachedEmbedder {
        CachedEmbedder::new(Arc::new(self), capacity)
    }

    /// Wrap with the default cache capacity (4096 entries).
    #[must_use]
    pub fn cached_default(self) -> CachedEmbedder {
        CachedEmbedder::new(Arc::new(self), 4096)
    }

    /// Make a single producer-authenticated API request for a batch, with retry.
    async fn request_batch(&self, cx: &Cx, texts: &[&str]) -> SearchResult<Vec<Vec<f32>>> {
        self.provider_contract
            .verify_unchanged(self.provider.as_ref())?;
        let headers = self.provider.request_headers();
        let url = self.provider.request_url();
        let safe_provider = self.provider_contract.provider_name.as_str();

        let mut last_err = None;
        'retry: for attempt in 0..=self.config.max_retries {
            embed_checkpoint(cx, "api.request_batch")?;
            let mut nonce_entropy = [0_u8; 32];
            cx.random_bytes(&mut nonce_entropy);
            let challenge = self.build_challenge_with_entropy(texts, &nonce_entropy)?;
            nonce_entropy.fill(0);
            let body = self
                .provider
                .serialize_attested_request(texts, &challenge)
                .map_err(|_error| {
                    bounded_embedding_failure(
                        self.provider.as_ref(),
                        "remote request serialization failed",
                    )
                })?;

            if attempt > 0 {
                let multiplier = 1_u32.checked_shl((attempt - 1).min(31)).unwrap_or(u32::MAX);
                let backoff = self.config.retry_base_delay.saturating_mul(multiplier);
                debug!(
                    provider = safe_provider,
                    attempt,
                    backoff_ms = backoff.as_millis(),
                    "retrying API request"
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
                    body,
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
                let attestation =
                    self.provider
                        .response_attestation(&response_body)
                        .map_err(|_error| {
                            unverifiable_remote_space(
                                &self.provider_contract.provider_name,
                                "per-response producer attestation is malformed",
                            )
                        })?;
                self.verify_response_vectors(&embeddings, texts.len())?;
                self.verify_response_attestation(attestation.as_ref(), &challenge, &embeddings)?;
                return Ok(embeddings);
            }

            if status == 429 || status >= 500 {
                warn!(
                    provider = safe_provider,
                    status, attempt, "transient API error"
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

    fn verify_response_attestation(
        &self,
        attestation: Option<&RemoteEmbeddingAttestationV1>,
        challenge: &RemoteEmbeddingChallengeV1,
        embeddings: &[Vec<f32>],
    ) -> SearchResult<()> {
        let attestation = attestation.ok_or_else(|| {
            unverifiable_remote_space(
                &self.provider_contract.provider_name,
                "remote response carried no producer-authenticated attestation",
            )
        })?;
        if !attestation.authenticate_hmac_sha256(&self.attester.authentication_key) {
            return Err(unverifiable_remote_space(
                &self.provider_contract.provider_name,
                "remote response producer authentication failed",
            ));
        }

        let vector_count = u32::try_from(embeddings.len()).map_err(|_| {
            unverifiable_remote_space(
                &self.provider_contract.provider_name,
                "remote response shape exceeds the attestation schema",
            )
        })?;
        let vector_dimension = embeddings.first().map_or(0, Vec::len);
        let vector_dimension = u32::try_from(vector_dimension).map_err(|_| {
            unverifiable_remote_space(
                &self.provider_contract.provider_name,
                "remote response dimension exceeds the attestation schema",
            )
        })?;

        let exact_match = attestation.key_id == self.attester.key_id
            && attestation.generation == self.attester.generation
            && attestation.request_nonce == challenge.request_nonce
            && attestation.ordered_request_sha256 == challenge.ordered_request_sha256
            && attestation.input_count == challenge.input_count
            && attestation.endpoint_fingerprint == challenge.endpoint_fingerprint
            && attestation.identity_fingerprint == challenge.identity_fingerprint
            && attestation.space_fingerprint == challenge.space_fingerprint
            && attestation.producer_fingerprint == challenge.producer_fingerprint
            && attestation.provider == self.provider_contract.provider_name
            && attestation.model == self.provider_contract.model
            && attestation.producer_backend == self.identity.producer.backend
            && attestation.protocol_revision == self.identity.producer.protocol_revision
            && attestation.vector_count == vector_count
            && attestation.vector_dimension == vector_dimension
            && attestation.response_payload_sha256 == remote_embedding_payload_sha256(embeddings);
        if exact_match {
            return Ok(());
        }
        Err(unverifiable_remote_space(
            &self.provider_contract.provider_name,
            "authenticated remote response drifted from its request or immutable epoch",
        ))
    }

    fn verify_response_vectors(
        &self,
        embeddings: &[Vec<f32>],
        expected_count: usize,
    ) -> SearchResult<()> {
        if embeddings.len() != expected_count {
            return Err(bounded_embedding_failure(
                self.provider.as_ref(),
                "remote response vector count disagrees with the request",
            ));
        }
        if embeddings
            .iter()
            .any(|vector| vector.len() != self.dimension())
        {
            return Err(bounded_embedding_failure(
                self.provider.as_ref(),
                "remote response vector dimension disagrees with the attested dimension",
            ));
        }
        Ok(())
    }
}

impl Embedder for ApiEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            embed_checkpoint(cx, "api.embed")?;
            let results = self.request_batch(cx, &[text]).await?;
            results
                .into_iter()
                .next()
                .ok_or_else(|| SearchError::EmbeddingFailed {
                    model: self.provider_contract.embedder_id.clone(),
                    source: "empty response from API".into(),
                })
                .map(|mut v| {
                    l2_normalize_in_place(&mut v);
                    v
                })
        })
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            embed_checkpoint(cx, "api.embed_batch")?;
            if texts.is_empty() {
                return Ok(Vec::new());
            }

            let batch_size = self.provider_contract.max_batch_size;
            let mut all_embeddings = Vec::with_capacity(texts.len());

            for chunk in texts.chunks(batch_size) {
                embed_checkpoint(cx, "api.embed_batch")?;
                let mut batch = self.request_batch(cx, chunk).await?;
                for v in &mut batch {
                    l2_normalize_in_place(v);
                }
                all_embeddings.extend(batch);
            }

            Ok(all_embeddings)
        })
    }

    /// Both bound operations are the core defaults over this producer's own
    /// `embed`/`embed_batch`, so a bound batch binds what singles would.
    fn bound_batch_is_native(&self) -> bool {
        true
    }

    fn dimension(&self) -> usize {
        self.provider_contract.dimension
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn id(&self) -> &str {
        &self.provider_contract.embedder_id
    }

    fn model_name(&self) -> &str {
        &self.provider_contract.model
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::ApiEmbedder
    }

    fn supports_mrl(&self) -> bool {
        self.provider_contract.supports_mrl
    }

    fn truncate_embedding(&self, embedding: &[f32], target_dim: usize) -> SearchResult<Vec<f32>> {
        if target_dim == 0 || target_dim > embedding.len() {
            return Err(SearchError::EmbeddingFailed {
                model: self.provider_contract.embedder_id.clone(),
                source: format!(
                    "target dimension {target_dim} must be between 1 and embedding dimension {}",
                    embedding.len()
                )
                .into(),
            });
        }
        let mut truncated = embedding[..target_dim].to_vec();
        l2_normalize_in_place(&mut truncated);
        Ok(truncated)
    }
}

#[cfg(test)]
mod tests;
