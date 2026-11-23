//! Daemon fallback wrappers for sync embedders and rerankers.
//!
//! These wrappers attempt daemon inference first and gracefully fall back to
//! local in-process models with bounded retry and jittered backoff.

use std::fs::File;
use std::io::Read as _;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use frankensearch_core::{
    AttestedDaemonEmbeddingResponseV1, DaemonChallengeV1, DaemonClient, DaemonConnectionIdentityV1,
    DaemonEmbeddingAttestationV1, DaemonError, DaemonOperationV1, DaemonRetryConfig,
    EmbeddingIdentityBundleV1, EmbeddingSpaceKindV1, MIN_DAEMON_ATTESTATION_KEY_BYTES,
    ModelCategory, RerankDocument, RerankScore, SearchError, SearchResult, SyncEmbed, SyncRerank,
    next_request_id,
};
use sha2::{Digest, Sha256};
use tracing::{debug, warn};

/// Trust level carried by daemon embedding APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DaemonTrustLevelV1 {
    /// Every accepted response is producer-authenticated.
    VerifiedRemote,
    /// Raw daemon vectors are explicit, transient, and ineligible for
    /// persistence/cache/index APIs.
    AssumedRemote,
}

/// Pinned HMAC verifier for one public daemon key identifier.
pub struct PinnedDaemonVerifierV1 {
    key_id: String,
    secret_key: Vec<u8>,
}

impl PinnedDaemonVerifierV1 {
    /// Construct a verifier with at least 256 bits of key material.
    ///
    /// # Errors
    ///
    /// Returns `UnverifiableRemoteSpace` for malformed identifiers or short
    /// keys.
    pub fn new(key_id: impl Into<String>, secret_key: Vec<u8>) -> SearchResult<Self> {
        let key_id = key_id.into();
        if !is_bounded_daemon_label(&key_id) || secret_key.len() < MIN_DAEMON_ATTESTATION_KEY_BYTES
        {
            return Err(SearchError::UnverifiableRemoteSpace {
                producer: "<redacted-daemon-producer>".to_owned(),
                reason: "invalid pinned daemon verifier".to_owned(),
            });
        }
        Ok(Self { key_id, secret_key })
    }

    fn validate_for(&self, connection: &DaemonConnectionIdentityV1) -> SearchResult<()> {
        if self.key_id != connection.key_id
            || self.secret_key.len() < MIN_DAEMON_ATTESTATION_KEY_BYTES
        {
            return Err(unverifiable_daemon_space(
                connection,
                "pinned key identifier does not match daemon connection",
            ));
        }
        Ok(())
    }
}

impl std::fmt::Debug for PinnedDaemonVerifierV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PinnedDaemonVerifierV1")
            .field("key_id_fingerprint", &label_fingerprint(&self.key_id))
            .field("secret_key", &"<redacted>")
            .finish()
    }
}

impl Drop for PinnedDaemonVerifierV1 {
    fn drop(&mut self) {
        self.secret_key.fill(0);
    }
}

/// Explicit wrapper around unverified transient daemon vectors.
///
/// This type deliberately carries no embedding identity and cannot implement
/// [`SyncEmbed`].
pub struct AssumedDaemonEmbeddingBatchV1 {
    vectors: Vec<Vec<f32>>,
}

impl AssumedDaemonEmbeddingBatchV1 {
    /// Explicit trust label.
    #[must_use]
    pub const fn trust_level(&self) -> DaemonTrustLevelV1 {
        DaemonTrustLevelV1::AssumedRemote
    }

    /// Borrow transient vectors.
    #[must_use]
    pub fn vectors(&self) -> &[Vec<f32>] {
        &self.vectors
    }

    /// Consume the transient wrapper.
    #[must_use]
    pub fn into_vectors(self) -> Vec<Vec<f32>> {
        self.vectors
    }
}

impl std::fmt::Debug for AssumedDaemonEmbeddingBatchV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AssumedDaemonEmbeddingBatchV1")
            .field("trust_level", &DaemonTrustLevelV1::AssumedRemote)
            .field("vector_count", &self.vectors.len())
            .field(
                "vector_dimension",
                &self.vectors.first().map_or(0, Vec::len),
            )
            .field("vectors", &"<redacted>")
            .finish()
    }
}

/// Explicit transient-only access to a raw daemon client.
///
/// It intentionally does not implement [`SyncEmbed`], expose an immutable
/// identity, or compose with verified cache/index APIs.
pub struct AssumedDaemonClient {
    daemon: Arc<dyn DaemonClient>,
}

impl AssumedDaemonClient {
    #[must_use]
    pub fn new(daemon: Arc<dyn DaemonClient>) -> Self {
        Self { daemon }
    }

    /// Embed one input for transient exploration.
    ///
    /// # Errors
    ///
    /// Returns a typed cancellation or a redacted embedding failure.
    pub fn embed_transient(&self, text: &str) -> SearchResult<AssumedDaemonEmbeddingBatchV1> {
        let request_id = next_request_id();
        let vector = self
            .daemon
            .embed(text, &request_id)
            .map_err(|error| map_assumed_daemon_error(&error))?;
        Ok(AssumedDaemonEmbeddingBatchV1 {
            vectors: vec![vector],
        })
    }

    /// Embed an ordered batch for transient exploration.
    ///
    /// # Errors
    ///
    /// Returns a typed cancellation or a redacted embedding failure.
    pub fn embed_batch_transient(
        &self,
        texts: &[&str],
    ) -> SearchResult<AssumedDaemonEmbeddingBatchV1> {
        let request_id = next_request_id();
        let vectors = self
            .daemon
            .embed_batch(texts, &request_id)
            .map_err(|error| map_assumed_daemon_error(&error))?;
        Ok(AssumedDaemonEmbeddingBatchV1 { vectors })
    }
}

impl std::fmt::Debug for AssumedDaemonClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AssumedDaemonClient")
            .field("trust_level", &DaemonTrustLevelV1::AssumedRemote)
            .field("daemon", &"<redacted>")
            .finish()
    }
}

/// No-op daemon client used when daemon config is missing.
pub struct NoopDaemonClient {
    id: String,
}

impl NoopDaemonClient {
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }
}

impl DaemonClient for NoopDaemonClient {
    fn id(&self) -> &str {
        &self.id
    }

    fn is_available(&self) -> bool {
        false
    }

    fn embed(&self, _text: &str, _request_id: &str) -> Result<Vec<f32>, DaemonError> {
        Err(DaemonError::Unavailable(
            "daemon not configured".to_string(),
        ))
    }

    fn embed_batch(
        &self,
        _texts: &[&str],
        _request_id: &str,
    ) -> Result<Vec<Vec<f32>>, DaemonError> {
        Err(DaemonError::Unavailable(
            "daemon not configured".to_string(),
        ))
    }

    fn rerank(
        &self,
        _query: &str,
        _documents: &[&str],
        _request_id: &str,
    ) -> Result<Vec<f32>, DaemonError> {
        Err(DaemonError::Unavailable(
            "daemon not configured".to_string(),
        ))
    }
}

#[derive(Debug)]
struct DaemonState {
    consecutive_failures: u32,
    next_retry_at: Option<Instant>,
}

impl DaemonState {
    const fn new() -> Self {
        Self {
            consecutive_failures: 0,
            next_retry_at: None,
        }
    }

    fn can_attempt(&self, now: Instant) -> bool {
        self.next_retry_at.is_none_or(|at| now >= at)
    }

    const fn record_success(&mut self) {
        self.consecutive_failures = 0;
        self.next_retry_at = None;
    }

    fn record_failure(&mut self, config: &DaemonRetryConfig, err: &DaemonError) {
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        let retry_after = match err {
            DaemonError::Overloaded { retry_after, .. } => *retry_after,
            _ => None,
        };
        let backoff = config.backoff_for_attempt(self.consecutive_failures, retry_after);
        self.next_retry_at = Some(Instant::now() + backoff);
    }
}

#[derive(Debug)]
struct DaemonFailure {
    error: DaemonError,
    attempts: u32,
    backoff: bool,
}

#[derive(Debug)]
enum DaemonEmbeddingAttemptError {
    Transport(DaemonError),
    Unverifiable,
    Cancelled,
}

#[derive(Debug)]
struct DaemonEmbeddingFailure {
    error: DaemonEmbeddingAttemptError,
    attempts: u32,
    backoff: bool,
}

fn lock_state(state: &Mutex<DaemonState>) -> std::sync::MutexGuard<'_, DaemonState> {
    state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Embedder wrapper that uses the daemon when available and falls back to a local embedder.
pub struct DaemonFallbackEmbedder {
    daemon: Arc<dyn DaemonClient>,
    fallback: Option<Arc<dyn SyncEmbed>>,
    expected_connection: DaemonConnectionIdentityV1,
    verifier: PinnedDaemonVerifierV1,
    model_id: String,
    dimension: usize,
    semantic: bool,
    config: DaemonRetryConfig,
    state: Mutex<DaemonState>,
}

impl DaemonFallbackEmbedder {
    /// Legacy caller-supplied identity cannot authenticate daemon responses.
    ///
    /// # Errors
    ///
    /// Always returns [`SearchError::UnverifiableRemoteSpace`]. Use
    /// [`Self::new_verified`] with a pinned authority, or
    /// [`AssumedDaemonClient`] for explicit transient exploration.
    pub fn new(
        daemon: Arc<dyn DaemonClient>,
        fallback: Arc<dyn SyncEmbed>,
        _config: DaemonRetryConfig,
    ) -> SearchResult<Self> {
        drop(daemon);
        drop(fallback);
        Err(SearchError::UnverifiableRemoteSpace {
            producer: "<redacted-daemon-producer>".to_owned(),
            reason: "caller-supplied daemon identity is not producer proof".to_owned(),
        })
    }

    /// Construct a verified daemon embedder after authenticating handshake and
    /// health boundaries.
    ///
    /// A local fallback is optional. When present, its complete identity must
    /// exactly equal the daemon identity because one `SyncEmbed` object cannot
    /// truthfully expose two producer identities.
    ///
    /// # Errors
    ///
    /// Returns a typed cancellation or `UnverifiableRemoteSpace` for invalid
    /// config, missing proof, identity/key/generation drift, or a mismatched
    /// local fallback.
    pub fn new_verified(
        daemon: Arc<dyn DaemonClient>,
        fallback: Option<Arc<dyn SyncEmbed>>,
        config: DaemonRetryConfig,
        expected_connection: DaemonConnectionIdentityV1,
        verifier: PinnedDaemonVerifierV1,
    ) -> SearchResult<Self> {
        expected_connection
            .validate()
            .map_err(|_| unverifiable_daemon_space(&expected_connection, "invalid connection"))?;
        verifier.validate_for(&expected_connection)?;
        if config.max_attempts == 0 {
            return Err(unverifiable_daemon_space(
                &expected_connection,
                "daemon retry attempts must be non-zero",
            ));
        }
        if let Some(local) = fallback.as_deref() {
            Self::validate_local_fallback(local, &expected_connection)?;
        }
        let model_id = expected_connection
            .embedding_identity
            .space
            .logical_model_id
            .clone();
        let dimension = usize::try_from(expected_connection.embedding_identity.space.dimension)
            .map_err(|_| {
                unverifiable_daemon_space(
                    &expected_connection,
                    "daemon dimension does not fit usize",
                )
            })?;
        let semantic = matches!(
            expected_connection.embedding_identity.space.kind,
            EmbeddingSpaceKindV1::Semantic
        );
        let embedder = Self {
            daemon,
            fallback,
            expected_connection,
            verifier,
            model_id,
            dimension,
            semantic,
            config,
            state: Mutex::new(DaemonState::new()),
        };
        embedder.verify_handshake_and_health()?;
        Ok(embedder)
    }

    /// Construct a verified daemon-only embedder with default retry policy.
    ///
    /// # Errors
    ///
    /// Returns a typed proof or cancellation error from [`Self::new_verified`].
    pub fn with_verified_defaults(
        daemon: Arc<dyn DaemonClient>,
        expected_connection: DaemonConnectionIdentityV1,
        verifier: PinnedDaemonVerifierV1,
    ) -> SearchResult<Self> {
        Self::new_verified(
            daemon,
            None,
            DaemonRetryConfig::default(),
            expected_connection,
            verifier,
        )
    }

    /// Explicit trust label for verified composition.
    #[must_use]
    pub const fn trust_level(&self) -> DaemonTrustLevelV1 {
        DaemonTrustLevelV1::VerifiedRemote
    }

    /// Pinned connection identity authenticated at construction and on every
    /// accepted response.
    #[must_use]
    pub const fn connection_identity(&self) -> &DaemonConnectionIdentityV1 {
        &self.expected_connection
    }

    fn validate_local_fallback(
        fallback: &dyn SyncEmbed,
        expected: &DaemonConnectionIdentityV1,
    ) -> SearchResult<()> {
        let local_identity = fallback
            .identity()
            .map_err(|_| unverifiable_daemon_space(expected, "fallback has no identity"))?;
        local_identity.validate().map_err(|_| {
            unverifiable_daemon_space(expected, "fallback identity failed validation")
        })?;
        let expected_dimension = usize::try_from(expected.embedding_identity.space.dimension)
            .map_err(|_| {
                unverifiable_daemon_space(expected, "daemon dimension does not fit usize")
            })?;
        let expected_semantic = matches!(
            expected.embedding_identity.space.kind,
            EmbeddingSpaceKindV1::Semantic
        );
        if local_identity != &expected.embedding_identity
            || fallback.dimension() != expected_dimension
            || fallback.is_semantic() != expected_semantic
            || fallback.category() != expected.model_category
        {
            return Err(unverifiable_daemon_space(
                expected,
                "fallback metadata or identity differs from daemon identity",
            ));
        }
        Ok(())
    }

    const fn should_retry(err: &DaemonError) -> bool {
        !matches!(
            err,
            DaemonError::InvalidInput(_)
                | DaemonError::Overloaded { .. }
                | DaemonError::Cancelled
                | DaemonError::UnverifiableRemoteSpace
        )
    }

    const fn fallback_reason(err: &DaemonError, backoff_active: bool) -> &'static str {
        if backoff_active {
            return "backoff";
        }
        match err {
            DaemonError::Unavailable(_) => "unavailable",
            DaemonError::Timeout(_) => "timeout",
            DaemonError::Overloaded { .. } => "overloaded",
            DaemonError::Failed(_) => "error",
            DaemonError::InvalidInput(_) => "invalid",
            DaemonError::Cancelled => "cancelled",
            DaemonError::UnverifiableRemoteSpace => "unverifiable",
        }
    }

    fn log_fallback(&self, request_id: &str, retries: u32, reason: &str) {
        warn!(
            daemon_endpoint_hash = self.expected_connection.endpoint_fingerprint,
            daemon_protocol_hash = label_fingerprint(&self.expected_connection.protocol_revision),
            daemon_key_hash = label_fingerprint(&self.expected_connection.key_id),
            daemon_identity_hash = self.expected_connection.embedding_identity.fingerprint(),
            daemon_generation = self.expected_connection.generation,
            request_id = request_id,
            retry_count = retries,
            fallback_reason = reason,
            "Daemon embed failed; using local embedder"
        );
    }

    fn log_attestation_rejection(&self, request_id: &str, reason: &str) {
        warn!(
            daemon_endpoint_hash = self.expected_connection.endpoint_fingerprint,
            daemon_protocol_hash = label_fingerprint(&self.expected_connection.protocol_revision),
            daemon_key_hash = label_fingerprint(&self.expected_connection.key_id),
            daemon_identity_hash = self.expected_connection.embedding_identity.fingerprint(),
            daemon_generation = self.expected_connection.generation,
            request_id,
            rejection = reason,
            "Rejected unverifiable daemon embedding response"
        );
    }

    fn fresh_challenge(
        &self,
        operation: DaemonOperationV1,
        inputs: &[&str],
    ) -> Result<DaemonChallengeV1, DaemonEmbeddingAttemptError> {
        let nonce = fresh_daemon_nonce().map_err(|_| DaemonEmbeddingAttemptError::Unverifiable)?;
        DaemonChallengeV1::for_inputs(nonce, operation, inputs, &self.expected_connection)
            .map_err(|_| DaemonEmbeddingAttemptError::Unverifiable)
    }

    fn verify_attestation(
        &self,
        attestation: &DaemonEmbeddingAttestationV1,
        challenge: &DaemonChallengeV1,
        vectors: &[Vec<f32>],
    ) -> Result<(), DaemonEmbeddingAttemptError> {
        attestation
            .validate_against(challenge, &self.expected_connection, vectors)
            .and_then(|()| attestation.authenticate_hmac_sha256(&self.verifier.secret_key))
            .map_err(|_| DaemonEmbeddingAttemptError::Unverifiable)
    }

    fn verify_control_boundary(
        &self,
        operation: DaemonOperationV1,
    ) -> Result<(), DaemonEmbeddingAttemptError> {
        let challenge = self.fresh_challenge(operation, &[])?;
        let response = match operation {
            DaemonOperationV1::Handshake => self.daemon.handshake_attested(&challenge),
            DaemonOperationV1::Health => self.daemon.health_attested(&challenge),
            DaemonOperationV1::Embed
            | DaemonOperationV1::EmbedBatch
            | DaemonOperationV1::Rerank => {
                return Err(DaemonEmbeddingAttemptError::Unverifiable);
            }
        }
        .map_err(classify_daemon_embedding_error)?;
        self.verify_attestation(&response, &challenge, &[])
    }

    fn verify_handshake_and_health(&self) -> SearchResult<()> {
        if !self.daemon.is_available() {
            return Err(unverifiable_daemon_space(
                &self.expected_connection,
                "daemon unavailable before authenticated handshake",
            ));
        }
        for operation in [DaemonOperationV1::Handshake, DaemonOperationV1::Health] {
            self.verify_control_boundary(operation).map_err(|error| {
                self.embedding_attempt_error_to_search(&error, "daemon.control")
            })?;
        }
        Ok(())
    }

    fn verify_embedding_response(
        &self,
        response: AttestedDaemonEmbeddingResponseV1,
        challenge: &DaemonChallengeV1,
    ) -> Result<Vec<Vec<f32>>, DaemonEmbeddingAttemptError> {
        self.verify_attestation(&response.attestation, challenge, &response.vectors)?;
        Ok(response.vectors)
    }

    fn wait_for_backoff(&self) {
        let next_retry_at = lock_state(&self.state).next_retry_at;
        if let Some(next_retry_at) = next_retry_at {
            let sleep_for = next_retry_at.saturating_duration_since(Instant::now());
            if !sleep_for.is_zero() {
                std::thread::sleep(sleep_for);
            }
        }
    }

    fn try_embed(&self, request_id: &str, text: &str) -> Result<Vec<f32>, DaemonEmbeddingFailure> {
        if !self.daemon.is_available() {
            return Err(DaemonEmbeddingFailure {
                error: DaemonEmbeddingAttemptError::Transport(DaemonError::Unavailable(
                    "daemon not available".to_owned(),
                )),
                attempts: 0,
                backoff: false,
            });
        }
        if !lock_state(&self.state).can_attempt(Instant::now()) {
            return Err(DaemonEmbeddingFailure {
                error: DaemonEmbeddingAttemptError::Transport(DaemonError::Unavailable(
                    "backoff active".to_owned(),
                )),
                attempts: 0,
                backoff: true,
            });
        }

        let mut attempts = 0;
        let mut last_error = DaemonError::Unavailable("daemon embed failed".to_owned());
        while attempts < self.config.max_attempts {
            attempts += 1;
            let challenge = self
                .fresh_challenge(DaemonOperationV1::Embed, &[text])
                .map_err(|error| DaemonEmbeddingFailure {
                    error,
                    attempts,
                    backoff: false,
                })?;
            debug!(
                daemon_endpoint_hash = self.expected_connection.endpoint_fingerprint,
                daemon_protocol_hash =
                    label_fingerprint(&self.expected_connection.protocol_revision),
                daemon_key_hash = label_fingerprint(&self.expected_connection.key_id),
                daemon_identity_hash = self.expected_connection.embedding_identity.fingerprint(),
                daemon_generation = self.expected_connection.generation,
                request_id,
                attempt = attempts,
                max_attempts = self.config.max_attempts,
                "Attempting authenticated daemon embed"
            );
            match self.daemon.embed_attested(text, &challenge) {
                Ok(response) => {
                    let mut vectors = self
                        .verify_embedding_response(response, &challenge)
                        .map_err(|error| DaemonEmbeddingFailure {
                            error,
                            attempts,
                            backoff: false,
                        })?;
                    let vector = vectors.pop().ok_or(DaemonEmbeddingFailure {
                        error: DaemonEmbeddingAttemptError::Unverifiable,
                        attempts,
                        backoff: false,
                    })?;
                    lock_state(&self.state).record_success();
                    return Ok(vector);
                }
                Err(error) => {
                    let classified = classify_daemon_embedding_error(error);
                    let error = match classified {
                        DaemonEmbeddingAttemptError::Transport(error) => error,
                        other => {
                            return Err(DaemonEmbeddingFailure {
                                error: other,
                                attempts,
                                backoff: false,
                            });
                        }
                    };
                    let should_retry = Self::should_retry(&error);
                    let should_backoff = !matches!(error, DaemonError::InvalidInput(_));
                    if should_backoff {
                        lock_state(&self.state).record_failure(&self.config, &error);
                    }
                    debug!(
                        daemon_endpoint_hash = self.expected_connection.endpoint_fingerprint,
                        request_id,
                        attempt = attempts,
                        max_attempts = self.config.max_attempts,
                        will_retry = should_retry && attempts < self.config.max_attempts,
                        error_kind = Self::fallback_reason(&error, false),
                        "Authenticated daemon embed failed"
                    );
                    last_error = error;
                    if !should_retry || attempts >= self.config.max_attempts {
                        break;
                    }
                    if should_backoff {
                        self.wait_for_backoff();
                    }
                    for operation in [DaemonOperationV1::Handshake, DaemonOperationV1::Health] {
                        self.verify_control_boundary(operation).map_err(|error| {
                            DaemonEmbeddingFailure {
                                error,
                                attempts,
                                backoff: false,
                            }
                        })?;
                    }
                }
            }
        }

        Err(DaemonEmbeddingFailure {
            error: DaemonEmbeddingAttemptError::Transport(last_error),
            attempts,
            backoff: false,
        })
    }

    fn try_embed_batch(
        &self,
        request_id: &str,
        texts: &[&str],
    ) -> Result<Vec<Vec<f32>>, DaemonEmbeddingFailure> {
        if !self.daemon.is_available() {
            return Err(DaemonEmbeddingFailure {
                error: DaemonEmbeddingAttemptError::Transport(DaemonError::Unavailable(
                    "daemon not available".to_owned(),
                )),
                attempts: 0,
                backoff: false,
            });
        }
        if !lock_state(&self.state).can_attempt(Instant::now()) {
            return Err(DaemonEmbeddingFailure {
                error: DaemonEmbeddingAttemptError::Transport(DaemonError::Unavailable(
                    "backoff active".to_owned(),
                )),
                attempts: 0,
                backoff: true,
            });
        }

        let mut attempts = 0;
        let mut last_error = DaemonError::Unavailable("daemon embed batch failed".to_owned());
        while attempts < self.config.max_attempts {
            attempts += 1;
            let challenge = self
                .fresh_challenge(DaemonOperationV1::EmbedBatch, texts)
                .map_err(|error| DaemonEmbeddingFailure {
                    error,
                    attempts,
                    backoff: false,
                })?;
            debug!(
                daemon_endpoint_hash = self.expected_connection.endpoint_fingerprint,
                daemon_protocol_hash =
                    label_fingerprint(&self.expected_connection.protocol_revision),
                daemon_key_hash = label_fingerprint(&self.expected_connection.key_id),
                daemon_identity_hash = self.expected_connection.embedding_identity.fingerprint(),
                daemon_generation = self.expected_connection.generation,
                request_id,
                attempt = attempts,
                max_attempts = self.config.max_attempts,
                input_count = texts.len(),
                "Attempting authenticated daemon embed batch"
            );
            match self.daemon.embed_batch_attested(texts, &challenge) {
                Ok(response) => {
                    let vectors = self
                        .verify_embedding_response(response, &challenge)
                        .map_err(|error| DaemonEmbeddingFailure {
                            error,
                            attempts,
                            backoff: false,
                        })?;
                    lock_state(&self.state).record_success();
                    return Ok(vectors);
                }
                Err(error) => {
                    let classified = classify_daemon_embedding_error(error);
                    let error = match classified {
                        DaemonEmbeddingAttemptError::Transport(error) => error,
                        other => {
                            return Err(DaemonEmbeddingFailure {
                                error: other,
                                attempts,
                                backoff: false,
                            });
                        }
                    };
                    let should_retry = Self::should_retry(&error);
                    let should_backoff = !matches!(error, DaemonError::InvalidInput(_));
                    if should_backoff {
                        lock_state(&self.state).record_failure(&self.config, &error);
                    }
                    debug!(
                        daemon_endpoint_hash = self.expected_connection.endpoint_fingerprint,
                        request_id,
                        attempt = attempts,
                        max_attempts = self.config.max_attempts,
                        will_retry = should_retry && attempts < self.config.max_attempts,
                        error_kind = Self::fallback_reason(&error, false),
                        "Authenticated daemon embed batch failed"
                    );
                    last_error = error;
                    if !should_retry || attempts >= self.config.max_attempts {
                        break;
                    }
                    if should_backoff {
                        self.wait_for_backoff();
                    }
                    for operation in [DaemonOperationV1::Handshake, DaemonOperationV1::Health] {
                        self.verify_control_boundary(operation).map_err(|error| {
                            DaemonEmbeddingFailure {
                                error,
                                attempts,
                                backoff: false,
                            }
                        })?;
                    }
                }
            }
        }

        Err(DaemonEmbeddingFailure {
            error: DaemonEmbeddingAttemptError::Transport(last_error),
            attempts,
            backoff: false,
        })
    }

    fn embedding_attempt_error_to_search(
        &self,
        error: &DaemonEmbeddingAttemptError,
        phase: &str,
    ) -> SearchError {
        match error {
            DaemonEmbeddingAttemptError::Cancelled => SearchError::Cancelled {
                phase: phase.to_owned(),
                reason: "daemon operation cancelled".to_owned(),
            },
            DaemonEmbeddingAttemptError::Unverifiable
            | DaemonEmbeddingAttemptError::Transport(_) => unverifiable_daemon_space(
                &self.expected_connection,
                "daemon could not authenticate the required protocol boundary",
            ),
        }
    }
}

impl SyncEmbed for DaemonFallbackEmbedder {
    fn embed_sync(&self, text: &str) -> SearchResult<Vec<f32>> {
        let request_id = next_request_id();
        match self.try_embed(&request_id, text) {
            Ok(vector) => Ok(vector),
            Err(failure) => {
                let retries = failure.attempts.saturating_sub(1);
                match failure.error {
                    DaemonEmbeddingAttemptError::Unverifiable => {
                        self.log_attestation_rejection(&request_id, "attestation");
                        Err(unverifiable_daemon_space(
                            &self.expected_connection,
                            "daemon response failed authenticated identity validation",
                        ))
                    }
                    DaemonEmbeddingAttemptError::Cancelled => Err(SearchError::Cancelled {
                        phase: "daemon.embedding".to_owned(),
                        reason: "daemon operation cancelled".to_owned(),
                    }),
                    DaemonEmbeddingAttemptError::Transport(error) => {
                        let reason = Self::fallback_reason(&error, failure.backoff);
                        self.fallback.as_ref().map_or_else(
                            || Err(map_verified_daemon_transport_error(&self.model_id, &error)),
                            |fallback| {
                                self.log_fallback(&request_id, retries, reason);
                                fallback.embed_sync(text)
                            },
                        )
                    }
                }
            }
        }
    }

    fn embed_batch_sync(&self, texts: &[&str]) -> SearchResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let request_id = next_request_id();
        match self.try_embed_batch(&request_id, texts) {
            Ok(vectors) => Ok(vectors),
            Err(failure) => {
                let retries = failure.attempts.saturating_sub(1);
                match failure.error {
                    DaemonEmbeddingAttemptError::Unverifiable => {
                        self.log_attestation_rejection(&request_id, "attestation");
                        Err(unverifiable_daemon_space(
                            &self.expected_connection,
                            "daemon response failed authenticated identity validation",
                        ))
                    }
                    DaemonEmbeddingAttemptError::Cancelled => Err(SearchError::Cancelled {
                        phase: "daemon.embedding_batch".to_owned(),
                        reason: "daemon operation cancelled".to_owned(),
                    }),
                    DaemonEmbeddingAttemptError::Transport(error) => {
                        let reason = Self::fallback_reason(&error, failure.backoff);
                        self.fallback.as_ref().map_or_else(
                            || Err(map_verified_daemon_transport_error(&self.model_id, &error)),
                            |fallback| {
                                self.log_fallback(&request_id, retries, reason);
                                fallback.embed_batch_sync(texts)
                            },
                        )
                    }
                }
            }
        }
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.expected_connection.embedding_identity)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn id(&self) -> &str {
        &self.model_id
    }

    fn model_name(&self) -> &str {
        &self.model_id
    }

    fn is_ready(&self) -> bool {
        self.verify_handshake_and_health().is_ok()
            || self
                .fallback
                .as_ref()
                .is_some_and(|fallback| fallback.is_ready())
    }

    fn is_semantic(&self) -> bool {
        self.semantic
    }

    fn category(&self) -> ModelCategory {
        self.expected_connection.model_category
    }
}

fn classify_daemon_embedding_error(error: DaemonError) -> DaemonEmbeddingAttemptError {
    match error {
        DaemonError::Cancelled => DaemonEmbeddingAttemptError::Cancelled,
        DaemonError::UnverifiableRemoteSpace => DaemonEmbeddingAttemptError::Unverifiable,
        transport => DaemonEmbeddingAttemptError::Transport(transport),
    }
}

fn fresh_daemon_nonce() -> std::io::Result<String> {
    let mut entropy = [0_u8; 32];
    File::open("/dev/urandom")?.read_exact(&mut entropy)?;
    Ok(encode_lower_hex(&entropy))
}

fn map_assumed_daemon_error(error: &DaemonError) -> SearchError {
    match error {
        DaemonError::Cancelled => SearchError::Cancelled {
            phase: "daemon.assumed_transient".to_owned(),
            reason: "daemon operation cancelled".to_owned(),
        },
        DaemonError::UnverifiableRemoteSpace => SearchError::UnverifiableRemoteSpace {
            producer: "<redacted-daemon-producer>".to_owned(),
            reason: "daemon rejected the transient request".to_owned(),
        },
        _ => SearchError::EmbeddingFailed {
            model: "<assumed-daemon>".to_owned(),
            source: std::io::Error::other("assumed daemon transport failed").into(),
        },
    }
}

fn map_verified_daemon_transport_error(model_id: &str, error: &DaemonError) -> SearchError {
    match error {
        DaemonError::Cancelled => SearchError::Cancelled {
            phase: "daemon.embedding".to_owned(),
            reason: "daemon operation cancelled".to_owned(),
        },
        DaemonError::UnverifiableRemoteSpace => SearchError::UnverifiableRemoteSpace {
            producer: "<redacted-daemon-producer>".to_owned(),
            reason: "daemon embedding space is unverifiable".to_owned(),
        },
        _ => SearchError::EmbeddingFailed {
            model: label_fingerprint(model_id),
            source: std::io::Error::other("verified daemon transport failed").into(),
        },
    }
}

fn unverifiable_daemon_space(connection: &DaemonConnectionIdentityV1, reason: &str) -> SearchError {
    SearchError::UnverifiableRemoteSpace {
        producer: connection.endpoint_fingerprint.clone(),
        reason: if reason.len() <= 256 && !reason.chars().any(char::is_control) {
            reason.to_owned()
        } else {
            "daemon identity validation failed".to_owned()
        },
    }
}

fn is_bounded_daemon_label(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'/' | b':')
        })
}

fn label_fingerprint(value: &str) -> String {
    encode_lower_hex(&Sha256::digest(value.as_bytes()))
}

fn encode_lower_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

/// Producer-authentication state for a verified rerank channel.
struct VerifiedRerankChannel {
    expected_connection: DaemonConnectionIdentityV1,
    verifier: PinnedDaemonVerifierV1,
}

/// Reranker wrapper that uses the daemon when available and falls back to a local reranker.
pub struct DaemonFallbackReranker {
    daemon: Arc<dyn DaemonClient>,
    fallback: Option<Arc<dyn SyncRerank>>,
    config: DaemonRetryConfig,
    state: Mutex<DaemonState>,
    verified: Option<VerifiedRerankChannel>,
}

impl DaemonFallbackReranker {
    /// Construct an UNAUTHENTICATED reranker (bd-5vult): scores are consumed
    /// from the raw `DaemonClient::rerank` primitive with shape validation but
    /// no producer proof, so any process that can bind the daemon endpoint
    /// controls result ordering. [`Self::trust_level`] reports
    /// [`DaemonTrustLevelV1::AssumedRemote`] to keep the asymmetry loud; use
    /// [`Self::new_verified`] wherever the embedder side is verified.
    #[must_use]
    pub fn new(
        daemon: Arc<dyn DaemonClient>,
        fallback: Option<Arc<dyn SyncRerank>>,
        config: DaemonRetryConfig,
    ) -> Self {
        Self {
            daemon,
            fallback,
            config,
            state: Mutex::new(DaemonState::new()),
            verified: None,
        }
    }

    /// Construct a reranker whose every accepted daemon response is
    /// producer-authenticated, mirroring
    /// [`DaemonFallbackEmbedder::new_verified`]: each attempt sends a fresh
    /// nonce challenge over `DaemonOperationV1::Rerank` and rejects any
    /// response whose envelope fails `validate_against` + HMAC
    /// authentication. Unlike the embedder constructor this does not probe
    /// handshake/health at build time — the reranker holds no channel state
    /// beyond the per-response proof, and every response is independently
    /// authenticated.
    ///
    /// # Errors
    ///
    /// Returns `UnverifiableRemoteSpace` for an invalid connection identity,
    /// a verifier/key mismatch, or a zero retry budget.
    pub fn new_verified(
        daemon: Arc<dyn DaemonClient>,
        fallback: Option<Arc<dyn SyncRerank>>,
        config: DaemonRetryConfig,
        expected_connection: DaemonConnectionIdentityV1,
        verifier: PinnedDaemonVerifierV1,
    ) -> SearchResult<Self> {
        expected_connection
            .validate()
            .map_err(|_| unverifiable_daemon_space(&expected_connection, "invalid connection"))?;
        verifier.validate_for(&expected_connection)?;
        if config.max_attempts == 0 {
            return Err(unverifiable_daemon_space(
                &expected_connection,
                "daemon retry attempts must be non-zero",
            ));
        }
        Ok(Self {
            daemon,
            fallback,
            config,
            state: Mutex::new(DaemonState::new()),
            verified: Some(VerifiedRerankChannel {
                expected_connection,
                verifier,
            }),
        })
    }

    /// Explicit trust label: [`DaemonTrustLevelV1::VerifiedRemote`] when
    /// constructed via [`Self::new_verified`], otherwise
    /// [`DaemonTrustLevelV1::AssumedRemote`].
    #[must_use]
    pub const fn trust_level(&self) -> DaemonTrustLevelV1 {
        if self.verified.is_some() {
            DaemonTrustLevelV1::VerifiedRemote
        } else {
            DaemonTrustLevelV1::AssumedRemote
        }
    }

    /// One authenticated rerank attempt: fresh nonce challenge over the
    /// ordered `[query, documents..]` inputs, then envelope validation and
    /// HMAC authentication before any score is released.
    fn attempt_verified_rerank(
        channel: &VerifiedRerankChannel,
        daemon: &dyn DaemonClient,
        query: &str,
        documents: &[&str],
    ) -> Result<Vec<f32>, DaemonError> {
        let mut inputs = Vec::with_capacity(documents.len() + 1);
        inputs.push(query);
        inputs.extend_from_slice(documents);
        let nonce = fresh_daemon_nonce().map_err(|_| DaemonError::UnverifiableRemoteSpace)?;
        let challenge = DaemonChallengeV1::for_inputs(
            nonce,
            DaemonOperationV1::Rerank,
            &inputs,
            &channel.expected_connection,
        )?;
        let response = daemon.rerank_attested(query, documents, &challenge)?;
        response.attestation.validate_against(
            &challenge,
            &channel.expected_connection,
            &response.vectors,
        )?;
        response
            .attestation
            .authenticate_hmac_sha256(&channel.verifier.secret_key)?;
        let mut vectors = response.vectors;
        let scores = vectors.pop().ok_or(DaemonError::UnverifiableRemoteSpace)?;
        if !vectors.is_empty() {
            return Err(DaemonError::UnverifiableRemoteSpace);
        }
        Ok(scores)
    }

    fn log_fallback(&self, request_id: &str, retries: u32, reason: &str) {
        warn!(
            daemon_id_hash = label_fingerprint(self.daemon.id()),
            request_id,
            retry_count = retries,
            fallback_reason = reason,
            "Daemon rerank failed; using local reranker"
        );
    }

    fn try_rerank(
        &self,
        request_id: &str,
        query: &str,
        documents: &[&str],
    ) -> Result<Vec<f32>, DaemonFailure> {
        if !self.daemon.is_available() {
            return Err(DaemonFailure {
                error: DaemonError::Unavailable("daemon not available".to_string()),
                attempts: 0,
                backoff: false,
            });
        }
        let now = Instant::now();
        if !lock_state(&self.state).can_attempt(now) {
            return Err(DaemonFailure {
                error: DaemonError::Unavailable("backoff active".to_string()),
                attempts: 0,
                backoff: true,
            });
        }

        let mut attempts = 0;
        let mut last_err: Option<DaemonError> = None;

        while attempts < self.config.max_attempts {
            attempts += 1;
            debug!(
                daemon_id_hash = label_fingerprint(self.daemon.id()),
                request_id,
                attempt = attempts,
                max_attempts = self.config.max_attempts,
                "Attempting daemon rerank"
            );
            let attempt = self.verified.as_ref().map_or_else(
                || self.daemon.rerank(query, documents, request_id),
                |channel| {
                    Self::attempt_verified_rerank(channel, self.daemon.as_ref(), query, documents)
                },
            );
            match attempt {
                Ok(scores) => {
                    lock_state(&self.state).record_success();
                    return Ok(scores);
                }
                Err(err) => {
                    let should_retry = DaemonFallbackEmbedder::should_retry(&err);
                    let should_backoff = !matches!(err, DaemonError::InvalidInput(_));
                    let backoff = if should_backoff {
                        lock_state(&self.state).record_failure(&self.config, &err);
                        true
                    } else {
                        false
                    };

                    debug!(
                        daemon_id_hash = label_fingerprint(self.daemon.id()),
                        request_id,
                        attempt = attempts,
                        max_attempts = self.config.max_attempts,
                        will_retry = should_retry && attempts < self.config.max_attempts,
                        error_kind = DaemonFallbackEmbedder::fallback_reason(&err, false),
                        "Daemon rerank failed"
                    );

                    last_err = Some(err);
                    if !should_retry || attempts >= self.config.max_attempts {
                        break;
                    }

                    if backoff && let Some(next_retry_at) = lock_state(&self.state).next_retry_at {
                        let sleep_for = next_retry_at.saturating_duration_since(Instant::now());
                        if !sleep_for.is_zero() {
                            std::thread::sleep(sleep_for);
                        }
                    }
                }
            }
        }

        Err(DaemonFailure {
            error: last_err
                .unwrap_or_else(|| DaemonError::Unavailable("daemon rerank failed".to_string())),
            attempts,
            backoff: false,
        })
    }

    /// Reject a structurally invalid daemon rerank response before any score
    /// is attributed to a document.
    ///
    /// `Vec<f32>` carries no length invariant on the wire: a daemon that caps
    /// its batch or truncates on a partial read returns a score prefix, and a
    /// positional zip would assign `0.0` to every document past it — sinking
    /// them to the bottom of a "successful" rerank with no error. Non-finite
    /// scores poison downstream ordering the same way. Both are daemon
    /// contract violations, so they take the existing failed-call fallback
    /// path as `InvalidInput`.
    fn validate_rerank_shape(scores: &[f32], expected: usize) -> Result<(), DaemonError> {
        if scores.len() != expected {
            return Err(DaemonError::InvalidInput(format!(
                "daemon rerank returned {} scores for {expected} documents",
                scores.len()
            )));
        }
        if let Some(index) = scores.iter().position(|score| !score.is_finite()) {
            return Err(DaemonError::InvalidInput(format!(
                "daemon rerank returned a non-finite score at index {index}"
            )));
        }
        Ok(())
    }
}

impl SyncRerank for DaemonFallbackReranker {
    fn rerank_sync(
        &self,
        query: &str,
        documents: &[RerankDocument],
    ) -> SearchResult<Vec<RerankScore>> {
        let texts: Vec<&str> = documents.iter().map(|doc| doc.text.as_str()).collect();
        let request_id = next_request_id();

        let outcome = self
            .try_rerank(&request_id, query, &texts)
            .and_then(
                |scores| match Self::validate_rerank_shape(&scores, documents.len()) {
                    Ok(()) => Ok(scores),
                    Err(error) => Err(DaemonFailure {
                        error,
                        attempts: 1,
                        backoff: false,
                    }),
                },
            );
        match outcome {
            Ok(scores) => Ok(documents
                .iter()
                .zip(scores)
                .enumerate()
                .map(|(index, (doc, score))| RerankScore {
                    doc_id: doc.doc_id.clone(),
                    score,
                    original_rank: index,
                    raw_logit: None,
                })
                .collect()),
            Err(failure) => {
                if matches!(&failure.error, DaemonError::Cancelled) {
                    return Err(SearchError::Cancelled {
                        phase: "daemon.rerank".to_owned(),
                        reason: "daemon operation cancelled".to_owned(),
                    });
                }
                // A failed producer proof is an authentication event, not an
                // availability event: falling back would mask an active
                // endpoint takeover, so it hard-errors (parity with the
                // embedder's attestation-failure handling).
                if self.verified.is_some()
                    && matches!(&failure.error, DaemonError::UnverifiableRemoteSpace)
                {
                    return Err(SearchError::UnverifiableRemoteSpace {
                        producer: "<redacted-daemon-producer>".to_owned(),
                        reason: "daemon rerank response failed producer authentication".to_owned(),
                    });
                }
                let retries = failure.attempts.saturating_sub(1);
                let reason =
                    DaemonFallbackEmbedder::fallback_reason(&failure.error, failure.backoff);
                self.log_fallback(&request_id, retries, reason);
                self.fallback.as_ref().map_or_else(
                    || {
                        Err(SearchError::RerankFailed {
                            model: "daemon-reranker".to_string(),
                            source: std::io::Error::other("no local reranker available").into(),
                        })
                    },
                    |reranker| reranker.rerank_sync(query, documents),
                )
            }
        }
    }

    fn id(&self) -> &str {
        self.fallback
            .as_ref()
            .map_or("daemon-reranker", |fallback| fallback.id())
    }

    fn model_name(&self) -> &str {
        self.fallback
            .as_ref()
            .map_or("daemon-reranker", |fallback| fallback.model_name())
    }

    fn max_length(&self) -> usize {
        self.fallback
            .as_ref()
            .map_or(512, |fallback| fallback.max_length())
    }

    fn is_available(&self) -> bool {
        self.daemon.is_available()
            || self
                .fallback
                .as_ref()
                .is_some_and(|reranker| reranker.is_available())
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::unnecessary_literal_bound
)]
mod tests;
