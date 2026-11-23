use std::fmt;
use std::time::Duration;

use sha2::{Digest, Sha256};

use frankensearch_core::error::SearchResult;

use crate::api_provider::MIN_REMOTE_ATTESTATION_KEY_BYTES;

use super::transport::{bounded_remote_identifier, encode_lower_hex, unverifiable_remote_space};

const DEFAULT_MAX_API_RESPONSE_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_API_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for API embedder HTTP behavior.
#[derive(Debug, Clone)]
pub struct ApiEmbedderConfig {
    /// Maximum retries on transient failure (429, 5xx).
    pub max_retries: u32,
    /// Base delay for exponential backoff.
    pub retry_base_delay: Duration,
    /// Requests per minute limit (0 = unlimited).
    pub requests_per_minute: u32,
    /// Total timeout for one HTTP attempt.
    pub request_timeout: Duration,
    /// Maximum response bytes admitted before parsing.
    pub max_response_bytes: usize,
}

impl Default for ApiEmbedderConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_base_delay: Duration::from_millis(500),
            requests_per_minute: 0,
            request_timeout: DEFAULT_API_REQUEST_TIMEOUT,
            max_response_bytes: DEFAULT_MAX_API_RESPONSE_BYTES,
        }
    }
}

/// Explicit trust classification for remote API vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteApiTrustLevelV1 {
    /// Every returned vector passed pinned producer authentication.
    VerifiedRemote,
    /// Caller explicitly requested transient, non-persistable exploration.
    AssumedRemote,
}

/// Explicitly unverified vectors returned only by the transient API.
///
/// This wrapper deliberately does not implement [`Embedder`](frankensearch_core::traits::Embedder) and cannot be
/// wrapped in [`CachedEmbedder`](crate::cached_embedder::CachedEmbedder). Callers must unwrap it consciously; verified
/// indexing/search APIs continue to accept only identity-bound `Embedder`
/// implementations.
#[derive(Clone)]
pub struct AssumedRemoteEmbeddingBatchV1 {
    pub(super) vectors: Vec<Vec<f32>>,
    pub(super) identity_fingerprint: String,
}

impl fmt::Debug for AssumedRemoteEmbeddingBatchV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AssumedRemoteEmbeddingBatchV1")
            .field("trust", &RemoteApiTrustLevelV1::AssumedRemote)
            .field("vector_count", &self.vectors.len())
            .field(
                "vector_dimension",
                &self.vectors.first().map_or(0, Vec::len),
            )
            .field("identity_fingerprint", &self.identity_fingerprint)
            .field("vectors", &"<redacted>")
            .finish()
    }
}

impl AssumedRemoteEmbeddingBatchV1 {
    /// Explicit trust label for this non-persistable result.
    #[must_use]
    pub const fn trust_level(&self) -> RemoteApiTrustLevelV1 {
        RemoteApiTrustLevelV1::AssumedRemote
    }

    /// Expected, caller-authored identity fingerprint.
    ///
    /// This is a label, not producer proof.
    #[must_use]
    pub fn assumed_identity_fingerprint(&self) -> &str {
        &self.identity_fingerprint
    }

    /// Borrow transient vectors for immediate exploratory use.
    #[must_use]
    pub fn vectors(&self) -> &[Vec<f32>] {
        &self.vectors
    }

    /// Consume the visibly unverified wrapper.
    #[must_use]
    pub fn into_vectors(self) -> Vec<Vec<f32>> {
        self.vectors
    }
}

/// Pinned HMAC-SHA256 gateway authority for verified remote embeddings.
///
/// The secret is never formatted or serialized. The generation and key ID are
/// immutable for one `ApiEmbedder`; rotation requires constructing a new
/// verified instance so midstream drift fails closed.
#[derive(Clone)]
pub struct PinnedRemoteAttesterV1 {
    pub(super) key_id: String,
    pub(super) authentication_key: Vec<u8>,
    pub(super) generation: u64,
}

impl fmt::Debug for PinnedRemoteAttesterV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PinnedRemoteAttesterV1")
            .field("key_id", &bounded_remote_identifier(&self.key_id))
            .field("authentication_key", &"<redacted>")
            .field("generation", &self.generation)
            .field("key_fingerprint", &self.key_fingerprint())
            .finish()
    }
}

impl PinnedRemoteAttesterV1 {
    /// Pin one gateway key ID, high-entropy shared key, and generation.
    ///
    /// # Errors
    ///
    /// Returns `UnverifiableRemoteSpace` for an unsafe key ID or a key shorter
    /// than 256 bits.
    pub fn new(
        key_id: impl Into<String>,
        authentication_key: impl Into<Vec<u8>>,
        generation: u64,
    ) -> SearchResult<Self> {
        let key_id = key_id.into();
        let authentication_key = authentication_key.into();
        if bounded_remote_identifier(&key_id) != key_id.as_str() {
            return Err(unverifiable_remote_space(
                "remote-api",
                "pinned attestation key ID is malformed",
            ));
        }
        if authentication_key.len() < MIN_REMOTE_ATTESTATION_KEY_BYTES {
            return Err(unverifiable_remote_space(
                "remote-api",
                "pinned attestation key has insufficient entropy",
            ));
        }
        Ok(Self {
            key_id,
            authentication_key,
            generation,
        })
    }

    /// Bounded key identifier expected in every accepted response.
    #[must_use]
    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    /// Pinned gateway generation expected in every accepted response.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Credential-safe fingerprint for structured diagnostics.
    #[must_use]
    pub fn key_fingerprint(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch.remote-attestation-key-fingerprint.v1");
        hasher.update(&self.authentication_key);
        encode_lower_hex(&hasher.finalize())
    }

    pub(super) fn fresh_nonce(&self, sequence: u64, entropy: &[u8; 32]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch.remote-attestation-request-nonce.v1");
        hasher.update(entropy);
        hasher.update(sequence.to_be_bytes());
        hasher.update(self.generation.to_be_bytes());
        hasher.update(self.key_id.as_bytes());
        encode_lower_hex(&hasher.finalize())
    }
}

impl Drop for PinnedRemoteAttesterV1 {
    fn drop(&mut self) {
        self.authentication_key.fill(0);
    }
}
