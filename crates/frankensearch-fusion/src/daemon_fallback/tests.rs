use std::collections::HashSet;
use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::*;

struct ConstEmbedder {
    id: &'static str,
    model_name: &'static str,
    dim: usize,
    value: f32,
    semantic: bool,
    category: ModelCategory,
    identity: EmbeddingIdentityBundleV1,
}

impl SyncEmbed for ConstEmbedder {
    fn embed_sync(&self, _text: &str) -> SearchResult<Vec<f32>> {
        Ok(vec![self.value; self.dim])
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn id(&self) -> &str {
        self.id
    }

    fn model_name(&self) -> &str {
        self.model_name
    }

    fn is_semantic(&self) -> bool {
        self.semantic
    }

    fn category(&self) -> ModelCategory {
        self.category
    }
}

struct ConstReranker {
    id: &'static str,
}

impl SyncRerank for ConstReranker {
    fn rerank_sync(
        &self,
        _query: &str,
        documents: &[RerankDocument],
    ) -> SearchResult<Vec<RerankScore>> {
        Ok(documents
            .iter()
            .enumerate()
            .map(|(idx, doc)| RerankScore {
                doc_id: doc.doc_id.clone(),
                score: 10.0 - idx as f32,
                original_rank: idx,
                raw_logit: None,
            })
            .collect())
    }

    fn id(&self) -> &str {
        self.id
    }

    fn model_name(&self) -> &str {
        self.id
    }
}

/// Daemon fixture whose rerank call succeeds with a caller-chosen score
/// vector, so tests can exercise arity/finiteness violations on the Ok
/// path.
struct ShapedRerankDaemon {
    scores: Vec<f32>,
    raw_calls: AtomicUsize,
}

impl ShapedRerankDaemon {
    fn new(scores: Vec<f32>) -> Self {
        Self {
            scores,
            raw_calls: AtomicUsize::new(0),
        }
    }
}

impl DaemonClient for ShapedRerankDaemon {
    fn id(&self) -> &str {
        "shaped-rerank-daemon"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn embed(&self, _text: &str, _request_id: &str) -> Result<Vec<f32>, DaemonError> {
        Err(DaemonError::Unavailable(
            "shaped rerank fixture has no embedder".to_string(),
        ))
    }

    fn embed_batch(
        &self,
        _texts: &[&str],
        _request_id: &str,
    ) -> Result<Vec<Vec<f32>>, DaemonError> {
        Err(DaemonError::Unavailable(
            "shaped rerank fixture has no embedder".to_string(),
        ))
    }

    fn rerank(
        &self,
        _query: &str,
        _documents: &[&str],
        _request_id: &str,
    ) -> Result<Vec<f32>, DaemonError> {
        self.raw_calls.fetch_add(1, Ordering::Relaxed);
        Ok(self.scores.clone())
    }
}

/// Daemon fixture that signs exact rerank envelopes, with optional
/// post-signing score tampering to exercise client-side rejection.
struct AttestedRerankDaemon {
    connection: DaemonConnectionIdentityV1,
    scores: Vec<f32>,
    tamper_scores: bool,
    raw_calls: AtomicUsize,
}

impl DaemonClient for AttestedRerankDaemon {
    fn id(&self) -> &str {
        "attested-rerank-daemon"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn rerank_attested(
        &self,
        _query: &str,
        _documents: &[&str],
        challenge: &DaemonChallengeV1,
    ) -> Result<AttestedDaemonEmbeddingResponseV1, DaemonError> {
        let mut response = AttestedDaemonEmbeddingResponseV1::signed(
            challenge.clone(),
            self.connection.clone(),
            vec![self.scores.clone()],
            TEST_KEY,
        )?;
        if self.tamper_scores {
            response.vectors[0][0] += 1.0;
        }
        Ok(response)
    }

    fn embed(&self, _text: &str, _request_id: &str) -> Result<Vec<f32>, DaemonError> {
        Err(DaemonError::Unavailable(
            "attested rerank fixture has no embedder".to_string(),
        ))
    }

    fn embed_batch(
        &self,
        _texts: &[&str],
        _request_id: &str,
    ) -> Result<Vec<Vec<f32>>, DaemonError> {
        Err(DaemonError::Unavailable(
            "attested rerank fixture has no embedder".to_string(),
        ))
    }

    fn rerank(
        &self,
        _query: &str,
        _documents: &[&str],
        _request_id: &str,
    ) -> Result<Vec<f32>, DaemonError> {
        self.raw_calls.fetch_add(1, Ordering::Relaxed);
        Err(DaemonError::Unavailable(
            "verified path must never touch the raw primitive".to_string(),
        ))
    }
}

#[derive(Clone, Copy)]
enum FailureMode {
    Timeout,
    Overloaded { retry_after: Duration },
    Failed,
    InvalidInput,
}

impl FailureMode {
    fn error(&self) -> DaemonError {
        match self {
            Self::Timeout => DaemonError::Timeout("daemon timeout".to_string()),
            Self::Overloaded { retry_after } => DaemonError::Overloaded {
                retry_after: Some(*retry_after),
                message: "queue full".to_string(),
            },
            Self::Failed => DaemonError::Failed("daemon failed".to_string()),
            Self::InvalidInput => DaemonError::InvalidInput("invalid input".to_string()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResponseMutation {
    None,
    MissingAttestation,
    WrongKey,
    PayloadTamper,
    ResponseReorder,
    SpaceDrift,
    EndpointDrift,
    GenerationDrift,
    KeyIdDrift,
    UnknownSchema,
    ReplayNonce,
    Cancelled,
}

const TEST_KEY: &[u8] = b"0123456789abcdef0123456789abcdef";
const WRONG_KEY: &[u8] = b"abcdef0123456789abcdef0123456789";

struct FixtureDaemon {
    attested_calls: AtomicUsize,
    raw_calls: AtomicUsize,
    control_calls: AtomicUsize,
    fail_first: usize,
    mode: FailureMode,
    available: AtomicBool,
    embed_value: f32,
    connection: DaemonConnectionIdentityV1,
    key: Vec<u8>,
    mutation: Mutex<ResponseMutation>,
    challenges: Mutex<Vec<DaemonChallengeV1>>,
}

impl FixtureDaemon {
    fn new(
        fail_first: usize,
        mode: FailureMode,
        available: bool,
        embed_value: f32,
        connection: DaemonConnectionIdentityV1,
    ) -> Self {
        Self {
            attested_calls: AtomicUsize::new(0),
            raw_calls: AtomicUsize::new(0),
            control_calls: AtomicUsize::new(0),
            fail_first,
            mode,
            available: AtomicBool::new(available),
            embed_value,
            connection,
            key: TEST_KEY.to_vec(),
            mutation: Mutex::new(ResponseMutation::None),
            challenges: Mutex::new(Vec::new()),
        }
    }

    fn exact(connection: DaemonConnectionIdentityV1, embed_value: f32) -> Self {
        Self::new(0, FailureMode::Failed, true, embed_value, connection)
    }

    fn set_mutation(&self, mutation: ResponseMutation) {
        *self
            .mutation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = mutation;
    }

    fn set_available(&self, available: bool) {
        self.available.store(available, Ordering::Relaxed);
    }

    fn record_challenge(&self, challenge: &DaemonChallengeV1) {
        self.challenges
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(challenge.clone());
    }

    fn signed_response(
        &self,
        challenge: &DaemonChallengeV1,
        vectors: Vec<Vec<f32>>,
    ) -> Result<AttestedDaemonEmbeddingResponseV1, DaemonError> {
        self.record_challenge(challenge);
        let mutation = *self
            .mutation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match mutation {
            ResponseMutation::MissingAttestation => {
                return Err(DaemonError::UnverifiableRemoteSpace);
            }
            ResponseMutation::Cancelled => return Err(DaemonError::Cancelled),
            _ => {}
        }
        let signing_key = if mutation == ResponseMutation::WrongKey {
            WRONG_KEY
        } else {
            &self.key
        };
        let mut response = AttestedDaemonEmbeddingResponseV1::signed(
            challenge.clone(),
            self.connection.clone(),
            vectors,
            signing_key,
        )?;
        match mutation {
            ResponseMutation::PayloadTamper => response.vectors[0][0] += 1.0,
            ResponseMutation::ResponseReorder => response.vectors.swap(0, 1),
            ResponseMutation::SpaceDrift => {
                let mut drifted = self.connection.embedding_identity.clone();
                drifted.space.tokenizer_fingerprint = "ab".repeat(32);
                drifted.producer.space_fingerprint = drifted.space.fingerprint();
                response.attestation.connection.embedding_identity = drifted;
            }
            ResponseMutation::EndpointDrift => {
                response.attestation.connection.endpoint_fingerprint =
                    frankensearch_core::daemon_endpoint_fingerprint("unix:/run/other-daemon.sock");
            }
            ResponseMutation::GenerationDrift => {
                response.attestation.connection.generation += 1;
            }
            ResponseMutation::KeyIdDrift => {
                "rotated-key".clone_into(&mut response.attestation.connection.key_id);
            }
            ResponseMutation::UnknownSchema => {
                response.attestation.schema_version = u16::MAX;
            }
            ResponseMutation::ReplayNonce => {
                response.attestation.challenge.request_nonce = "11".repeat(32);
            }
            ResponseMutation::None
            | ResponseMutation::MissingAttestation
            | ResponseMutation::WrongKey
            | ResponseMutation::Cancelled => {}
        }
        Ok(response)
    }

    fn control_attestation(
        &self,
        challenge: &DaemonChallengeV1,
    ) -> Result<DaemonEmbeddingAttestationV1, DaemonError> {
        self.control_calls.fetch_add(1, Ordering::Relaxed);
        self.signed_response(challenge, Vec::new())
            .map(|response| response.attestation)
    }

    fn maybe_fail_attested(&self) -> Result<(), DaemonError> {
        let call = self.attested_calls.fetch_add(1, Ordering::Relaxed);
        if call < self.fail_first {
            Err(self.mode.error())
        } else {
            Ok(())
        }
    }
}

impl DaemonClient for FixtureDaemon {
    fn id(&self) -> &str {
        "fixture-daemon"
    }

    fn is_available(&self) -> bool {
        self.available.load(Ordering::Relaxed)
    }

    fn handshake_attested(
        &self,
        challenge: &DaemonChallengeV1,
    ) -> Result<DaemonEmbeddingAttestationV1, DaemonError> {
        self.control_attestation(challenge)
    }

    fn health_attested(
        &self,
        challenge: &DaemonChallengeV1,
    ) -> Result<DaemonEmbeddingAttestationV1, DaemonError> {
        self.control_attestation(challenge)
    }

    fn embed_attested(
        &self,
        _text: &str,
        challenge: &DaemonChallengeV1,
    ) -> Result<AttestedDaemonEmbeddingResponseV1, DaemonError> {
        if let Err(error) = self.maybe_fail_attested() {
            self.record_challenge(challenge);
            return Err(error);
        }
        self.signed_response(challenge, vec![vec![self.embed_value; 4]])
    }

    fn embed_batch_attested(
        &self,
        texts: &[&str],
        challenge: &DaemonChallengeV1,
    ) -> Result<AttestedDaemonEmbeddingResponseV1, DaemonError> {
        if let Err(error) = self.maybe_fail_attested() {
            self.record_challenge(challenge);
            return Err(error);
        }
        let vectors = (0..texts.len())
            .map(|index| vec![self.embed_value + index as f32; 4])
            .collect();
        self.signed_response(challenge, vectors)
    }

    fn embed(&self, _text: &str, _request_id: &str) -> Result<Vec<f32>, DaemonError> {
        let call = self.raw_calls.fetch_add(1, Ordering::Relaxed);
        if call < self.fail_first {
            Err(self.mode.error())
        } else {
            Ok(vec![self.embed_value; 4])
        }
    }

    fn embed_batch(&self, texts: &[&str], _request_id: &str) -> Result<Vec<Vec<f32>>, DaemonError> {
        let call = self.raw_calls.fetch_add(1, Ordering::Relaxed);
        if call < self.fail_first {
            Err(self.mode.error())
        } else {
            Ok(vec![vec![self.embed_value; 4]; texts.len()])
        }
    }

    fn rerank(
        &self,
        _query: &str,
        documents: &[&str],
        _request_id: &str,
    ) -> Result<Vec<f32>, DaemonError> {
        let call = self.raw_calls.fetch_add(1, Ordering::Relaxed);
        if call < self.fail_first {
            Err(self.mode.error())
        } else {
            Ok((0..documents.len())
                .map(|idx| (documents.len() - idx) as f32)
                .collect())
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct WireRequest {
    challenge: DaemonChallengeV1,
    inputs: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
enum WireResponse {
    Control(DaemonEmbeddingAttestationV1),
    Embedding(AttestedDaemonEmbeddingResponseV1),
}

struct TcpDaemonClient {
    address: SocketAddr,
}

impl TcpDaemonClient {
    fn round_trip(
        &self,
        challenge: &DaemonChallengeV1,
        inputs: &[&str],
    ) -> Result<WireResponse, DaemonError> {
        let mut stream = TcpStream::connect(self.address)
            .map_err(|_| DaemonError::Failed("tcp connection failed".to_owned()))?;
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .map_err(|_| DaemonError::Failed("tcp timeout setup failed".to_owned()))?;
        let request = WireRequest {
            challenge: challenge.clone(),
            inputs: inputs.iter().map(|input| (*input).to_owned()).collect(),
        };
        let payload = serde_json::to_vec(&request)
            .map_err(|_| DaemonError::Failed("wire request encoding failed".to_owned()))?;
        stream
            .write_all(&payload)
            .map_err(|_| DaemonError::Failed("wire request write failed".to_owned()))?;
        stream
            .shutdown(Shutdown::Write)
            .map_err(|_| DaemonError::Failed("wire request shutdown failed".to_owned()))?;
        let mut response = Vec::new();
        stream
            .read_to_end(&mut response)
            .map_err(|_| DaemonError::Failed("wire response read failed".to_owned()))?;
        serde_json::from_slice(&response)
            .map_err(|_| DaemonError::Failed("wire response decoding failed".to_owned()))
    }
}

impl DaemonClient for TcpDaemonClient {
    fn id(&self) -> &str {
        "tcp-test-daemon"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn handshake_attested(
        &self,
        challenge: &DaemonChallengeV1,
    ) -> Result<DaemonEmbeddingAttestationV1, DaemonError> {
        match self.round_trip(challenge, &[])? {
            WireResponse::Control(attestation) => Ok(attestation),
            WireResponse::Embedding(_) => Err(DaemonError::UnverifiableRemoteSpace),
        }
    }

    fn health_attested(
        &self,
        challenge: &DaemonChallengeV1,
    ) -> Result<DaemonEmbeddingAttestationV1, DaemonError> {
        match self.round_trip(challenge, &[])? {
            WireResponse::Control(attestation) => Ok(attestation),
            WireResponse::Embedding(_) => Err(DaemonError::UnverifiableRemoteSpace),
        }
    }

    fn embed_attested(
        &self,
        text: &str,
        challenge: &DaemonChallengeV1,
    ) -> Result<AttestedDaemonEmbeddingResponseV1, DaemonError> {
        match self.round_trip(challenge, &[text])? {
            WireResponse::Embedding(response) => Ok(response),
            WireResponse::Control(_) => Err(DaemonError::UnverifiableRemoteSpace),
        }
    }

    fn embed_batch_attested(
        &self,
        texts: &[&str],
        challenge: &DaemonChallengeV1,
    ) -> Result<AttestedDaemonEmbeddingResponseV1, DaemonError> {
        match self.round_trip(challenge, texts)? {
            WireResponse::Embedding(response) => Ok(response),
            WireResponse::Control(_) => Err(DaemonError::UnverifiableRemoteSpace),
        }
    }

    fn embed(&self, _text: &str, _request_id: &str) -> Result<Vec<f32>, DaemonError> {
        Err(DaemonError::UnverifiableRemoteSpace)
    }

    fn embed_batch(
        &self,
        _texts: &[&str],
        _request_id: &str,
    ) -> Result<Vec<Vec<f32>>, DaemonError> {
        Err(DaemonError::UnverifiableRemoteSpace)
    }

    fn rerank(
        &self,
        _query: &str,
        _documents: &[&str],
        _request_id: &str,
    ) -> Result<Vec<f32>, DaemonError> {
        Err(DaemonError::Unavailable(
            "test gateway has no reranker".to_owned(),
        ))
    }
}

fn spawn_authenticated_tcp_gateway(
    connection: DaemonConnectionIdentityV1,
) -> (SocketAddr, thread::JoinHandle<()>) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        for _ in 0..3 {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut payload = Vec::new();
            stream.read_to_end(&mut payload).unwrap();
            let request: WireRequest = serde_json::from_slice(&payload).unwrap();
            let input_refs = request
                .inputs
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            let expected = DaemonChallengeV1::for_inputs(
                request.challenge.request_nonce.clone(),
                request.challenge.operation,
                &input_refs,
                &connection,
            )
            .unwrap();
            assert_eq!(request.challenge, expected);
            let response = match request.challenge.operation {
                DaemonOperationV1::Handshake | DaemonOperationV1::Health => {
                    let signed = AttestedDaemonEmbeddingResponseV1::signed(
                        request.challenge,
                        connection.clone(),
                        Vec::new(),
                        TEST_KEY,
                    )
                    .unwrap();
                    WireResponse::Control(signed.attestation)
                }
                DaemonOperationV1::Embed => {
                    let signed = AttestedDaemonEmbeddingResponseV1::signed(
                        request.challenge,
                        connection.clone(),
                        vec![vec![4.0, 3.0, 2.0, 1.0]],
                        TEST_KEY,
                    )
                    .unwrap();
                    WireResponse::Embedding(signed)
                }
                DaemonOperationV1::EmbedBatch => {
                    let vectors = (0..input_refs.len())
                        .map(|index| vec![index as f32 + 1.0; 4])
                        .collect();
                    let signed = AttestedDaemonEmbeddingResponseV1::signed(
                        request.challenge,
                        connection.clone(),
                        vectors,
                        TEST_KEY,
                    )
                    .unwrap();
                    WireResponse::Embedding(signed)
                }
                DaemonOperationV1::Rerank => {
                    panic!("tcp gateway fixture serves no rerank operation")
                }
            };
            stream
                .write_all(&serde_json::to_vec(&response).unwrap())
                .unwrap();
        }
    });
    (address, handle)
}

fn test_connection(model_id: &str) -> DaemonConnectionIdentityV1 {
    DaemonConnectionIdentityV1 {
        schema_version: frankensearch_core::DAEMON_CONNECTION_IDENTITY_SCHEMA_V1,
        endpoint_fingerprint: frankensearch_core::daemon_endpoint_fingerprint(
            "unix:/run/frankensearch.sock",
        ),
        executable_fingerprint: frankensearch_core::daemon_executable_fingerprint(
            b"fixture-daemon-v1",
        ),
        protocol_revision: "frankensearch-daemon-v1".to_owned(),
        key_id: "fixture-key-v1".to_owned(),
        generation: 9,
        embedding_identity: EmbeddingIdentityBundleV1::explicit_test_model(model_id, 4),
        model_category: ModelCategory::HashEmbedder,
    }
}

fn verifier() -> PinnedDaemonVerifierV1 {
    PinnedDaemonVerifierV1::new("fixture-key-v1", TEST_KEY.to_vec()).unwrap()
}

fn fallback_embedder(value: f32, identity: EmbeddingIdentityBundleV1) -> Arc<dyn SyncEmbed> {
    Arc::new(ConstEmbedder {
        id: "fixture-space",
        model_name: "fixture-space",
        dim: 4,
        value,
        semantic: false,
        category: ModelCategory::HashEmbedder,
        identity,
    })
}

#[test]
fn caller_supplied_epoch_without_producer_proof_fails_closed() {
    let connection = test_connection("fixture-space");
    let daemon = Arc::new(FixtureDaemon::exact(connection.clone(), 2.0));
    let fallback = fallback_embedder(1.0, connection.embedding_identity);
    assert!(matches!(
        DaemonFallbackEmbedder::new(daemon, fallback, DaemonRetryConfig::default()),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));
}

#[test]
fn verified_daemon_only_accepts_exact_authenticated_single_and_batch() {
    let connection = test_connection("fixture-space");
    let daemon = Arc::new(FixtureDaemon::exact(connection.clone(), 2.0));
    let embedder = DaemonFallbackEmbedder::with_verified_defaults(
        daemon.clone(),
        connection.clone(),
        verifier(),
    )
    .unwrap();
    assert_eq!(embedder.trust_level(), DaemonTrustLevelV1::VerifiedRemote);
    assert_eq!(
        embedder.identity().unwrap().fingerprint(),
        connection.embedding_identity.fingerprint()
    );
    assert_eq!(embedder.embed_sync("hello").unwrap(), vec![2.0; 4]);
    assert_eq!(
        embedder.embed_batch_sync(&["first", "second"]).unwrap(),
        vec![vec![2.0; 4], vec![3.0; 4]]
    );
    assert_eq!(
        embedder.embed_batch_sync(&[]).unwrap(),
        Vec::<Vec<f32>>::new()
    );
    assert_eq!(daemon.control_calls.load(Ordering::Relaxed), 2);
    assert_eq!(daemon.attested_calls.load(Ordering::Relaxed), 2);

    let challenges = daemon
        .challenges
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let nonce_count = challenges
        .iter()
        .map(|challenge| challenge.request_nonce.as_str())
        .collect::<HashSet<_>>()
        .len();
    assert_eq!(nonce_count, challenges.len());
}

#[test]
fn daemon_only_readiness_requires_fresh_authenticated_control_proof() {
    let connection = test_connection("fixture-space");
    let daemon = Arc::new(FixtureDaemon::exact(connection.clone(), 2.0));
    let embedder =
        DaemonFallbackEmbedder::with_verified_defaults(daemon.clone(), connection, verifier())
            .unwrap();

    assert!(embedder.is_ready());
    daemon.set_mutation(ResponseMutation::GenerationDrift);
    assert!(!embedder.is_ready());
}

#[test]
fn authenticated_tcp_gateway_e2e_admits_exact_vectors_without_a_mock_transport() {
    let connection = test_connection("fixture-space");
    let (address, gateway) = spawn_authenticated_tcp_gateway(connection.clone());
    let daemon: Arc<dyn DaemonClient> = Arc::new(TcpDaemonClient { address });
    let embedder =
        DaemonFallbackEmbedder::with_verified_defaults(daemon, connection, verifier()).unwrap();
    assert_eq!(
        embedder.embed_sync("query over a real TCP socket").unwrap(),
        vec![4.0, 3.0, 2.0, 1.0]
    );
    gateway.join().unwrap();
}

#[test]
fn verified_constructor_rejects_missing_attestation_wrong_key_and_key_id() {
    let connection = test_connection("fixture-space");
    for mutation in [
        ResponseMutation::MissingAttestation,
        ResponseMutation::WrongKey,
    ] {
        let daemon = Arc::new(FixtureDaemon::exact(connection.clone(), 2.0));
        daemon.set_mutation(mutation);
        assert!(matches!(
            DaemonFallbackEmbedder::with_verified_defaults(daemon, connection.clone(), verifier()),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
    }

    let daemon = Arc::new(FixtureDaemon::exact(connection.clone(), 2.0));
    let wrong_key_id = PinnedDaemonVerifierV1::new("other-key", TEST_KEY.to_vec()).unwrap();
    assert!(matches!(
        DaemonFallbackEmbedder::with_verified_defaults(daemon, connection, wrong_key_id),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));
}

#[test]
fn every_response_drift_and_tamper_class_fails_typed() {
    let connection = test_connection("fixture-space");
    let daemon = Arc::new(FixtureDaemon::exact(connection.clone(), 2.0));
    let embedder =
        DaemonFallbackEmbedder::with_verified_defaults(daemon.clone(), connection, verifier())
            .unwrap();
    for mutation in [
        ResponseMutation::MissingAttestation,
        ResponseMutation::WrongKey,
        ResponseMutation::PayloadTamper,
        ResponseMutation::SpaceDrift,
        ResponseMutation::EndpointDrift,
        ResponseMutation::GenerationDrift,
        ResponseMutation::KeyIdDrift,
        ResponseMutation::UnknownSchema,
        ResponseMutation::ReplayNonce,
    ] {
        daemon.set_mutation(mutation);
        assert!(
            matches!(
                embedder.embed_sync("secret query"),
                Err(SearchError::UnverifiableRemoteSpace { .. })
            ),
            "mutation {mutation:?} must fail closed"
        );
    }
}

#[test]
fn batch_order_and_shape_are_authenticated() {
    let connection = test_connection("fixture-space");
    let daemon = Arc::new(FixtureDaemon::exact(connection.clone(), 2.0));
    let embedder =
        DaemonFallbackEmbedder::with_verified_defaults(daemon.clone(), connection, verifier())
            .unwrap();
    daemon.set_mutation(ResponseMutation::ResponseReorder);
    assert!(matches!(
        embedder.embed_batch_sync(&["first", "second"]),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));
}

#[test]
fn reconnect_reauthenticates_and_every_attempt_uses_a_fresh_nonce() {
    let connection = test_connection("fixture-space");
    let daemon = Arc::new(FixtureDaemon::new(
        1,
        FailureMode::Failed,
        true,
        2.0,
        connection.clone(),
    ));
    let config = DaemonRetryConfig {
        max_attempts: 2,
        base_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(5),
        jitter_pct: 0.0,
    };
    let embedder =
        DaemonFallbackEmbedder::new_verified(daemon.clone(), None, config, connection, verifier())
            .unwrap();

    let result = embedder.embed_sync("hello").unwrap();
    assert_eq!(result, vec![2.0; 4]);
    assert_eq!(daemon.attested_calls.load(Ordering::Relaxed), 2);
    assert_eq!(daemon.control_calls.load(Ordering::Relaxed), 4);
    let challenges = daemon
        .challenges
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let nonces = challenges
        .iter()
        .map(|challenge| challenge.request_nonce.as_str())
        .collect::<HashSet<_>>();
    assert_eq!(nonces.len(), challenges.len());
}

#[test]
fn attestation_failure_and_cancellation_never_use_local_fallback() {
    let connection = test_connection("fixture-space");
    let daemon = Arc::new(FixtureDaemon::exact(connection.clone(), 2.0));
    let fallback = fallback_embedder(1.0, connection.embedding_identity.clone());
    let embedder = DaemonFallbackEmbedder::new_verified(
        daemon.clone(),
        Some(fallback),
        DaemonRetryConfig::default(),
        connection,
        verifier(),
    )
    .unwrap();

    daemon.set_mutation(ResponseMutation::PayloadTamper);
    assert!(matches!(
        embedder.embed_sync("hello"),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));
    daemon.set_mutation(ResponseMutation::Cancelled);
    assert!(matches!(
        embedder.embed_sync("hello"),
        Err(SearchError::Cancelled { .. })
    ));
}

#[test]
fn transport_failure_uses_only_an_exact_identity_local_fallback() {
    let connection = test_connection("fixture-space");
    let daemon = Arc::new(FixtureDaemon::exact(connection.clone(), 2.0));
    let fallback = fallback_embedder(1.0, connection.embedding_identity.clone());
    let embedder = DaemonFallbackEmbedder::new_verified(
        daemon.clone(),
        Some(fallback),
        DaemonRetryConfig::default(),
        connection.clone(),
        verifier(),
    )
    .unwrap();
    daemon.set_available(false);
    assert_eq!(embedder.embed_sync("hello").unwrap(), vec![1.0; 4]);

    let daemon = Arc::new(FixtureDaemon::exact(connection.clone(), 2.0));
    let drifted_fallback = fallback_embedder(
        1.0,
        EmbeddingIdentityBundleV1::explicit_test_model("different-space", 4),
    );
    assert!(matches!(
        DaemonFallbackEmbedder::new_verified(
            daemon,
            Some(drifted_fallback),
            DaemonRetryConfig::default(),
            connection,
            verifier()
        ),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));
}

#[test]
fn invalid_input_does_not_retry_and_uses_exact_local_fallback() {
    let connection = test_connection("fixture-space");
    let daemon = Arc::new(FixtureDaemon::new(
        10,
        FailureMode::InvalidInput,
        true,
        2.0,
        connection.clone(),
    ));
    let fallback = fallback_embedder(1.0, connection.embedding_identity.clone());
    let config = DaemonRetryConfig {
        max_attempts: 3,
        ..DaemonRetryConfig::default()
    };
    let embedder = DaemonFallbackEmbedder::new_verified(
        daemon.clone(),
        Some(fallback),
        config,
        connection,
        verifier(),
    )
    .unwrap();

    let result = embedder.embed_sync("hello").unwrap();
    assert_eq!(result, vec![1.0; 4]);
    assert_eq!(daemon.attested_calls.load(Ordering::Relaxed), 1);
}

#[test]
fn assumed_daemon_is_explicit_transient_and_debug_redacts_vectors() {
    let connection = test_connection("fixture-space");
    let daemon = Arc::new(FixtureDaemon::exact(connection, 7.0));
    let assumed = AssumedDaemonClient::new(daemon);
    let batch = assumed.embed_batch_transient(&["first", "second"]).unwrap();
    assert_eq!(batch.trust_level(), DaemonTrustLevelV1::AssumedRemote);
    assert_eq!(batch.vectors(), &[vec![7.0; 4], vec![7.0; 4]]);
    let rendered = format!("{batch:?} {assumed:?}");
    assert!(rendered.contains("AssumedRemote"));
    assert!(rendered.contains("<redacted>"));
    assert!(!rendered.contains("[7.0, 7.0, 7.0, 7.0]"));
    assert!(!rendered.contains("fixture-daemon"));
}

#[test]
fn verifier_debug_redacts_secret_material() {
    let verifier = verifier();
    let rendered = format!("{verifier:?}");
    assert!(rendered.contains("<redacted>"));
    assert!(!rendered.contains("0123456789abcdef"));
}

#[test]
fn reranker_falls_back_when_daemon_fails() {
    let daemon = Arc::new(FixtureDaemon::new(
        10,
        FailureMode::Timeout,
        true,
        2.0,
        test_connection("fixture-space"),
    ));
    let fallback: Arc<dyn SyncRerank> = Arc::new(ConstReranker {
        id: "fallback-reranker",
    });
    let reranker = DaemonFallbackReranker::new(
        daemon.clone(),
        Some(fallback.clone()),
        DaemonRetryConfig {
            max_attempts: 1,
            ..DaemonRetryConfig::default()
        },
    );

    let docs = vec![
        RerankDocument {
            doc_id: "a".to_string(),
            text: "doc a".to_string(),
        },
        RerankDocument {
            doc_id: "b".to_string(),
            text: "doc b".to_string(),
        },
    ];
    let result = reranker.rerank_sync("query", &docs).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].doc_id, "a");
    assert_eq!(result[0].score, 10.0);
    assert_eq!(daemon.raw_calls.load(Ordering::Relaxed), 1);
}

fn rerank_docs() -> Vec<RerankDocument> {
    vec![
        RerankDocument {
            doc_id: "a".to_string(),
            text: "doc a".to_string(),
        },
        RerankDocument {
            doc_id: "b".to_string(),
            text: "doc b".to_string(),
        },
    ]
}

fn shaped_reranker(
    scores: Vec<f32>,
    fallback: Option<Arc<dyn SyncRerank>>,
) -> (Arc<ShapedRerankDaemon>, DaemonFallbackReranker) {
    let daemon = Arc::new(ShapedRerankDaemon::new(scores));
    let reranker = DaemonFallbackReranker::new(
        daemon.clone(),
        fallback,
        DaemonRetryConfig {
            max_attempts: 1,
            ..DaemonRetryConfig::default()
        },
    );
    (daemon, reranker)
}

#[test]
fn reranker_uses_daemon_scores_when_shape_is_valid() {
    let fallback: Arc<dyn SyncRerank> = Arc::new(ConstReranker {
        id: "fallback-reranker",
    });
    let (daemon, reranker) = shaped_reranker(vec![0.25, 0.75], Some(fallback));
    let result = reranker.rerank_sync("query", &rerank_docs()).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].score, 0.25);
    assert_eq!(result[1].score, 0.75);
    assert_eq!(daemon.raw_calls.load(Ordering::Relaxed), 1);
}

#[test]
fn reranker_falls_back_when_daemon_returns_score_prefix() {
    let fallback: Arc<dyn SyncRerank> = Arc::new(ConstReranker {
        id: "fallback-reranker",
    });
    let (daemon, reranker) = shaped_reranker(vec![0.9], Some(fallback));
    let result = reranker.rerank_sync("query", &rerank_docs()).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(
        result[0].score, 10.0,
        "fallback scores expected, not the daemon prefix"
    );
    assert_eq!(
        result[1].score, 9.0,
        "fallback scores expected, not a silent 0.0 pad"
    );
    assert_eq!(daemon.raw_calls.load(Ordering::Relaxed), 1);
}

#[test]
fn reranker_falls_back_when_daemon_returns_extra_scores() {
    let fallback: Arc<dyn SyncRerank> = Arc::new(ConstReranker {
        id: "fallback-reranker",
    });
    let (_daemon, reranker) = shaped_reranker(vec![0.1, 0.2, 0.3], Some(fallback));
    let result = reranker.rerank_sync("query", &rerank_docs()).unwrap();
    assert_eq!(result[0].score, 10.0);
    assert_eq!(result[1].score, 9.0);
}

#[test]
fn reranker_falls_back_when_daemon_returns_non_finite_score() {
    let fallback: Arc<dyn SyncRerank> = Arc::new(ConstReranker {
        id: "fallback-reranker",
    });
    let (_daemon, reranker) = shaped_reranker(vec![0.5, f32::NAN], Some(fallback));
    let result = reranker.rerank_sync("query", &rerank_docs()).unwrap();
    assert_eq!(result[0].score, 10.0);
    assert_eq!(result[1].score, 9.0);
}

#[test]
fn reranker_surfaces_shape_violation_when_no_fallback_exists() {
    let (_daemon, reranker) = shaped_reranker(vec![0.9], None);
    let result = reranker.rerank_sync("query", &rerank_docs());
    assert!(
        result.is_err(),
        "a short daemon response must never zero-pad into a successful rerank"
    );
}

fn verified_reranker(
    daemon: Arc<dyn DaemonClient>,
    connection: DaemonConnectionIdentityV1,
    fallback: Option<Arc<dyn SyncRerank>>,
) -> DaemonFallbackReranker {
    DaemonFallbackReranker::new_verified(
        daemon,
        fallback,
        DaemonRetryConfig {
            max_attempts: 1,
            ..DaemonRetryConfig::default()
        },
        connection,
        verifier(),
    )
    .unwrap()
}

#[test]
fn verified_reranker_accepts_exact_signed_scores() {
    let connection = test_connection("fixture-space");
    let daemon = Arc::new(AttestedRerankDaemon {
        connection: connection.clone(),
        scores: vec![0.25, 0.75],
        tamper_scores: false,
        raw_calls: AtomicUsize::new(0),
    });
    let fallback: Arc<dyn SyncRerank> = Arc::new(ConstReranker {
        id: "fallback-reranker",
    });
    let reranker = verified_reranker(daemon.clone(), connection, Some(fallback));
    assert_eq!(reranker.trust_level(), DaemonTrustLevelV1::VerifiedRemote);
    let result = reranker.rerank_sync("query", &rerank_docs()).unwrap();
    assert_eq!(result[0].score, 0.25);
    assert_eq!(result[1].score, 0.75);
    assert_eq!(
        daemon.raw_calls.load(Ordering::Relaxed),
        0,
        "verified path must never touch the raw primitive"
    );
}

#[test]
fn verified_reranker_rejects_tampered_scores_without_fallback() {
    let connection = test_connection("fixture-space");
    let daemon = Arc::new(AttestedRerankDaemon {
        connection: connection.clone(),
        scores: vec![0.25, 0.75],
        tamper_scores: true,
        raw_calls: AtomicUsize::new(0),
    });
    let fallback: Arc<dyn SyncRerank> = Arc::new(ConstReranker {
        id: "fallback-reranker",
    });
    let reranker = verified_reranker(daemon, connection, Some(fallback));
    let result = reranker.rerank_sync("query", &rerank_docs());
    assert!(
        matches!(result, Err(SearchError::UnverifiableRemoteSpace { .. })),
        "tampered scores must hard-error, not fall back: {result:?}"
    );
}

#[test]
fn raw_only_client_is_ineligible_for_verified_rerank() {
    let connection = test_connection("fixture-space");
    let daemon = Arc::new(ShapedRerankDaemon::new(vec![0.9, 0.1]));
    let fallback: Arc<dyn SyncRerank> = Arc::new(ConstReranker {
        id: "fallback-reranker",
    });
    let reranker = verified_reranker(daemon.clone(), connection, Some(fallback));
    let result = reranker.rerank_sync("query", &rerank_docs());
    assert!(
        matches!(result, Err(SearchError::UnverifiableRemoteSpace { .. })),
        "the fail-closed rerank_attested default must reject raw-only clients: {result:?}"
    );
    assert_eq!(
        daemon.raw_calls.load(Ordering::Relaxed),
        0,
        "verified path must never touch the raw primitive"
    );
}

#[test]
fn unverified_reranker_reports_assumed_trust() {
    let (_daemon, reranker) = shaped_reranker(vec![0.5, 0.5], None);
    assert_eq!(reranker.trust_level(), DaemonTrustLevelV1::AssumedRemote);
}

#[test]
fn overloaded_sets_backoff_and_skips_immediate_retry() {
    let connection = test_connection("fixture-space");
    let daemon = Arc::new(FixtureDaemon::new(
        1,
        FailureMode::Overloaded {
            retry_after: Duration::from_millis(25),
        },
        true,
        2.0,
        connection.clone(),
    ));
    let fallback = fallback_embedder(1.0, connection.embedding_identity.clone());
    let config = DaemonRetryConfig {
        max_attempts: 1,
        base_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(50),
        jitter_pct: 0.0,
    };
    let embedder = DaemonFallbackEmbedder::new_verified(
        daemon.clone(),
        Some(fallback),
        config,
        connection,
        verifier(),
    )
    .unwrap();

    let _ = embedder.embed_sync("first").unwrap();
    let calls_after_first = daemon.attested_calls.load(Ordering::Relaxed);
    let _ = embedder.embed_sync("second").unwrap();
    let calls_after_second = daemon.attested_calls.load(Ordering::Relaxed);

    assert_eq!(calls_after_first, calls_after_second);
}
