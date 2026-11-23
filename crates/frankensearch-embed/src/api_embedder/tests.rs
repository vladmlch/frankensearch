use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::Mutex;
use std::sync::atomic::AtomicU8;
use std::thread;
use std::time::Duration;

use asupersync::http::h1::ClientError;
use asupersync::test_utils::run_test_with_cx;
use serde::Deserialize;

use frankensearch_core::generation::{
    EmbeddingArtifactIdentityV1, EmbeddingSpaceKindV1, QuantizationFormat,
};

use crate::api_provider::{OpenAiProvider, remote_ordered_request_sha256};

use super::transport::{
    API_OUTPUT_NORMALIZATION_V1, API_STORAGE_ENDIANNESS_V1, API_STORAGE_FORMAT_V1, RateLimiter,
    api_client_error_to_search, unverifiable_remote_space,
};
use super::*;

const TEST_ATTESTATION_KEY: &[u8; 32] = b"0123456789abcdef0123456789abcdef";
const TEST_KEY_ID: &str = "gateway-key-2026-07";
const TEST_GENERATION: u64 = 41;

/// The PUBLIC entry points must observe cancel themselves.
///
/// `embed_checkpoint` is also called inside `request_batch`, so asserting
/// only that some `Cancelled` came back would still pass if the `api.embed`
/// and `api.embed_batch` call sites were deleted. Pinning the phase is what
/// proves the refusal happened at the public boundary, before any HTTP work
/// is attempted: this embedder points at the real provider endpoint, so a
/// request that was actually started would surface as a transport failure
/// rather than as `Cancelled`.
#[test]
fn public_embed_entrypoints_observe_cancel_before_any_request() {
    let embedder = verified_test_embedder(4);
    run_test_with_cx(|cx| async move {
        cx.cancel_fast(asupersync::CancelKind::User);
        match embedder.embed(&cx, "alpha").await.unwrap_err() {
            SearchError::Cancelled { phase, .. } => assert_eq!(phase, "api.embed"),
            other => panic!("expected Cancelled at api.embed, got {other:?}"),
        }
        match embedder.embed_batch(&cx, &["alpha"]).await.unwrap_err() {
            SearchError::Cancelled { phase, .. } => {
                assert_eq!(phase, "api.embed_batch");
            }
            other => panic!("expected Cancelled at api.embed_batch, got {other:?}"),
        }
    });
}

#[test]
fn embed_checkpoint_observes_cancel_before_http() {
    run_test_with_cx(|cx| async move {
        cx.cancel_fast(asupersync::CancelKind::User);
        let err = super::embed_checkpoint(&cx, "api.embed").unwrap_err();
        match err {
            SearchError::Cancelled { phase, .. } => {
                assert_eq!(phase, "api.embed");
            }
            other => panic!("expected Cancelled, got {other:?}"),
        }
    });
}

#[derive(Debug, Clone)]
struct TestGatewayProvider {
    endpoint: String,
    dimension: usize,
}

#[derive(Debug)]
struct MidstreamDriftProvider {
    drift_mode: Arc<AtomicU8>,
}

impl ApiProvider for MidstreamDriftProvider {
    fn provider_name(&self) -> &'static str {
        "openai"
    }

    fn api_model_id(&self) -> &'static str {
        if self.drift_mode.load(Ordering::Relaxed) == 1 {
            "same-dimension-wrong-model"
        } else {
            "text-embedding-3-small"
        }
    }

    fn embedder_id(&self) -> &'static str {
        "midstream-drift-test"
    }

    fn identity_backend(&self) -> &'static str {
        "remote-api-openai"
    }

    fn identity_protocol_revision(&self) -> &'static str {
        "openai-embeddings-json-v1"
    }

    fn dimension(&self) -> usize {
        4
    }

    fn max_batch_size(&self) -> usize {
        16
    }

    fn supports_mrl(&self) -> bool {
        false
    }

    fn endpoint_url(&self) -> &'static str {
        if self.drift_mode.load(Ordering::Relaxed) == 2 {
            "http://127.0.0.1:2/changed"
        } else {
            "http://127.0.0.1:1/expected"
        }
    }

    fn request_headers(&self) -> Vec<(String, String)> {
        Vec::new()
    }

    fn serialize_request(&self, _texts: &[&str]) -> SearchResult<Vec<u8>> {
        Ok(Vec::new())
    }

    fn deserialize_response(&self, _body: &[u8]) -> SearchResult<Vec<Vec<f32>>> {
        Ok(Vec::new())
    }
}

impl ApiProvider for TestGatewayProvider {
    fn provider_name(&self) -> &'static str {
        "openai"
    }

    fn api_model_id(&self) -> &'static str {
        "text-embedding-3-small"
    }

    fn embedder_id(&self) -> &'static str {
        "authenticated-test-gateway"
    }

    fn identity_backend(&self) -> &'static str {
        "remote-api-openai"
    }

    fn identity_protocol_revision(&self) -> &'static str {
        "openai-embeddings-json-v1"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn max_batch_size(&self) -> usize {
        16
    }

    fn supports_mrl(&self) -> bool {
        false
    }

    fn endpoint_url(&self) -> &str {
        &self.endpoint
    }

    fn request_headers(&self) -> Vec<(String, String)> {
        vec![("content-type".to_owned(), "application/json".to_owned())]
    }

    fn serialize_request(&self, texts: &[&str]) -> SearchResult<Vec<u8>> {
        serde_json::to_vec(&serde_json::json!({ "inputs": texts })).map_err(|error| {
            SearchError::EmbeddingFailed {
                model: self.embedder_id().to_owned(),
                source: error.into(),
            }
        })
    }

    fn serialize_attested_request(
        &self,
        texts: &[&str],
        challenge: &RemoteEmbeddingChallengeV1,
    ) -> SearchResult<Vec<u8>> {
        serde_json::to_vec(&serde_json::json!({
            "inputs": texts,
            "challenge": challenge,
        }))
        .map_err(|error| SearchError::EmbeddingFailed {
            model: self.embedder_id().to_owned(),
            source: error.into(),
        })
    }

    fn deserialize_response(&self, body: &[u8]) -> SearchResult<Vec<Vec<f32>>> {
        serde_json::from_slice::<TestGatewayResponse>(body)
            .map(|response| response.vectors)
            .map_err(|error| SearchError::EmbeddingFailed {
                model: self.embedder_id().to_owned(),
                source: error.into(),
            })
    }

    fn response_attestation(
        &self,
        body: &[u8],
    ) -> SearchResult<Option<RemoteEmbeddingAttestationV1>> {
        serde_json::from_slice::<TestGatewayResponse>(body)
            .map(|response| response.attestation)
            .map_err(|error| SearchError::EmbeddingFailed {
                model: self.embedder_id().to_owned(),
                source: error.into(),
            })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TestGatewayRequest {
    inputs: Vec<String>,
    challenge: RemoteEmbeddingChallengeV1,
}

#[derive(Debug, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct TestGatewayResponse {
    vectors: Vec<Vec<f32>>,
    attestation: Option<RemoteEmbeddingAttestationV1>,
}

fn spawn_authenticated_gateway(
    identity: EmbeddingIdentityBundleV1,
    vectors: Vec<Vec<f32>>,
    transient_failures: usize,
    include_attestation: bool,
) -> (String, thread::JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        let mut nonces = Vec::new();
        for attempt in 0..=transient_failures {
            let (mut stream, _) = listener.accept().unwrap();
            let body = read_http_request_body(&mut stream);
            let request: TestGatewayRequest = serde_json::from_slice(&body).unwrap();
            let input_refs: Vec<&str> = request.inputs.iter().map(String::as_str).collect();
            assert_eq!(
                request.challenge.ordered_request_sha256,
                remote_ordered_request_sha256(&input_refs)
            );
            assert_eq!(
                request.challenge.input_count,
                u32::try_from(input_refs.len()).unwrap()
            );
            nonces.push(request.challenge.request_nonce.clone());

            if attempt < transient_failures {
                write_http_response(&mut stream, 503, b"{\"error\":\"transient\"}");
                continue;
            }

            let attestation = include_attestation
                .then(|| {
                    RemoteEmbeddingAttestationV1::unsigned(
                        &request.challenge,
                        &identity,
                        "openai",
                        "text-embedding-3-small",
                        TEST_GENERATION,
                        &vectors,
                        TEST_KEY_ID,
                    )
                    .unwrap()
                })
                .map(|mut attestation| {
                    attestation.sign_hmac_sha256(TEST_ATTESTATION_KEY).unwrap();
                    attestation
                });
            let response = serde_json::to_vec(&TestGatewayResponse {
                vectors: vectors.clone(),
                attestation,
            })
            .unwrap();
            write_http_response(&mut stream, 200, &response);
        }
        nonces
    });
    (format!("http://{address}/embeddings"), handle)
}

fn read_http_request_body(stream: &mut TcpStream) -> Vec<u8> {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4096];
    let header_end = loop {
        let read = stream.read(&mut buffer).unwrap();
        assert!(read > 0, "request ended before HTTP headers");
        request.extend_from_slice(&buffer[..read]);
        if let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n") {
            break index + 4;
        }
        assert!(request.len() <= 64 * 1024, "request headers too large");
    };
    let headers = std::str::from_utf8(&request[..header_end]).unwrap();
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap();
    while request.len() - header_end < content_length {
        let read = stream.read(&mut buffer).unwrap();
        assert!(read > 0, "request ended before declared body");
        request.extend_from_slice(&buffer[..read]);
    }
    request[header_end..header_end + content_length].to_vec()
}

fn write_http_response(stream: &mut TcpStream, status: u16, body: &[u8]) {
    let reason = if status == 200 {
        "OK"
    } else {
        "Service Unavailable"
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .unwrap();
    stream.write_all(body).unwrap();
    stream.flush().unwrap();
    let _ = stream.shutdown(Shutdown::Both);
}

fn remote_test_identity(dimension: u32) -> FrozenEmbeddingIdentityBundleV1 {
    let model = "text-embedding-3-small";
    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(model, dimension);
    identity.space.kind = EmbeddingSpaceKindV1::Semantic;
    identity.space.hash_control = None;
    identity.space.artifact_manifest_fingerprint = "1".repeat(64);
    identity.space.artifacts = vec![
        EmbeddingArtifactIdentityV1 {
            role: "weights".to_owned(),
            sha256: "2".repeat(64),
            size: 1,
        },
        EmbeddingArtifactIdentityV1 {
            role: "tokenizer".to_owned(),
            sha256: "3".repeat(64),
            size: 1,
        },
    ];
    identity.space.tokenizer_fingerprint = "3".repeat(64);
    identity.space.vocabulary_fingerprint = "4".repeat(64);
    identity.space.model_config_fingerprint = "5".repeat(64);
    identity.space.output_normalization = API_OUTPUT_NORMALIZATION_V1.to_owned();
    identity.storage.format = API_STORAGE_FORMAT_V1.to_owned();
    identity.storage.quantization = QuantizationFormat::F32;
    identity.storage.endianness = API_STORAGE_ENDIANNESS_V1.to_owned();
    identity.storage.vector_normalization = API_OUTPUT_NORMALIZATION_V1.to_owned();
    identity.producer.backend = "remote-api-openai".to_owned();
    identity.producer.protocol_revision = "openai-embeddings-json-v1".to_owned();
    identity.producer.provenance_manifest_fingerprint = "6".repeat(64);
    identity.producer.space_fingerprint = identity.space.fingerprint();
    identity
        .validate()
        .expect("valid explicit remote test epoch");
    identity
        .freeze()
        .expect("freeze explicit remote test epoch")
}

fn test_attester() -> PinnedRemoteAttesterV1 {
    PinnedRemoteAttesterV1::new(TEST_KEY_ID, TEST_ATTESTATION_KEY.to_vec(), TEST_GENERATION)
        .expect("valid pinned test attester")
}

fn verified_test_embedder(dimension: u32) -> ApiEmbedder {
    let provider = Box::new(OpenAiProvider::text_embedding_3_small(
        "redacted-test-key",
        Some(usize::try_from(dimension).unwrap()),
    ));
    ApiEmbedder::with_verified_defaults(
        provider,
        Some(remote_test_identity(dimension)),
        test_attester(),
    )
    .expect("valid verified test embedder")
}

fn signed_test_attestation(
    embedder: &ApiEmbedder,
    challenge: &RemoteEmbeddingChallengeV1,
    vectors: &[Vec<f32>],
) -> RemoteEmbeddingAttestationV1 {
    let mut attestation = RemoteEmbeddingAttestationV1::unsigned(
        challenge,
        &embedder.identity,
        embedder.provider.provider_name(),
        embedder.provider.api_model_id(),
        TEST_GENERATION,
        vectors,
        TEST_KEY_ID,
    )
    .expect("valid unsigned test attestation");
    attestation
        .sign_hmac_sha256(TEST_ATTESTATION_KEY)
        .expect("sign test attestation");
    attestation
}

#[test]
fn l2_normalize_unit_vector() {
    let mut v = vec![1.0, 0.0, 0.0];
    l2_normalize_in_place(&mut v);
    assert!((v[0] - 1.0).abs() < f32::EPSILON);
}

#[test]
fn l2_normalize_general() {
    let mut v = vec![3.0, 4.0];
    l2_normalize_in_place(&mut v);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
}

#[test]
fn l2_normalize_zero_vector() {
    let mut v = vec![0.0, 0.0, 0.0];
    l2_normalize_in_place(&mut v);
    assert!(v.iter().all(|&x| x == 0.0));
}

#[test]
fn rate_limiter_unlimited() {
    let rl = RateLimiter::new(0);
    assert!(rl.acquire().is_none());
    assert!(rl.acquire().is_none());
}

#[test]
fn rate_limiter_exhausts_tokens() {
    let rl = RateLimiter::new(2);
    // Should get 2 immediate tokens.
    assert!(rl.acquire().is_none());
    assert!(rl.acquire().is_none());
    // Third should require waiting.
    assert!(rl.acquire().is_some());
}

#[test]
fn api_embedder_properties() {
    let embedder = verified_test_embedder(256);
    assert_eq!(embedder.dimension(), 256);
    assert_eq!(embedder.id(), "openai-text-embedding-3-small-256d");
    assert!(embedder.is_semantic());
    assert_eq!(embedder.category(), ModelCategory::ApiEmbedder);
    assert!(embedder.supports_mrl());
    assert_eq!(
        embedder.remote_trust_level(),
        RemoteApiTrustLevelV1::VerifiedRemote
    );
    assert_eq!(embedder.attester.key_id(), TEST_KEY_ID);
    assert_eq!(embedder.attester.generation(), TEST_GENERATION);
}

#[test]
fn truncate_embedding_works() {
    let embedder = verified_test_embedder(4);
    let emb = vec![1.0, 2.0, 3.0, 4.0];
    let truncated = embedder.truncate_embedding(&emb, 2).unwrap();
    assert_eq!(truncated.len(), 2);
    let norm: f32 = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
}

#[test]
fn truncate_embedding_rejects_larger_dim() {
    let embedder = verified_test_embedder(4);
    let emb = vec![1.0, 2.0];
    assert!(embedder.truncate_embedding(&emb, 0).is_err());
    assert!(embedder.truncate_embedding(&emb, 4).is_err());
}

#[test]
fn api_embedder_rejects_unverifiable_or_drifted_epoch() {
    let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(4)));
    assert!(matches!(
        ApiEmbedder::with_defaults(provider, None),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));

    let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(4)));
    let drifted = remote_test_identity(3);
    assert!(matches!(
        ApiEmbedder::with_verified_defaults(provider, Some(drifted), test_attester()),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));

    let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(4)));
    let mut drifted = remote_test_identity(4).identity;
    drifted.space.logical_model_id = "text-embedding-3-large".to_owned();
    drifted.producer.space_fingerprint = drifted.space.fingerprint();
    assert!(matches!(
        ApiEmbedder::with_verified_defaults(
            provider,
            Some(drifted.freeze().unwrap()),
            test_attester()
        ),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));

    let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(4)));
    let mut drifted = remote_test_identity(4).identity;
    drifted.producer.protocol_revision = "unregistered-wire-protocol".to_owned();
    assert!(matches!(
        ApiEmbedder::with_verified_defaults(
            provider,
            Some(drifted.freeze().unwrap()),
            test_attester()
        ),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));

    let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(4)));
    let mut drifted = remote_test_identity(4).identity;
    drifted.storage.format = "fsvi-v2".to_owned();
    drifted.storage.quantization = QuantizationFormat::F16;
    drifted.storage.endianness = "little-endian".to_owned();
    assert!(matches!(
        ApiEmbedder::with_verified_defaults(
            provider,
            Some(drifted.freeze().unwrap()),
            test_attester()
        ),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));
}

#[test]
fn direct_provider_identity_without_pinned_authority_fails_closed() {
    let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(4)));
    let error = ApiEmbedder::with_defaults(provider, Some(remote_test_identity(4)))
        .expect_err("caller-authored identity must not authenticate a provider");
    assert!(matches!(error, SearchError::UnverifiableRemoteSpace { .. }));
}

#[test]
fn per_response_attestation_accepts_exact_authenticated_envelope() {
    let embedder = verified_test_embedder(4);
    let vectors = vec![vec![1.0, 2.0, 3.0, 4.0]];
    let challenge = embedder.build_challenge(&["alpha"]).unwrap();
    let attestation = signed_test_attestation(&embedder, &challenge, &vectors);
    assert!(
        embedder
            .verify_response_attestation(Some(&attestation), &challenge, &vectors)
            .is_ok()
    );
}

#[test]
fn per_response_attestation_rejects_authenticated_epoch_drift() {
    let embedder = verified_test_embedder(4);
    let vectors = vec![vec![1.0, 2.0, 3.0, 4.0]];
    let challenge = embedder.build_challenge(&["alpha"]).unwrap();
    let mut attestation = signed_test_attestation(&embedder, &challenge, &vectors);
    attestation.producer_fingerprint = "f".repeat(64);
    attestation.sign_hmac_sha256(TEST_ATTESTATION_KEY).unwrap();
    assert!(matches!(
        embedder.verify_response_attestation(Some(&attestation), &challenge, &vectors),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));

    let mut attestation = signed_test_attestation(&embedder, &challenge, &vectors);
    attestation.endpoint_fingerprint = "e".repeat(64);
    attestation.sign_hmac_sha256(TEST_ATTESTATION_KEY).unwrap();
    assert!(matches!(
        embedder.verify_response_attestation(Some(&attestation), &challenge, &vectors),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));

    let mut attestation = signed_test_attestation(&embedder, &challenge, &vectors);
    attestation.model = "same-dimension-wrong-model".to_owned();
    attestation.sign_hmac_sha256(TEST_ATTESTATION_KEY).unwrap();
    assert!(matches!(
        embedder.verify_response_attestation(Some(&attestation), &challenge, &vectors),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));

    let mut attestation = signed_test_attestation(&embedder, &challenge, &vectors);
    attestation.generation += 1;
    attestation.sign_hmac_sha256(TEST_ATTESTATION_KEY).unwrap();
    assert!(matches!(
        embedder.verify_response_attestation(Some(&attestation), &challenge, &vectors),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));

    let mut attestation = signed_test_attestation(&embedder, &challenge, &vectors);
    attestation.key_id = "rotated-key".to_owned();
    attestation.sign_hmac_sha256(TEST_ATTESTATION_KEY).unwrap();
    assert!(matches!(
        embedder.verify_response_attestation(Some(&attestation), &challenge, &vectors),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));
}

#[test]
fn authenticated_envelope_mutation_matrix_fails_closed() {
    let embedder = verified_test_embedder(4);
    let vectors = vec![vec![1.0, 2.0, 3.0, 4.0]];
    let challenge = embedder.build_challenge(&["alpha"]).unwrap();
    let mutations: [fn(&mut RemoteEmbeddingAttestationV1); 7] = [
        |attestation| attestation.identity_fingerprint = "1".repeat(64),
        |attestation| attestation.space_fingerprint = "2".repeat(64),
        |attestation| attestation.provider = "other-provider".to_owned(),
        |attestation| attestation.producer_backend = "other-backend".to_owned(),
        |attestation| attestation.protocol_revision = "other-protocol-v1".to_owned(),
        |attestation| attestation.ordered_request_sha256 = "3".repeat(64),
        |attestation| attestation.response_payload_sha256 = "4".repeat(64),
    ];

    for mutate in mutations {
        let mut attestation = signed_test_attestation(&embedder, &challenge, &vectors);
        mutate(&mut attestation);
        attestation.sign_hmac_sha256(TEST_ATTESTATION_KEY).unwrap();
        assert!(matches!(
            embedder.verify_response_attestation(Some(&attestation), &challenge, &vectors),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
    }
}

#[test]
fn same_model_and_dimension_reject_every_canonical_space_component_drift() {
    let embedder = verified_test_embedder(4);
    let vectors = vec![vec![1.0, 2.0, 3.0, 4.0]];
    let expected_challenge = embedder.build_challenge(&["alpha"]).unwrap();
    let mutations: [fn(&mut EmbeddingIdentityBundleV1); 5] = [
        |identity| {
            identity.space.artifact_manifest_fingerprint = "7".repeat(64);
            identity.space.artifacts[0].sha256 = "8".repeat(64);
        },
        |identity| {
            identity.space.tokenizer_fingerprint = "7".repeat(64);
            identity.space.vocabulary_fingerprint = "8".repeat(64);
            identity.space.artifacts[1].sha256 = "9".repeat(64);
        },
        |identity| {
            identity.space.model_preprocessing = "changed-preprocessing-v2".to_owned();
        },
        |identity| {
            identity.space.pooling = "changed-pooling-v2".to_owned();
        },
        |identity| {
            identity.space.output_normalization = "changed-normalization-v2".to_owned();
            identity.storage.vector_normalization = "changed-normalization-v2".to_owned();
        },
    ];

    for mutate in mutations {
        let mut drifted_identity = embedder.identity.clone();
        mutate(&mut drifted_identity);
        drifted_identity.producer.space_fingerprint = drifted_identity.space.fingerprint();
        drifted_identity.validate().unwrap();
        assert_eq!(
            drifted_identity.space.logical_model_id,
            embedder.identity.space.logical_model_id
        );
        assert_eq!(
            drifted_identity.space.dimension,
            embedder.identity.space.dimension
        );
        assert_ne!(
            drifted_identity.space.fingerprint(),
            embedder.identity.space.fingerprint()
        );

        let mut drifted_challenge = expected_challenge.clone();
        drifted_challenge.identity_fingerprint = drifted_identity.fingerprint();
        drifted_challenge.space_fingerprint = drifted_identity.space.fingerprint();
        drifted_challenge.producer_fingerprint = drifted_identity.producer.fingerprint();
        let mut attestation = RemoteEmbeddingAttestationV1::unsigned(
            &drifted_challenge,
            &drifted_identity,
            embedder.provider_contract.provider_name.as_str(),
            embedder.provider_contract.model.as_str(),
            TEST_GENERATION,
            &vectors,
            TEST_KEY_ID,
        )
        .unwrap();
        attestation.sign_hmac_sha256(TEST_ATTESTATION_KEY).unwrap();

        assert!(matches!(
            embedder.verify_response_attestation(Some(&attestation), &expected_challenge, &vectors),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
    }
}

#[test]
fn verified_provider_contract_rejects_midstream_model_and_endpoint_drift() {
    let drift_mode = Arc::new(AtomicU8::new(0));
    let provider = Box::new(MidstreamDriftProvider {
        drift_mode: Arc::clone(&drift_mode),
    });
    let embedder = ApiEmbedder::with_verified_defaults(
        provider,
        Some(remote_test_identity(4)),
        test_attester(),
    )
    .unwrap();

    for mode in [1, 2] {
        drift_mode.store(mode, Ordering::Relaxed);
        assert!(matches!(
            embedder.build_challenge(&["alpha"]),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
    }
}

#[test]
fn per_response_attestation_rejects_missing_wrong_key_and_unknown_schema() {
    let embedder = verified_test_embedder(4);
    let vectors = vec![vec![1.0, 2.0, 3.0, 4.0]];
    let challenge = embedder.build_challenge(&["alpha"]).unwrap();
    assert!(matches!(
        embedder.verify_response_attestation(None, &challenge, &vectors),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));

    let mut wrong_key = RemoteEmbeddingAttestationV1::unsigned(
        &challenge,
        &embedder.identity,
        embedder.provider.provider_name(),
        embedder.provider.api_model_id(),
        TEST_GENERATION,
        &vectors,
        TEST_KEY_ID,
    )
    .unwrap();
    wrong_key.sign_hmac_sha256(&[0x55; 32]).unwrap();
    assert!(matches!(
        embedder.verify_response_attestation(Some(&wrong_key), &challenge, &vectors),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));

    let mut unknown_schema = signed_test_attestation(&embedder, &challenge, &vectors);
    unknown_schema.schema_version = 2;
    assert!(matches!(
        embedder.verify_response_attestation(Some(&unknown_schema), &challenge, &vectors),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));
}

#[test]
fn attestation_rejects_replay_request_reorder_and_payload_tamper() {
    let embedder = verified_test_embedder(4);
    let vectors = vec![vec![1.0, 2.0, 3.0, 4.0], vec![4.0, 3.0, 2.0, 1.0]];
    let original = embedder.build_challenge(&["alpha", "beta"]).unwrap();
    let attestation = signed_test_attestation(&embedder, &original, &vectors);

    let replay_target = embedder.build_challenge(&["alpha", "beta"]).unwrap();
    assert_ne!(original.request_nonce, replay_target.request_nonce);
    assert!(matches!(
        embedder.verify_response_attestation(Some(&attestation), &replay_target, &vectors),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));

    let reordered_request = embedder.build_challenge(&["beta", "alpha"]).unwrap();
    assert_ne!(
        original.ordered_request_sha256,
        reordered_request.ordered_request_sha256
    );
    assert!(matches!(
        embedder.verify_response_attestation(Some(&attestation), &reordered_request, &vectors),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));

    let reordered_vectors = vec![vectors[1].clone(), vectors[0].clone()];
    assert!(matches!(
        embedder.verify_response_attestation(Some(&attestation), &original, &reordered_vectors),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));

    let truncated_vectors = vec![vectors[0].clone()];
    assert!(matches!(
        embedder.verify_response_attestation(Some(&attestation), &original, &truncated_vectors),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));
}

#[test]
fn assumed_remote_is_explicit_transient_and_debug_redacts_vectors() {
    let provider = Box::new(OpenAiProvider::text_embedding_3_small(
        "assumed-secret",
        Some(4),
    ));
    let assumed = AssumedRemoteApi::with_defaults(provider, Some(remote_test_identity(4))).unwrap();
    assert_eq!(
        assumed.remote_trust_level(),
        RemoteApiTrustLevelV1::AssumedRemote
    );
    let batch = AssumedRemoteEmbeddingBatchV1 {
        vectors: vec![vec![123_456.0, 654_321.0, 0.0, 0.0]],
        identity_fingerprint: assumed.identity.fingerprint(),
    };
    let rendered = format!("{batch:?}");
    assert!(rendered.contains("AssumedRemote"));
    assert!(!rendered.contains("123456"));
    assert!(!rendered.contains("654321"));

    let attester = test_attester();
    let rendered = format!("{attester:?}");
    assert!(!rendered.contains("0123456789abcdef"));
}

#[test]
fn authenticated_local_gateway_e2e_admits_exact_vectors() {
    let frozen_identity = remote_test_identity(4);
    let raw_vectors = vec![vec![3.0, 4.0, 0.0, 0.0], vec![0.0, 0.0, 5.0, 12.0]];
    let (endpoint, handle) =
        spawn_authenticated_gateway(frozen_identity.identity.clone(), raw_vectors, 0, true);
    let provider = Box::new(TestGatewayProvider {
        endpoint,
        dimension: 4,
    });
    let embedder =
        ApiEmbedder::with_verified_defaults(provider, Some(frozen_identity), test_attester())
            .unwrap();

    let captured = Arc::new(Mutex::new(None));
    let captured_for_task = Arc::clone(&captured);
    run_test_with_cx(|cx| async move {
        let vectors = embedder.embed_batch(&cx, &["alpha", "beta"]).await.unwrap();
        *captured_for_task.lock().unwrap() = Some(vectors);
    });
    let vectors = captured.lock().unwrap().take().unwrap();
    let nonces = handle.join().unwrap();
    assert_eq!(nonces.len(), 1);
    assert_eq!(vectors.len(), 2);
    for (actual, expected) in vectors[0].iter().zip([0.6, 0.8, 0.0, 0.0]) {
        assert!((actual - expected).abs() < 1e-6);
    }
    for (actual, expected) in vectors[1].iter().zip([0.0, 0.0, 5.0 / 13.0, 12.0 / 13.0]) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn successful_gateway_response_without_attestation_fails_strict_mode() {
    let frozen_identity = remote_test_identity(4);
    let raw_vectors = vec![vec![1.0, 0.0, 0.0, 0.0]];
    let (endpoint, handle) =
        spawn_authenticated_gateway(frozen_identity.identity.clone(), raw_vectors, 0, false);
    let provider = Box::new(TestGatewayProvider {
        endpoint,
        dimension: 4,
    });
    let embedder =
        ApiEmbedder::with_verified_defaults(provider, Some(frozen_identity), test_attester())
            .unwrap();

    let rejected = Arc::new(Mutex::new(false));
    let rejected_for_task = Arc::clone(&rejected);
    run_test_with_cx(|cx| async move {
        let error = embedder
            .embed_batch(&cx, &["alpha"])
            .await
            .expect_err("strict mode must reject a successful unattested response");
        *rejected_for_task.lock().unwrap() =
            matches!(error, SearchError::UnverifiableRemoteSpace { .. });
    });
    assert!(*rejected.lock().unwrap());
    assert_eq!(handle.join().unwrap().len(), 1);
}

#[test]
fn authenticated_retry_uses_a_fresh_nonce_and_admits_only_final_response() {
    let frozen_identity = remote_test_identity(4);
    let raw_vectors = vec![vec![1.0, 0.0, 0.0, 0.0]];
    let (endpoint, handle) =
        spawn_authenticated_gateway(frozen_identity.identity.clone(), raw_vectors, 1, true);
    let provider = Box::new(TestGatewayProvider {
        endpoint,
        dimension: 4,
    });
    let config = ApiEmbedderConfig {
        max_retries: 1,
        retry_base_delay: Duration::from_millis(1),
        ..ApiEmbedderConfig::default()
    };
    let embedder =
        ApiEmbedder::new_verified(provider, config, Some(frozen_identity), test_attester())
            .unwrap();

    let captured = Arc::new(Mutex::new(None));
    let captured_for_task = Arc::clone(&captured);
    run_test_with_cx(|cx| async move {
        let vectors = embedder.embed_batch(&cx, &["alpha"]).await.unwrap();
        *captured_for_task.lock().unwrap() = Some(vectors);
    });
    let vectors = captured.lock().unwrap().take().unwrap();
    let nonces = handle.join().unwrap();
    assert_eq!(vectors, vec![vec![1.0, 0.0, 0.0, 0.0]]);
    assert_eq!(nonces.len(), 2);
    assert_ne!(nonces[0], nonces[1]);
}

#[test]
fn transport_cancellation_and_timeout_are_typed_and_redacted() {
    let provider = OpenAiProvider::custom(
        "credential-canary",
        "model",
        4,
        "https://user:password@example.invalid/private?token=secret",
    );
    let cancelled = api_client_error_to_search(&ClientError::Cancelled, &provider);
    assert!(matches!(cancelled, SearchError::Cancelled { .. }));

    let timed_out = api_client_error_to_search(&ClientError::DeadlineExceeded, &provider);
    let rendered = timed_out.to_string();
    assert!(matches!(timed_out, SearchError::EmbeddingFailed { .. }));
    for secret in [
        "credential-canary",
        "password",
        "example.invalid",
        "private",
        "secret",
    ] {
        assert!(!rendered.contains(secret));
    }
}

#[test]
fn unverifiable_remote_space_redacts_untrusted_labels() {
    let error =
        unverifiable_remote_space("provider\nforged-log-line", "reason\nforged-reason-line");
    assert!(matches!(
        &error,
        SearchError::UnverifiableRemoteSpace { producer, reason }
            if producer == "<redacted-remote-producer>"
                && reason == "remote identity validation failed"
    ));

    let rendered = error.to_string();
    assert!(rendered.contains("Embedding space identity is unverifiable"));
    assert!(!rendered.contains("<redacted-remote-producer>"));
    assert!(!rendered.contains("remote identity validation failed"));
    assert!(!rendered.contains("forged-log-line"));
    assert!(!rendered.contains("forged-reason-line"));
}
