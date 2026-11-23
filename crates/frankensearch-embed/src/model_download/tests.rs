use std::collections::VecDeque;
use std::io::{ErrorKind, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use asupersync::http::h1::ClientError;
use asupersync::test_utils::run_test_with_cx;
use frankensearch_core::error::{SearchError, SearchResult};

use crate::model_manifest::{
    ConsentSource, DownloadConsent, FrozenModelArtifactManifestV1, ModelArtifactManifestV1,
    ModelArtifactRoleV1, ModelLifecycle, ModelState,
};

use super::helpers::{
    AcquisitionLock, LocalCopyErrorGuard, acquisition_transport_progress, client_error_to_search,
    format_bytes, huggingface_url, legacy_manifest_from_frozen, sha256_hex,
};
use super::{
    DEFAULT_MAX_MODEL_ARTIFACT_BYTES, DownloadConfig, DownloadProgress,
    ModelAcquisitionCacheReasonV1, ModelAcquisitionOutcomeV1, ModelAcquisitionRequest,
    ModelAcquisitionSource, ModelAcquisitionSourceKindV1, ModelAcquisitionStageV1,
    ModelAcquisitionVerificationResultV1, ModelDownloader, diagnose_model_acquisition,
};

const LOCK_CHILD_DESTINATION_ENV: &str = "FRANKENSEARCH_TEST_ACQUISITION_LOCK_CHILD_DESTINATION";
const LOCK_CHILD_READY_ENV: &str = "FRANKENSEARCH_TEST_ACQUISITION_LOCK_CHILD_READY";
const LOCK_CHILD_RELEASE_ENV: &str = "FRANKENSEARCH_TEST_ACQUISITION_LOCK_CHILD_RELEASE";

#[derive(Debug, Clone)]
pub(super) struct TestHttpResponse {
    pub(super) status: u16,
    pub(super) reason: &'static str,
    pub(super) body: Vec<u8>,
}

pub(super) fn spawn_test_http_server(
    responses: Vec<TestHttpResponse>,
) -> (String, Arc<AtomicUsize>, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let queue = Arc::new(Mutex::new(VecDeque::from(responses)));
    let served = Arc::new(AtomicUsize::new(0));
    let served_for_thread = Arc::clone(&served);
    let queue_for_thread = Arc::clone(&queue);

    let handle = thread::spawn(move || {
        while let Ok((mut stream, _)) = listener.accept() {
            if read_http_headers(&mut stream).is_err() {
                break;
            }

            let response = {
                let mut guard = queue_for_thread.lock().unwrap();
                guard.pop_front()
            };
            let Some(response) = response else {
                break;
            };

            served_for_thread.fetch_add(1, Ordering::SeqCst);
            if write_http_response(&mut stream, &response).is_err() {
                break;
            }
            let _ = stream.shutdown(Shutdown::Both);
            if queue_for_thread.lock().unwrap().is_empty() {
                break;
            }
        }
    });

    (format!("http://{addr}"), served, handle)
}

fn read_http_headers(stream: &mut TcpStream) -> std::io::Result<()> {
    let mut buf = [0_u8; 1024];
    let mut request = Vec::new();
    loop {
        let read = stream.read(&mut buf)?;
        if read == 0 {
            break;
        }
        request.extend_from_slice(&buf[..read]);
        if request.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
        if request.len() > 64 * 1024 {
            break;
        }
    }
    Ok(())
}

fn write_http_response(stream: &mut TcpStream, response: &TestHttpResponse) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.status,
        response.reason,
        response.body.len()
    )?;
    stream.write_all(&response.body)?;
    stream.flush()?;
    Ok(())
}

fn acquisition_fixture() -> FrozenModelArtifactManifestV1 {
    let mut manifest = ModelArtifactManifestV1::potion_128m_native().unwrap();
    manifest.logical_model_id = "acquisition-fixture".to_owned();
    manifest.artifacts.retain(|artifact| {
        matches!(
            artifact.role,
            ModelArtifactRoleV1::Weights | ModelArtifactRoleV1::Tokenizer
        )
    });
    for artifact in &mut manifest.artifacts {
        let (path, bytes) = match artifact.role {
            ModelArtifactRoleV1::Weights => ("model.safetensors", fixture_weights()),
            ModelArtifactRoleV1::Tokenizer => ("tokenizer.json", fixture_tokenizer()),
            _ => unreachable!("fixture retains only required roles"),
        };
        artifact.relative_path = path.to_owned();
        artifact.upstream_url = format!("https://models.example.invalid/{path}");
        artifact.size = u64::try_from(bytes.len()).unwrap();
        artifact.sha256 = sha256_hex(bytes);
    }
    manifest.freeze().unwrap()
}

const fn fixture_weights() -> &'static [u8] {
    b"fixture-weights-v1"
}

const fn fixture_tokenizer() -> &'static [u8] {
    br#"{"fixture":"tokenizer-v1"}"#
}

fn fixture_bytes(role: ModelArtifactRoleV1) -> &'static [u8] {
    match role {
        ModelArtifactRoleV1::Weights => fixture_weights(),
        ModelArtifactRoleV1::Tokenizer => fixture_tokenizer(),
        _ => unreachable!("fixture retains only required roles"),
    }
}

fn write_acquisition_fixture(source_dir: &Path, frozen: &FrozenModelArtifactManifestV1) {
    for artifact in &frozen.manifest.artifacts {
        let path = source_dir.join(&artifact.relative_path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, fixture_bytes(artifact.role)).unwrap();
    }
}

fn fixture_lifecycle(frozen: &FrozenModelArtifactManifestV1) -> ModelLifecycle {
    ModelLifecycle::new(
        legacy_manifest_from_frozen(frozen).unwrap(),
        DownloadConsent::granted(ConsentSource::Programmatic),
    )
}

fn verify_fixture_load(path: &Path, frozen: &FrozenModelArtifactManifestV1) -> SearchResult<()> {
    frozen.manifest.verify_dir(path).map(|_| ())
}

#[test]
fn huggingface_url_format() {
    let url = huggingface_url(
        "sentence-transformers/all-MiniLM-L6-v2",
        "abc123",
        "onnx/model.onnx",
    );
    assert_eq!(
        url,
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/abc123/onnx/model.onnx"
    );
}

#[test]
fn sha256_hex_known_value() {
    let hash = sha256_hex(b"hello world");
    assert_eq!(
        hash,
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    );
}

#[test]
fn format_bytes_units() {
    assert_eq!(format_bytes(0), "0 B");
    assert_eq!(format_bytes(512), "512 B");
    assert_eq!(format_bytes(1024), "1.0 KB");
    assert_eq!(format_bytes(1_048_576), "1.0 MB");
    assert_eq!(format_bytes(1_073_741_824), "1.0 GB");
}

#[test]
fn download_config_defaults() {
    let config = DownloadConfig::default();
    assert_eq!(config.max_retries, 3);
    assert_eq!(config.retry_base_delay, Duration::from_secs(1));
    assert_eq!(config.max_redirects, 5);
    assert_eq!(config.max_response_bytes, DEFAULT_MAX_MODEL_ARTIFACT_BYTES);
    assert!(config.max_response_bytes > 16 * 1024 * 1024);
    assert!(config.user_agent.starts_with("frankensearch/"));
}

#[test]
fn frozen_local_acquisition_publishes_then_reuses_warm_cache_without_network() {
    let temp = tempfile::tempdir().unwrap();
    let source_dir = temp.path().join("private-source");
    let destination_dir = temp.path().join("cache").join("model");
    let frozen = acquisition_fixture();
    write_acquisition_fixture(&source_dir, &frozen);
    let downloader = ModelDownloader::with_defaults();

    run_test_with_cx(|cx| async move {
        let mut lifecycle = fixture_lifecycle(&frozen);
        let local_events = Arc::new(Mutex::new(Vec::new()));
        let local_events_for_callback = Arc::clone(&local_events);
        let receipt = downloader
            .acquire_frozen_model(
                &cx,
                ModelAcquisitionRequest {
                    frozen_manifest: &frozen,
                    source: ModelAcquisitionSource::LocalBundle(&source_dir),
                    destination_dir: &destination_dir,
                },
                &mut lifecycle,
                move |progress| {
                    local_events_for_callback
                        .lock()
                        .unwrap()
                        .push(progress.clone());
                },
                verify_fixture_load,
            )
            .await
            .unwrap();

        assert_eq!(receipt.source, ModelAcquisitionSourceKindV1::LocalBundle);
        assert_eq!(receipt.outcome, ModelAcquisitionOutcomeV1::Published);
        assert!(receipt.requires_reindex);
        assert!(!receipt.prior_generation_preserved);
        assert!(matches!(
            lifecycle.state(),
            ModelState::AcquiredNeedsReindex
        ));
        frozen.manifest.verify_dir(&destination_dir).unwrap();

        let serialized = serde_json::to_string(&receipt).unwrap();
        assert!(!serialized.contains(&source_dir.to_string_lossy().to_string()));
        assert!(!serialized.contains(&destination_dir.to_string_lossy().to_string()));
        assert!(!serialized.contains("fixture-weights-v1"));
        assert!(serialized.contains(&frozen.fingerprint));
        let events = local_events.lock().unwrap().clone();
        assert!(events.iter().any(|event| {
            event.stage == ModelAcquisitionStageV1::Published
                && event.verification_result == ModelAcquisitionVerificationResultV1::Passed
        }));
        assert!(
            events
                .iter()
                .filter(|event| event.stage == ModelAcquisitionStageV1::Transporting)
                .all(|event| {
                    event.source_host.is_none()
                        && event.artifact_role.is_some()
                        && event.artifact_sha256.is_some()
                        && event.bytes_processed <= event.total_bytes
                })
        );
        let serialized_events = serde_json::to_string(&events).unwrap();
        assert!(!serialized_events.contains(&source_dir.to_string_lossy().to_string()));
        assert!(!serialized_events.contains("fixture-weights-v1"));

        let mut warm_lifecycle = fixture_lifecycle(&frozen);
        let warm_events = Arc::new(Mutex::new(Vec::new()));
        let warm_events_for_callback = Arc::clone(&warm_events);
        let warm_receipt = downloader
            .acquire_frozen_model(
                &cx,
                ModelAcquisitionRequest {
                    frozen_manifest: &frozen,
                    source: ModelAcquisitionSource::Network,
                    destination_dir: &destination_dir,
                },
                &mut warm_lifecycle,
                move |progress| {
                    warm_events_for_callback
                        .lock()
                        .unwrap()
                        .push(progress.clone());
                },
                verify_fixture_load,
            )
            .await
            .unwrap();
        assert_eq!(warm_receipt.source, ModelAcquisitionSourceKindV1::WarmCache);
        assert_eq!(
            warm_receipt.outcome,
            ModelAcquisitionOutcomeV1::VerifiedWarmCache
        );
        assert!(warm_receipt.source_hosts.is_empty());
        let events = warm_events.lock().unwrap().clone();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].stage, ModelAcquisitionStageV1::WarmCacheVerified);
        assert_eq!(
            events[0].cache_reason,
            Some(ModelAcquisitionCacheReasonV1::FrozenGenerationVerified)
        );
        assert!(events[0].source_host.is_none());
        assert!(events[0].artifact_sha256.is_none());
        let serialized_progress = serde_json::to_string(&events[0]).unwrap();
        assert!(!serialized_progress.contains(&destination_dir.to_string_lossy().to_string()));
        assert!(matches!(
            warm_lifecycle.state(),
            ModelState::AcquiredNeedsReindex
        ));
    });
}

#[test]
fn corrupt_warm_cache_is_rejected_and_preserved_during_replacement() {
    let temp = tempfile::tempdir().unwrap();
    let source_dir = temp.path().join("source");
    let destination_dir = temp.path().join("model");
    let frozen = acquisition_fixture();
    write_acquisition_fixture(&source_dir, &frozen);
    let downloader = ModelDownloader::with_defaults();

    run_test_with_cx(|cx| async move {
        let mut initial_lifecycle = fixture_lifecycle(&frozen);
        downloader
            .acquire_frozen_model(
                &cx,
                ModelAcquisitionRequest {
                    frozen_manifest: &frozen,
                    source: ModelAcquisitionSource::LocalBundle(&source_dir),
                    destination_dir: &destination_dir,
                },
                &mut initial_lifecycle,
                |_| {},
                verify_fixture_load,
            )
            .await
            .unwrap();
        std::fs::write(
            destination_dir.join("model.safetensors"),
            b"corrupt-cached-weights",
        )
        .unwrap();

        let events = Arc::new(Mutex::new(Vec::new()));
        let events_for_callback = Arc::clone(&events);
        let mut replacement_lifecycle = fixture_lifecycle(&frozen);
        let receipt = downloader
            .acquire_frozen_model(
                &cx,
                ModelAcquisitionRequest {
                    frozen_manifest: &frozen,
                    source: ModelAcquisitionSource::LocalBundle(&source_dir),
                    destination_dir: &destination_dir,
                },
                &mut replacement_lifecycle,
                move |progress| {
                    events_for_callback.lock().unwrap().push(progress.clone());
                },
                verify_fixture_load,
            )
            .await
            .unwrap();

        assert_eq!(receipt.outcome, ModelAcquisitionOutcomeV1::Published);
        assert!(receipt.prior_generation_preserved);
        frozen.manifest.verify_dir(&destination_dir).unwrap();
        assert!(events.lock().unwrap().iter().all(|event| {
            event.cache_reason == Some(ModelAcquisitionCacheReasonV1::DestinationRejected)
        }));
        let backup = std::fs::read_dir(temp.path())
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("model.backup."))
            })
            .expect("corrupt cache retained as prior generation");
        assert_eq!(
            std::fs::read(backup.join("model.safetensors")).unwrap(),
            b"corrupt-cached-weights"
        );
    });
}

#[test]
fn network_progress_exposes_only_registered_host_and_integrity_metadata() {
    let frozen = acquisition_fixture();
    let artifact = &frozen.manifest.artifacts[0];
    let raw = DownloadProgress {
        file_name: artifact.relative_path.clone(),
        bytes_downloaded: artifact.size.saturating_add(100),
        total_bytes: Some(artifact.size),
        files_completed: 0,
        files_total: frozen.manifest.artifacts.len(),
        speed_bytes_per_sec: 42.0,
        eta_seconds: None,
    };
    let progress = acquisition_transport_progress(
        &frozen,
        ModelAcquisitionSourceKindV1::Network,
        &raw,
        Duration::from_millis(7),
        Some(ModelAcquisitionCacheReasonV1::DestinationMissing),
    )
    .unwrap();
    assert_eq!(
        progress.source_host.as_deref(),
        Some("models.example.invalid")
    );
    assert_eq!(progress.bytes_processed, artifact.size);
    assert_eq!(progress.artifact_sha256.as_deref(), Some(&*artifact.sha256));
    let serialized = serde_json::to_string(&progress).unwrap();
    assert!(!serialized.contains(&artifact.upstream_url));
    assert!(!serialized.contains(&artifact.relative_path));
}

#[test]
fn local_acquisition_rejects_truncation_and_overrun_before_publication() {
    for replacement in [
        b"short".as_slice(),
        b"fixture-weights-v1-with-unregistered-tail".as_slice(),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let source_dir = temp.path().join("source");
        let destination_dir = temp.path().join("model");
        let frozen = acquisition_fixture();
        write_acquisition_fixture(&source_dir, &frozen);
        std::fs::write(source_dir.join("model.safetensors"), replacement).unwrap();
        let downloader = ModelDownloader::with_defaults();

        run_test_with_cx(|cx| async move {
            let mut lifecycle = fixture_lifecycle(&frozen);
            let error = downloader
                .acquire_frozen_model(
                    &cx,
                    ModelAcquisitionRequest {
                        frozen_manifest: &frozen,
                        source: ModelAcquisitionSource::LocalBundle(&source_dir),
                        destination_dir: &destination_dir,
                    },
                    &mut lifecycle,
                    |_| {},
                    verify_fixture_load,
                )
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::HashMismatch { .. }));
            assert!(!destination_dir.exists());
            assert!(matches!(
                lifecycle.state(),
                ModelState::VerificationFailed { .. }
            ));
        });
    }
}

#[test]
fn failed_local_acquisition_preserves_prior_generation_and_retains_stage() {
    let temp = tempfile::tempdir().unwrap();
    let source_dir = temp.path().join("source");
    let destination_dir = temp.path().join("model");
    let frozen = acquisition_fixture();
    write_acquisition_fixture(&source_dir, &frozen);
    std::fs::write(source_dir.join("model.safetensors"), b"fixture-weights-v2").unwrap();
    std::fs::create_dir_all(&destination_dir).unwrap();
    std::fs::write(destination_dir.join("prior-generation"), b"prior").unwrap();
    let downloader = ModelDownloader::with_defaults();

    run_test_with_cx(|cx| async move {
        let mut lifecycle = fixture_lifecycle(&frozen);
        let error = downloader
            .acquire_frozen_model(
                &cx,
                ModelAcquisitionRequest {
                    frozen_manifest: &frozen,
                    source: ModelAcquisitionSource::LocalBundle(&source_dir),
                    destination_dir: &destination_dir,
                },
                &mut lifecycle,
                |_| {},
                verify_fixture_load,
            )
            .await
            .unwrap_err();
        assert!(matches!(error, SearchError::HashMismatch { .. }));
        assert_eq!(
            std::fs::read(destination_dir.join("prior-generation")).unwrap(),
            b"prior"
        );
        let recovery = diagnose_model_acquisition(&destination_dir).unwrap();
        assert_eq!(recovery.orphan_staging_generations, 1);
        assert!(recovery.destination_present);
        assert!(recovery.advisory_lock_present);
        let serialized = serde_json::to_string(&recovery).unwrap();
        assert!(!serialized.contains(&destination_dir.to_string_lossy().to_string()));
    });
}

#[test]
fn recovery_diagnostics_inventory_every_retained_generation_kind() {
    let temp = tempfile::tempdir().unwrap();
    let destination_dir = temp.path().join("model");
    std::fs::create_dir_all(&destination_dir).unwrap();
    std::fs::create_dir(temp.path().join(".model-download-orphan")).unwrap();
    std::fs::create_dir(temp.path().join(".model.installing.interrupted")).unwrap();
    std::fs::create_dir(temp.path().join("model.backup.prior")).unwrap();
    std::fs::write(temp.path().join(".model.acquisition.lock"), b"").unwrap();

    let recovery = diagnose_model_acquisition(&destination_dir).unwrap();

    assert!(recovery.destination_present);
    assert_eq!(recovery.orphan_staging_generations, 1);
    assert_eq!(recovery.interrupted_installing_generations, 1);
    assert_eq!(recovery.preserved_backup_generations, 1);
    assert!(recovery.advisory_lock_present);
    let serialized = serde_json::to_string(&recovery).unwrap();
    assert!(!serialized.contains(&temp.path().to_string_lossy().to_string()));
}

#[test]
fn permission_and_storage_exhaustion_preserve_prior_generation() {
    for injected_kind in [ErrorKind::PermissionDenied, ErrorKind::StorageFull] {
        let temp = tempfile::tempdir().unwrap();
        let source_dir = temp.path().join("source");
        let destination_dir = temp.path().join("model");
        let frozen = acquisition_fixture();
        write_acquisition_fixture(&source_dir, &frozen);
        std::fs::create_dir_all(&destination_dir).unwrap();
        std::fs::write(destination_dir.join("prior-generation"), b"prior").unwrap();
        let downloader = ModelDownloader::with_defaults();

        run_test_with_cx(|cx| async move {
            let _fault = LocalCopyErrorGuard::install(injected_kind);
            let mut lifecycle = fixture_lifecycle(&frozen);
            let error = downloader
                .acquire_frozen_model(
                    &cx,
                    ModelAcquisitionRequest {
                        frozen_manifest: &frozen,
                        source: ModelAcquisitionSource::LocalBundle(&source_dir),
                        destination_dir: &destination_dir,
                    },
                    &mut lifecycle,
                    |_| {},
                    verify_fixture_load,
                )
                .await
                .unwrap_err();

            assert!(matches!(
                error,
                SearchError::Io(ref io_error) if io_error.kind() == injected_kind
            ));
            assert_eq!(
                std::fs::read(destination_dir.join("prior-generation")).unwrap(),
                b"prior"
            );
            assert!(matches!(
                lifecycle.state(),
                ModelState::VerificationFailed { .. }
            ));
        });
    }
}

#[test]
fn successful_replacement_preserves_prior_generation_as_backup() {
    let temp = tempfile::tempdir().unwrap();
    let source_dir = temp.path().join("source");
    let destination_dir = temp.path().join("model");
    let frozen = acquisition_fixture();
    write_acquisition_fixture(&source_dir, &frozen);
    std::fs::create_dir_all(&destination_dir).unwrap();
    std::fs::write(destination_dir.join("prior-generation"), b"prior").unwrap();
    let downloader = ModelDownloader::with_defaults();

    run_test_with_cx(|cx| async move {
        let mut lifecycle = fixture_lifecycle(&frozen);
        let receipt = downloader
            .acquire_frozen_model(
                &cx,
                ModelAcquisitionRequest {
                    frozen_manifest: &frozen,
                    source: ModelAcquisitionSource::LocalBundle(&source_dir),
                    destination_dir: &destination_dir,
                },
                &mut lifecycle,
                |_| {},
                verify_fixture_load,
            )
            .await
            .unwrap();
        assert!(receipt.prior_generation_preserved);
        frozen.manifest.verify_dir(&destination_dir).unwrap();
        let recovery = diagnose_model_acquisition(&destination_dir).unwrap();
        assert_eq!(recovery.preserved_backup_generations, 1);

        let backup = std::fs::read_dir(temp.path())
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("model.backup."))
            })
            .expect("prior generation backup");
        assert_eq!(
            std::fs::read(backup.join("prior-generation")).unwrap(),
            b"prior"
        );
    });
}

#[test]
fn load_self_test_failure_never_replaces_prior_generation() {
    let temp = tempfile::tempdir().unwrap();
    let source_dir = temp.path().join("source");
    let destination_dir = temp.path().join("model");
    let frozen = acquisition_fixture();
    write_acquisition_fixture(&source_dir, &frozen);
    std::fs::create_dir_all(&destination_dir).unwrap();
    std::fs::write(destination_dir.join("prior-generation"), b"prior").unwrap();
    let downloader = ModelDownloader::with_defaults();

    run_test_with_cx(|cx| async move {
        let mut lifecycle = fixture_lifecycle(&frozen);
        let error = downloader
            .acquire_frozen_model(
                &cx,
                ModelAcquisitionRequest {
                    frozen_manifest: &frozen,
                    source: ModelAcquisitionSource::LocalBundle(&source_dir),
                    destination_dir: &destination_dir,
                },
                &mut lifecycle,
                |_| {},
                |_, _| {
                    Err(SearchError::InvalidConfig {
                        field: "model_acquisition.load_self_test".to_owned(),
                        value: "failed".to_owned(),
                        reason: "fixture rejected by load self-test".to_owned(),
                    })
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(error, SearchError::InvalidConfig { .. }));
        assert_eq!(
            std::fs::read(destination_dir.join("prior-generation")).unwrap(),
            b"prior"
        );
    });
}

#[test]
fn concurrent_acquisition_lock_fails_closed_without_touching_destination() {
    let temp = tempfile::tempdir().unwrap();
    let source_dir = temp.path().join("source");
    let destination_dir = temp.path().join("model");
    let frozen = acquisition_fixture();
    write_acquisition_fixture(&source_dir, &frozen);
    let downloader = ModelDownloader::with_defaults();

    run_test_with_cx(|cx| async move {
        let _held_lock = AcquisitionLock::acquire(&destination_dir).unwrap();
        let mut lifecycle = fixture_lifecycle(&frozen);
        let error = downloader
            .acquire_frozen_model(
                &cx,
                ModelAcquisitionRequest {
                    frozen_manifest: &frozen,
                    source: ModelAcquisitionSource::LocalBundle(&source_dir),
                    destination_dir: &destination_dir,
                },
                &mut lifecycle,
                |_| {},
                verify_fixture_load,
            )
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            SearchError::InvalidConfig { ref field, .. }
                if field == "model_acquisition.lock"
        ));
        assert!(!destination_dir.exists());
    });
}

#[test]
fn acquisition_lock_process_helper() {
    let Some(destination) = std::env::var_os(LOCK_CHILD_DESTINATION_ENV) else {
        return;
    };
    let ready =
        PathBuf::from(std::env::var_os(LOCK_CHILD_READY_ENV).expect("child ready signal path"));
    let release =
        PathBuf::from(std::env::var_os(LOCK_CHILD_RELEASE_ENV).expect("child release signal path"));
    let _lock = AcquisitionLock::acquire(Path::new(&destination)).unwrap();
    std::fs::write(&ready, b"locked").unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while !release.exists() {
        assert!(
            Instant::now() < deadline,
            "parent did not release child acquisition lock"
        );
        thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn two_process_acquisition_race_fails_closed_and_preserves_prior_generation() {
    let temp = tempfile::tempdir().unwrap();
    let source_dir = temp.path().join("source");
    let destination_dir = temp.path().join("model");
    let ready = temp.path().join("child-ready");
    let release = temp.path().join("child-release");
    let frozen = acquisition_fixture();
    write_acquisition_fixture(&source_dir, &frozen);
    std::fs::create_dir_all(&destination_dir).unwrap();
    std::fs::write(destination_dir.join("prior-generation"), b"prior").unwrap();

    let mut child = Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("model_download::tests::acquisition_lock_process_helper")
        .arg("--nocapture")
        .env(LOCK_CHILD_DESTINATION_ENV, &destination_dir)
        .env(LOCK_CHILD_READY_ENV, &ready)
        .env(LOCK_CHILD_RELEASE_ENV, &release)
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while !ready.exists() {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(
                status.success() && ready.exists(),
                "lock helper exited before readiness with {status}"
            );
        }
        assert!(
            Instant::now() < deadline,
            "lock helper did not become ready"
        );
        thread::sleep(Duration::from_millis(10));
    }

    let downloader = ModelDownloader::with_defaults();
    let observed_lock_failure = Arc::new(AtomicBool::new(false));
    let observed_lock_failure_for_task = Arc::clone(&observed_lock_failure);
    run_test_with_cx(|cx| async move {
        let mut lifecycle = fixture_lifecycle(&frozen);
        let result = downloader
            .acquire_frozen_model(
                &cx,
                ModelAcquisitionRequest {
                    frozen_manifest: &frozen,
                    source: ModelAcquisitionSource::LocalBundle(&source_dir),
                    destination_dir: &destination_dir,
                },
                &mut lifecycle,
                |_| {},
                verify_fixture_load,
            )
            .await;
        observed_lock_failure_for_task.store(
            matches!(
                result,
                Err(SearchError::InvalidConfig { ref field, .. })
                    if field == "model_acquisition.lock"
            ),
            Ordering::SeqCst,
        );
    });
    std::fs::write(&release, b"release").unwrap();
    assert!(child.wait().unwrap().success());

    assert!(observed_lock_failure.load(Ordering::SeqCst));
    assert_eq!(
        std::fs::read(temp.path().join("model").join("prior-generation")).unwrap(),
        b"prior"
    );
}

#[test]
fn structured_cancellation_prevents_stage_or_publication() {
    let temp = tempfile::tempdir().unwrap();
    let source_dir = temp.path().join("source");
    let destination_dir = temp.path().join("model");
    let frozen = acquisition_fixture();
    write_acquisition_fixture(&source_dir, &frozen);
    let downloader = ModelDownloader::with_defaults();

    run_test_with_cx(|cx| async move {
        cx.set_cancel_requested(true);
        let mut lifecycle = fixture_lifecycle(&frozen);
        let error = downloader
            .acquire_frozen_model(
                &cx,
                ModelAcquisitionRequest {
                    frozen_manifest: &frozen,
                    source: ModelAcquisitionSource::LocalBundle(&source_dir),
                    destination_dir: &destination_dir,
                },
                &mut lifecycle,
                |_| {},
                verify_fixture_load,
            )
            .await
            .unwrap_err();
        assert!(matches!(error, SearchError::Cancelled { .. }));
        assert!(matches!(lifecycle.state(), ModelState::Cancelled));
        assert!(!destination_dir.exists());
    });
}

#[test]
fn structured_cancellation_is_observed_at_every_precommit_stage() {
    for cancellation_stage in [
        ModelAcquisitionStageV1::Transporting,
        ModelAcquisitionStageV1::StagedVerified,
        ModelAcquisitionStageV1::LoadSelfTestPassed,
    ] {
        let temp = tempfile::tempdir().unwrap();
        let source_dir = temp.path().join("source");
        let destination_dir = temp.path().join("model");
        let frozen = acquisition_fixture();
        write_acquisition_fixture(&source_dir, &frozen);
        std::fs::create_dir_all(&destination_dir).unwrap();
        std::fs::write(destination_dir.join("prior-generation"), b"prior").unwrap();
        let downloader = ModelDownloader::with_defaults();

        run_test_with_cx(|cx| async move {
            let cancellation_cx = cx.clone();
            let mut lifecycle = fixture_lifecycle(&frozen);
            let error = downloader
                .acquire_frozen_model(
                    &cx,
                    ModelAcquisitionRequest {
                        frozen_manifest: &frozen,
                        source: ModelAcquisitionSource::LocalBundle(&source_dir),
                        destination_dir: &destination_dir,
                    },
                    &mut lifecycle,
                    move |progress| {
                        if progress.stage == cancellation_stage {
                            cancellation_cx.set_cancel_requested(true);
                        }
                    },
                    verify_fixture_load,
                )
                .await
                .unwrap_err();

            assert!(matches!(error, SearchError::Cancelled { .. }));
            assert!(matches!(lifecycle.state(), ModelState::Cancelled));
            assert_eq!(
                std::fs::read(destination_dir.join("prior-generation")).unwrap(),
                b"prior"
            );
        });
    }
}

#[test]
fn cancellation_after_publication_does_not_revoke_committed_generation() {
    let temp = tempfile::tempdir().unwrap();
    let source_dir = temp.path().join("source");
    let destination_dir = temp.path().join("model");
    let frozen = acquisition_fixture();
    write_acquisition_fixture(&source_dir, &frozen);
    let downloader = ModelDownloader::with_defaults();

    run_test_with_cx(|cx| async move {
        let cancellation_cx = cx.clone();
        let mut lifecycle = fixture_lifecycle(&frozen);
        let receipt = downloader
            .acquire_frozen_model(
                &cx,
                ModelAcquisitionRequest {
                    frozen_manifest: &frozen,
                    source: ModelAcquisitionSource::LocalBundle(&source_dir),
                    destination_dir: &destination_dir,
                },
                &mut lifecycle,
                move |progress| {
                    if progress.stage == ModelAcquisitionStageV1::Published {
                        cancellation_cx.set_cancel_requested(true);
                    }
                },
                verify_fixture_load,
            )
            .await
            .unwrap();

        assert_eq!(receipt.outcome, ModelAcquisitionOutcomeV1::Published);
        assert!(matches!(
            lifecycle.state(),
            ModelState::AcquiredNeedsReindex
        ));
        frozen.manifest.verify_dir(&destination_dir).unwrap();
    });
}

#[test]
fn download_progress_display_with_total() {
    let progress = DownloadProgress {
        file_name: "model.onnx".to_owned(),
        bytes_downloaded: 524_288,
        total_bytes: Some(1_048_576),
        files_completed: 0,
        files_total: 3,
        speed_bytes_per_sec: 1_048_576.0,
        eta_seconds: Some(0.5),
    };
    let display = progress.to_string();
    assert!(display.contains("[1/3]"));
    assert!(display.contains("model.onnx"));
    assert!(display.contains("50%"));
}

#[test]
fn download_progress_display_without_total() {
    let progress = DownloadProgress {
        file_name: "config.json".to_owned(),
        bytes_downloaded: 1024,
        total_bytes: None,
        files_completed: 2,
        files_total: 3,
        speed_bytes_per_sec: 0.0,
        eta_seconds: None,
    };
    let display = progress.to_string();
    assert!(display.contains("[3/3]"));
    assert!(display.contains("config.json"));
    assert!(display.contains("1.0 KB"));
}

#[test]
fn client_error_converts_to_search_error() {
    let err = client_error_to_search(
        ClientError::InvalidUrl("bad".to_owned()),
        "https://example.com",
    );
    assert!(matches!(err, SearchError::ModelLoadFailed { .. }));
}

#[test]
fn tls_and_cancellation_errors_are_typed_bounded_and_redacted() {
    let tls_error = client_error_to_search(
        ClientError::TlsError("certificate detail from /private/cert.pem".to_owned()),
        "https://models.example.invalid/private/model?token=secret",
    );
    let rendered = tls_error.to_string();
    assert!(matches!(tls_error, SearchError::ModelLoadFailed { .. }));
    assert!(rendered.contains("models.example.invalid"));
    assert!(rendered.contains("tls-handshake-failure"));
    assert!(!rendered.contains("/private"));
    assert!(!rendered.contains("secret"));
    assert!(!rendered.contains("certificate detail"));

    let cancelled = client_error_to_search(
        ClientError::Cancelled,
        "https://models.example.invalid/model",
    );
    assert!(matches!(cancelled, SearchError::Cancelled { .. }));
}
