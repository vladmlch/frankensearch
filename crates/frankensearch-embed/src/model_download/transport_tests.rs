use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use asupersync::test_utils::run_test_with_cx;
use frankensearch_core::error::SearchError;

use crate::model_manifest::{
    ModelFile, ModelLifecycle, ModelManifest, PLACEHOLDER_PINNED_REVISION,
    PLACEHOLDER_VERIFY_AFTER_DOWNLOAD,
};

use super::helpers::{
    TempFileGuard, create_unique_staging_dir, format_bytes, response_content_length, sha256_hex,
};
use super::tests::{TestHttpResponse, spawn_test_http_server};
use super::{DEFAULT_MAX_MODEL_ARTIFACT_BYTES, DownloadConfig, DownloadProgress, ModelDownloader};

#[test]
fn download_single_file_success_writes_file_and_reports_progress() {
    let body = b"hello-model".to_vec();
    let file = ModelFile {
        name: "model.onnx".to_owned(),
        sha256: sha256_hex(&body),
        size: u64::try_from(body.len()).unwrap(),
        url: None,
    };

    let (base_url, served, handle) = spawn_test_http_server(vec![TestHttpResponse {
        status: 200,
        reason: "OK",
        body: body.clone(),
    }]);
    let url = format!("{base_url}/model.onnx");
    let dest_dir = tempfile::tempdir().unwrap();
    let dest = dest_dir.path().join("model.onnx");
    let dest_for_task = dest.clone();
    let progress = Arc::new(Mutex::new(Vec::<DownloadProgress>::new()));
    let progress_for_cb = Arc::clone(&progress);
    let downloader = ModelDownloader::new(DownloadConfig {
        max_retries: 0,
        retry_base_delay: Duration::from_millis(1),
        user_agent: "frankensearch-test".to_owned(),
        max_redirects: 0,
        max_response_bytes: DEFAULT_MAX_MODEL_ARTIFACT_BYTES,
    });

    run_test_with_cx(|cx| async move {
        downloader
            .download_single_file(&cx, &url, &dest_for_task, &file, 0, 1, &|p| {
                progress_for_cb.lock().unwrap().push(p.clone());
            })
            .await
            .unwrap();
    });

    handle.join().unwrap();
    assert_eq!(served.load(Ordering::SeqCst), 1);
    assert_eq!(std::fs::read(dest).unwrap(), body);

    let events = progress.lock().unwrap();
    assert!(events.len() >= 2);
    assert_eq!(events[0].bytes_downloaded, 0);
    assert_eq!(events[0].file_name, "model.onnx");
    let last = events.last().unwrap();
    let expected_size = u64::try_from(body.len()).unwrap();
    assert_eq!(last.bytes_downloaded, expected_size);
    assert_eq!(last.total_bytes, Some(expected_size));
    drop(events);
}

#[test]
fn download_single_file_succeeds_beyond_legacy_16mib_codec_cap() {
    // Regression for #27: the default HTTP codec caps response bodies at
    // 16 MiB, which rejected production model artifacts (the default Potion
    // tokenizer is ~17.8 MiB). A body just over the legacy cap must now
    // stream to disk and verify.
    let body = vec![0x41_u8; 16 * 1024 * 1024 + 4096];
    let file = ModelFile {
        name: "big.onnx".to_owned(),
        sha256: sha256_hex(&body),
        size: u64::try_from(body.len()).unwrap(),
        url: None,
    };

    let (base_url, served, handle) = spawn_test_http_server(vec![TestHttpResponse {
        status: 200,
        reason: "OK",
        body: body.clone(),
    }]);
    let url = format!("{base_url}/big.onnx");
    let dest_dir = tempfile::tempdir().unwrap();
    let dest = dest_dir.path().join("big.onnx");
    let dest_for_task = dest.clone();
    // Default config carries the 2 GiB cap.
    let downloader = ModelDownloader::with_defaults();

    run_test_with_cx(|cx| async move {
        downloader
            .download_single_file(&cx, &url, &dest_for_task, &file, 0, 1, &|_| {})
            .await
            .unwrap();
    });

    handle.join().unwrap();
    assert_eq!(served.load(Ordering::SeqCst), 1);
    assert_eq!(std::fs::read(dest).unwrap().len(), body.len());
}

#[test]
fn download_model_rejects_artifact_larger_than_response_cap() {
    // A manifest whose declared artifact exceeds the configured cap must
    // fail fast with an actionable InvalidConfig error, not a cryptic
    // mid-stream BodyTooLarge after retries.
    let mut manifest = ModelManifest::minilm_v2();
    let oversized = u64::try_from(manifest_cap_test_bytes()).unwrap() + 1;
    manifest.files[0].size = oversized;
    // Skip the sum-of-sizes cross-check so the oversized size reaches the cap guard.
    manifest.download_size_bytes = 0;
    let consent = crate::model_manifest::DownloadConsent::granted(
        crate::model_manifest::ConsentSource::Environment,
    );
    let mut lifecycle = ModelLifecycle::new(manifest.clone(), consent);
    let dest = tempfile::tempdir().unwrap();
    let downloader = ModelDownloader::new(DownloadConfig {
        max_retries: 0,
        retry_base_delay: Duration::from_millis(1),
        user_agent: "frankensearch-test".to_owned(),
        max_redirects: 0,
        max_response_bytes: manifest_cap_test_bytes(),
    });

    run_test_with_cx(|cx| async move {
        let err = downloader
            .download_model(&cx, &manifest, dest.path(), &mut lifecycle, |_| {})
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            SearchError::InvalidConfig { ref field, .. }
                if field == "download.max_response_bytes"
        ));
        assert!(err.to_string().contains("max_response_bytes"));
        assert!(matches!(
            lifecycle.state(),
            crate::model_manifest::ModelState::VerificationFailed { .. }
        ));
    });
}

const fn manifest_cap_test_bytes() -> usize {
    4 * 1024
}

#[test]
fn download_file_with_retry_succeeds_after_transient_http_error() {
    let body = b"retry-success".to_vec();
    let file = ModelFile {
        name: "model.onnx".to_owned(),
        sha256: sha256_hex(&body),
        size: u64::try_from(body.len()).unwrap(),
        url: None,
    };

    let (base_url, served, handle) = spawn_test_http_server(vec![
        TestHttpResponse {
            status: 500,
            reason: "Internal Server Error",
            body: b"server error".to_vec(),
        },
        TestHttpResponse {
            status: 200,
            reason: "OK",
            body: body.clone(),
        },
    ]);
    let url = format!("{base_url}/model.onnx");
    let dest_dir = tempfile::tempdir().unwrap();
    let dest = dest_dir.path().join("model.onnx");
    let dest_for_task = dest.clone();
    let downloader = ModelDownloader::new(DownloadConfig {
        max_retries: 1,
        retry_base_delay: Duration::from_millis(1),
        user_agent: "frankensearch-test".to_owned(),
        max_redirects: 0,
        max_response_bytes: DEFAULT_MAX_MODEL_ARTIFACT_BYTES,
    });

    run_test_with_cx(|cx| async move {
        downloader
            .download_file_with_retry(&cx, &url, &dest_for_task, &file, 0, 1, &|_| {})
            .await
            .unwrap();
    });

    handle.join().unwrap();
    assert_eq!(served.load(Ordering::SeqCst), 2);
    assert_eq!(std::fs::read(dest).unwrap(), body);
}

#[test]
fn download_file_with_retry_returns_error_after_max_attempts() {
    let file = ModelFile {
        name: "model.onnx".to_owned(),
        sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
        size: 4,
        url: None,
    };

    let (base_url, served, handle) = spawn_test_http_server(vec![
        TestHttpResponse {
            status: 500,
            reason: "Internal Server Error",
            body: b"error".to_vec(),
        },
        TestHttpResponse {
            status: 500,
            reason: "Internal Server Error",
            body: b"error".to_vec(),
        },
    ]);
    let url = format!("{base_url}/model.onnx");
    let dest_dir = tempfile::tempdir().unwrap();
    let dest = dest_dir.path().join("model.onnx");
    let dest_for_task = dest.clone();
    let downloader = ModelDownloader::new(DownloadConfig {
        max_retries: 1,
        retry_base_delay: Duration::from_millis(1),
        user_agent: "frankensearch-test".to_owned(),
        max_redirects: 0,
        max_response_bytes: DEFAULT_MAX_MODEL_ARTIFACT_BYTES,
    });

    run_test_with_cx(|cx| async move {
        let err = downloader
            .download_file_with_retry(&cx, &url, &dest_for_task, &file, 0, 1, &|_| {})
            .await
            .unwrap_err();
        assert!(matches!(err, SearchError::ModelLoadFailed { .. }));
        assert!(err.to_string().contains("HTTP 500"));
    });

    handle.join().unwrap();
    assert_eq!(served.load(Ordering::SeqCst), 2);
    assert!(!dest.exists());
}

#[test]
fn download_single_file_hash_mismatch_does_not_write_destination() {
    let expected = b"expected-content".to_vec();
    let file = ModelFile {
        name: "model.onnx".to_owned(),
        sha256: sha256_hex(&expected),
        size: u64::try_from(expected.len()).unwrap(),
        url: None,
    };

    let (base_url, served, handle) = spawn_test_http_server(vec![TestHttpResponse {
        status: 200,
        reason: "OK",
        body: b"different-content".to_vec(),
    }]);
    let url = format!("{base_url}/model.onnx");
    let dest_dir = tempfile::tempdir().unwrap();
    let dest = dest_dir.path().join("model.onnx");
    let dest_for_task = dest.clone();
    let downloader = ModelDownloader::new(DownloadConfig {
        max_retries: 0,
        retry_base_delay: Duration::from_millis(1),
        user_agent: "frankensearch-test".to_owned(),
        max_redirects: 0,
        max_response_bytes: DEFAULT_MAX_MODEL_ARTIFACT_BYTES,
    });

    run_test_with_cx(|cx| async move {
        let err = downloader
            .download_single_file(&cx, &url, &dest_for_task, &file, 0, 1, &|_| {})
            .await
            .unwrap_err();
        assert!(matches!(err, SearchError::HashMismatch { .. }));
    });

    handle.join().unwrap();
    assert_eq!(served.load(Ordering::SeqCst), 1);
    assert!(!dest.exists());
    assert!(!dest.with_extension("tmp").exists());
}

#[test]
fn download_model_failure_transitions_lifecycle_to_verification_failed() {
    let manifest = ModelManifest {
        id: "test-model".to_owned(),
        version: "test-v1".to_owned(),
        display_name: None,
        description: None,
        repo: "owner/repo".to_owned(),
        revision: "d".repeat(40),
        files: vec![ModelFile {
            name: "bad-file.bin".to_owned(),
            sha256: "0".repeat(64),
            size: 1,
            // The manifest remains valid and production-ready, while the
            // missing authority forces an immediate client error.
            url: Some("https://".to_owned()),
        }],
        license: "Apache-2.0".to_owned(),
        dimension: None,
        tier: None,
        download_size_bytes: 0,
    };
    assert!(manifest.validate().is_ok());
    assert!(manifest.is_production_ready());
    let consent = crate::model_manifest::DownloadConsent::granted(
        crate::model_manifest::ConsentSource::Environment,
    );
    let mut lifecycle = ModelLifecycle::new(manifest.clone(), consent);
    let dest = tempfile::tempdir().unwrap();
    let downloader = ModelDownloader::new(DownloadConfig {
        max_retries: 0,
        retry_base_delay: Duration::from_millis(1),
        user_agent: "frankensearch-test".to_owned(),
        max_redirects: 0,
        max_response_bytes: DEFAULT_MAX_MODEL_ARTIFACT_BYTES,
    });

    run_test_with_cx(|cx| async move {
        let err = downloader
            .download_model(&cx, &manifest, dest.path(), &mut lifecycle, |_| {})
            .await
            .unwrap_err();
        assert!(matches!(err, SearchError::ModelLoadFailed { .. }));
        assert!(matches!(
            lifecycle.state(),
            crate::model_manifest::ModelState::VerificationFailed { .. }
        ));
        assert!(lifecycle.begin_download(1).is_ok());
    });
}

#[test]
fn create_unique_staging_dir_returns_distinct_paths() {
    let temp = tempfile::tempdir().unwrap();
    let first = create_unique_staging_dir(temp.path()).expect("first staging dir");
    let second = create_unique_staging_dir(temp.path()).expect("second staging dir");

    assert_ne!(first, second);
    assert!(first.is_dir());
    assert!(second.is_dir());
}

// ─── bd-r476 tests begin ───

#[test]
fn response_content_length_found() {
    let headers = vec![("Content-Length".to_owned(), "42".to_owned())];
    assert_eq!(response_content_length(&headers), Some(42));
}

#[test]
fn response_content_length_missing() {
    let headers = vec![("X-Custom".to_owned(), "value".to_owned())];
    assert_eq!(response_content_length(&headers), None);
}

#[test]
fn response_content_length_invalid_value() {
    let headers = vec![("Content-Length".to_owned(), "not-a-number".to_owned())];
    assert_eq!(response_content_length(&headers), None);
}

#[test]
fn response_content_length_case_insensitive() {
    let headers = vec![("content-length".to_owned(), "100".to_owned())];
    assert_eq!(response_content_length(&headers), Some(100));

    let headers_upper = vec![("CONTENT-LENGTH".to_owned(), "200".to_owned())];
    assert_eq!(response_content_length(&headers_upper), Some(200));
}

#[test]
fn response_content_length_trims_whitespace() {
    let headers = vec![("Content-Length".to_owned(), "  300  ".to_owned())];
    assert_eq!(response_content_length(&headers), Some(300));
}

#[test]
fn sha256_digest_hex_known_empty() {
    // SHA-256 of empty data
    let hash = sha256_hex(b"");
    assert_eq!(
        hash,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn sha256_digest_hex_always_64_chars() {
    let hash = sha256_hex(b"test");
    assert_eq!(hash.len(), 64);
    assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn sha256_digest_hex_lowercase() {
    let hash = sha256_hex(b"ABC");
    // Hex should be lowercase
    assert_eq!(hash, hash.to_lowercase());
}

#[test]
fn format_bytes_boundary_values() {
    // Just below KB
    assert_eq!(format_bytes(1023), "1023 B");
    // Exactly KB
    assert_eq!(format_bytes(1024), "1.0 KB");
    // Just below MB
    assert!(format_bytes(1024 * 1024 - 1).contains("KB"));
    // Exactly MB
    assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
    // Just below GB
    assert!(format_bytes(1024 * 1024 * 1024 - 1).contains("MB"));
    // Exactly GB
    assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
}

#[test]
fn format_bytes_large_values() {
    let ten_gb = 10 * 1024 * 1024 * 1024_u64;
    let result = format_bytes(ten_gb);
    assert!(result.contains("GB"));
    assert!(result.contains("10.0"));
}

#[test]
fn temp_file_guard_armed_cleans_up() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("temp.bin");
    std::fs::write(&path, b"data").unwrap();
    assert!(path.exists());
    {
        let _guard = TempFileGuard::new(path.clone());
        // guard drops here, armed=true
    }
    assert!(!path.exists(), "armed guard should remove file on drop");
}

#[test]
fn temp_file_guard_disarmed_does_not_clean_up() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("keep.bin");
    std::fs::write(&path, b"data").unwrap();
    assert!(path.exists());
    {
        let mut guard = TempFileGuard::new(path.clone());
        guard.disarm();
        // guard drops here, armed=false
    }
    assert!(path.exists(), "disarmed guard should leave file intact");
}

#[test]
fn temp_file_guard_armed_nonexistent_file_does_not_panic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nonexistent.bin");
    {
        let _guard = TempFileGuard::new(path);
        // guard drops on nonexistent file — should not panic
    }
}

#[test]
fn download_progress_debug() {
    let progress = DownloadProgress {
        file_name: "model.onnx".to_owned(),
        bytes_downloaded: 0,
        total_bytes: Some(100),
        files_completed: 0,
        files_total: 1,
        speed_bytes_per_sec: 0.0,
        eta_seconds: None,
    };
    let debug = format!("{progress:?}");
    assert!(debug.contains("DownloadProgress"));
    assert!(debug.contains("model.onnx"));
}

#[test]
fn download_progress_clone() {
    let progress = DownloadProgress {
        file_name: "file.bin".to_owned(),
        bytes_downloaded: 42,
        total_bytes: Some(100),
        files_completed: 1,
        files_total: 2,
        speed_bytes_per_sec: 1000.0,
        eta_seconds: Some(0.058),
    };
    #[allow(clippy::redundant_clone)]
    let cloned = progress.clone();
    assert_eq!(cloned.file_name, "file.bin");
    assert_eq!(cloned.bytes_downloaded, 42);
    assert_eq!(cloned.total_bytes, Some(100));
    assert_eq!(cloned.files_completed, 1);
    assert_eq!(cloned.files_total, 2);
}

#[test]
fn download_config_clone_and_debug() {
    let config = DownloadConfig::default();
    let cloned = config.clone();
    assert_eq!(cloned.max_retries, config.max_retries);
    assert_eq!(cloned.user_agent, config.user_agent);
    let debug = format!("{config:?}");
    assert!(debug.contains("DownloadConfig"));
    assert!(debug.contains("max_retries"));
}

#[test]
fn model_downloader_with_defaults_creates_valid_instance() {
    let _downloader = ModelDownloader::with_defaults();
}

#[test]
fn download_progress_display_zero_total_bytes_no_percent() {
    let progress = DownloadProgress {
        file_name: "test.bin".to_owned(),
        bytes_downloaded: 500,
        total_bytes: Some(0),
        files_completed: 0,
        files_total: 1,
        speed_bytes_per_sec: 0.0,
        eta_seconds: None,
    };
    let display = progress.to_string();
    // total_bytes=0 is filtered out, so no percentage
    assert!(!display.contains('%'));
}

#[test]
fn create_unique_staging_dir_creates_parent_if_needed() {
    let temp = tempfile::tempdir().unwrap();
    let nested = temp.path().join("deeply").join("nested").join("dir");
    let result = create_unique_staging_dir(&nested).expect("should create nested dir");
    assert!(result.is_dir());
    // The function creates the parent of dest_dir (for sibling staging), not dest_dir itself.
    assert!(nested.parent().unwrap().is_dir());
}

// ─── bd-r476 tests end ───

#[test]
fn download_model_rejects_non_production_ready_manifest() {
    let mut manifest = ModelManifest::minilm_v2();
    manifest.revision = PLACEHOLDER_PINNED_REVISION.to_owned();
    manifest.files[0].sha256 = PLACEHOLDER_VERIFY_AFTER_DOWNLOAD.to_owned();
    manifest.files[0].size = 0;
    manifest.files[0].url = None;
    manifest.download_size_bytes = 0;
    let consent = crate::model_manifest::DownloadConsent::granted(
        crate::model_manifest::ConsentSource::Environment,
    );
    let mut lifecycle = ModelLifecycle::new(manifest.clone(), consent);
    let dest = tempfile::tempdir().unwrap();
    let downloader = ModelDownloader::new(DownloadConfig {
        max_retries: 0,
        retry_base_delay: Duration::from_millis(1),
        user_agent: "frankensearch-test".to_owned(),
        max_redirects: 0,
        max_response_bytes: DEFAULT_MAX_MODEL_ARTIFACT_BYTES,
    });

    run_test_with_cx(|cx| async move {
        let err = downloader
            .download_model(&cx, &manifest, dest.path(), &mut lifecycle, |_| {})
            .await
            .unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
        assert!(err.to_string().contains("production-ready"));
        assert!(matches!(
            lifecycle.state(),
            crate::model_manifest::ModelState::VerificationFailed { .. }
        ));
    });
}

#[test]
fn download_model_rejects_manifest_with_path_traversal_filename() {
    let mut manifest = ModelManifest::minilm_v2();
    manifest.files[0].name = "../escape.bin".to_owned();
    let consent = crate::model_manifest::DownloadConsent::granted(
        crate::model_manifest::ConsentSource::Environment,
    );
    let mut lifecycle = ModelLifecycle::new(manifest.clone(), consent);
    let dest = tempfile::tempdir().unwrap();
    let downloader = ModelDownloader::new(DownloadConfig {
        max_retries: 0,
        retry_base_delay: Duration::from_millis(1),
        user_agent: "frankensearch-test".to_owned(),
        max_redirects: 0,
        max_response_bytes: DEFAULT_MAX_MODEL_ARTIFACT_BYTES,
    });

    run_test_with_cx(|cx| async move {
        let err = downloader
            .download_model(&cx, &manifest, dest.path(), &mut lifecycle, |_| {})
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            SearchError::InvalidConfig { ref field, .. } if field == "files[].name"
        ));
        assert!(matches!(
            lifecycle.state(),
            crate::model_manifest::ModelState::VerificationFailed { .. }
        ));
    });
}
