#![allow(clippy::wildcard_imports)]

pub use std::fs::{self, OpenOptions};
pub use std::io::Write;
pub use std::path::{Path, PathBuf};
pub use std::sync::Arc;
pub use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub use frankensearch_core::SearchError;
pub use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, EmbeddingArtifactIdentityV1, EmbeddingSpaceKindV1,
    QuantizationFormat,
};
pub use sha2::{Digest, Sha256};

pub use super::*;

mod unit_and_locks;
mod v1_roundtrip_vacuum;
mod v2_admission_publication;
mod v2_owner_snapshot;
mod wal_compaction;

pub fn temp_index_path(name: &str) -> PathBuf {
    static NEXT_FIXTURE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let serial = NEXT_FIXTURE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "frankensearch-index-{name}-{}-{now}-{serial}",
        std::process::id(),
    ));
    fs::create_dir(&directory).expect("create private index fixture directory");
    directory.join("index.fsvi")
}

pub fn sample_vector(base: f32, dim: usize) -> Vec<f32> {
    vec![base; dim]
}

pub fn fsvi_v2_binding(
    model_id: &str,
    dimension: u32,
    quantization: Quantization,
    sequence: u64,
    nonce_byte: u8,
) -> FsviV2IdentityBinding {
    let mut identity =
        frankensearch_core::generation::EmbeddingIdentityBundleV1::explicit_test_model(
            model_id, dimension,
        );
    identity.storage.format = "fsvi-v2".to_owned();
    identity.storage.quantization = match quantization {
        Quantization::F32 => QuantizationFormat::F32,
        Quantization::F16 => QuantizationFormat::F16,
    };
    identity.storage.endianness = "little-endian".to_owned();
    let generation = ArtifactGenerationIdentityV1::new(sequence, [nonce_byte; 16])
        .expect("valid test generation");
    FsviV2IdentityBinding::new(
        generation,
        identity.freeze().expect("valid frozen test identity"),
    )
    .expect("valid FSVI v2 binding")
}

pub fn semantic_fsvi_v2_binding(
    model_id: &str,
    dimension: u32,
    sequence: u64,
    nonce_byte: u8,
) -> FsviV2IdentityBinding {
    let mut identity =
        frankensearch_core::generation::EmbeddingIdentityBundleV1::explicit_test_model(
            model_id, dimension,
        );
    identity.space.kind = EmbeddingSpaceKindV1::Semantic;
    identity.space.hash_control = None;
    identity.space.artifact_manifest_fingerprint = "1".repeat(64);
    identity.space.artifacts = vec![
        EmbeddingArtifactIdentityV1 {
            role: "weights".to_owned(),
            sha256: "2".repeat(64),
            size: 20,
        },
        EmbeddingArtifactIdentityV1 {
            role: "tokenizer".to_owned(),
            sha256: "3".repeat(64),
            size: 10,
        },
    ];
    identity.space.tokenizer_fingerprint = "3".repeat(64);
    identity.space.vocabulary_fingerprint = "4".repeat(64);
    identity.space.model_config_fingerprint = "5".repeat(64);
    identity.producer.space_fingerprint = identity.space.fingerprint();
    identity.storage.format = "fsvi-v2".to_owned();
    identity.storage.quantization = QuantizationFormat::F16;
    identity.storage.endianness = "little-endian".to_owned();
    let generation = ArtifactGenerationIdentityV1::new(sequence, [nonce_byte; 16])
        .expect("valid semantic test generation");
    FsviV2IdentityBinding::new(
        generation,
        identity.freeze().expect("valid semantic test identity"),
    )
    .expect("valid semantic FSVI v2 binding")
}

pub fn fsvi_v2_binding_with_input_variant(
    model_id: &str,
    dimension: u32,
    sequence: u64,
    nonce_byte: u8,
    chunking: &str,
) -> FsviV2IdentityBinding {
    let mut identity =
        frankensearch_core::generation::EmbeddingIdentityBundleV1::explicit_test_model(
            model_id, dimension,
        );
    identity.input.chunking = chunking.to_owned();
    identity.space.input_contract_fingerprint = identity.input.fingerprint();
    identity.producer.space_fingerprint = identity.space.fingerprint();
    identity.storage.format = "fsvi-v2".to_owned();
    identity.storage.quantization = QuantizationFormat::F16;
    identity.storage.endianness = "little-endian".to_owned();
    FsviV2IdentityBinding::new(
        ArtifactGenerationIdentityV1::new(sequence, [nonce_byte; 16])
            .expect("valid variant generation"),
        identity.freeze().expect("valid variant frozen identity"),
    )
    .expect("valid variant FSVI v2 binding")
}

pub fn refresh_v2_header_crc(bytes: &mut [u8]) {
    let header_size = usize::try_from(u32::from_le_bytes(
        bytes[6..10].try_into().expect("header-size field"),
    ))
    .expect("header size fits usize");
    let crc_offset = header_size.checked_sub(4).expect("CRC offset");
    let crc = crc32(&bytes[..crc_offset]);
    bytes[crc_offset..header_size].copy_from_slice(&crc.to_le_bytes());
}

pub fn assert_inspection_corrupted(path: &Path) {
    assert!(
        matches!(
            VectorIndex::inspect(path),
            Err(SearchError::IndexCorrupted { .. })
        ),
        "expected typed inspection to reject corrupted bytes at {}",
        path.display()
    );
}

pub fn admit_owned_v2_fixture(
    path: &Path,
    binding: &FsviV2IdentityBinding,
) -> Result<ValidatedFsviBytes, FsviAdmissionError> {
    let bytes = fs::read(path).map_err(SearchError::Io)?;
    ValidatedFsviBytes::from_arc(Arc::<[u8]>::from(bytes), binding)
}

pub fn assert_owned_admission_corrupted(path: &Path, binding: &FsviV2IdentityBinding) {
    assert!(
        matches!(
            admit_owned_v2_fixture(path, binding),
            Err(FsviAdmissionError::Index(
                SearchError::IndexCorrupted { .. }
            ))
        ),
        "expected admitted open to reject corrupted bytes at {}",
        path.display()
    );
}

pub fn assert_snapshot_rejection<T: std::fmt::Debug>(
    result: Result<T, FsviAdmissionError>,
    expected_reason: FsviSnapshotRejectionReason,
) {
    match result {
        Err(FsviAdmissionError::SnapshotRejected(rejected)) => {
            assert_eq!(rejected.reason, expected_reason);
            assert!(!rejected.detail.is_empty());
        }
        other => panic!("expected snapshot rejection {expected_reason:?}, observed {other:?}"),
    }
}

pub fn directory_entry_names(path: &Path) -> Vec<std::ffi::OsString> {
    let mut names: Vec<_> = fs::read_dir(path)
        .expect("read private fixture directory")
        .map(|entry| entry.expect("read fixture entry").file_name())
        .collect();
    names.sort_unstable();
    names
}

pub fn write_v2_fixture(
    name: &str,
    binding: FsviV2IdentityBinding,
) -> (PathBuf, FsviV2IdentityBinding) {
    let path = temp_index_path(name);
    let expected = binding.clone();
    let mut writer = VectorIndex::create_v2(&path, binding).expect("create v2 writer");
    writer
        .write_record("doc-beta", &[0.0, 1.0, 0.0, 0.0])
        .expect("write beta");
    writer
        .write_record("doc-alpha", &[1.0, 0.0, 0.0, 0.0])
        .expect("write alpha");
    writer.finish().expect("finish v2 fixture");
    (path, expected)
}
