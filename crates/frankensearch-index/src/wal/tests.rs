use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, QuantizationFormat,
};

use super::*;

mod core;
mod strict_and_locking;

pub(super) fn temp_wal_path(name: &str) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "frankensearch-wal-{name}-{}-{now}.fsvi.wal",
        std::process::id()
    ))
}

pub(super) fn make_entry(doc_id: &str, base: f32, dim: usize) -> WalEntry {
    WalEntry {
        doc_id: doc_id.into(),
        doc_id_hash: crate::fnv1a_hash(doc_id.as_bytes()),
        embedding: vec![base; dim],
    }
}

pub(super) fn identity_binding_v2(
    dimension: u32,
    quantization: QuantizationFormat,
) -> WalIdentityBindingV2 {
    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("wal-identity-v2", dimension);
    identity.storage.format = "fsvi-v2".to_owned();
    identity.storage.quantization = quantization;
    identity.storage.endianness = "little-endian".to_owned();
    let frozen = identity.freeze().expect("valid FSVI v2 identity");
    let generation =
        ArtifactGenerationIdentityV1::new(u64::MAX, [0xa5; 16]).expect("valid generation");
    WalIdentityBindingV2::new(generation, &frozen).expect("valid WAL identity binding")
}

pub(super) fn refresh_identity_header_crc(bytes: &mut [u8; IDENTITY_BOUND_WAL_HEADER_SIZE]) {
    let crc = crc32_of(&bytes[..IDENTITY_BOUND_WAL_CRC_OFFSET]);
    bytes[IDENTITY_BOUND_WAL_CRC_OFFSET..].copy_from_slice(&crc.to_le_bytes());
}
