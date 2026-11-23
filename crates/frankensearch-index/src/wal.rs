//! Write-ahead log for incremental FSVI index updates.
//!
//! New vectors are appended to a `.fsvi.wal` sidecar file in batches.
//! Each batch is CRC32-protected so partial writes from crashes are
//! detected and discarded on reload.
//!
//! # File Layout
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │ WAL Header (20 bytes)               │
//! │   magic: b"FWAL" (4)               │
//! │   version: u16 LE                  │
//! │   dimension: u32 LE                │
//! │   quantization: u8                 │
//! │   reserved: [u8; 5]               │
//! │   header_crc32: u32 LE             │
//! ├─────────────────────────────────────┤
//! │ Batch 0                             │
//! │   batch_magic: b"FWB1" (4)         │
//! │   entry_count: u32 LE              │
//! │   entries...                        │
//! │   batch_crc32: u32 LE              │
//! ├─────────────────────────────────────┤
//! │ Batch 1 ...                         │
//! └─────────────────────────────────────┘
//! ```

use std::path::{Path, PathBuf};

use crc32fast::Hasher as Crc32Hasher;
use frankensearch_core::{SearchError, SearchResult};

use crate::Quantization;

mod identity;
mod io;
mod strict;
#[cfg(test)]
mod tests;

pub use self::identity::{WalIdentityBindingV2, WalIdentityHeaderV2};
pub use self::strict::inspect_wal_strict;

pub(crate) fn read_wal(
    path: &Path,
    expected_dimension: usize,
    quantization: Quantization,
) -> SearchResult<(Vec<WalEntry>, u8, u64)> {
    io::read_wal(path, expected_dimension, quantization)
}

pub(crate) fn read_wal_recovering_torn_header(
    path: &Path,
    expected_dimension: usize,
    quantization: Quantization,
) -> SearchResult<(Vec<WalEntry>, u8, u64)> {
    io::read_wal_recovering_torn_header(path, expected_dimension, quantization)
}

pub(crate) fn append_wal_batch(
    wal_path: &Path,
    entries: &[WalEntry],
    dimension: usize,
    quantization: Quantization,
    compaction_gen: u8,
    fsync: bool,
) -> SearchResult<()> {
    io::append_wal_batch(
        wal_path,
        entries,
        dimension,
        quantization,
        compaction_gen,
        fsync,
    )
}

pub(crate) fn remove_wal(path: &Path) -> SearchResult<()> {
    io::remove_wal(path)
}

// ─── Constants ──────────────────────────────────────────────────────────────

const WAL_MAGIC: [u8; 4] = *b"FWAL";
const WAL_VERSION: u16 = 1;
const BATCH_MAGIC: [u8; 4] = *b"FWB1";
/// Minimum WAL file size (header only).
const WAL_HEADER_SIZE: usize = 20;

/// Version of the WAL header that binds an exact FSVI v2 generation and
/// embedding identity.
pub const IDENTITY_BOUND_WAL_VERSION: u16 = 2;
/// Fixed byte length of an identity-bound WAL v2 header.
pub const IDENTITY_BOUND_WAL_HEADER_SIZE: usize = 208;

const IDENTITY_BOUND_WAL_HEADER_SIZE_U16: u16 = 208;
const SHA256_FINGERPRINT_BYTES: usize = 32;
const IDENTITY_BUNDLE_FINGERPRINT_OFFSET: usize = 44;
const SPACE_FINGERPRINT_OFFSET: usize =
    IDENTITY_BUNDLE_FINGERPRINT_OFFSET + SHA256_FINGERPRINT_BYTES;
const PRODUCER_FINGERPRINT_OFFSET: usize = SPACE_FINGERPRINT_OFFSET + SHA256_FINGERPRINT_BYTES;
const INPUT_FINGERPRINT_OFFSET: usize = PRODUCER_FINGERPRINT_OFFSET + SHA256_FINGERPRINT_BYTES;
const STORAGE_FINGERPRINT_OFFSET: usize = INPUT_FINGERPRINT_OFFSET + SHA256_FINGERPRINT_BYTES;
const IDENTITY_BOUND_WAL_CRC_OFFSET: usize = STORAGE_FINGERPRINT_OFFSET + SHA256_FINGERPRINT_BYTES;

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for WAL-based incremental updates.
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Maximum WAL entries before compaction is recommended.
    pub compaction_threshold: usize,
    /// Compaction threshold as fraction of main index size.
    pub compaction_ratio: f64,
    /// Whether to fsync after each batch write.
    pub fsync_on_write: bool,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            compaction_threshold: 1000,
            compaction_ratio: 0.10,
            fsync_on_write: true,
        }
    }
}

// ─── Types ──────────────────────────────────────────────────────────────────

/// A single WAL entry (in-memory representation).
#[derive(Debug, Clone)]
pub(crate) struct WalEntry {
    pub doc_id: String,
    /// Stored for future dedup/lookup optimizations during compaction.
    #[allow(dead_code)]
    pub doc_id_hash: u64,
    pub embedding: Vec<f32>,
}

/// Statistics from a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactionStats {
    /// Records in the main index before compaction.
    pub main_records_before: usize,
    /// Records from the WAL merged in.
    pub wal_records: usize,
    /// Total records in the compacted index.
    pub total_records_after: usize,
    /// Time taken in milliseconds.
    pub elapsed_ms: f64,
}

/// Strict, side-effect-free classification of a WAL pathname.
///
/// Unlike crash-recovery loading, this diagnostic never truncates, removes,
/// repairs, or silently discards a corrupt suffix. Every present byte must form
/// one complete valid header/batch sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StrictWalInspection {
    /// No directory entry existed for the WAL path during the stable snapshot.
    Absent,
    /// A zero-length regular file existed. This is staging state only and is
    /// still forbidden beside a published FSVI v2 generation.
    Empty {
        /// SHA-256 of the empty byte image.
        whole_image_sha256: [u8; 32],
    },
    /// A complete legacy WAL v1 byte image.
    LegacyV1(StrictWalImage),
    /// A complete identity-bound WAL v2 byte image.
    IdentityBoundV2 {
        /// Exact validated identity header.
        header: WalIdentityHeaderV2,
        /// Redacted byte/batch witness.
        image: StrictWalImage,
    },
}

/// Redacted strict WAL byte/batch witness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrictWalImage {
    /// Exact byte length.
    pub byte_len: u64,
    /// SHA-256 of every WAL byte.
    pub whole_image_sha256: [u8; 32],
    /// Stored vector dimension.
    pub dimension: u32,
    /// Stored quantization.
    pub quantization: Quantization,
    /// Number of complete batches.
    pub batch_count: u64,
    /// Total entries across complete batches.
    pub entry_count: u64,
}

// ─── Index tagging ──────────────────────────────────────────────────────────

/// Sentinel bit marking WAL-sourced entries in the search heap.
/// Uses the MSB of usize so main index positions (always < 2^63) are unaffected.
pub(crate) const WAL_INDEX_BIT: usize = 1_usize << (usize::BITS - 1);

pub(crate) const fn is_wal_index(index: usize) -> bool {
    index & WAL_INDEX_BIT != 0
}

pub(crate) const fn to_wal_index(wal_pos: usize) -> usize {
    wal_pos | WAL_INDEX_BIT
}

pub(crate) const fn from_wal_index(tagged: usize) -> usize {
    tagged & !WAL_INDEX_BIT
}

// ─── Path helpers ───────────────────────────────────────────────────────────

/// Derive the WAL sidecar path from the main FSVI index path.
#[must_use]
pub fn wal_path_for(fsvi_path: &Path) -> PathBuf {
    let mut p = fsvi_path.as_os_str().to_os_string();
    p.push(".wal");
    PathBuf::from(p)
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn crc32_of(data: &[u8]) -> u32 {
    let mut hasher = Crc32Hasher::new();
    hasher.update(data);
    hasher.finalize()
}

const fn quantization_to_wire(quantization: Quantization) -> u8 {
    match quantization {
        Quantization::F32 => 0,
        Quantization::F16 => 1,
    }
}

fn copy_array<const N: usize>(bytes: &[u8], path: &Path, field: &str) -> SearchResult<[u8; N]> {
    bytes
        .try_into()
        .map_err(|_| wal_corrupted(path, format!("identity-bound WAL {field} is truncated")))
}

fn decode_sha256_fingerprint(field: &str, fingerprint: &str) -> SearchResult<[u8; 32]> {
    if fingerprint.len() != 64 {
        return Err(wal_identity_config_error(
            field,
            "must be a 64-character lowercase SHA-256 fingerprint",
        ));
    }
    let mut decoded = [0; 32];
    for (output, pair) in decoded
        .iter_mut()
        .zip(fingerprint.as_bytes().as_chunks::<2>().0)
    {
        let high = decode_lower_hex_nibble(pair[0]).ok_or_else(|| {
            wal_identity_config_error(
                field,
                "must be a 64-character lowercase SHA-256 fingerprint",
            )
        })?;
        let low = decode_lower_hex_nibble(pair[1]).ok_or_else(|| {
            wal_identity_config_error(
                field,
                "must be a 64-character lowercase SHA-256 fingerprint",
            )
        })?;
        *output = (high << 4) | low;
    }
    Ok(decoded)
}

const fn decode_lower_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn wal_identity_config_error(field: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("wal_identity_v2.{field}"),
        value: "redacted".to_owned(),
        reason: reason.to_owned(),
    }
}

fn wal_corrupted(path: &Path, detail: impl Into<String>) -> SearchError {
    SearchError::IndexCorrupted {
        path: path.to_path_buf(),
        detail: detail.into(),
    }
}
