use std::fs;
#[cfg(unix)]
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crc32fast::Hasher as Crc32;
use frankensearch_core::generation::QuantizationFormat;
use frankensearch_core::{SearchError, SearchResult};
use half::f16;
use sha2::{Digest, Sha256};

use super::mutations::PendingRecord;
#[cfg(test)]
use super::types::take_f16_destination_reservation_failure;
use super::types::{
    FSVI_MAGIC, FSVI_V2_FIXED_PREFIX_BYTES, FSVI_V2_IDENTITY_BINDING_SCHEMA,
    FSVI_V2_MAX_HEADER_BYTES, FSVI_V2_VERSION, FSVI_VERSION, FsviV2IdentityBinding,
    ORDERED_DOCSET_DIGEST_DOMAIN, Quantization, RECORD_FLAG_TOMBSTONE, SHA256_BYTES,
    VECTOR_CONTENT_DIGEST_DOMAIN,
};

pub(super) fn fsvi_v2_header_len(binding: &FsviV2IdentityBinding) -> SearchResult<usize> {
    FSVI_V2_FIXED_PREFIX_BYTES
        .checked_add(binding.frozen_identity.canonical_bytes.len())
        .and_then(|length| length.checked_add(binding.space_canonical_bytes.len()))
        .and_then(|length| length.checked_add(binding.storage_canonical_bytes.len()))
        .and_then(|length| length.checked_add(4))
        .filter(|length| *length <= FSVI_V2_MAX_HEADER_BYTES)
        .ok_or_else(|| {
            fsvi_v2_config_error(
                "header_size",
                "canonical identity payloads overflow the bounded v2 header",
            )
        })
}

pub(super) fn build_v2_header(
    binding: &FsviV2IdentityBinding,
    publication_nonce: u16,
    record_count: usize,
    vectors_offset: u64,
    ordered_docset_digest: [u8; SHA256_BYTES],
    vector_content_digest: [u8; SHA256_BYTES],
) -> SearchResult<Vec<u8>> {
    let header_len = fsvi_v2_header_len(binding)?;
    let header_size_u32 = u32::try_from(header_len)
        .map_err(|_| fsvi_v2_config_error("header_size", "must fit in the v2 u32 header field"))?;
    let dimension_u32 = u32::try_from(binding.dimension)
        .map_err(|_| fsvi_v2_config_error("dimension", "must fit in u32"))?;
    let record_count_u64 = u64::try_from(record_count)
        .map_err(|_| fsvi_v2_config_error("record_count", "must fit in u64"))?;
    let bundle_len = u32::try_from(binding.frozen_identity.canonical_bytes.len())
        .map_err(|_| fsvi_v2_config_error("identity_bundle_len", "must fit in u32"))?;
    let space_len = u32::try_from(binding.space_canonical_bytes.len())
        .map_err(|_| fsvi_v2_config_error("space_identity_len", "must fit in u32"))?;
    let storage_len = u32::try_from(binding.storage_canonical_bytes.len())
        .map_err(|_| fsvi_v2_config_error("storage_identity_len", "must fit in u32"))?;

    let mut header = Vec::with_capacity(header_len);
    header.extend_from_slice(&FSVI_MAGIC);
    header.extend_from_slice(&FSVI_V2_VERSION.to_le_bytes());
    header.extend_from_slice(&header_size_u32.to_le_bytes());
    header.extend_from_slice(&FSVI_V2_IDENTITY_BINDING_SCHEMA.to_le_bytes());
    header.push(binding.quantization as u8);
    header.push(0);
    // Cross-tier publication nonce (bd-miio8), mirroring the v1 reserved
    // bytes: every tier of one multi-tier v2 publication carries the same
    // non-zero value; zero marks a single-tier or pre-nonce file.
    header.extend_from_slice(&publication_nonce.to_le_bytes());
    header.extend_from_slice(&dimension_u32.to_le_bytes());
    header.extend_from_slice(&record_count_u64.to_le_bytes());
    header.extend_from_slice(&vectors_offset.to_le_bytes());
    header.extend_from_slice(&binding.generation.schema_version.to_le_bytes());
    header.extend_from_slice(&0_u16.to_le_bytes());
    header.extend_from_slice(&binding.generation.sequence.to_le_bytes());
    header.extend_from_slice(&binding.generation.nonce);
    header.extend_from_slice(&bundle_len.to_le_bytes());
    header.extend_from_slice(&space_len.to_le_bytes());
    header.extend_from_slice(&storage_len.to_le_bytes());
    header.extend_from_slice(&binding.bundle_fingerprint);
    header.extend_from_slice(&binding.space_fingerprint);
    header.extend_from_slice(&binding.producer_fingerprint);
    header.extend_from_slice(&binding.input_fingerprint);
    header.extend_from_slice(&binding.storage_fingerprint);
    header.extend_from_slice(&binding.generation_fingerprint);
    header.extend_from_slice(&ordered_docset_digest);
    header.extend_from_slice(&vector_content_digest);
    header.extend_from_slice(&binding.frozen_identity.canonical_bytes);
    header.extend_from_slice(&binding.space_canonical_bytes);
    header.extend_from_slice(&binding.storage_canonical_bytes);
    let crc = crc32(&header);
    header.extend_from_slice(&crc.to_le_bytes());
    if header.len() != header_len {
        return Err(fsvi_v2_config_error(
            "header_size",
            "encoded bytes disagree with the precomputed v2 header size",
        ));
    }
    Ok(header)
}

pub(super) fn read_array<const N: usize>(
    path: &Path,
    data: &[u8],
    cursor: &mut usize,
    field: &str,
) -> SearchResult<[u8; N]> {
    let slice = read_slice(path, data, cursor, N, field)?;
    let mut out = [0_u8; N];
    out.copy_from_slice(slice);
    Ok(out)
}

pub(super) fn read_slice<'a>(
    path: &Path,
    data: &'a [u8],
    cursor: &mut usize,
    len: usize,
    field: &str,
) -> SearchResult<&'a [u8]> {
    let end = cursor
        .checked_add(len)
        .ok_or_else(|| index_corrupted(path, format!("{field} offset overflow")))?;
    if end > data.len() {
        return Err(index_corrupted(
            path,
            format!("{field} is truncated (wanted {len} bytes)"),
        ));
    }
    let out = &data[*cursor..end];
    *cursor = end;
    Ok(out)
}

pub(super) fn build_header_prefix(
    embedder_id: &str,
    embedder_revision: &str,
    dimension: usize,
    quantization: Quantization,
    compaction_gen: u8,
    publication_nonce: u16,
    record_count: usize,
    vectors_offset: u64,
) -> SearchResult<Vec<u8>> {
    validate_header_string(embedder_id, "embedder_id")?;
    validate_header_string(embedder_revision, "embedder_revision")?;
    let dimension_u32 = u32::try_from(dimension).map_err(|_| SearchError::InvalidConfig {
        field: "dimension".to_owned(),
        value: dimension.to_string(),
        reason: "dimension must fit in u32".to_owned(),
    })?;
    let record_count_u64 = u64::try_from(record_count).map_err(|_| SearchError::InvalidConfig {
        field: "record_count".to_owned(),
        value: record_count.to_string(),
        reason: "record_count must fit in u64".to_owned(),
    })?;
    let mut out = Vec::with_capacity(
        4 + 2 + 2 + embedder_id.len() + 2 + embedder_revision.len() + 4 + 1 + 3 + 8 + 8,
    );
    out.extend_from_slice(&FSVI_MAGIC);
    out.extend_from_slice(&FSVI_VERSION.to_le_bytes());
    out.extend_from_slice(
        &u16::try_from(embedder_id.len())
            .map_err(|_| SearchError::InvalidConfig {
                field: "embedder_id".to_owned(),
                value: embedder_id.to_owned(),
                reason: "embedder_id byte length must fit in u16".to_owned(),
            })?
            .to_le_bytes(),
    );
    out.extend_from_slice(embedder_id.as_bytes());
    out.extend_from_slice(
        &u16::try_from(embedder_revision.len())
            .map_err(|_| SearchError::InvalidConfig {
                field: "embedder_revision".to_owned(),
                value: embedder_revision.to_owned(),
                reason: "embedder_revision byte length must fit in u16".to_owned(),
            })?
            .to_le_bytes(),
    );
    out.extend_from_slice(embedder_revision.as_bytes());
    out.extend_from_slice(&dimension_u32.to_le_bytes());
    out.push(quantization as u8);
    out.push(compaction_gen);
    out.extend_from_slice(&publication_nonce.to_le_bytes());
    out.extend_from_slice(&record_count_u64.to_le_bytes());
    out.extend_from_slice(&vectors_offset.to_le_bytes());
    Ok(out)
}

pub fn validate_header_string(value: &str, field: &str) -> SearchResult<()> {
    if value.is_empty() && field == "embedder_id" {
        return Err(SearchError::InvalidConfig {
            field: field.to_owned(),
            value: value.to_owned(),
            reason: "embedder_id cannot be empty".to_owned(),
        });
    }
    let _ = u16::try_from(value.len()).map_err(|_| SearchError::InvalidConfig {
        field: field.to_owned(),
        value: value.to_owned(),
        reason: "value length must fit in u16".to_owned(),
    })?;
    Ok(())
}

pub(super) fn quantization_from_identity(
    quantization: QuantizationFormat,
) -> SearchResult<Quantization> {
    match quantization {
        QuantizationFormat::F32 => Ok(Quantization::F32),
        QuantizationFormat::F16 => Ok(Quantization::F16),
        QuantizationFormat::Int8 | QuantizationFormat::Int4 => Err(fsvi_v2_config_error(
            "storage.quantization",
            "FSVI v2 vector slabs currently support only F32 or F16",
        )),
    }
}

pub(super) fn fsvi_v2_config_error(field: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("fsvi_v2.{field}"),
        value: "redacted".to_owned(),
        reason: reason.to_owned(),
    }
}

pub(super) fn sha256_bytes(bytes: &[u8]) -> [u8; SHA256_BYTES] {
    Sha256::digest(bytes).into()
}

pub fn fingerprint_hex(fingerprint: &[u8; SHA256_BYTES]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(SHA256_BYTES * 2);
    for byte in fingerprint {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

pub(super) fn decode_sha256_fingerprint(
    field: &str,
    value: &str,
) -> SearchResult<[u8; SHA256_BYTES]> {
    if value.len() != SHA256_BYTES * 2 {
        return Err(fsvi_v2_config_error(
            field,
            "must be a lowercase 64-character SHA-256 fingerprint",
        ));
    }
    let mut decoded = [0_u8; SHA256_BYTES];
    for (output, pair) in decoded.iter_mut().zip(value.as_bytes().as_chunks::<2>().0) {
        let high = decode_lower_hex_nibble(pair[0]).ok_or_else(|| {
            fsvi_v2_config_error(
                field,
                "must be a lowercase 64-character SHA-256 fingerprint",
            )
        })?;
        let low = decode_lower_hex_nibble(pair[1]).ok_or_else(|| {
            fsvi_v2_config_error(
                field,
                "must be a lowercase 64-character SHA-256 fingerprint",
            )
        })?;
        *output = (high << 4) | low;
    }
    Ok(decoded)
}

pub(super) fn decode_sha256_fingerprint_from_index(
    path: &Path,
    field: &str,
    value: &str,
) -> SearchResult<[u8; SHA256_BYTES]> {
    decode_sha256_fingerprint(field, value).map_err(|_| {
        index_corrupted(
            path,
            format!("v2 {field} fingerprint is not lowercase SHA-256"),
        )
    })
}

const fn decode_lower_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

pub fn update_digest_domain(hasher: &mut Sha256, domain: &[u8]) {
    hasher.update(
        u64::try_from(domain.len())
            .unwrap_or(u64::MAX)
            .to_be_bytes(),
    );
    hasher.update(domain);
}

pub(super) fn ordered_docset_digest(records: &[PendingRecord]) -> [u8; SHA256_BYTES] {
    let mut hasher = Sha256::new();
    update_digest_domain(&mut hasher, ORDERED_DOCSET_DIGEST_DOMAIN);
    let live_count = records
        .iter()
        .filter(|record| !is_tombstoned_flags(record.flags))
        .count();
    hasher.update(u64::try_from(live_count).unwrap_or(u64::MAX).to_be_bytes());
    for record in records
        .iter()
        .filter(|record| !is_tombstoned_flags(record.flags))
    {
        hasher.update(
            u64::try_from(record.doc_id.len())
                .unwrap_or(u64::MAX)
                .to_be_bytes(),
        );
        hasher.update(record.doc_id.as_bytes());
    }
    hasher.finalize().into()
}

fn vector_content_hasher(
    record_count: usize,
    dimension: usize,
    quantization: Quantization,
) -> SearchResult<Sha256> {
    let mut hasher = Sha256::new();
    update_digest_domain(&mut hasher, VECTOR_CONTENT_DIGEST_DOMAIN);
    hasher.update(
        u64::try_from(record_count)
            .map_err(|_| fsvi_v2_config_error("record_count", "must fit in u64"))?
            .to_be_bytes(),
    );
    hasher.update(
        u64::try_from(dimension)
            .map_err(|_| fsvi_v2_config_error("dimension", "must fit in u64"))?
            .to_be_bytes(),
    );
    hasher.update([quantization as u8]);
    Ok(hasher)
}

pub(super) fn write_vector_slab_v2<W: Write>(
    writer: &mut W,
    records: &[PendingRecord],
    dimension: usize,
    quantization: Quantization,
) -> SearchResult<[u8; SHA256_BYTES]> {
    let mut hasher = vector_content_hasher(records.len(), dimension, quantization)?;
    let element_bytes = quantization.bytes_per_element();
    let record_bytes = dimension
        .checked_mul(element_bytes)
        .ok_or_else(|| fsvi_v2_config_error("vector_slab", "record byte length overflow"))?;
    let mut encoded = Vec::with_capacity(record_bytes);
    match quantization {
        Quantization::F16 => {
            let mut scratch = Vec::<f16>::with_capacity(dimension);
            for record in records {
                scratch.clear();
                crate::simd::encode_f32_to_f16_extend(&record.embedding, &mut scratch);
                encoded.clear();
                for value in &scratch {
                    encoded.extend_from_slice(&value.to_le_bytes());
                }
                hasher.update(&encoded);
                writer.write_all(&encoded)?;
            }
        }
        Quantization::F32 => {
            for record in records {
                encoded.clear();
                for value in &record.embedding {
                    encoded.extend_from_slice(&value.to_le_bytes());
                }
                hasher.update(&encoded);
                writer.write_all(&encoded)?;
            }
        }
    }
    Ok(hasher.finalize().into())
}

pub(super) fn vector_content_digest_from_bytes(
    record_count: usize,
    dimension: usize,
    quantization: Quantization,
    vector_bytes: &[u8],
) -> SearchResult<[u8; SHA256_BYTES]> {
    let expected_len = record_count
        .checked_mul(dimension)
        .and_then(|count| count.checked_mul(quantization.bytes_per_element()))
        .ok_or_else(|| fsvi_v2_config_error("vector_slab", "byte length overflow"))?;
    if vector_bytes.len() != expected_len {
        return Err(fsvi_v2_config_error(
            "vector_slab",
            "bytes disagree with the declared shape",
        ));
    }
    let mut hasher = vector_content_hasher(record_count, dimension, quantization)?;
    hasher.update(vector_bytes);
    Ok(hasher.finalize().into())
}

pub(super) fn write_vector_slab<W: Write>(
    writer: &mut W,
    records: &[PendingRecord],
    quantization: Quantization,
) -> SearchResult<()> {
    match quantization {
        Quantization::F16 => {
            // Encode each record's f32 → f16 via the F16C-dispatched kernel and
            // write the whole record's bytes in one `write_all`, instead of a
            // per-element `from_f32` + 2-byte `write_all` (which was ~38M tiny
            // writes for a 100k×384 index). The on-disk format is LE f16.
            let dim = records.first().map_or(0, |r| r.embedding.len());
            let mut scratch: Vec<f16> = Vec::with_capacity(dim);
            for record in records {
                scratch.clear();
                crate::simd::encode_f32_to_f16_extend(&record.embedding, &mut scratch);
                #[cfg(target_endian = "little")]
                {
                    // SAFETY: `half::f16` is `repr(transparent)` over `u16`; on a
                    // little-endian target the native bytes equal `to_le_bytes`, so
                    // the slab is byte-identical to the per-element path.
                    #[allow(unsafe_code)]
                    let bytes = unsafe {
                        core::slice::from_raw_parts(
                            scratch.as_ptr().cast::<u8>(),
                            scratch.len() * 2,
                        )
                    };
                    writer.write_all(bytes)?;
                }
                #[cfg(not(target_endian = "little"))]
                {
                    for &h in &scratch {
                        writer.write_all(&h.to_le_bytes())?;
                    }
                }
            }
        }
        Quantization::F32 => {
            for record in records {
                for value in &record.embedding {
                    writer.write_all(&value.to_le_bytes())?;
                }
            }
        }
    }
    Ok(())
}

pub fn align_up(value: u64, alignment: u64) -> SearchResult<u64> {
    if alignment == 0 {
        return Ok(value);
    }
    let add = alignment
        .checked_sub(1)
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "alignment".to_owned(),
            value: alignment.to_string(),
            reason: "alignment underflow".to_owned(),
        })?;
    let padded = value
        .checked_add(add)
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "alignment".to_owned(),
            value: format!("{value}+{add}"),
            reason: "alignment overflow".to_owned(),
        })?;
    Ok((padded / alignment) * alignment)
}

pub fn temporary_output_path(path: &Path) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let pid = std::process::id();
    let mut os = path.as_os_str().to_os_string();
    os.push(format!(".tmp.{pid}.{now}"));
    PathBuf::from(os)
}

#[cfg(unix)]
pub fn sync_parent_directory(path: &Path) -> SearchResult<()> {
    if let Some(parent) = path.parent() {
        let dir = File::open(parent)?;
        dir.sync_all()?;
    }
    Ok(())
}

/// Drop the durability crate's `<fsvi>.fec` repair sidecar after the main
/// file's bytes change (compaction, vacuum, `finish`, in-place tombstones).
///
/// The sidecar is a `RaptorQ` encoding of the exact bytes it was built from;
/// a stale one would let a later repair "restore" the previous contents and
/// silently discard everything written since (observed: a watcher's shutdown
/// compaction followed by `fsfs compact` dropped a merged record). The file
/// owner drops it here; publishers re-protect after they finish writing.
pub(super) fn remove_durability_sidecar(fsvi_path: &Path) {
    let mut sidecar = fsvi_path.as_os_str().to_os_string();
    sidecar.push(".fec");
    let sidecar = PathBuf::from(sidecar);
    match fs::remove_file(&sidecar) {
        Ok(()) => tracing::debug!(
            path = %sidecar.display(),
            "durability sidecar removed after the FSVI bytes changed"
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => tracing::warn!(
            path = %sidecar.display(),
            error = %error,
            "stale durability sidecar could not be removed; a later repair would restore old bytes"
        ),
    }
}

pub(super) fn map_lock_error(
    path: &Path,
    requested_access: &str,
    error: &dyn std::fmt::Display,
) -> SearchError {
    SearchError::InvalidConfig {
        field: "fsvi.map_lock".to_owned(),
        value: path.display().to_string(),
        reason: format!(
            "cannot acquire {requested_access} lock before mapping this FSVI: {error}; drop live readers/writers before retrying"
        ),
    }
}

fn f16_destination_allocation_error(dimension: usize) -> SearchError {
    SearchError::InvalidConfig {
        field: "vectors.f16_destination".to_owned(),
        value: dimension.to_string(),
        reason: "destination allocation failed".to_owned(),
    }
}

pub fn reserve_f16_destination(destination: &mut Vec<f16>, dimension: usize) -> SearchResult<()> {
    #[cfg(test)]
    if take_f16_destination_reservation_failure() {
        return Err(f16_destination_allocation_error(dimension));
    }
    destination
        .try_reserve_exact(dimension)
        .map_err(|_| f16_destination_allocation_error(dimension))
}

pub fn index_corrupted(path: &Path, detail: impl Into<String>) -> SearchError {
    SearchError::IndexCorrupted {
        path: path.to_path_buf(),
        detail: detail.into(),
    }
}

pub fn crc32(data: &[u8]) -> u32 {
    let mut hasher = Crc32::new();
    hasher.update(data);
    hasher.finalize()
}

pub fn fnv1a_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3_u64);
    }
    hash
}

/// Whether a vector can contribute retrieval signal: every component is
/// finite and the squared norm is non-zero and finite. An all-zero vector
/// scores 0.0 against everything, so ranking it is arbitrary tie-breaking;
/// a norm that overflows to infinity poisons downstream scores.
pub(super) fn vector_signal_usable(vector: &[f32]) -> bool {
    let mut norm_sq = 0.0f32;
    for &value in vector {
        if !value.is_finite() {
            return false;
        }
        norm_sq += value * value;
    }
    norm_sq > 0.0 && norm_sq.is_finite()
}

pub const fn is_tombstoned_flags(flags: u16) -> bool {
    flags & RECORD_FLAG_TOMBSTONE != 0
}

/// Successor in the wrapping compaction-generation sequence.
///
/// Zero is reserved for legacy/identity-v2 writers, so the wrap skips it. A
/// WAL sidecar binds to main generation `G` iff its own tag equals
/// `next_generation(G)`; a replacement built with
/// `next_generation(destination)` therefore atomically invalidates the
/// destination's sidecar at install time.
#[must_use]
pub const fn next_generation(current: u8) -> u8 {
    if current == 255 { 1 } else { current + 1 }
}

/// Successor in the wrapping publication-nonce sequence (bd-miio8).
///
/// Zero is reserved for legacy/single-tier files, so the wrap skips it.
/// Choosing the successor of the destination pair's current nonce guarantees
/// successive multi-tier publications never share an identity, which is what
/// makes a crash between per-tier installs detectable at open time.
#[must_use]
pub const fn next_publication_nonce(current: u16) -> u16 {
    if current == u16::MAX { 1 } else { current + 1 }
}
