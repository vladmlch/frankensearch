use std::fs;
use std::io::Read;
use std::path::Path;

use frankensearch_core::{SearchError, SearchResult};
use sha2::{Digest, Sha256};

use crate::{
    FsviAdmissionError, FsviSnapshotRejectionReason, Quantization, open_readonly_noatime_nofollow,
    snapshot_parent_or_current, snapshot_rejected, stable_file_identity,
    validate_single_link_regular_file,
};

use super::{
    BATCH_MAGIC, IDENTITY_BOUND_WAL_HEADER_SIZE, IDENTITY_BOUND_WAL_VERSION, StrictWalImage,
    StrictWalInspection, WAL_HEADER_SIZE, WAL_MAGIC, WAL_VERSION, WalIdentityHeaderV2, crc32_of,
    wal_corrupted,
};

/// Strictly inspect one WAL pathname without mutating bytes, timestamps, or
/// directory entries.
///
/// The present-file path uses the same no-atime/no-follow, single-link,
/// pre/post inode and containing-directory checks as immutable FSVI admission.
/// Crash-recovery tolerance is deliberately not applied: a truncated or corrupt
/// final batch is an error rather than an implicitly discarded suffix.
///
/// # Errors
///
/// Returns typed snapshot rejection for unsafe path topology or concurrent
/// mutation and [`SearchError::IndexCorrupted`] for every malformed WAL field.
pub fn inspect_wal_strict(path: &Path) -> Result<StrictWalInspection, FsviAdmissionError> {
    let parent = snapshot_parent_or_current(path);
    let parent_metadata = fs::symlink_metadata(parent).map_err(SearchError::Io)?;
    if !parent_metadata.file_type().is_dir() {
        return Err(snapshot_rejected(
            FsviSnapshotRejectionReason::DirectoryChangedDuringRead,
            "the WAL parent must be a real directory, not a symlink or special file",
        ));
    }
    let parent_identity = stable_file_identity(&parent_metadata);
    let path_metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let after = fs::symlink_metadata(parent).map_err(SearchError::Io)?;
            if !after.file_type().is_dir() || stable_file_identity(&after) != parent_identity {
                return Err(snapshot_rejected(
                    FsviSnapshotRejectionReason::DirectoryChangedDuringRead,
                    "the WAL parent changed while absence was being inspected",
                ));
            }
            return Ok(StrictWalInspection::Absent);
        }
        Err(error) => return Err(SearchError::Io(error).into()),
    };
    let path_identity = validate_single_link_regular_file(&path_metadata)?;
    let mut file = open_readonly_noatime_nofollow(path)?;
    let opened_identity =
        validate_single_link_regular_file(&file.metadata().map_err(SearchError::Io)?)?;
    if opened_identity != path_identity {
        return Err(snapshot_rejected(
            FsviSnapshotRejectionReason::PathChangedDuringRead,
            "the WAL pathname and opened descriptor identify different bytes",
        ));
    }
    let byte_len = usize::try_from(path_identity.size)
        .map_err(|_| wal_corrupted(path, "WAL byte length does not fit in usize"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(byte_len)
        .map_err(|_| SearchError::InvalidConfig {
            field: "wal_snapshot.byte_len".to_owned(),
            value: byte_len.to_string(),
            reason: "unable to reserve the exact immutable WAL byte image".to_owned(),
        })?;
    bytes.resize(byte_len, 0);
    file.read_exact(&mut bytes).map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            snapshot_rejected(
                FsviSnapshotRejectionReason::PathChangedDuringRead,
                "the WAL inode was truncated while being inspected",
            )
        } else {
            FsviAdmissionError::Index(SearchError::Io(error))
        }
    })?;
    let mut trailing = [0_u8; 1];
    if file.read(&mut trailing).map_err(SearchError::Io)? != 0 {
        return Err(snapshot_rejected(
            FsviSnapshotRejectionReason::PathChangedDuringRead,
            "the WAL inode grew while being inspected",
        ));
    }
    let descriptor_identity =
        validate_single_link_regular_file(&file.metadata().map_err(SearchError::Io)?)?;
    let final_path_metadata = fs::symlink_metadata(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            snapshot_rejected(
                FsviSnapshotRejectionReason::PathChangedDuringRead,
                "the WAL pathname disappeared while being inspected",
            )
        } else {
            FsviAdmissionError::Index(SearchError::Io(error))
        }
    })?;
    let final_path_identity = validate_single_link_regular_file(&final_path_metadata)?;
    if descriptor_identity != path_identity || final_path_identity != path_identity {
        return Err(snapshot_rejected(
            FsviSnapshotRejectionReason::PathChangedDuringRead,
            "the WAL inode identity, size, mode, links, or timestamps changed during inspection",
        ));
    }
    let final_parent = fs::symlink_metadata(parent).map_err(SearchError::Io)?;
    if !final_parent.file_type().is_dir() || stable_file_identity(&final_parent) != parent_identity
    {
        return Err(snapshot_rejected(
            FsviSnapshotRejectionReason::DirectoryChangedDuringRead,
            "the WAL containing directory changed during inspection",
        ));
    }
    parse_strict_wal_bytes(path, &bytes).map_err(FsviAdmissionError::Index)
}

fn parse_strict_wal_bytes(path: &Path, data: &[u8]) -> SearchResult<StrictWalInspection> {
    if data.is_empty() {
        return Ok(StrictWalInspection::Empty {
            whole_image_sha256: Sha256::digest(data).into(),
        });
    }
    if data.len() < 6 {
        return Err(wal_corrupted(path, "WAL magic/version prefix is truncated"));
    }
    if data[..4] != WAL_MAGIC {
        return Err(wal_corrupted(path, "bad magic bytes"));
    }
    let version = u16::from_le_bytes([data[4], data[5]]);
    let (cursor, dimension, quantization, format) = match version {
        WAL_VERSION => {
            if data.len() < WAL_HEADER_SIZE {
                return Err(wal_corrupted(path, "legacy WAL header is truncated"));
            }
            let dimension = u32::from_le_bytes([data[6], data[7], data[8], data[9]]);
            if dimension == 0 {
                return Err(wal_corrupted(
                    path,
                    "legacy WAL dimension must be greater than zero",
                ));
            }
            let quantization = Quantization::from_wire(data[10], path)?;
            if data[12..16] != [0; 4] {
                return Err(wal_corrupted(
                    path,
                    "legacy WAL reserved bytes must be zero",
                ));
            }
            let stored_crc = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
            if stored_crc != crc32_of(&data[..16]) {
                return Err(wal_corrupted(path, "legacy WAL header CRC mismatch"));
            }
            (
                WAL_HEADER_SIZE,
                dimension,
                quantization,
                StrictWalFormat::Legacy,
            )
        }
        IDENTITY_BOUND_WAL_VERSION => {
            let header = WalIdentityHeaderV2::decode(data, path)?;
            let dimension = header.dimension();
            let quantization = header.quantization();
            (
                IDENTITY_BOUND_WAL_HEADER_SIZE,
                dimension,
                quantization,
                StrictWalFormat::IdentityBound(header),
            )
        }
        found => {
            return Err(wal_corrupted(
                path,
                format!("unsupported WAL version {found}"),
            ));
        }
    };
    let dimension_usize = usize::try_from(dimension)
        .map_err(|_| wal_corrupted(path, "WAL dimension does not fit in usize"))?;
    let vector_bytes = dimension_usize
        .checked_mul(quantization.bytes_per_element())
        .ok_or_else(|| wal_corrupted(path, "WAL vector byte length overflow"))?;
    let (batch_count, entry_count) = inspect_complete_batches(path, &data[cursor..], vector_bytes)?;
    let image = StrictWalImage {
        byte_len: u64::try_from(data.len())
            .map_err(|_| wal_corrupted(path, "WAL byte length does not fit in u64"))?,
        whole_image_sha256: Sha256::digest(data).into(),
        dimension,
        quantization,
        batch_count,
        entry_count,
    };
    Ok(match format {
        StrictWalFormat::Legacy => StrictWalInspection::LegacyV1(image),
        StrictWalFormat::IdentityBound(header) => {
            StrictWalInspection::IdentityBoundV2 { header, image }
        }
    })
}

enum StrictWalFormat {
    Legacy,
    IdentityBound(WalIdentityHeaderV2),
}

fn inspect_complete_batches(
    path: &Path,
    data: &[u8],
    vector_bytes: usize,
) -> SearchResult<(u64, u64)> {
    let mut cursor = 0usize;
    let mut batch_count = 0_u64;
    let mut total_entries = 0_u64;
    while cursor < data.len() {
        let remaining = &data[cursor..];
        if remaining.len() < 8 {
            return Err(wal_corrupted(path, "WAL batch header is truncated"));
        }
        if remaining[..4] != BATCH_MAGIC {
            return Err(wal_corrupted(path, "WAL batch magic mismatch"));
        }
        let entry_count =
            u32::from_le_bytes([remaining[4], remaining[5], remaining[6], remaining[7]]);
        let mut batch_cursor = 8usize;
        for _ in 0..entry_count {
            let len_end = batch_cursor
                .checked_add(2)
                .ok_or_else(|| wal_corrupted(path, "WAL entry length offset overflow"))?;
            if len_end > remaining.len() {
                return Err(wal_corrupted(path, "WAL entry length is truncated"));
            }
            let doc_id_len = usize::from(u16::from_le_bytes([
                remaining[batch_cursor],
                remaining[batch_cursor + 1],
            ]));
            batch_cursor = len_end;
            let doc_end = batch_cursor
                .checked_add(doc_id_len)
                .ok_or_else(|| wal_corrupted(path, "WAL document id offset overflow"))?;
            let vector_end = doc_end
                .checked_add(vector_bytes)
                .ok_or_else(|| wal_corrupted(path, "WAL vector offset overflow"))?;
            if vector_end > remaining.len() {
                return Err(wal_corrupted(path, "WAL entry payload is truncated"));
            }
            std::str::from_utf8(&remaining[batch_cursor..doc_end]).map_err(|error| {
                wal_corrupted(path, format!("WAL document id is not UTF-8: {error}"))
            })?;
            batch_cursor = vector_end;
        }
        let crc_end = batch_cursor
            .checked_add(4)
            .ok_or_else(|| wal_corrupted(path, "WAL batch CRC offset overflow"))?;
        if crc_end > remaining.len() {
            return Err(wal_corrupted(path, "WAL batch CRC is truncated"));
        }
        let stored_crc = u32::from_le_bytes(
            remaining[batch_cursor..crc_end]
                .try_into()
                .map_err(|_| wal_corrupted(path, "WAL batch CRC is truncated"))?,
        );
        if stored_crc != crc32_of(&remaining[..batch_cursor]) {
            return Err(wal_corrupted(path, "WAL batch CRC mismatch"));
        }
        cursor = cursor
            .checked_add(crc_end)
            .ok_or_else(|| wal_corrupted(path, "WAL batch cursor overflow"))?;
        batch_count = batch_count
            .checked_add(1)
            .ok_or_else(|| wal_corrupted(path, "WAL batch count overflow"))?;
        total_entries = total_entries
            .checked_add(u64::from(entry_count))
            .ok_or_else(|| wal_corrupted(path, "WAL entry count overflow"))?;
    }
    Ok((batch_count, total_entries))
}
