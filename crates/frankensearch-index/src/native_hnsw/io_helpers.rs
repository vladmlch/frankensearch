use std::fs::File;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

use frankensearch_core::{
    error::{SearchError, SearchResult},
    sha256_checksum,
};

use super::{NATIVE_HNSW_MAX_BASENAME_BYTES, SHA256_BYTES, SHA256_HEX_LEN};

pub(super) fn canonical_graph_basename(graph_path: &Path) -> SearchResult<String> {
    let basename = graph_path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .ok_or_else(|| {
            native_hnsw_receipt_config_error(
                "graph_path",
                "redacted-invalid-basename",
                "must end in one canonical UTF-8 basename",
            )
        })?
        .to_owned();
    validate_graph_basename(&basename)?;
    Ok(basename)
}

pub(super) fn validate_graph_basename(basename: &str) -> SearchResult<()> {
    if basename.is_empty() || basename.len() > NATIVE_HNSW_MAX_BASENAME_BYTES {
        return Err(native_hnsw_receipt_config_error(
            "graph_basename",
            "redacted-invalid-length",
            "must be non-empty and leave room for the adjacent receipt suffix",
        ));
    }
    if basename.chars().any(char::is_control) {
        return Err(native_hnsw_receipt_config_error(
            "graph_basename",
            "redacted-control-character",
            "must not contain control characters",
        ));
    }
    let mut components = Path::new(basename).components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(native_hnsw_receipt_config_error(
            "graph_basename",
            "redacted-noncanonical",
            "must contain exactly one normal path component",
        ));
    }
    if Path::new(basename)
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        != Some("fshnsw")
    {
        return Err(native_hnsw_receipt_config_error(
            "graph_basename",
            basename,
            "must use the canonical .fshnsw extension",
        ));
    }
    Ok(())
}

pub(super) fn validate_native_hnsw_graph_path(graph_path: &Path) -> SearchResult<()> {
    if graph_path.as_os_str().is_empty() {
        return Err(native_hnsw_receipt_config_error(
            "graph_path",
            "redacted-empty",
            "must not be empty",
        ));
    }
    if graph_path
        .components()
        .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        return Err(native_hnsw_receipt_config_error(
            "graph_path",
            "redacted-dot-relative",
            "must not contain '.' or '..' components",
        ));
    }
    let _ = canonical_graph_basename(graph_path)?;
    Ok(())
}

pub(super) fn reject_symlink_ancestors(path: &Path) -> SearchResult<()> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    let mut current = PathBuf::new();
    for component in parent.components() {
        current.push(component.as_os_str());
        if matches!(component, Component::Prefix(_) | Component::RootDir) {
            continue;
        }
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(SearchError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "native HNSW path ancestor '{}' must not be a symbolic link",
                        current.display()
                    ),
                )));
            }
            Ok(metadata) if metadata.file_type().is_dir() => {}
            Ok(_) => {
                return Err(SearchError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "native HNSW path ancestor '{}' must be a directory",
                        current.display()
                    ),
                )));
            }
            // ubs:ignore — this compares a public I/O error enum, not secret material.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(SearchError::Io(error)),
        }
    }
    Ok(())
}

pub(super) fn read_regular_file_bytes(path: &Path, kind: &str) -> SearchResult<Vec<u8>> {
    let path_metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        // ubs:ignore — this compares a public I/O error enum, not secret material.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(SearchError::IndexNotFound {
                path: path.to_path_buf(),
            });
        }
        Err(error) => return Err(SearchError::Io(error)),
    };
    if !path_metadata.file_type().is_file() {
        return Err(SearchError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "{kind} '{}' must be a regular file, not a symlink or special file",
                path.display()
            ),
        )));
    }
    let mut file = File::open(path).map_err(SearchError::Io)?;
    let opened_metadata = file.metadata().map_err(SearchError::Io)?;
    ensure_same_open_file(&path_metadata, &opened_metadata).map_err(SearchError::Io)?;
    let expected_len = opened_metadata.len();
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(usize::try_from(expected_len).map_err(|_| {
            SearchError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{kind} '{}' is too large for this platform", path.display()),
            ))
        })?)
        .map_err(|error| {
            SearchError::Io(std::io::Error::other(format!(
                "could not allocate {expected_len} bytes for {kind} '{}': {error}",
                path.display()
            )))
        })?;
    file.read_to_end(&mut bytes).map_err(SearchError::Io)?;
    if u64::try_from(bytes.len()).ok() != Some(expected_len) {
        return Err(SearchError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!(
                "{kind} '{}' changed length while being read",
                path.display()
            ),
        )));
    }
    let final_path_metadata = std::fs::symlink_metadata(path).map_err(SearchError::Io)?;
    ensure_same_open_file(&final_path_metadata, &opened_metadata).map_err(SearchError::Io)?;
    Ok(bytes)
}

pub(super) fn persist_native_hnsw_receipt(path: &Path, bytes: &[u8]) -> SearchResult<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent).map_err(SearchError::Io)?;
    reject_non_regular_receipt_destination(path)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(SearchError::Io)?;
    temporary.write_all(bytes).map_err(SearchError::Io)?;
    temporary.as_file().sync_all().map_err(SearchError::Io)?;
    temporary.persist(path).map_err(|error| {
        SearchError::Io(std::io::Error::new(
            error.error.kind(),
            format!(
                "failed to atomically publish native HNSW generation receipt '{}': {}",
                path.display(),
                error.error
            ),
        ))
    })?;
    #[cfg(unix)]
    sync_parent_directory(path)?;
    Ok(())
}

pub(super) fn reject_non_regular_receipt_destination(path: &Path) -> SearchResult<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => Ok(()),
        Ok(_) => Err(SearchError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "native HNSW generation receipt destination '{}' must be a regular file, not a \
                 symlink or special file",
                path.display()
            ),
        ))),
        // ubs:ignore — this compares a public I/O error enum, not secret material.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(SearchError::Io(error)),
    }
}

pub(super) fn append_canonical_bytes(destination: &mut Vec<u8>, value: &[u8]) -> SearchResult<()> {
    destination
        .extend_from_slice(&usize_to_u64(value.len(), "topology.domain_length")?.to_be_bytes());
    destination.extend_from_slice(value);
    Ok(())
}

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    let checksum = sha256_checksum(bytes);
    checksum
        .strip_prefix("sha256:")
        .unwrap_or(checksum.as_str())
        .to_owned()
}

pub(super) fn validate_sha256_hex(field: &str, value: &str) -> SearchResult<()> {
    if value.len() == SHA256_HEX_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Ok(());
    }
    Err(native_hnsw_receipt_config_error(
        field,
        "redacted-invalid-sha256",
        "must be a lowercase 64-character SHA-256 digest",
    ))
}

pub(super) fn decode_sha256_hex(field: &str, value: &str) -> SearchResult<[u8; SHA256_BYTES]> {
    validate_sha256_hex(field, value)?;
    let mut decoded = [0_u8; SHA256_BYTES];
    let (pairs, remainder) = value.as_bytes().as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(native_hnsw_receipt_config_error(
            field,
            "redacted-invalid-sha256",
            "must be a lowercase 64-character SHA-256 digest",
        ));
    }
    for (index, pair) in pairs.iter().enumerate() {
        let (Some(high), Some(low)) = (
            decode_lower_hex_nibble(pair[0]),
            decode_lower_hex_nibble(pair[1]),
        ) else {
            return Err(native_hnsw_receipt_config_error(
                field,
                "redacted-invalid-sha256",
                "must be a lowercase 64-character SHA-256 digest",
            ));
        };
        decoded[index] = (high << 4) | low;
    }
    Ok(decoded)
}

pub(super) const fn decode_lower_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

pub(super) fn encode_lower_hex(bytes: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bytes = bytes.as_ref();
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

pub(super) fn native_hnsw_receipt_config_error(
    field: &str,
    value: &str,
    reason: &str,
) -> SearchError {
    let bounded_value = if value.len() <= 128 && !value.chars().any(char::is_control) {
        value.to_owned()
    } else {
        "redacted".to_owned()
    };
    SearchError::InvalidConfig {
        field: format!("native_hnsw_receipt.{field}"),
        value: bounded_value,
        reason: reason.to_owned(),
    }
}

pub(super) fn native_hnsw_receipt_corrupted(path: &Path, detail: impl Into<String>) -> SearchError {
    SearchError::IndexCorrupted {
        path: path.to_path_buf(),
        detail: detail.into(),
    }
}

pub(super) fn receipt_usize(value: u64, field: &str) -> SearchResult<usize> {
    usize::try_from(value).map_err(|_| {
        native_hnsw_receipt_config_error(
            field,
            &value.to_string(),
            "does not fit this platform's usize",
        )
    })
}

pub(super) fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

pub(super) fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

pub(super) fn get_u32(bytes: &[u8], offset: usize) -> u32 {
    let mut value = [0_u8; 4];
    value.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_le_bytes(value)
}

pub(super) fn get_u64(bytes: &[u8], offset: usize) -> u64 {
    let mut value = [0_u8; 8];
    value.copy_from_slice(&bytes[offset..offset + 8]);
    u64::from_le_bytes(value)
}

pub(super) fn usize_to_u64(value: usize, field: &str) -> SearchResult<u64> {
    u64::try_from(value).map_err(|_| SearchError::InvalidConfig {
        field: format!("native_hnsw_{field}"),
        value: value.to_string(),
        reason: "value must fit in the persisted u64 field".to_owned(),
    })
}

pub(super) fn persisted_usize(path: &Path, value: u64, field: &str) -> SearchResult<usize> {
    usize::try_from(value).map_err(|_| {
        native_hnsw_corrupted(
            path,
            format!("native HNSW {field} value {value} does not fit this platform"),
        )
    })
}

pub(super) fn native_hnsw_corrupted(path: &Path, detail: impl Into<String>) -> SearchError {
    SearchError::IndexCorrupted {
        path: path.to_path_buf(),
        detail: detail.into(),
    }
}

pub(super) fn write_payload_u32(
    file: &mut File,
    hasher: &mut crc32fast::Hasher,
    payload_len: &mut u64,
    value: u32,
) -> SearchResult<()> {
    let next_len = payload_len
        .checked_add(4)
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "native_hnsw_payload_len".to_owned(),
            value: payload_len.to_string(),
            reason: "payload length overflow".to_owned(),
        })?;
    let bytes = value.to_le_bytes();
    file.write_all(&bytes).map_err(SearchError::Io)?;
    hasher.update(&bytes);
    *payload_len = next_len;
    Ok(())
}

pub(super) fn reject_non_regular_destination(path: &Path) -> SearchResult<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => Ok(()),
        Ok(_) => Err(SearchError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "native HNSW destination '{}' must be a regular file, not a symlink or special \
                 file",
                path.display()
            ),
        ))),
        // ubs:ignore — this compares a public I/O error enum, not secret material.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(SearchError::Io(error)),
    }
}

pub(super) fn open_regular_file(path: &Path) -> SearchResult<File> {
    let path_metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        // ubs:ignore — this compares a public I/O error enum, not secret material.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(SearchError::IndexNotFound {
                path: path.to_path_buf(),
            });
        }
        Err(error) => return Err(SearchError::Io(error)),
    };
    if !path_metadata.file_type().is_file() {
        return Err(SearchError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "native HNSW artifact '{}' must be a regular file, not a symlink or special file",
                path.display()
            ),
        )));
    }
    let file = File::open(path).map_err(SearchError::Io)?;
    ensure_same_open_file(&path_metadata, &file.metadata().map_err(SearchError::Io)?)
        .map_err(SearchError::Io)?;
    Ok(file)
}

#[cfg(unix)]
fn ensure_same_open_file(
    path_metadata: &std::fs::Metadata,
    opened_metadata: &std::fs::Metadata,
) -> std::io::Result<()> {
    use std::os::unix::fs::MetadataExt;

    if path_metadata.dev() != opened_metadata.dev() || path_metadata.ino() != opened_metadata.ino()
    {
        return Err(std::io::Error::other(
            "native HNSW artifact changed while it was being opened",
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn ensure_same_open_file(
    _: &std::fs::Metadata,
    opened_metadata: &std::fs::Metadata,
) -> std::io::Result<()> {
    if !opened_metadata.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "native HNSW artifact must be a regular file",
        ));
    }
    Ok(())
}

#[cfg(unix)]
pub(super) fn sync_parent_directory(path: &Path) -> SearchResult<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(SearchError::Io)
}
