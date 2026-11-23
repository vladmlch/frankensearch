use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use frankensearch_core::{SearchError, SearchResult};
use hnsw_rs::prelude::{DistDot, HnswIo};

use super::fingerprint::{
    ann_corrupted, fingerprint_doc_ids, fingerprint_hnsw_sidecar, fingerprint_source_map,
};
use super::topology::validate_hnsw_topology;
use super::types::{
    HNSW_GENERATION_RECEIPT_FILENAME, HNSW_GENERATION_RECEIPT_MAX_BYTES,
    HNSW_GENERATION_RECEIPT_VERSION, HNSW_META_FORMAT_CURRENT, HNSW_SAVE_LOCK_DIRECTORY,
    HnswConfig, HnswGenerationReceipt, HnswIndex, HnswMeta, HnswSourceIdentityV1,
    ValidatedHnswGeneration,
};

/// Derive the `hnsw_rs` sidecar basename from the metadata path.
///
/// `hnsw_rs` writes `{basename}.hnsw.graph` and `{basename}.hnsw.data`, so for
/// a metadata path of `dir/vector.fast.hnsw` we use the stem `vector.fast`,
/// yielding `dir/vector.fast.hnsw.graph` / `.data` — distinct from the
/// metadata file itself.
pub(super) fn hnsw_sidecar_basename(path: &Path) -> SearchResult<String> {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| ann_corrupted(path, "ANN sidecar path has no usable file stem"))
}

pub(super) fn hnsw_metadata_file_name(path: &Path) -> SearchResult<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| ann_corrupted(path, "ANN metadata path has no usable UTF-8 file name"))
}

pub(super) fn hnsw_generation_prefix(requested_basename: &str, vector_fingerprint: u64) -> String {
    format!(".{requested_basename}.generation-{vector_fingerprint:016x}-")
}

pub(super) fn hnsw_save_lock_path(path: &Path) -> SearchResult<PathBuf> {
    if path.components().any(|component| {
        component
            .as_os_str()
            .to_str()
            .is_some_and(|name| name.eq_ignore_ascii_case(HNSW_SAVE_LOCK_DIRECTORY))
    }) {
        return Err(ann_corrupted(
            path,
            format!(
                "ANN metadata paths cannot be inside the reserved '{HNSW_SAVE_LOCK_DIRECTORY}' directory"
            ),
        ));
    }
    let file_name = path
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| ann_corrupted(path, "ANN metadata path has no usable file name"))?;
    let mut lock_name = file_name.to_os_string();
    lock_name.push(".lock");
    let parent = path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    Ok(parent.join(HNSW_SAVE_LOCK_DIRECTORY).join(lock_name))
}

pub fn hnsw_save_lock_artifact_path(path: &Path) -> SearchResult<PathBuf> {
    hnsw_save_lock_path(path)
}

pub fn materialize_hnsw_save_lock_artifact(lock_path: &Path) -> SearchResult<PathBuf> {
    let lock_directory = lock_path
        .parent()
        .ok_or_else(|| ann_corrupted(lock_path, "HNSW save lock has no parent directory"))?;
    let artifact_parent = lock_directory
        .parent()
        .ok_or_else(|| ann_corrupted(lock_path, "HNSW save lock directory has no parent"))?;
    std::fs::create_dir_all(artifact_parent)?;
    match std::fs::create_dir(lock_directory) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => {
            return Err(SearchError::Io(std::io::Error::new(
                error.kind(),
                format!(
                    "failed to create persistent HNSW save-lock directory '{}': {error}",
                    lock_directory.display()
                ),
            )));
        }
    }
    let lock_directory_metadata = std::fs::symlink_metadata(lock_directory).map_err(|error| {
        SearchError::Io(std::io::Error::new(
            error.kind(),
            format!(
                "failed to inspect persistent HNSW save-lock directory '{}': {error}",
                lock_directory.display()
            ),
        ))
    })?;
    if lock_directory_metadata.file_type().is_symlink() || !lock_directory_metadata.is_dir() {
        return Err(SearchError::Io(std::io::Error::other(format!(
            "persistent HNSW save-lock directory '{}' is not a local directory",
            lock_directory.display()
        ))));
    }
    match std::fs::symlink_metadata(lock_path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
            return Err(SearchError::Io(std::io::Error::other(format!(
                "persistent HNSW save lock '{}' is not a local regular file",
                lock_path.display()
            ))));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(SearchError::Io(error)),
    }
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock_path)
        .map_err(|error| {
            SearchError::Io(std::io::Error::new(
                error.kind(),
                format!(
                    "failed to open persistent HNSW save lock '{}': {error}",
                    lock_path.display()
                ),
            ))
        })?;
    lock.sync_all().map_err(SearchError::Io)?;
    sync_hnsw_directory(lock_directory)?;
    std::fs::canonicalize(lock_path).map_err(SearchError::Io)
}

pub(super) fn acquire_hnsw_save_lock(path: &Path) -> SearchResult<std::fs::File> {
    let lock_path = hnsw_save_lock_path(path)?;
    let _identity = materialize_hnsw_save_lock_artifact(&lock_path)?;
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|error| {
            SearchError::Io(std::io::Error::new(
                error.kind(),
                format!(
                    "failed to open persistent HNSW save lock '{}': {error}",
                    lock_path.display()
                ),
            ))
        })?;
    lock.try_lock().map_err(|error| {
        let error: std::io::Error = error.into();
        SearchError::Io(std::io::Error::new(
            error.kind(),
            format!(
                "failed to acquire HNSW save lock '{}': {error}; another writer may be saving",
                lock_path.display()
            ),
        ))
    })?;
    Ok(lock)
}

pub(super) fn serialize_hnsw_metadata(meta: &HnswMeta) -> SearchResult<Vec<u8>> {
    serde_json::to_vec(meta)
        .map_err(|error| SearchError::Io(std::io::Error::other(error.to_string())))
}

pub(super) fn find_reusable_hnsw_generation(
    index: &HnswIndex,
    metadata_path: &Path,
    parent: &Path,
    requested_basename: &str,
    metadata_file_name: &str,
) -> SearchResult<Option<HnswMeta>> {
    let prefix = hnsw_generation_prefix(requested_basename, index.vector_fingerprint);
    let mut candidates = Vec::new();
    for entry in std::fs::read_dir(parent).map_err(SearchError::Io)? {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                tracing::debug!(
                    path = %metadata_path.display(),
                    ?error,
                    "ignoring unreadable HNSW generation directory entry"
                );
                continue;
            }
        };
        let file_name = entry.file_name();
        if file_name
            .to_str()
            .is_some_and(|name| name.starts_with(&prefix))
        {
            candidates.push(entry.path());
        }
    }
    candidates.sort_unstable();

    for generation in candidates {
        match reusable_hnsw_generation(index, metadata_path, &generation, metadata_file_name) {
            Ok(Some(meta)) => return Ok(Some(meta)),
            Ok(None) => {}
            Err(error) => {
                tracing::debug!(
                    path = %metadata_path.display(),
                    generation = %generation.display(),
                    ?error,
                    "ignoring invalid HNSW READY generation"
                );
            }
        }
    }
    Ok(None)
}

fn reusable_hnsw_generation(
    index: &HnswIndex,
    metadata_path: &Path,
    generation_path: &Path,
    metadata_file_name: &str,
) -> SearchResult<Option<HnswMeta>> {
    let Some(validated) = validate_hnsw_generation_receipt(
        metadata_path,
        generation_path,
        metadata_file_name,
        &index.doc_ids,
        index.vector_fingerprint,
        fingerprint_source_map(&index.source_positions, index.source_record_count),
        index.dimension,
        index.config,
    )?
    else {
        return Ok(None);
    };

    // PUBLICATION-TIME IDENTITY RE-VALIDATION (bd-21zyj). Everything the
    // receipt validator just checked is CONTENT — doc-id fingerprint, vector
    // fingerprint, dimension, params. A READY generation left on disk by a
    // DIFFERENT published FSVI generation can match every one of them, because
    // two generations may hold byte-identical live vectors under identical doc
    // ids (a re-publication carries a new nonce; a model revision bump moves
    // the space fingerprint without moving a vector). Adopting it here would
    // publish metadata that names THIS graph's identity while pointing at that
    // other generation's directory — an identity claim the bytes do not
    // support. Re-validate before reuse, failing closed in both directions
    // exactly as the load path does.
    if !HnswSourceIdentityV1::admits(
        validated.source_identity.as_ref(),
        index.source_identity.as_ref(),
    ) {
        tracing::debug!(
            path = %metadata_path.display(),
            generation = %generation_path.display(),
            receipt_bound = validated.source_identity.is_some(),
            graph_bound = index.source_identity.is_some(),
            "ignoring HNSW READY generation dumped from a different source generation"
        );
        return Ok(None);
    }

    // Byte identity alone is not enough: a complete pair can still be
    // semantically unreadable after a native-format change or a faulty dump.
    // Republishing such a receipt would make every fallback rebuild select the
    // same broken generation again. Prove that the current reader can load the
    // pair before treating it as reusable.
    if !hnsw_generation_is_loadable(
        generation_path,
        &validated.basename,
        &validated.graph,
        index.doc_ids.len(),
        index.dimension,
    ) {
        tracing::debug!(
            path = %metadata_path.display(),
            generation = %generation_path.display(),
            "ignoring digest-valid but unloadable HNSW READY generation"
        );
        return Ok(None);
    }

    Ok(Some(index.metadata_for_generation(
        &validated.generation,
        &validated.basename,
    )))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_hnsw_generation_receipt(
    metadata_path: &Path,
    generation_path: &Path,
    metadata_file_name: &str,
    doc_ids: &[String],
    vector_fingerprint: u64,
    source_map_fingerprint: u64,
    dimension: usize,
    config: HnswConfig,
) -> SearchResult<Option<ValidatedHnswGeneration>> {
    let Some(generation_name) = generation_path.file_name().and_then(|name| name.to_str()) else {
        return Ok(None);
    };
    let generation_name = validate_hnsw_sidecar_basename(metadata_path, generation_name)?;
    let receipt_path = generation_path.join(HNSW_GENERATION_RECEIPT_FILENAME);
    let Ok(generation_metadata) = std::fs::symlink_metadata(generation_path) else {
        return Ok(None);
    };
    let Ok(receipt_metadata) = std::fs::symlink_metadata(&receipt_path) else {
        return Ok(None);
    };
    if generation_metadata.file_type().is_symlink()
        || !generation_metadata.is_dir()
        || receipt_metadata.file_type().is_symlink()
        || !receipt_metadata.is_file()
    {
        return Ok(None);
    }
    let Ok(receipt_len) = usize::try_from(receipt_metadata.len()) else {
        return Ok(None);
    };
    if receipt_len > HNSW_GENERATION_RECEIPT_MAX_BYTES {
        return Ok(None);
    }

    let receipt_file = std::fs::File::open(&receipt_path).map_err(SearchError::Io)?;
    let opened_receipt_metadata = receipt_file.metadata().map_err(SearchError::Io)?;
    if !opened_receipt_metadata.is_file() || opened_receipt_metadata.len() != receipt_metadata.len()
    {
        return Ok(None);
    }
    let receipt_read_limit = u64::try_from(HNSW_GENERATION_RECEIPT_MAX_BYTES)
        .map_err(|_| SearchError::Io(std::io::Error::other("HNSW receipt limit exceeds u64")))?
        .saturating_add(1);
    let mut receipt_bytes = Vec::with_capacity(receipt_len);
    receipt_file
        .take(receipt_read_limit)
        .read_to_end(&mut receipt_bytes)
        .map_err(SearchError::Io)?;
    if receipt_bytes.len() > HNSW_GENERATION_RECEIPT_MAX_BYTES {
        return Ok(None);
    }
    let receipt: HnswGenerationReceipt =
        serde_json::from_slice(&receipt_bytes).map_err(|error| {
            ann_corrupted(
                metadata_path,
                format!("failed to parse HNSW generation receipt: {error}"),
            )
        })?;

    if receipt.receipt_version != HNSW_GENERATION_RECEIPT_VERSION
        || receipt.metadata_file_name != metadata_file_name
        || receipt.format_version != HNSW_META_FORMAT_CURRENT
        || receipt.generation.as_str().ne(generation_name.as_str())
        || receipt.doc_count != doc_ids.len()
        || receipt.doc_ids_fingerprint != fingerprint_doc_ids(doc_ids)
        || receipt.vector_fingerprint != vector_fingerprint
        || receipt.source_map_fingerprint != source_map_fingerprint
        || receipt.dimension != dimension
        || receipt.config != config
    {
        return Ok(None);
    }

    let basename = validate_hnsw_sidecar_basename(metadata_path, &receipt.sidecar_basename)?;
    let graph = generation_path.join(format!("{basename}.hnsw.graph"));
    let data = generation_path.join(format!("{basename}.hnsw.data"));
    if !native_sidecar_pair_is_local(metadata_path, generation_path, &graph, &data) {
        return Ok(None);
    }
    if fingerprint_hnsw_sidecar(&graph)? != receipt.graph
        || fingerprint_hnsw_sidecar(&data)? != receipt.data
    {
        return Ok(None);
    }

    Ok(Some(ValidatedHnswGeneration {
        generation: generation_name,
        basename,
        source_identity: receipt.source_identity.clone(),
        graph,
    }))
}

fn hnsw_generation_is_loadable(
    generation_path: &Path,
    basename: &str,
    graph_path: &Path,
    expected_points: usize,
    expected_dimension: usize,
) -> bool {
    let Ok(graph_file) = std::fs::File::open(graph_path) else {
        return false;
    };
    let mut graph_reader = std::io::BufReader::new(graph_file);
    let description = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        hnsw_rs::prelude::load_description(&mut graph_reader)
    }));
    let Ok(Ok(description)) = description else {
        return false;
    };
    if description.nb_point != expected_points
        || (expected_points != 0 && description.dimension != expected_dimension)
    {
        return false;
    }

    // `hnsw_rs` currently unwraps a few native-parser results internally. Keep
    // corrupt retained generations on the normal reject-and-redump path rather
    // than letting a validation-only reuse probe unwind through `save()`.
    let mut native_io = HnswIo::new(generation_path, basename);
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Ok(candidate) = native_io.load_hnsw::<f32, DistDot>() else {
            return false;
        };
        candidate.get_nb_point() == expected_points
            && validate_hnsw_topology(&candidate, expected_points).is_ok()
    }))
    .unwrap_or(false)
}

pub(super) fn write_hnsw_generation_receipt(
    generation_path: &Path,
    receipt: &HnswGenerationReceipt,
) -> SearchResult<()> {
    let bytes = serde_json::to_vec(receipt)
        .map_err(|error| SearchError::Io(std::io::Error::other(error.to_string())))?;
    if bytes.len() > HNSW_GENERATION_RECEIPT_MAX_BYTES {
        return Err(SearchError::Io(std::io::Error::other(format!(
            "HNSW generation receipt exceeds {} bytes",
            HNSW_GENERATION_RECEIPT_MAX_BYTES
        ))));
    }

    let receipt_path = generation_path.join(HNSW_GENERATION_RECEIPT_FILENAME);
    let mut receipt_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&receipt_path)
        .map_err(SearchError::Io)?;
    receipt_file.write_all(&bytes).map_err(SearchError::Io)?;
    receipt_file.sync_all().map_err(SearchError::Io)?;
    sync_hnsw_directory(generation_path)
}

pub(super) fn persisted_hnsw_sidecar_location(
    path: &Path,
    meta: &HnswMeta,
) -> SearchResult<(PathBuf, String)> {
    let parent = path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let generation = meta
        .sidecar_generation
        .as_deref()
        .ok_or_else(|| ann_corrupted(path, "current HNSW metadata has no sidecar generation"))?;
    let generation = validate_hnsw_sidecar_basename(path, generation)?;
    let basename = meta
        .sidecar_basename
        .as_deref()
        .ok_or_else(|| ann_corrupted(path, "current HNSW metadata has no sidecar basename"))?;
    let basename = validate_hnsw_sidecar_basename(path, basename)?;
    Ok((parent.join(generation), basename))
}

pub(super) fn validate_hnsw_sidecar_basename(path: &Path, basename: &str) -> SearchResult<String> {
    let candidate = Path::new(basename);
    if candidate.file_name() != Some(candidate.as_os_str()) {
        return Err(ann_corrupted(
            path,
            "HNSW native sidecar basename must be one non-empty path component",
        ));
    }
    Ok(basename.to_owned())
}

pub(super) fn native_sidecar_pair_is_local(
    metadata_path: &Path,
    generation: &Path,
    graph: &Path,
    data: &Path,
) -> bool {
    let metadata_parent = metadata_path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let Ok(canonical_parent) = std::fs::canonicalize(metadata_parent) else {
        return false;
    };

    let Ok(generation_metadata) = std::fs::symlink_metadata(generation) else {
        return false;
    };
    if generation_metadata.file_type().is_symlink() || !generation_metadata.is_dir() {
        return false;
    }
    let Ok(canonical_generation) = std::fs::canonicalize(generation) else {
        return false;
    };
    if canonical_generation.parent() != Some(canonical_parent.as_path()) {
        return false;
    }

    [graph, data].into_iter().all(|sidecar| {
        let Ok(sidecar_metadata) = std::fs::symlink_metadata(sidecar) else {
            return false;
        };
        if sidecar_metadata.file_type().is_symlink() || !sidecar_metadata.is_file() {
            return false;
        }
        std::fs::canonicalize(sidecar).is_ok_and(|canonical_sidecar| {
            canonical_sidecar.parent() == Some(canonical_generation.as_path())
        })
    })
}

pub(super) fn sync_hnsw_sidecars(parent: &Path, basename: &str) -> SearchResult<()> {
    for suffix in [".hnsw.graph", ".hnsw.data"] {
        let sidecar_path = parent.join(format!("{basename}{suffix}"));
        let sidecar = std::fs::File::open(&sidecar_path).map_err(|error| {
            SearchError::Io(std::io::Error::new(
                error.kind(),
                format!(
                    "failed to open dumped HNSW sidecar '{}': {error}",
                    sidecar_path.display()
                ),
            ))
        })?;
        sidecar.sync_all().map_err(|error| {
            SearchError::Io(std::io::Error::new(
                error.kind(),
                format!(
                    "failed to sync dumped HNSW sidecar '{}': {error}",
                    sidecar_path.display()
                ),
            ))
        })?;
    }
    sync_hnsw_directory(parent)?;
    Ok(())
}

pub(super) fn sync_hnsw_directory(directory: &Path) -> SearchResult<()> {
    #[cfg(unix)]
    {
        let handle = std::fs::File::open(directory).map_err(|error| {
            SearchError::Io(std::io::Error::new(
                error.kind(),
                format!(
                    "failed to open HNSW parent directory '{}': {error}",
                    directory.display()
                ),
            ))
        })?;
        handle.sync_all().map_err(|error| {
            SearchError::Io(std::io::Error::new(
                error.kind(),
                format!(
                    "failed to sync HNSW parent directory '{}': {error}",
                    directory.display()
                ),
            ))
        })?;
    }
    #[cfg(not(unix))]
    {
        let _ = directory;
    }
    Ok(())
}

pub(super) fn publish_hnsw_metadata(path: &Path, parent: &Path, bytes: &[u8]) -> SearchResult<()> {
    install_hnsw_metadata(path, parent, bytes)?;
    sync_hnsw_directory(parent)
}

/// Remove stale native HNSW generations after metadata has durably selected a
/// replacement.
///
/// The save lock is held by the caller for the complete publication and
/// cleanup sequence.  A failed or interrupted metadata publication never
/// reaches this function, preserving READY generations for a retry.  Only
/// sibling directories in this metadata role's private generation namespace
/// are eligible; symlinks and unrelated filesystem entries are left intact.
pub(super) fn gc_superseded_hnsw_generations(
    parent: &Path,
    requested_basename: &str,
    published_metadata: &HnswMeta,
) -> SearchResult<()> {
    let retained_generation = published_metadata
        .sidecar_generation
        .as_deref()
        .ok_or_else(|| ann_corrupted(parent, "published HNSW metadata has no generation"))?;
    let generation_prefix = format!(".{requested_basename}.generation-");

    for entry in std::fs::read_dir(parent).map_err(SearchError::Io)? {
        let entry = entry.map_err(SearchError::Io)?;
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            continue;
        };
        if file_name == retained_generation || !file_name.starts_with(&generation_prefix) {
            continue;
        }

        let file_type = entry.file_type().map_err(SearchError::Io)?;
        if file_type.is_symlink() || !file_type.is_dir() {
            continue;
        }

        std::fs::remove_dir_all(entry.path()).map_err(|error| {
            SearchError::Io(std::io::Error::new(
                error.kind(),
                format!(
                    "failed to remove superseded HNSW generation '{}': {error}",
                    entry.path().display()
                ),
            ))
        })?;
    }

    sync_hnsw_directory(parent)
}

pub(super) fn install_hnsw_metadata(path: &Path, parent: &Path, bytes: &[u8]) -> SearchResult<()> {
    let mut temporary_builder = tempfile::Builder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        // Keep the atomically persisted metadata consistent with ordinary
        // index files: 0666 constrained by the deployment's umask, rather
        // than tempfile's private 0600 default.
        temporary_builder.permissions(std::fs::Permissions::from_mode(0o666));
    }
    let mut temporary = temporary_builder
        .tempfile_in(parent)
        .map_err(SearchError::Io)?;
    temporary.write_all(bytes).map_err(SearchError::Io)?;
    temporary.as_file().sync_all().map_err(SearchError::Io)?;
    temporary.persist(path).map_err(|error| {
        SearchError::Io(std::io::Error::new(
            error.error.kind(),
            format!(
                "failed to atomically publish HNSW metadata '{}': {}",
                path.display(),
                error.error
            ),
        ))
    })?;
    Ok(())
}
