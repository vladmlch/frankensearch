use std::fs::{self, OpenOptions};
use std::io::{Seek, Write};
use std::path::{Path, PathBuf};

use frankensearch_core::{SearchError, SearchResult};
use half::f16;
use tracing::{debug, warn};

use crate::{Quantization, snapshot_parent_or_current};

use super::{
    BATCH_MAGIC, WAL_HEADER_SIZE, WAL_MAGIC, WAL_VERSION, WalEntry, crc32_of, wal_corrupted,
};

// ─── WAL reading ────────────────────────────────────────────────────────────

/// Load all valid WAL entries from disk.
///
/// Partially-written batches (crash recovery) are silently discarded.
/// Returns an empty vec if the WAL file does not exist.
pub(super) fn read_wal(
    path: &Path,
    expected_dimension: usize,
    quantization: Quantization,
) -> SearchResult<(Vec<WalEntry>, u8, u64)> {
    if !path.exists() {
        return Ok((Vec::new(), 0, 0));
    }
    let data = std::fs::read(path)?;
    if data.len() < WAL_HEADER_SIZE {
        warn!(path = %path.display(), len = data.len(), "WAL file too small, ignoring");
        return Ok((Vec::new(), 0, 0));
    }
    parse_wal_bytes(&data, expected_dimension, quantization, path)
}

/// Load WAL entries for a WRITE-capable open, quarantining a torn header.
///
/// [`read_wal`] is deliberately side-effect-free because the classification
/// path ([`crate::two_tier::observe_tier`]) shares it, so a sidecar whose
/// header is torn at or above [`super::WAL_HEADER_SIZE`] makes it — and therefore
/// every future open — fail forever, while a sidecar SHORTER than the header
/// is ignored and a torn batch TAIL is tolerated. This resolves that
/// asymmetry for the writer path only: the corrupt sidecar is moved aside and
/// the open continues with no WAL entries.
///
/// The corrupt bytes are never destroyed. They are moved, whole, to a
/// no-clobber `<sidecar>.corrupt.<n>` sibling: stage the successor name with
/// `create_new`, move authority to it with one atomic rename, and only then
/// is the live pathname retired. A crash at any point leaves either the
/// original sidecar or the quarantine copy — never neither, and never a
/// truncated remnant. This is why the recovery cannot be a `set_len` or a
/// `remove_file`: a green recovery test that discarded the evidence would be
/// blessing the loss of every append the sidecar still held.
///
/// The quarantine runs only while holding the exclusive writer lock, so it
/// cannot race a live appender. If another writer holds the sidecar, the
/// original corruption error is returned unchanged rather than fighting it.
///
/// # Errors
///
/// Returns the underlying corruption error when the sidecar cannot be
/// quarantined, and I/O errors from the quarantine itself.
pub(super) fn read_wal_recovering_torn_header(
    path: &Path,
    expected_dimension: usize,
    quantization: Quantization,
) -> SearchResult<(Vec<WalEntry>, u8, u64)> {
    match read_wal(path, expected_dimension, quantization) {
        Ok(loaded) => Ok(loaded),
        Err(error @ SearchError::IndexCorrupted { .. }) => {
            quarantine_torn_wal_header(path, error)?;
            Ok((Vec::new(), 0, 0))
        }
        Err(error) => Err(error),
    }
}

/// Move a sidecar with an unusable header aside, preserving every byte.
fn quarantine_torn_wal_header(path: &Path, cause: SearchError) -> SearchResult<()> {
    let live = match OpenOptions::new().write(true).open(path) {
        Ok(file) => file,
        // The sidecar vanished under us; there is nothing left to quarantine
        // and nothing was lost that a fresh append cannot recreate.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    // Proving no appender is live is what makes moving the pathname safe.
    // Without it this recovery would race a writer mid-batch.
    let live = match acquire_wal_writer_lock(path, live) {
        Ok(guard) => guard,
        Err(lock_error) => {
            warn!(
                path = %path.display(),
                %lock_error,
                "torn WAL header cannot be quarantined while another writer holds the sidecar"
            );
            return Err(cause);
        }
    };

    let quarantine = stage_quarantine_path(path)?;
    fs::rename(path, &quarantine)?;
    drop(live);
    #[cfg(unix)]
    crate::sync_parent_directory(path)?;
    warn!(
        path = %path.display(),
        quarantine = %quarantine.display(),
        cause = %cause,
        "WAL header is unusable; sidecar quarantined and the open continues without it"
    );
    Ok(())
}

/// Reserve the first free `<sidecar>.corrupt.<n>` sibling.
///
/// `create_new` is what makes this no-clobber: an earlier quarantine copy is
/// never overwritten, so repeated recoveries accumulate evidence instead of
/// destroying it.
fn stage_quarantine_path(path: &Path) -> SearchResult<PathBuf> {
    let file_name = path
        .file_name()
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "wal.quarantine".to_owned(),
            value: path.display().to_string(),
            reason: "WAL sidecar path has no file name component".to_owned(),
        })?
        .to_owned();
    let directory = snapshot_parent_or_current(path).to_path_buf();
    for attempt in 0..u16::MAX {
        let mut candidate_name = file_name.clone();
        candidate_name.push(format!(".corrupt.{attempt}"));
        let candidate = directory.join(candidate_name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(_) => return Ok(candidate),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => (),
            Err(error) => return Err(error.into()),
        }
    }
    Err(SearchError::InvalidConfig {
        field: "wal.quarantine".to_owned(),
        value: path.display().to_string(),
        reason: "every quarantine name for this sidecar is already taken".to_owned(),
    })
}

fn parse_wal_bytes(
    data: &[u8],
    expected_dimension: usize,
    quantization: Quantization,
    path: &Path,
) -> SearchResult<(Vec<WalEntry>, u8, u64)> {
    // Header validation.
    if data[..4] != WAL_MAGIC {
        return Err(wal_corrupted(path, "bad magic bytes"));
    }
    let version = u16::from_le_bytes([data[4], data[5]]);
    if version != WAL_VERSION {
        return Err(wal_corrupted(
            path,
            format!("version mismatch: expected {WAL_VERSION}, got {version}"),
        ));
    }
    let dimension = u32::from_le_bytes([data[6], data[7], data[8], data[9]]) as usize;
    if dimension != expected_dimension {
        return Err(wal_corrupted(
            path,
            format!("dimension mismatch: expected {expected_dimension}, got {dimension}"),
        ));
    }
    let quant_byte = data[10];
    let wal_quant = Quantization::from_wire(quant_byte, path)?;
    if wal_quant != quantization {
        return Err(wal_corrupted(path, "quantization mismatch"));
    }
    let compaction_gen = data[11];
    // reserved: data[12..16]
    let header_crc_stored = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
    let header_crc_computed = crc32_of(&data[..16]);
    if header_crc_stored != header_crc_computed {
        return Err(wal_corrupted(path, "header CRC mismatch"));
    }

    // Parse batches.
    let mut entries = Vec::new();
    let mut cursor = WAL_HEADER_SIZE;
    let vector_bytes = dimension * quantization.bytes_per_element();

    while cursor + 8 <= data.len() {
        if let Ok((batch_entries, batch_len)) =
            parse_batch(&data[cursor..], dimension, vector_bytes, quantization)
        {
            entries.extend(batch_entries);
            cursor += batch_len;
        } else {
            warn!(
                path = %path.display(),
                offset = cursor,
                entries_recovered = entries.len(),
                "WAL batch corrupt or truncated, discarding remainder"
            );
            break;
        }
    }

    debug!(path = %path.display(), entries = entries.len(), "loaded WAL entries");
    Ok((entries, compaction_gen, cursor as u64))
}

fn parse_batch(
    data: &[u8],
    dimension: usize,
    vector_bytes: usize,
    quantization: Quantization,
) -> Result<(Vec<WalEntry>, usize), ()> {
    if data.len() < 8 {
        return Err(());
    }
    if data[..4] != BATCH_MAGIC {
        return Err(());
    }
    let entry_count = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

    // Bound capacity to what the data can actually hold (minimum 2 bytes header
    // + vector_bytes per entry) to prevent OOM from malicious entry_count values.
    let min_entry_bytes = 2_usize.saturating_add(vector_bytes).max(1);
    let max_possible = data.len().saturating_sub(8) / min_entry_bytes;
    let mut cursor: usize = 8;
    let mut entries = Vec::with_capacity(entry_count.min(max_possible));

    for _ in 0..entry_count {
        if cursor.checked_add(2).is_none_or(|end| end > data.len()) {
            return Err(());
        }
        let doc_id_len = u16::from_le_bytes([data[cursor], data[cursor + 1]]) as usize;
        cursor += 2;

        let needed = doc_id_len.checked_add(vector_bytes).ok_or(())?;
        if cursor
            .checked_add(needed)
            .is_none_or(|end| end > data.len())
        {
            return Err(());
        }
        let doc_id = std::str::from_utf8(&data[cursor..cursor + doc_id_len])
            .map_err(|_| ())?
            .to_owned();
        cursor += doc_id_len;

        let embedding = decode_vector(
            &data[cursor..cursor + vector_bytes],
            dimension,
            quantization,
        );
        cursor += vector_bytes;

        entries.push(WalEntry {
            doc_id_hash: crate::fnv1a_hash(doc_id.as_bytes()),
            doc_id,
            embedding,
        });
    }

    // Verify batch CRC.
    if cursor + 4 > data.len() {
        return Err(());
    }
    let stored_crc = u32::from_le_bytes([
        data[cursor],
        data[cursor + 1],
        data[cursor + 2],
        data[cursor + 3],
    ]);
    let computed_crc = crc32_of(&data[..cursor]);
    if stored_crc != computed_crc {
        return Err(());
    }
    cursor += 4;

    Ok((entries, cursor))
}

pub(super) fn decode_vector(
    bytes: &[u8],
    dimension: usize,
    quantization: Quantization,
) -> Vec<f32> {
    let vec = match quantization {
        Quantization::F16 => {
            // SIMD-widen 8 little-endian f16 per 16-byte block
            // (`widen8_f16_bytes`, the same magic-factor widen the f16 dot
            // kernels use — bit-identical to the scalar `f16::to_f32`), then a
            // scalar tail for the last < 8. The last remaining scalar consumer
            // of the vector_at_f32 SIMD-widen route-next (bd-y3hf).
            let f16_bytes = dimension * 2;
            let source = &bytes[..f16_bytes.min(bytes.len())];
            let mut out = Vec::with_capacity(dimension);
            let (blocks, remainder) = source.as_chunks::<16>();
            for arr in blocks {
                out.extend_from_slice(&crate::simd::widen8_f16_bytes(arr).to_array());
            }
            for chunk in remainder.as_chunks::<2>().0 {
                out.push(f16::from_le_bytes(*chunk).to_f32());
            }
            out.truncate(dimension);
            out
        }
        Quantization::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .take(dimension)
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect(),
    };
    debug_assert_eq!(
        Vec::<f32>::len(&vec),
        dimension,
        "decode_vector produced {actual} elements, expected {dimension}",
        actual = Vec::<f32>::len(&vec),
    );
    vec
}

// ─── WAL writing ────────────────────────────────────────────────────────────

pub(super) fn wal_entry_count(entries_len: usize) -> SearchResult<u32> {
    u32::try_from(entries_len).map_err(|_| SearchError::InvalidConfig {
        field: "wal_batch_entries".to_owned(),
        value: entries_len.to_string(),
        reason: "WAL batch entry count exceeds u32 maximum".to_owned(),
    })
}

fn wal_doc_id_len(doc_bytes_len: usize) -> SearchResult<u16> {
    u16::try_from(doc_bytes_len).map_err(|_| SearchError::InvalidConfig {
        field: "doc_id_length".to_owned(),
        value: doc_bytes_len.to_string(),
        reason: format!("document ID length {doc_bytes_len} exceeds WAL u16 limit (65535 bytes)"),
    })
}

fn wal_dimension(dimension: usize) -> SearchResult<u32> {
    u32::try_from(dimension).map_err(|_| SearchError::InvalidConfig {
        field: "dimension".to_owned(),
        value: dimension.to_string(),
        reason: "vector dimension exceeds u32 maximum for WAL header".to_owned(),
    })
}

/// Append a batch of entries to the WAL file.
///
/// If the WAL file does not exist, a header is written first.
/// The batch is CRC32-protected so partial writes are detectable.
pub(super) fn append_wal_batch(
    wal_path: &Path,
    entries: &[WalEntry],
    dimension: usize,
    quantization: Quantization,
    compaction_gen: u8,
    fsync: bool,
) -> SearchResult<()> {
    // Try to create a new WAL file atomically. If successful, we are the creator
    // and responsible for writing the header.
    let created = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(wal_path);

    let mut created_fresh = false;
    let mut writer = match created {
        Ok(file) => {
            let mut writer = acquire_wal_writer_lock(wal_path, file)?;
            created_fresh = true;
            write_wal_header(&mut writer.file, dimension, quantization, compaction_gen)?;
            writer
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            // File exists — open in write mode (not append) so we can
            // seek and truncate in-place without releasing the handle.
            //
            // We use .create(true) here to handle the TOCTOU race where the file
            // is deleted by another process (e.g. compaction) between our
            // create_new check above and this open. If that happens, we create
            // a new empty file, which `existing_len` check below will catch
            // and initialize with a header.
            let file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(false)
                .open(wal_path)?;
            // Before any length is observed or offset resolved: a second
            // appender that reads the end offset concurrently would clobber
            // this batch, and the refusal must happen before a single byte
            // of this one is written.
            let mut writer = acquire_wal_writer_lock(wal_path, file)?;
            let existing_len = writer.file.metadata()?.len();
            if existing_len < WAL_HEADER_SIZE as u64 {
                // Empty or truncated file — the creator likely crashed before
                // flushing the header. Truncate in-place and write a fresh
                // header. We keep the file handle open to avoid a TOCTOU race
                // where a concurrent writer could land valid data between a
                // drop and a reopen-with-truncate.
                warn!(
                    path = %wal_path.display(),
                    existing_len,
                    "WAL file smaller than header; truncating and writing fresh header"
                );
                writer.file.set_len(0)?;
                writer.file.seek(std::io::SeekFrom::Start(0))?;
                write_wal_header(&mut writer.file, dimension, quantization, compaction_gen)?;
                // `.create(true)` above may have re-created a dirent that a
                // concurrent delete removed after our `create_new` failed, or
                // repaired one a crashed creator never durably linked; either
                // way the dirent needs the same parent sync as a fresh create.
                created_fresh = true;
            } else {
                // Seek to end so the batch is appended after existing data.
                writer.file.seek(std::io::SeekFrom::End(0))?;
            }
            writer
        }
        Err(error) => return Err(error.into()),
    };

    // Build batch bytes.
    let mut batch = Vec::new();
    batch.extend_from_slice(&BATCH_MAGIC);
    let entry_count = wal_entry_count(entries.len())?;
    batch.extend_from_slice(&entry_count.to_le_bytes());

    for entry in entries {
        let doc_bytes = entry.doc_id.as_bytes();
        let doc_id_len = wal_doc_id_len(doc_bytes.len())?;
        batch.extend_from_slice(&doc_id_len.to_le_bytes());
        batch.extend_from_slice(doc_bytes);
        encode_vector(&mut batch, &entry.embedding, quantization);
    }

    let crc = crc32_of(&batch);
    batch.extend_from_slice(&crc.to_le_bytes());

    writer.file.write_all(&batch)?;

    if fsync {
        writer.file.sync_all()?;
        if created_fresh {
            // A durable first append also needs the CREATE dirent persisted:
            // file fsync alone does not guarantee the new sidecar survives a
            // crash on every filesystem, and a vanished sidecar silently
            // loses every acknowledged append.
            #[cfg(unix)]
            crate::sync_parent_directory(wal_path)?;
        }
    }

    debug!(
        path = %wal_path.display(),
        batch_entries = entries.len(),
        batch_bytes = batch.len(),
        "appended WAL batch"
    );
    Ok(())
}

pub(super) fn write_wal_header(
    writer: &mut impl Write,
    dimension: usize,
    quantization: Quantization,
    compaction_gen: u8,
) -> SearchResult<()> {
    let mut header = Vec::with_capacity(WAL_HEADER_SIZE);
    header.extend_from_slice(&WAL_MAGIC);
    header.extend_from_slice(&WAL_VERSION.to_le_bytes());
    let dim_u32 = wal_dimension(dimension)?;
    header.extend_from_slice(&dim_u32.to_le_bytes());
    #[allow(clippy::cast_possible_truncation)]
    header.push(quantization as u8);
    header.push(compaction_gen);
    header.extend_from_slice(&[0u8; 4]); // reserved
    let crc = crc32_of(&header);
    header.extend_from_slice(&crc.to_le_bytes());
    writer.write_all(&header)?;
    Ok(())
}

pub(super) fn encode_vector(buf: &mut Vec<u8>, embedding: &[f32], quantization: Quantization) {
    match quantization {
        Quantization::F16 => {
            for &val in embedding {
                buf.extend_from_slice(&f16::from_f32(val).to_le_bytes());
            }
        }
        Quantization::F32 => {
            for &val in embedding {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }
    }
}

/// Remove the WAL sidecar file.
pub(super) fn remove_wal(path: &Path) -> SearchResult<()> {
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    Ok(())
}

fn wal_writer_lock_error(path: &Path, error: &dyn std::fmt::Display) -> SearchError {
    SearchError::InvalidConfig {
        field: "wal.writer_lock".to_owned(),
        value: path.display().to_string(),
        reason: format!(
            "cannot acquire the exclusive WAL writer lock: {error}; this sidecar is \
             single-writer, so the appender that holds it must finish or exit first"
        ),
    }
}

/// Take the sidecar's exclusive advisory lock for the duration of a mutation.
///
/// The lock must be held BEFORE the append offset is resolved. Two appenders
/// that each resolve `SeekFrom::End(0)` independently derive the same offset
/// and the second `write_all` lands on top of the first batch, so an
/// acknowledged append disappears. Serializing after the seek would not help:
/// the lost update is decided by when the offset is read, not by when the
/// bytes are written.
///
/// This is the same retained-descriptor idiom the FSVI main file uses
/// (`fsvi.map_lock`, bd-x06p5); it arbitrates cooperating writers and makes
/// no claim about a process that ignores the lock entirely.
pub(super) fn acquire_wal_writer_lock(path: &Path, file: fs::File) -> SearchResult<WalWriterLock> {
    file.try_lock()
        .map_err(|error| wal_writer_lock_error(path, &error))?;
    Ok(WalWriterLock {
        file,
        creator_pid: std::process::id(),
    })
}

/// Own the locked file through every mutation and error exit. Closing a file
/// alone does not release its open-description lock while an inherited or
/// duplicated descriptor remains alive. A child must only close its copy;
/// explicitly unlocking there would also release the creator's lock.
pub(super) struct WalWriterLock {
    pub(super) file: fs::File,
    pub(super) creator_pid: u32,
}

impl Drop for WalWriterLock {
    fn drop(&mut self) {
        if self.creator_pid == std::process::id()
            && let Err(error) = self.file.unlock()
        {
            warn!(%error, "WAL writer explicit unlock failed; closing its descriptor");
        }
    }
}
