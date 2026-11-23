use std::fs::File;
use std::io::Read;
use std::path::Path;

use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
use safetensors::Dtype;
use safetensors::tensor::Metadata;

/// Tensor name candidates, tried in order when discovering the embedding matrix.
const TENSOR_NAME_CANDIDATES: [&str; 5] =
    ["embeddings", "embedding", "word_embeddings", "embed", "emb"];

/// Discover the embedding tensor name among a safetensors file's tensor names.
///
/// Tries known names first, then falls back to using the only tensor
/// if the file contains exactly one.
pub(super) fn discover_tensor_name(names: &[&str]) -> Option<String> {
    // Try known candidate names
    for candidate in &TENSOR_NAME_CANDIDATES {
        if names.contains(candidate) {
            return Some((*candidate).to_owned());
        }
    }

    // Fallback: if exactly one tensor exists, use it regardless of name
    if let [only] = names {
        return Some((*only).to_owned());
    }

    None
}

/// Length prefix of a safetensors file: a little-endian `u64` header size.
const SAFETENSORS_HEADER_LEN_PREFIX: usize = 8;

/// Upper bound on the safetensors JSON header, matching the `MAX_HEADER_SIZE`
/// the `safetensors` crate enforces in `SafeTensors::read_metadata`. Kept so a
/// corrupt length prefix cannot ask for a huge allocation.
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 100_000_000;

/// Bytes per positional read while streaming the matrix. A multiple of 4, so
/// every chunk decodes into whole `f32` values with no carry-over; small
/// enough that one buffer per worker adds little to the load's peak memory.
pub(super) const MATRIX_READ_CHUNK_BYTES: usize = 1 << 18;

/// A located, validated F32 matrix inside a safetensors file, not yet read.
///
/// Splitting "locate" from "read" is what keeps the peak allocation at one
/// copy of the matrix: the header is parsed from a few hundred bytes, the
/// caller runs every admission check against the shape it reports, and only
/// then are the tensor bytes streamed into their final `Vec<f32>`.
pub(super) struct SafetensorsF32Matrix {
    file: File,
    /// Absolute offset of the tensor's first byte within the file.
    data_start: u64,
    /// Tensor byte length; always `vocab_size * dimensions * 4`.
    byte_len: usize,
    pub(super) vocab_size: usize,
    pub(super) dimensions: usize,
}

impl SafetensorsF32Matrix {
    /// Parse the safetensors header and locate the embedding tensor.
    ///
    /// Performs the same structural validation the whole-file path did —
    /// header bounds, offset contiguity (via the `safetensors` crate's own
    /// `Metadata` deserializer), file-length completeness, 2-D shape, exact
    /// byte count — plus an EXPLICIT dtype check. The old path enforced F32
    /// only implicitly, through the byte-count comparison in
    /// `parse_f32_matrix`; a same-shaped tensor of another dtype was rejected
    /// with a size-mismatch message rather than a dtype one.
    pub(super) fn locate(path: &Path) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| format!("failed to open safetensors: {e}"))?;
        let file_len = file
            .metadata()
            .map_err(|e| format!("failed to stat safetensors: {e}"))?
            .len();

        let mut prefix = [0_u8; SAFETENSORS_HEADER_LEN_PREFIX];
        file.read_exact(&mut prefix)
            .map_err(|e| format!("failed to read safetensors header length: {e}"))?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len > MAX_SAFETENSORS_HEADER_BYTES {
            return Err(format!(
                "safetensors header of {header_len} bytes exceeds the {MAX_SAFETENSORS_HEADER_BYTES}-byte limit"
            ));
        }
        // Bound the header against the file too, so a corrupt prefix cannot
        // ask for a 100 MB allocation that the read is then going to fail.
        if header_len > file_len.saturating_sub(SAFETENSORS_HEADER_LEN_PREFIX as u64) {
            return Err(format!(
                "safetensors header of {header_len} bytes exceeds the {file_len}-byte file"
            ));
        }
        let header_len_usize = usize::try_from(header_len)
            .map_err(|_| "safetensors header length exceeds this platform's usize".to_owned())?;

        let mut header = vec![0_u8; header_len_usize];
        file.read_exact(&mut header)
            .map_err(|e| format!("failed to read safetensors header: {e}"))?;
        let metadata: Metadata = serde_json::from_slice(&header)
            .map_err(|e| format!("failed to parse safetensors: {e}"))?;

        // Completeness check, identical in effect to the one
        // `SafeTensors::read_metadata` performs over a whole-file buffer: a
        // truncated or over-long file is rejected rather than silently
        // yielding a short matrix.
        let data_len = u64::try_from(metadata.data_len())
            .map_err(|_| "safetensors data length exceeds u64".to_owned())?;
        let expected_len = header_len
            .checked_add(data_len)
            .and_then(|n| n.checked_add(SAFETENSORS_HEADER_LEN_PREFIX as u64))
            .ok_or_else(|| "safetensors declared length overflows".to_owned())?;
        if expected_len != file_len {
            return Err(format!(
                "safetensors file length mismatch: header declares {expected_len} bytes, file is {file_len}"
            ));
        }

        let names = metadata.offset_keys();
        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let tensor_name = discover_tensor_name(&name_refs).ok_or_else(|| {
            format!(
                "no embedding tensor found. Tried: {TENSOR_NAME_CANDIDATES:?}. Available: {names:?}"
            )
        })?;
        let info = metadata
            .info(&tensor_name)
            .ok_or_else(|| format!("failed to get tensor '{tensor_name}'"))?;

        if info.dtype != Dtype::F32 {
            return Err(format!(
                "expected an F32 embedding tensor, got {:?} for '{tensor_name}'",
                info.dtype
            ));
        }
        if info.shape.len() != 2 {
            return Err(format!(
                "expected 2D tensor, got {}D with shape {:?}",
                info.shape.len(),
                info.shape
            ));
        }
        let vocab_size = info.shape[0];
        let dimensions = info.shape[1];

        let (start, end) = info.data_offsets;
        let byte_len = end
            .checked_sub(start)
            .ok_or_else(|| format!("tensor '{tensor_name}' has inverted data offsets"))?;
        let expected_bytes = vocab_size
            .checked_mul(dimensions)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| format!("byte size overflow for [{vocab_size} x {dimensions}] f32"))?;
        if byte_len != expected_bytes {
            return Err(format!(
                "tensor data size mismatch: expected {expected_bytes} bytes for [{vocab_size} x {dimensions}] f32, got {byte_len}"
            ));
        }

        let data_start = u64::try_from(start)
            .ok()
            .and_then(|s| s.checked_add(header_len))
            .and_then(|s| s.checked_add(SAFETENSORS_HEADER_LEN_PREFIX as u64))
            .ok_or_else(|| format!("tensor '{tensor_name}' offset overflows the file"))?;

        Ok(Self {
            file,
            data_start,
            byte_len,
            vocab_size,
            dimensions,
        })
    }

    /// Stream the tensor into a freshly allocated `Vec<f32>`.
    ///
    /// Decoding is byte-for-byte what `parse_f32_matrix` did — the same
    /// little-endian `f32::from_le_bytes` over the same bytes — so the
    /// resulting matrix is bit-identical to the old path's. Chunks of
    /// `MATRIX_READ_CHUNK_BYTES` are read with positional reads and decoded in
    /// parallel, each straight into its slice of the final matrix; besides the
    /// matrix, at most one chunk buffer per worker is live.
    pub(super) fn read_values(self) -> Result<Vec<f32>, String> {
        const CHUNK_VALUES: usize = MATRIX_READ_CHUNK_BYTES / 4;
        let mut values = vec![0.0_f32; self.byte_len / 4];
        let (file, data_start) = (&self.file, self.data_start);
        values
            .par_chunks_mut(CHUNK_VALUES)
            .enumerate()
            .try_for_each_init(
                || vec![0_u8; MATRIX_READ_CHUNK_BYTES],
                |buffer, (index, chunk)| {
                    // Every chunk but the last is full, and `byte_len` is a
                    // multiple of 4, so `as_chunks` leaves no remainder.
                    let bytes = &mut buffer[..chunk.len() * 4];
                    let offset = u64::try_from(index * MATRIX_READ_CHUNK_BYTES)
                        .ok()
                        .and_then(|offset| data_start.checked_add(offset))
                        .ok_or_else(|| "embedding tensor offset overflows".to_owned())?;
                    read_exact_at(file, bytes, offset)
                        .map_err(|e| format!("failed to read the embedding tensor: {e}"))?;
                    for (value, &raw) in chunk.iter_mut().zip(bytes.as_chunks::<4>().0) {
                        *value = f32::from_le_bytes(raw);
                    }
                    Ok::<(), String>(())
                },
            )?;
        Ok(values)
    }
}

/// Read exactly `buffer.len()` bytes at `offset` without moving a shared
/// cursor, so several threads can read one handle at once.
#[cfg(unix)]
fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buffer, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut buffer: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buffer.is_empty() {
        match file.seek_read(buffer, offset) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(read) => {
                buffer = &mut buffer[read..];
                offset += read as u64;
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

/// Targets without positional reads share the handle's cursor, so each
/// seek-and-read runs under one lock.
#[cfg(not(any(unix, windows)))]
fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom};
    use std::sync::{Mutex, PoisonError};
    static CURSOR: Mutex<()> = Mutex::new(());
    let _cursor = CURSOR.lock().unwrap_or_else(PoisonError::into_inner);
    let mut handle = file;
    handle.seek(SeekFrom::Start(offset))?;
    handle.read_exact(buffer)
}

/// Parse raw bytes from a safetensors tensor into a flat `Vec<f32>` matrix.
///
/// Expects little-endian f32 data with shape `[vocab_size, dimensions]`.
///
/// This is the pre-#46 whole-buffer decoder. The shipping loader streams the
/// tensor instead (`SafetensorsF32Matrix::read_values`) so the file bytes and
/// the decoded matrix are never both resident; this is retained as the
/// BIT-IDENTITY ORACLE that stream decoding is checked against, which is why
/// it must keep decoding exactly the way it always did.
#[cfg(test)]
pub(super) fn parse_f32_matrix(
    data: &[u8],
    vocab_size: usize,
    dimensions: usize,
) -> Result<Vec<f32>, String> {
    let expected_elements = vocab_size
        .checked_mul(dimensions)
        .ok_or_else(|| format!("matrix size overflow for [{vocab_size} x {dimensions}]"))?;
    let expected_bytes = expected_elements
        .checked_mul(4)
        .ok_or_else(|| format!("byte size overflow for [{vocab_size} x {dimensions}] f32"))?;
    if data.len() != expected_bytes {
        return Err(format!(
            "tensor data size mismatch: expected {expected_bytes} bytes for [{vocab_size} x {dimensions}] f32, got {}",
            data.len()
        ));
    }

    // Pre-allocate the exact size
    let mut matrix = Vec::with_capacity(expected_elements);

    // Parse bytes in 4-byte chunks (length is validated above to be a
    // multiple of 4, so `as_chunks` leaves no remainder)
    for &bytes in data.as_chunks::<4>().0 {
        matrix.push(f32::from_le_bytes(bytes));
    }

    if matrix.len() != expected_elements {
        return Err(format!(
            "parsed element count mismatch: expected {}, got {}",
            expected_elements,
            matrix.len()
        ));
    }

    Ok(matrix)
}
