#[cfg(target_os = "linux")]
use std::ffi::OsString;
#[cfg(target_os = "linux")]
use std::fs::File;
#[cfg(target_os = "linux")]
use std::io::{Read, Write};
#[cfg(target_os = "linux")]
use std::os::fd::{AsRawFd, OwnedFd};
#[cfg(target_os = "linux")]
use std::path::Path;

use frankensearch_core::{SearchError, SearchResult};
#[cfg(target_os = "linux")]
use sha2::{Digest, Sha256};

#[cfg(target_os = "linux")]
use super::residual_sidecar::{
    ExactResidualLayout, ExactResidualPublication, ExactResidualPublicationCapability,
    ExactResidualSidecar, ResidualSourceBinding,
};
#[cfg(all(test, target_os = "linux"))]
use super::take_exact_residual_publication_acquisition_failure;
#[cfg(target_os = "linux")]
use super::{
    EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES, EXACT_RESIDUAL_SIDECAR_HEADER_BYTES,
    EXACT_RESIDUAL_SIDECAR_MAGIC, EXACT_RESIDUAL_SIDECAR_VERSION,
};

#[cfg(target_os = "linux")]
fn write_sidecar_piece(file: &mut File, digest: &mut Sha256, bytes: &[u8]) -> SearchResult<()> {
    file.write_all(bytes)
        .map_err(|error| residual_sidecar_error("publish", &error.to_string()))?;
    digest.update(bytes);
    Ok(())
}

#[cfg(target_os = "linux")]
fn write_sidecar_u32_values(
    file: &mut File,
    digest: &mut Sha256,
    values: &[u32],
) -> SearchResult<()> {
    const U32_BYTES: usize = std::mem::size_of::<u32>();

    let mut buffer = [0_u8; 4096];
    for chunk in values.chunks(buffer.len() / U32_BYTES) {
        let (slots, _) = buffer.as_chunks_mut::<U32_BYTES>();
        for (slot, value) in slots.iter_mut().zip(chunk) {
            *slot = value.to_le_bytes();
        }
        write_sidecar_piece(file, digest, &buffer[..std::mem::size_of_val(chunk)])?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn write_sidecar_f32_values(
    file: &mut File,
    digest: &mut Sha256,
    values: &[f32],
) -> SearchResult<()> {
    const F32_BYTES: usize = std::mem::size_of::<f32>();

    let mut buffer = [0_u8; 4096];
    for chunk in values.chunks(buffer.len() / F32_BYTES) {
        let (slots, _) = buffer.as_chunks_mut::<F32_BYTES>();
        for (slot, value) in slots.iter_mut().zip(chunk) {
            *slot = value.to_le_bytes();
        }
        write_sidecar_piece(file, digest, &buffer[..std::mem::size_of_val(chunk)])?;
    }
    Ok(())
}

/// Serialize directly into an owned unnamed inode. Publication never needs a
/// second `Vec<u8>` alongside the source-derived transform; the exact layout
/// and whole-file digest match the test-only sidecar encoder's wire format.
#[cfg(target_os = "linux")]
fn write_exact_residual_sidecar_stream(
    file: &mut File,
    sidecar: &ExactResidualSidecar,
) -> SearchResult<()> {
    let layout = sidecar.validated_layout()?;
    let mut digest = Sha256::new();
    write_sidecar_piece(file, &mut digest, &EXACT_RESIDUAL_SIDECAR_MAGIC)?;
    write_sidecar_piece(
        file,
        &mut digest,
        &EXACT_RESIDUAL_SIDECAR_VERSION.to_le_bytes(),
    )?;
    for value in [
        sidecar.source.generation_fingerprint,
        sidecar.source.vector_content_digest,
        sidecar.source.ordered_live_docset_digest,
        sidecar.source.space_fingerprint,
    ] {
        write_sidecar_piece(file, &mut digest, &value)?;
    }
    for value in [
        u64::try_from(sidecar.count)
            .map_err(|_| residual_sidecar_error("record_count", "does not fit u64"))?,
        u64::try_from(sidecar.dimension)
            .map_err(|_| residual_sidecar_error("dimension", "does not fit u64"))?,
    ] {
        write_sidecar_piece(file, &mut digest, &value.to_le_bytes())?;
    }
    for value in [
        u32::try_from(sidecar.block)
            .map_err(|_| residual_sidecar_error("block", "does not fit u32"))?,
        u32::try_from(sidecar.lanes)
            .map_err(|_| residual_sidecar_error("lanes", "does not fit u32"))?,
    ] {
        write_sidecar_piece(file, &mut digest, &value.to_le_bytes())?;
    }
    write_sidecar_u32_values(file, &mut digest, &sidecar.permutation)?;
    for values in [
        &sidecar.centroids,
        &sidecar.residuals,
        &sidecar.suffix_norms,
        &sidecar.correction_norms,
    ] {
        write_sidecar_f32_values(file, &mut digest, values)?;
    }
    debug_assert!(layout.encoded_bytes >= EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES);
    file.write_all(&digest.finalize())
        .map_err(|error| residual_sidecar_error("publish", &error.to_string()))
}

#[cfg(any(test, target_os = "linux"))]
pub(super) struct SidecarCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

#[cfg(any(test, target_os = "linux"))]
impl<'a> SidecarCursor<'a> {
    pub(super) const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    pub(super) fn take(&mut self, field: &str, len: usize) -> SearchResult<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| residual_sidecar_error(field, "payload offset overflow"))?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| residual_sidecar_error(field, "payload is truncated"))?;
        self.offset = end;
        Ok(bytes)
    }

    pub(super) fn take_array<const N: usize>(&mut self, field: &str) -> SearchResult<[u8; N]> {
        self.take(field, N)?
            .try_into()
            .map_err(|_| residual_sidecar_error(field, "fixed-width field is truncated"))
    }

    pub(super) fn u32(&mut self, field: &str) -> SearchResult<u32> {
        Ok(u32::from_le_bytes(self.take_array(field)?))
    }

    pub(super) fn u64(&mut self, field: &str) -> SearchResult<u64> {
        Ok(u64::from_le_bytes(self.take_array(field)?))
    }

    #[cfg(test)]
    pub(super) fn f32_vec(
        &mut self,
        field: &str,
        len: usize,
        nonnegative: bool,
    ) -> SearchResult<Vec<f32>> {
        const F32_BYTES: usize = std::mem::size_of::<f32>();

        let byte_len = len
            .checked_mul(F32_BYTES)
            .ok_or_else(|| residual_sidecar_error(field, "payload length overflow"))?;
        let bytes = self.take(field, byte_len)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(len)
            .map_err(|_| residual_sidecar_error(field, "allocation failed"))?;
        let (chunks, _) = bytes.as_chunks::<F32_BYTES>();
        for chunk in chunks {
            let value = f32::from_le_bytes(*chunk);
            if !value.is_finite() || (nonnegative && value < 0.0) {
                return Err(residual_sidecar_error(
                    field,
                    "must contain finite values in the declared range",
                ));
            }
            values.push(value);
        }
        Ok(values)
    }

    pub(super) const fn is_exhausted(&self) -> bool {
        self.offset == self.bytes.len()
    }

    #[cfg(test)]
    pub(super) const fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
}

pub(super) fn residual_sidecar_error(field: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("exact_residual_sidecar.{field}"),
        value: "redacted".to_owned(),
        reason: reason.to_owned(),
    }
}

#[cfg(test)]
pub(super) fn f32_bits_equal(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

pub(super) fn try_filled_sidecar_vec<T: Clone>(
    len: usize,
    value: T,
    field: &str,
) -> SearchResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| residual_sidecar_error(field, "allocation failed"))?;
    values.resize(len, value);
    Ok(values)
}

/// Open a sidecar's parent once, without resolving a symlink in the route, so
/// all subsequent reads/writes remain bound to this held directory descriptor.
/// The optimization intentionally declines unsupported/unsafe path routes
/// instead of falling back to a path-reopen race.
#[cfg(target_os = "linux")]
pub(super) fn open_exact_residual_sidecar_parent(path: &Path) -> SearchResult<(OwnedFd, OsString)> {
    use rustix::fs::{CWD, FileType, Mode, OFlags, ResolveFlags, fstat, openat2};

    if path
        .components()
        .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err(residual_sidecar_error(
            "path",
            "parent-directory traversal is not admissible for a sidecar",
        ));
    }
    let name = path.file_name().ok_or_else(|| {
        residual_sidecar_error("path", "destination must name one regular sidecar file")
    })?;
    if name == std::ffi::OsStr::new(".") || name == std::ffi::OsStr::new("..") {
        return Err(residual_sidecar_error(
            "path",
            "destination must name one regular sidecar file",
        ));
    }
    let parent = path
        .parent()
        .filter(|candidate| !candidate.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let directory = openat2(
        CWD,
        parent,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NONBLOCK | OFlags::NOFOLLOW,
        Mode::empty(),
        ResolveFlags::NO_SYMLINKS,
    )
    .map_err(|error| residual_sidecar_error("path", &error.to_string()))?;
    let stat =
        fstat(&directory).map_err(|error| residual_sidecar_error("path", &error.to_string()))?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
        return Err(residual_sidecar_error(
            "path",
            "sidecar parent must be a regular directory",
        ));
    }
    Ok((directory, name.to_os_string()))
}

#[cfg(target_os = "linux")]
pub(super) fn open_exact_residual_sidecar_file(
    path: &Path,
) -> SearchResult<Option<(File, rustix::fs::Stat)>> {
    use rustix::fs::{FileType, Mode, OFlags, fstat, openat};
    use rustix::io::Errno;

    let Ok((directory, name)) = open_exact_residual_sidecar_parent(path) else {
        return Ok(None);
    };
    let descriptor = match openat(
        &directory,
        &name,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NONBLOCK | OFlags::NOFOLLOW,
        Mode::empty(),
    ) {
        Ok(descriptor) => descriptor,
        Err(Errno::NOENT | Errno::LOOP | Errno::NOTDIR) => return Ok(None),
        Err(error) => return Err(residual_sidecar_error("open", &error.to_string())),
    };
    let before =
        fstat(&descriptor).map_err(|error| residual_sidecar_error("open", &error.to_string()))?;
    if FileType::from_raw_mode(before.st_mode) != FileType::RegularFile || before.st_nlink != 1 {
        return Ok(None);
    }
    Ok(Some((File::from(descriptor), before)))
}

#[cfg(target_os = "linux")]
fn same_exact_residual_sidecar_file(
    file: &File,
    before: &rustix::fs::Stat,
    expected_len: usize,
) -> SearchResult<bool> {
    use rustix::fs::fstat;

    let after = fstat(file).map_err(|error| residual_sidecar_error("open", &error.to_string()))?;
    let actual_len = usize::try_from(after.st_size)
        .map_err(|_| residual_sidecar_error("open", "sidecar size does not fit this platform"))?;
    Ok(actual_len == expected_len
        && after.st_dev == before.st_dev
        && after.st_ino == before.st_ino
        && after.st_size == before.st_size
        && after.st_nlink == before.st_nlink)
}

#[cfg(target_os = "linux")]
fn read_sidecar_piece<R: Read>(
    reader: &mut R,
    digest: &mut Sha256,
    expected: &[u8],
) -> SearchResult<bool> {
    let mut actual = [0_u8; 4096];
    let mut offset = 0_usize;
    while offset < expected.len() {
        let take = (expected.len() - offset).min(actual.len());
        match reader.read_exact(&mut actual[..take]) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(false),
            Err(error) => return Err(residual_sidecar_error("open", &error.to_string())),
        }
        digest.update(&actual[..take]);
        if actual[..take] != expected[offset..offset + take] {
            return Ok(false);
        }
        offset += take;
    }
    Ok(true)
}

#[cfg(target_os = "linux")]
fn read_sidecar_u32_values<R: Read>(
    reader: &mut R,
    digest: &mut Sha256,
    values: &[u32],
) -> SearchResult<bool> {
    const U32_BYTES: usize = std::mem::size_of::<u32>();

    let mut expected = [0_u8; 4096];
    for values in values.chunks(expected.len() / U32_BYTES) {
        let (slots, _) = expected.as_chunks_mut::<U32_BYTES>();
        for (slot, value) in slots.iter_mut().zip(values) {
            *slot = value.to_le_bytes();
        }
        let byte_len = std::mem::size_of_val(values);
        if !read_sidecar_piece(reader, digest, &expected[..byte_len])? {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(target_os = "linux")]
fn read_sidecar_f32_values<R: Read>(
    reader: &mut R,
    digest: &mut Sha256,
    values: &[f32],
) -> SearchResult<bool> {
    const F32_BYTES: usize = std::mem::size_of::<f32>();

    let mut expected = [0_u8; 4096];
    for values in values.chunks(expected.len() / F32_BYTES) {
        let (slots, _) = expected.as_chunks_mut::<F32_BYTES>();
        for (slot, value) in slots.iter_mut().zip(values) {
            *slot = value.to_le_bytes();
        }
        let byte_len = std::mem::size_of_val(values);
        if !read_sidecar_piece(reader, digest, &expected[..byte_len])? {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(target_os = "linux")]
pub(super) fn exact_residual_sidecar_header_matches_source(
    path: &Path,
    source: &ResidualSourceBinding,
    count: usize,
    dimension: usize,
) -> SearchResult<bool> {
    let layout = ExactResidualLayout::for_shape(count, dimension)?;
    let Some((mut file, before)) = open_exact_residual_sidecar_file(path)? else {
        return Ok(false);
    };
    let byte_len = usize::try_from(before.st_size)
        .map_err(|_| residual_sidecar_error("open", "sidecar size does not fit this platform"))?;
    if byte_len != layout.encoded_bytes {
        return Ok(false);
    }
    let mut header = [0_u8; EXACT_RESIDUAL_SIDECAR_HEADER_BYTES];
    match file.read_exact(&mut header) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(false),
        Err(error) => return Err(residual_sidecar_error("open", &error.to_string())),
    }
    Ok(
        ExactResidualSidecar::header_matches_source(&header, source, count, dimension)?
            && same_exact_residual_sidecar_file(&file, &before, layout.encoded_bytes)?,
    )
}

/// Compare a stream to one source-derived sidecar after the caller has fixed
/// its descriptor size with `fstat`. Every payload read is exact and the final
/// one-byte probe rejects growth without materializing a raw sidecar image.
/// The generic reader is an intentional deterministic truncation-after-fstat
/// test seam; production passes the held no-follow descriptor itself.
#[cfg(target_os = "linux")]
pub(super) fn exact_residual_sidecar_stream_matches_reader<R: Read>(
    reader: &mut R,
    expected: &ExactResidualSidecar,
) -> SearchResult<bool> {
    let _ = expected.validated_layout()?;
    let mut header = [0_u8; EXACT_RESIDUAL_SIDECAR_HEADER_BYTES];
    match reader.read_exact(&mut header) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(false),
        Err(error) => return Err(residual_sidecar_error("open", &error.to_string())),
    }
    if !ExactResidualSidecar::header_matches_source(
        &header,
        &expected.source,
        expected.count,
        expected.dimension,
    )? {
        return Ok(false);
    }

    let mut digest = Sha256::new();
    digest.update(header);
    if !read_sidecar_u32_values(reader, &mut digest, &expected.permutation)? {
        return Ok(false);
    }
    for values in [
        &expected.centroids,
        &expected.residuals,
        &expected.suffix_norms,
        &expected.correction_norms,
    ] {
        if !read_sidecar_f32_values(reader, &mut digest, values)? {
            return Ok(false);
        }
    }
    let mut supplied_digest = [0_u8; EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES];
    match reader.read_exact(&mut supplied_digest) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(false),
        Err(error) => return Err(residual_sidecar_error("open", &error.to_string())),
    }
    let mut trailing = [0_u8; 1];
    if reader
        .read(&mut trailing)
        .map_err(|error| residual_sidecar_error("open", &error.to_string()))?
        != 0
    {
        return Ok(false);
    }
    let expected_digest: [u8; EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES] = digest.finalize().into();
    Ok(supplied_digest == expected_digest)
}

#[cfg(target_os = "linux")]
pub(super) fn exact_residual_sidecar_stream_matches_derived(
    path: &Path,
    expected: &ExactResidualSidecar,
) -> SearchResult<bool> {
    let layout = expected.validated_layout()?;
    let Some((mut file, before)) = open_exact_residual_sidecar_file(path)? else {
        return Ok(false);
    };
    let byte_len = usize::try_from(before.st_size)
        .map_err(|_| residual_sidecar_error("open", "sidecar size does not fit this platform"))?;
    if byte_len != layout.encoded_bytes {
        return Ok(false);
    }
    Ok(
        exact_residual_sidecar_stream_matches_reader(&mut file, expected)?
            && same_exact_residual_sidecar_file(&file, &before, layout.encoded_bytes)?,
    )
}

/// Reject a corrupt candidate using only its own trailing digest.
///
/// Candidate discovery already filters on the header, so a corrupt body reaches
/// the byte-exact comparison and forces a full source derivation — up to
/// [`super::EXACT_RESIDUAL_SIDECAR_MAX_BYTES`] — purely to learn the candidate was
/// never reusable. That derivation is then discarded whenever the optional
/// cache also cannot publish, which is exactly the read-only and
/// `O_TMPFILE`-unsupported case.
///
/// [`write_exact_residual_sidecar_stream`] digests every byte it writes before
/// appending the digest itself, so bytes `[0, encoded_bytes - DIGEST)` are
/// self-covering and one bounded streaming read decides corruption with zero
/// derivation. This is an admission gate, not an authentication: passing it
/// proves internal consistency only, and the derived byte-exact comparison
/// remains the sole basis for reuse.
#[cfg(target_os = "linux")]
pub(super) fn exact_residual_sidecar_stream_is_self_consistent(
    path: &Path,
    layout: &ExactResidualLayout,
) -> SearchResult<bool> {
    /// Bounded streaming chunk: the read is streaming precisely so an untrusted
    /// candidate never turns into a second full-size heap image. The buffer is
    /// this fixed size regardless of the candidate's declared length.
    const CHUNK_BYTES: usize = 64 * 1024;

    let Some((mut file, before)) = open_exact_residual_sidecar_file(path)? else {
        return Ok(false);
    };
    let byte_len = usize::try_from(before.st_size)
        .map_err(|_| residual_sidecar_error("open", "sidecar size does not fit this platform"))?;
    if byte_len != layout.encoded_bytes {
        return Ok(false);
    }
    let Some(digested_bytes) = layout
        .encoded_bytes
        .checked_sub(EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES)
    else {
        return Ok(false);
    };

    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; CHUNK_BYTES];
    let mut remaining = digested_bytes;
    while remaining != 0 {
        let wanted = remaining.min(CHUNK_BYTES);
        match file.read_exact(&mut buffer[..wanted]) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(false),
            Err(error) => return Err(residual_sidecar_error("open", &error.to_string())),
        }
        digest.update(&buffer[..wanted]);
        remaining -= wanted;
    }

    let mut supplied_digest = [0_u8; EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES];
    match file.read_exact(&mut supplied_digest) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(false),
        Err(error) => return Err(residual_sidecar_error("open", &error.to_string())),
    }
    if !same_exact_residual_sidecar_file(&file, &before, layout.encoded_bytes)? {
        return Ok(false);
    }
    let expected_digest: [u8; EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES] = digest.finalize().into();
    Ok(supplied_digest == expected_digest)
}

/// Acquire an unnamed inode through a no-follow parent descriptor before any
/// large source-derived sidecar allocation. Failure proves the optional cache
/// cannot publish and callers must retain the exact flat route without deriving
/// a payload that would be discarded.
#[cfg(target_os = "linux")]
pub(super) fn acquire_exact_residual_sidecar_publication(
    path: &Path,
) -> SearchResult<ExactResidualPublicationCapability> {
    use rustix::fs::{Mode, OFlags, openat};

    #[cfg(test)]
    if take_exact_residual_publication_acquisition_failure() {
        return Err(residual_sidecar_error(
            "publish",
            "injected publication-capability acquisition failure",
        ));
    }

    let (directory, destination) = open_exact_residual_sidecar_parent(path)?;
    let descriptor = openat(
        &directory,
        ".",
        OFlags::WRONLY | OFlags::TMPFILE | OFlags::CLOEXEC | OFlags::NONBLOCK,
        Mode::RUSR | Mode::WUSR,
    )
    .map_err(|error| residual_sidecar_error("publish", &error.to_string()))?;
    Ok(ExactResidualPublicationCapability {
        directory,
        destination,
        file: File::from(descriptor),
    })
}

/// Consume a previously acquired publication capability, then fsync the held
/// parent directory. `linkat` never replaces a destination, and the unnamed
/// inode is reclaimed automatically on failures before that link. Once linked,
/// a parent-directory `sync_all` failure is reported but the visible destination
/// is deliberately left intact: unlinking it would race the now-published
/// immutable artifact and could erase a durable cache.
#[cfg(target_os = "linux")]
pub(super) fn publish_exact_residual_sidecar_with_capability(
    capability: ExactResidualPublicationCapability,
    sidecar: &ExactResidualSidecar,
) -> SearchResult<ExactResidualPublication> {
    use rustix::fs::{AtFlags, CWD, linkat};
    use rustix::io::Errno;

    let mut file = capability.file;
    write_exact_residual_sidecar_stream(&mut file, sidecar)?;
    file.sync_all()
        .map_err(|error| residual_sidecar_error("publish", &error.to_string()))?;

    // Link from `/proc/self/fd/<n>` while `file` remains open: this source is
    // the anonymous owned inode, not a temporary pathname an attacker could
    // replace. `linkat` fails with EEXIST rather than replacing `destination`.
    let descriptor_path = format!("/proc/self/fd/{}", file.as_raw_fd());
    match linkat(
        CWD,
        Path::new(&descriptor_path),
        &capability.directory,
        &capability.destination,
        AtFlags::SYMLINK_FOLLOW,
    ) {
        Ok(()) => {}
        Err(Errno::EXIST) => return Ok(ExactResidualPublication::DestinationExists),
        Err(error) => return Err(residual_sidecar_error("publish", &error.to_string())),
    }
    drop(file);
    File::from(capability.directory)
        .sync_all()
        .map_err(|error| residual_sidecar_error("publish", &error.to_string()))?;
    Ok(ExactResidualPublication::Published)
}

/// One-shot public writer wrapper. Product construction may acquire and hold
/// the capability before it derives the sidecar; callers of the explicit write
/// API simply acquire and consume it in this call.
#[cfg(target_os = "linux")]
pub(super) fn publish_exact_residual_sidecar(
    path: &Path,
    sidecar: &ExactResidualSidecar,
) -> SearchResult<ExactResidualPublication> {
    let capability = acquire_exact_residual_sidecar_publication(path)?;
    publish_exact_residual_sidecar_with_capability(capability, sidecar)
}
