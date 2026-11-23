use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use frankensearch_index::mapped_file::ReadOnlyMappedFile;
use xxhash_rust::xxh3::{Xxh3, xxh3_64};

use crate::error::QuillError;
use crate::schema::SchemaDescriptor;

use super::codec::{corrupted, parse_container, published_segment_id, resource, section_bytes};
use super::writer::EncodedSegment;
use super::{
    FILE_PREFIX_LEN, FIXED_HEADER_LEN, HEADER_CRC_LEN, MEMORY_PATH, SectionEntry, SectionKind,
    SegmentHeader, SegmentLimits, TRAILER_LEN,
};

/// Validating FSLX reader over an owned, borrowed, or memory-mapped byte source.
pub struct SegmentReader<B: AsRef<[u8]>> {
    source: B,
    path: PathBuf,
    schema: SchemaDescriptor,
    limits: SegmentLimits,
    header: SegmentHeader,
    sections: Vec<SectionEntry>,
    section_checks: Vec<OnceLock<Result<(), String>>>,
    file_xxh3: u64,
}

impl SegmentReader<Vec<u8>> {
    /// Parse an owned in-memory segment with default resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns a typed corruption, schema, or resource error.
    pub fn from_owned(bytes: Vec<u8>, schema: SchemaDescriptor) -> Result<Self, QuillError> {
        Self::from_owned_with_limits(bytes, schema, SegmentLimits::default())
    }

    /// Parse an owned in-memory segment with explicit resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::from_owned`].
    pub fn from_owned_with_limits(
        bytes: Vec<u8>,
        schema: SchemaDescriptor,
        limits: SegmentLimits,
    ) -> Result<Self, QuillError> {
        Self::parse_source(bytes, PathBuf::from(MEMORY_PATH), schema, limits)
    }
}

impl SegmentReader<EncodedSegment> {
    /// Parse an encoded in-memory segment, retaining its shared byte backing.
    ///
    /// The reader keeps the [`EncodedSegment`] alive instead of copying its
    /// bytes into a fresh buffer, so any other holder of the same encoded
    /// segment (for example retained publication state) shares one immutable
    /// allocation with this reader.
    ///
    /// # Errors
    ///
    /// Returns the same typed corruption, schema, or resource errors as
    /// `SegmentReader::from_owned`.
    pub fn from_encoded(
        encoded: EncodedSegment,
        schema: SchemaDescriptor,
    ) -> Result<Self, QuillError> {
        Self::parse_source(
            encoded,
            PathBuf::from(MEMORY_PATH),
            schema,
            SegmentLimits::default(),
        )
    }
}

impl<'a> SegmentReader<&'a [u8]> {
    /// Parse borrowed segment bytes with default resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns a typed corruption, schema, or resource error.
    pub fn from_bytes(bytes: &'a [u8], schema: SchemaDescriptor) -> Result<Self, QuillError> {
        Self::parse_source(
            bytes,
            PathBuf::from(MEMORY_PATH),
            schema,
            SegmentLimits::default(),
        )
    }
}

/// Read size for the same-descriptor streamed prefix hash.
const STREAMED_WITNESS_READ_BYTES: usize = 1 << 20;

impl SegmentReader<ReadOnlyMappedFile> {
    /// Used only inside the same-descriptor open callback, after MANIFEST checks.
    pub(crate) fn verify_streamed_file_witness(
        &self,
        file: &mut fs::File,
    ) -> Result<u64, QuillError> {
        use std::io::Read;

        let prefix_len = self
            .source
            .len()
            .checked_sub(TRAILER_LEN)
            .ok_or_else(|| corrupted(&self.path, "file is shorter than its trailer"))?;
        let mut hasher = Xxh3::new();
        let mut remaining = prefix_len;
        // One heap buffer of up to 1 MiB: 64x fewer read syscalls than the
        // former 16 KiB stack buffer on multi-GB segments.
        let mut buffer = vec![0_u8; STREAMED_WITNESS_READ_BYTES.min(prefix_len.max(1))];
        while remaining != 0 {
            let count = remaining.min(buffer.len());
            file.read_exact(&mut buffer[..count])?;
            hasher.update(&buffer[..count]);
            remaining -= count;
        }
        let actual = hasher.digest();
        if actual != self.file_xxh3 {
            return Err(corrupted(
                &self.path,
                format!(
                    "file checksum mismatch: expected {:#018x}, got {actual:#018x}",
                    self.file_xxh3
                ),
            ));
        }
        Ok(actual)
    }

    /// Open and validate an immutable, published FSLX file through the shared mmap facade.
    ///
    /// Only canonical `seg-<hex16>.fslx` generation files are accepted. Keeper
    /// must finish the temp-file sync and atomic rename before calling this
    /// method, and published generations must never be mutated in place.
    ///
    /// # Errors
    ///
    /// Returns a typed I/O, corruption, schema, or resource error.
    pub fn open_published(path: &Path, schema: SchemaDescriptor) -> Result<Self, QuillError> {
        Self::open_published_with_limits(path, schema, SegmentLimits::default())
    }

    /// Open with explicit resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::open_published`].
    pub fn open_published_with_limits(
        path: &Path,
        schema: SchemaDescriptor,
        limits: SegmentLimits,
    ) -> Result<Self, QuillError> {
        Self::open_published_checked(path, schema, limits, |_, _| Ok(())).map(|(reader, ())| reader)
    }

    /// Run admission while the exact descriptor backing the mapping is alive.
    /// The descriptor is released before returning the reader and admission result.
    pub(crate) fn open_published_checked<T>(
        path: &Path,
        schema: SchemaDescriptor,
        limits: SegmentLimits,
        check: impl FnOnce(&Self, &mut fs::File) -> Result<T, QuillError>,
    ) -> Result<(Self, T), QuillError> {
        let expected_segment_id =
            published_segment_id(path).map_err(|detail| corrupted(path, detail))?;
        let metadata = fs::symlink_metadata(path)?;
        if !metadata.file_type().is_file() {
            return Err(corrupted(path, "published segment is not a regular file"));
        }
        let file_len = metadata.len();
        if file_len > limits.max_file_bytes {
            return Err(resource(
                "segment file",
                format!(
                    "{} has length {file_len}, exceeding limit {}",
                    path.display(),
                    limits.max_file_bytes
                ),
            ));
        }
        if file_len
            < u64::try_from(FILE_PREFIX_LEN + FIXED_HEADER_LEN + HEADER_CRC_LEN + TRAILER_LEN)
                .unwrap_or(u64::MAX)
        {
            return Err(corrupted(path, format!("truncated file length {file_len}")));
        }
        ReadOnlyMappedFile::open_published_with_file(path, |mapped, file| {
            let reader = Self::parse_source(mapped, path.to_path_buf(), schema, limits)?;
            if reader.header.segment_id != expected_segment_id {
                return Err(corrupted(
                    path,
                    format!(
                        "published filename identifies segment {expected_segment_id:#018x}, but the header identifies {:#018x}",
                        reader.header.segment_id
                    ),
                ));
            }
            let witness = check(&reader, file)?;
            Ok((reader, witness))
        })
    }
}

impl<B: AsRef<[u8]>> SegmentReader<B> {
    fn parse_source(
        source: B,
        path: PathBuf,
        schema: SchemaDescriptor,
        limits: SegmentLimits,
    ) -> Result<Self, QuillError> {
        schema.validate()?;
        let parsed = parse_container(source.as_ref(), &path, schema, limits)?;
        let mut section_checks = Vec::new();
        section_checks
            .try_reserve_exact(parsed.sections.len())
            .map_err(|_| {
                resource(
                    "lazy section checks",
                    format!("allocation failed for {}", path.display()),
                )
            })?;
        for _ in &parsed.sections {
            section_checks.push(OnceLock::new());
        }
        Ok(Self {
            source,
            path,
            schema,
            limits,
            header: parsed.header,
            sections: parsed.sections,
            section_checks,
            file_xxh3: parsed.file_xxh3,
        })
    }

    /// Parsed immutable segment header.
    #[must_use]
    pub const fn header(&self) -> SegmentHeader {
        self.header
    }

    /// Validated section-table metadata in durable order.
    #[must_use]
    pub fn section_entries(&self) -> &[SectionEntry] {
        &self.sections
    }

    /// Exact complete file length.
    #[must_use]
    pub fn file_len(&self) -> u64 {
        u64::try_from(self.source.as_ref().len()).unwrap_or(u64::MAX)
    }

    /// Borrow the complete canonical FSLX image backing this reader.
    #[must_use]
    pub fn source_bytes(&self) -> &[u8] {
        self.source.as_ref()
    }

    /// xxh3-64 over the file prefix before the trailer, stored in the trailer.
    #[must_use]
    pub const fn file_xxh3(&self) -> u64 {
        self.file_xxh3
    }

    /// Recompute and verify the whole-file FSLX integrity witness.
    ///
    /// This hashes the exact file prefix before the fixed trailer, which is
    /// the same domain used by [`Self::verify`]. It deliberately leaves every
    /// lazy per-section checksum untouched. XXH3 detects corruption here; it
    /// does not authenticate untrusted content.
    ///
    /// # Errors
    ///
    /// Returns a typed corruption error when the actual backing bytes do not
    /// match the trailer witness parsed at open time.
    pub(crate) fn verify_file_witness(&self) -> Result<u64, QuillError> {
        let bytes = self.source.as_ref();
        let trailer_start = bytes
            .len()
            .checked_sub(TRAILER_LEN)
            .ok_or_else(|| corrupted(&self.path, "file is shorter than its trailer"))?;
        let actual = xxh3_64(&bytes[..trailer_start]);
        if actual != self.file_xxh3 {
            return Err(corrupted(
                &self.path,
                format!(
                    "file checksum mismatch: expected {:#018x}, got {actual:#018x}",
                    self.file_xxh3
                ),
            ));
        }
        Ok(actual)
    }

    /// Verify one section once and borrow its exact payload bytes.
    ///
    /// Unknown optional kinds remain visible in [`Self::section_entries`] and
    /// are hashed by [`Self::verify`], but normal reads skip their payloads.
    /// `Ok(None)` means the known kind is absent or the requested kind is
    /// unknown to this build.
    ///
    /// # Errors
    ///
    /// Returns a typed corruption error on a checksum mismatch.
    pub fn section(&self, kind: SectionKind) -> Result<Option<&[u8]>, QuillError> {
        if !kind.is_known() {
            return Ok(None);
        }
        let Ok(index) = self
            .sections
            .binary_search_by_key(&kind, |entry| entry.kind)
        else {
            return Ok(None);
        };
        let entry = self.sections[index];
        let bytes = section_bytes(self.source.as_ref(), entry)
            .ok_or_else(|| corrupted(&self.path, "validated section range became invalid"))?;
        let result = self.section_checks[index].get_or_init(|| {
            let actual = xxh3_64(bytes);
            if actual == entry.xxh3 {
                Ok(())
            } else {
                Err(format!(
                    "section {} checksum mismatch: expected {:#018x}, got {actual:#018x}",
                    entry.kind.raw(),
                    entry.xxh3
                ))
            }
        });
        match result {
            Ok(()) => Ok(Some(bytes)),
            Err(detail) => Err(corrupted(&self.path, detail.clone())),
        }
    }

    /// Eagerly revalidate the structure, every section, and file-prefix witness.
    ///
    /// Unlike [`Self::section`], this always recomputes hashes so doctor flows
    /// do not trust an earlier lazy result. The returned prefix witness is the
    /// value freshly recomputed from the exact bytes that passed this complete
    /// validation; crate-internal manifest binding carries it forward instead
    /// of hashing the same prefix again.
    ///
    /// # Errors
    ///
    /// Returns a typed corruption error for any checksum mismatch.
    pub(crate) fn verify_with_file_witness(&self) -> Result<u64, QuillError> {
        let bytes = self.source.as_ref();
        let parsed = parse_container(bytes, &self.path, self.schema, self.limits)?;
        for entry in &parsed.sections {
            let payload = section_bytes(bytes, *entry)
                .ok_or_else(|| corrupted(&self.path, "validated section range became invalid"))?;
            let actual = xxh3_64(payload);
            if actual != entry.xxh3 {
                return Err(corrupted(
                    &self.path,
                    format!(
                        "section {} checksum mismatch: expected {:#018x}, got {actual:#018x}",
                        entry.kind.raw(),
                        entry.xxh3
                    ),
                ));
            }
        }
        self.verify_file_witness()
    }

    /// Eagerly revalidate the structure, every section, and file-prefix witness.
    ///
    /// Unlike [`Self::section`], this always recomputes hashes so doctor flows
    /// do not trust an earlier lazy result.
    ///
    /// # Errors
    ///
    /// Returns a typed corruption error for any checksum mismatch.
    pub fn verify(&self) -> Result<(), QuillError> {
        self.verify_with_file_witness().map(|_| ())
    }
}
