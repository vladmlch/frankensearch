use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use xxhash_rust::xxh3::{Xxh3, xxh3_64};

use crate::error::QuillError;

use super::codec::{
    align_up, crc32, header_len, invalid_segment, resource, validate_header_range,
    validate_section_sequence, write_header_fixed, write_section_entry,
};
use super::{
    FILE_PREFIX_LEN, FSLX_FORMAT_VERSION, FSLX_SECTION_ALIGNMENT, FSLX_SEGMENT_MAGIC,
    HEADER_CRC_LEN, PendingSegmentFile, SectionEntry, SectionInput, SectionKind, SegmentHeader,
    SegmentHeaderInput, SegmentLimits, SegmentWriteCheckpoint, TRAILER_LEN,
};

/// Owned canonical FSLX bytes ready for Keeper publication.
///
/// The canonical byte payload is backing-shared: [`Clone`] bumps a reference
/// count on the immutable byte buffer instead of copying it, so retaining a
/// pending segment across publication attempts and binding it into reader
/// snapshots are O(1) in the payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedSegment {
    pub(super) bytes: Arc<Vec<u8>>,
    header: SegmentHeader,
    sections: Vec<SectionEntry>,
    file_xxh3: u64,
    source_xxh3: u64,
}

/// Exact layout declaration for one section emitted directly into a segment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlannedSection {
    kind: SectionKind,
    flags: u16,
    len: usize,
}

impl PlannedSection {
    /// Declare one ordinary known section with no flags.
    pub const fn new(kind: SectionKind, len: usize) -> Self {
        Self {
            kind,
            flags: 0,
            len,
        }
    }
}

/// Pre-sized canonical FSLX container for direct section emission.
///
/// The assembler owns the final allocation from the outset. Callers write each
/// declared payload into its exact final slice, then [`Self::finish`] patches
/// the section hashes, header CRC, and trailer checksums.
pub struct SegmentAssembler {
    bytes: Vec<u8>,
    header: SegmentHeader,
    sections: Vec<SectionEntry>,
    next_section: usize,
    header_len: usize,
}

impl SegmentAssembler {
    /// Validate a complete section plan and allocate its exact canonical file.
    pub fn new(header: SegmentHeaderInput, planned: &[PlannedSection]) -> Result<Self, QuillError> {
        let limits = SegmentLimits::default();
        header.schema.validate()?;
        validate_header_range(header.docid_lo, header.docid_hi, header.doc_count)
            .map_err(invalid_segment)?;
        validate_section_sequence(
            header.schema,
            planned.len(),
            |index| {
                let section = planned[index];
                (section.kind, section.flags)
            },
            false,
        )
        .map_err(invalid_segment)?;

        if planned.len() > limits.max_sections {
            return Err(resource(
                "section table",
                format!(
                    "section count {} exceeds limit {}",
                    planned.len(),
                    limits.max_sections
                ),
            ));
        }
        let section_count = u16::try_from(planned.len())
            .map_err(|_| invalid_segment("section count does not fit u16"))?;
        let header_len = header_len(planned.len()).map_err(invalid_segment)?;
        if header_len > limits.max_header_bytes {
            return Err(resource(
                "segment header",
                format!(
                    "header length {header_len} exceeds limit {}",
                    limits.max_header_bytes
                ),
            ));
        }
        let mut cursor = FILE_PREFIX_LEN
            .checked_add(header_len)
            .and_then(|value| value.checked_add(HEADER_CRC_LEN))
            .ok_or_else(|| invalid_segment("header end overflow"))?;
        let mut sections = Vec::new();
        sections
            .try_reserve_exact(planned.len())
            .map_err(|_| resource("section table", "allocation failed"))?;
        for section in planned {
            cursor = align_up(cursor, FSLX_SECTION_ALIGNMENT)
                .ok_or_else(|| invalid_segment("section alignment overflow"))?;
            let offset = u64::try_from(cursor)
                .map_err(|_| invalid_segment("section offset does not fit u64"))?;
            let len = u64::try_from(section.len)
                .map_err(|_| invalid_segment("section length does not fit u64"))?;
            sections.push(SectionEntry {
                kind: section.kind,
                flags: section.flags,
                offset,
                len,
                xxh3: 0,
            });
            cursor = cursor
                .checked_add(section.len)
                .ok_or_else(|| invalid_segment("section end overflow"))?;
        }
        let file_len = cursor
            .checked_add(TRAILER_LEN)
            .ok_or_else(|| invalid_segment("file length overflow"))?;
        let file_len_u64 =
            u64::try_from(file_len).map_err(|_| invalid_segment("file length does not fit u64"))?;
        if file_len_u64 > limits.max_file_bytes {
            return Err(resource(
                "segment file",
                format!(
                    "file length {file_len_u64} exceeds limit {}",
                    limits.max_file_bytes
                ),
            ));
        }

        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(file_len)
            .map_err(|_| resource("segment file", "allocation failed"))?;
        let header_end = FILE_PREFIX_LEN
            .checked_add(header_len)
            .and_then(|value| value.checked_add(HEADER_CRC_LEN))
            .ok_or_else(|| invalid_segment("header end overflow"))?;
        bytes.resize(header_end, 0);
        Ok(Self {
            bytes,
            header: SegmentHeader {
                segment_id: header.segment_id,
                schema_id: header.schema.schema_id()?,
                docid_lo: header.docid_lo,
                docid_hi: header.docid_hi,
                doc_count: header.doc_count,
                created_unix_s: header.created_unix_s,
                engine_version: header.engine_version,
                section_count,
            },
            sections,
            next_section: 0,
            header_len,
        })
    }

    /// Append one payload at its exact planned offset without reallocating.
    pub fn write_section(
        &mut self,
        kind: SectionKind,
        write: impl FnOnce(&mut Vec<u8>),
    ) -> Result<(), QuillError> {
        let entry = self
            .sections
            .get(self.next_section)
            .copied()
            .ok_or_else(|| invalid_segment(format!("unplanned section kind {}", kind.raw())))?;
        if entry.kind != kind {
            return Err(invalid_segment(format!(
                "section writer supplied kind {}, expected {}",
                kind.raw(),
                entry.kind.raw()
            )));
        }
        let offset = usize::try_from(entry.offset)
            .map_err(|_| invalid_segment("section offset does not fit usize"))?;
        let len = usize::try_from(entry.len)
            .map_err(|_| invalid_segment("section length does not fit usize"))?;
        let expected_end = offset
            .checked_add(len)
            .ok_or_else(|| invalid_segment("section end overflow"))?;
        self.bytes.resize(offset, 0);
        write(&mut self.bytes);
        if self.bytes.len() != expected_end {
            return Err(invalid_segment(format!(
                "section kind {} writer ended at {}, expected {}",
                kind.raw(),
                self.bytes.len(),
                expected_end
            )));
        }
        self.next_section += 1;
        Ok(())
    }

    /// Copy one small prebuilt payload into its exact final section slice.
    pub fn copy_section(&mut self, kind: SectionKind, bytes: &[u8]) -> Result<(), QuillError> {
        self.write_section(kind, |output| output.extend_from_slice(bytes))
    }

    /// Borrow one fully emitted section before the remaining plan is written.
    pub fn written_section(&self, kind: SectionKind) -> Result<&[u8], QuillError> {
        let index = self
            .sections
            .iter()
            .position(|entry| entry.kind == kind)
            .ok_or_else(|| invalid_segment(format!("unplanned section kind {}", kind.raw())))?;
        if index >= self.next_section {
            return Err(invalid_segment(format!(
                "section kind {} has not been written",
                kind.raw()
            )));
        }
        let entry = self.sections[index];
        let start = usize::try_from(entry.offset)
            .map_err(|_| invalid_segment("section offset does not fit usize"))?;
        let len = usize::try_from(entry.len)
            .map_err(|_| invalid_segment("section length does not fit usize"))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| invalid_segment("section end overflow"))?;
        self.bytes
            .get(start..end)
            .ok_or_else(|| invalid_segment("written section exceeds segment bytes"))
    }

    /// Finalize hashes and framing after every planned section was emitted.
    pub fn finish(mut self) -> Result<EncodedSegment, QuillError> {
        if self.next_section != self.sections.len() {
            return Err(invalid_segment(format!(
                "planned section kind {} was not written",
                self.sections[self.next_section].kind.raw()
            )));
        }
        for entry in &mut self.sections {
            let start = usize::try_from(entry.offset)
                .map_err(|_| invalid_segment("section offset does not fit usize"))?;
            let len = usize::try_from(entry.len)
                .map_err(|_| invalid_segment("section length does not fit usize"))?;
            let end = start
                .checked_add(len)
                .ok_or_else(|| invalid_segment("section end overflow"))?;
            let section = self
                .bytes
                .get(start..end)
                .ok_or_else(|| invalid_segment("planned section exceeds segment bytes"))?;
            entry.xxh3 = xxh3_64(section);
        }

        let header_len_u32 = u32::try_from(self.header_len)
            .map_err(|_| invalid_segment("header length does not fit u32"))?;
        let mut header_bytes = Vec::new();
        header_bytes
            .try_reserve_exact(self.header_len)
            .map_err(|_| resource("segment header", "allocation failed"))?;
        write_header_fixed(&mut header_bytes, self.header);
        for entry in &self.sections {
            write_section_entry(&mut header_bytes, *entry);
        }
        debug_assert_eq!(header_bytes.len(), self.header_len);
        self.bytes[..FSLX_SEGMENT_MAGIC.len()].copy_from_slice(&FSLX_SEGMENT_MAGIC);
        self.bytes[8..12].copy_from_slice(&FSLX_FORMAT_VERSION.to_le_bytes());
        self.bytes[12..FILE_PREFIX_LEN].copy_from_slice(&header_len_u32.to_le_bytes());
        let header_end = FILE_PREFIX_LEN
            .checked_add(self.header_len)
            .ok_or_else(|| invalid_segment("header end overflow"))?;
        self.bytes[FILE_PREFIX_LEN..header_end].copy_from_slice(&header_bytes);
        let crc_end = header_end
            .checked_add(HEADER_CRC_LEN)
            .ok_or_else(|| invalid_segment("header CRC end overflow"))?;
        self.bytes[header_end..crc_end].copy_from_slice(&crc32(&header_bytes).to_le_bytes());

        let mut source_hasher = Xxh3::new();
        source_hasher.update(&self.bytes);
        let file_xxh3 = source_hasher.digest();
        let hash_bytes = file_xxh3.to_le_bytes();
        let trailer_crc_bytes = crc32(&hash_bytes).to_le_bytes();
        self.bytes.extend_from_slice(&hash_bytes);
        self.bytes.extend_from_slice(&trailer_crc_bytes);
        source_hasher.update(&hash_bytes);
        source_hasher.update(&trailer_crc_bytes);
        let source_xxh3 = source_hasher.digest();

        Ok(EncodedSegment {
            bytes: Arc::new(self.bytes),
            header: self.header,
            sections: self.sections,
            file_xxh3,
            source_xxh3,
        })
    }
}

impl EncodedSegment {
    /// Encode one canonical FSLX segment with default resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns a typed invariant error for invalid metadata, section presence,
    /// flags, ordering, layout overflow, or resource exhaustion.
    pub fn encode(
        header: SegmentHeaderInput,
        sections: &[SectionInput<'_>],
    ) -> Result<Self, QuillError> {
        Self::encode_with_limits(header, sections, SegmentLimits::default())
    }

    /// Encode one canonical FSLX segment with explicit resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::encode`].
    pub fn encode_with_limits(
        header: SegmentHeaderInput,
        sections: &[SectionInput<'_>],
        limits: SegmentLimits,
    ) -> Result<Self, QuillError> {
        Self::encode_with_limits_impl(header, sections, limits, false)
    }

    pub(super) fn encode_with_limits_impl(
        header: SegmentHeaderInput,
        sections: &[SectionInput<'_>],
        limits: SegmentLimits,
        allow_unregistered_extensions: bool,
    ) -> Result<Self, QuillError> {
        header.schema.validate()?;
        validate_header_range(header.docid_lo, header.docid_hi, header.doc_count)
            .map_err(invalid_segment)?;
        validate_section_sequence(
            header.schema,
            sections.len(),
            |index| {
                let section = sections[index];
                (section.kind, section.flags)
            },
            allow_unregistered_extensions,
        )
        .map_err(invalid_segment)?;

        if sections.len() > limits.max_sections {
            return Err(resource(
                "section table",
                format!(
                    "section count {} exceeds limit {}",
                    sections.len(),
                    limits.max_sections
                ),
            ));
        }
        let section_count = u16::try_from(sections.len())
            .map_err(|_| invalid_segment("section count does not fit u16"))?;
        let header_len = header_len(sections.len()).map_err(invalid_segment)?;
        if header_len > limits.max_header_bytes {
            return Err(resource(
                "segment header",
                format!(
                    "header length {header_len} exceeds limit {}",
                    limits.max_header_bytes
                ),
            ));
        }
        let header_len_u32 = u32::try_from(header_len)
            .map_err(|_| invalid_segment("header length does not fit u32"))?;
        let mut cursor = FILE_PREFIX_LEN
            .checked_add(header_len)
            .and_then(|value| value.checked_add(HEADER_CRC_LEN))
            .ok_or_else(|| invalid_segment("header end overflow"))?;

        let mut table = Vec::new();
        table
            .try_reserve_exact(sections.len())
            .map_err(|_| resource("section table", "allocation failed"))?;
        for section in sections {
            cursor = align_up(cursor, FSLX_SECTION_ALIGNMENT)
                .ok_or_else(|| invalid_segment("section alignment overflow"))?;
            let offset = u64::try_from(cursor)
                .map_err(|_| invalid_segment("section offset does not fit u64"))?;
            let len = u64::try_from(section.bytes.len())
                .map_err(|_| invalid_segment("section length does not fit u64"))?;
            table.push(SectionEntry {
                kind: section.kind,
                flags: section.flags,
                offset,
                len,
                xxh3: xxh3_64(section.bytes),
            });
            cursor = cursor
                .checked_add(section.bytes.len())
                .ok_or_else(|| invalid_segment("section end overflow"))?;
        }
        let file_len = cursor
            .checked_add(TRAILER_LEN)
            .ok_or_else(|| invalid_segment("file length overflow"))?;
        let file_len_u64 =
            u64::try_from(file_len).map_err(|_| invalid_segment("file length does not fit u64"))?;
        if file_len_u64 > limits.max_file_bytes {
            return Err(resource(
                "segment file",
                format!(
                    "file length {file_len_u64} exceeds limit {}",
                    limits.max_file_bytes
                ),
            ));
        }

        let schema_id = header.schema.schema_id()?;
        let parsed_header = SegmentHeader {
            segment_id: header.segment_id,
            schema_id,
            docid_lo: header.docid_lo,
            docid_hi: header.docid_hi,
            doc_count: header.doc_count,
            created_unix_s: header.created_unix_s,
            engine_version: header.engine_version,
            section_count,
        };
        let mut header_bytes = Vec::new();
        header_bytes
            .try_reserve_exact(header_len)
            .map_err(|_| resource("segment header", "allocation failed"))?;
        write_header_fixed(&mut header_bytes, parsed_header);
        for entry in &table {
            write_section_entry(&mut header_bytes, *entry);
        }
        debug_assert_eq!(header_bytes.len(), header_len);

        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(file_len)
            .map_err(|_| resource("segment file", "allocation failed"))?;
        bytes.extend_from_slice(&FSLX_SEGMENT_MAGIC);
        bytes.extend_from_slice(&FSLX_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&header_len_u32.to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&crc32(&header_bytes).to_le_bytes());
        for (section, entry) in sections.iter().zip(&table) {
            let offset = usize::try_from(entry.offset)
                .map_err(|_| invalid_segment("section offset does not fit usize"))?;
            bytes.resize(offset, 0);
            bytes.extend_from_slice(section.bytes);
        }
        debug_assert_eq!(bytes.len(), file_len - TRAILER_LEN);
        let mut source_hasher = Xxh3::new();
        source_hasher.update(&bytes);
        let file_xxh3 = source_hasher.digest();
        let hash_bytes = file_xxh3.to_le_bytes();
        let trailer_crc_bytes = crc32(&hash_bytes).to_le_bytes();
        bytes.extend_from_slice(&hash_bytes);
        bytes.extend_from_slice(&trailer_crc_bytes);
        source_hasher.update(&hash_bytes);
        source_hasher.update(&trailer_crc_bytes);
        let source_xxh3 = source_hasher.digest();
        debug_assert_eq!(bytes.len(), file_len);

        Ok(Self {
            bytes: Arc::new(bytes),
            header: parsed_header,
            sections: table,
            file_xxh3,
            source_xxh3,
        })
    }

    /// Exact canonical segment bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the wrapper and return its canonical bytes.
    ///
    /// A uniquely owned segment gives back its buffer without copying; a
    /// segment whose byte backing is still shared with readers or retained
    /// publication state clones the bytes instead.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        Arc::try_unwrap(self.bytes).unwrap_or_else(|arc| (*arc).clone())
    }

    /// Parsed header represented by these bytes.
    #[must_use]
    pub const fn header(&self) -> SegmentHeader {
        self.header
    }

    /// Canonical section-table entries.
    #[must_use]
    pub fn section_entries(&self) -> &[SectionEntry] {
        &self.sections
    }

    /// Exact complete file length.
    #[must_use]
    pub fn file_len(&self) -> u64 {
        u64::try_from(self.bytes.len()).unwrap_or(u64::MAX)
    }

    /// xxh3-64 over the file prefix before the trailer, stored in the trailer.
    #[must_use]
    pub const fn file_xxh3(&self) -> u64 {
        self.file_xxh3
    }

    /// xxh3-64 over every canonical FSLX byte, including the trailer.
    ///
    /// This derived witness is not persisted in FSLX. It binds a durability
    /// sidecar to the complete source file without hashing the encoded bytes a
    /// second time.
    #[must_use]
    pub const fn source_xxh3(&self) -> u64 {
        self.source_xxh3
    }

    /// Create and durably sync the canonical Keeper temp file.
    ///
    /// This method never renames or publishes the file. It uses `create_new`,
    /// so an existing temp is preserved and reported as an I/O error.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the directory is unavailable, the temp already
    /// exists, a write fails, or `sync_all` fails.
    pub fn write_temp(&self, directory: &Path) -> Result<PendingSegmentFile, QuillError> {
        self.write_temp_with_observer(directory, |_, _| Ok(()))
    }

    /// Write the canonical temp, or adopt an exact synced temp from a prior attempt.
    ///
    /// A differing same-name artifact is preserved and rejected. This method is
    /// crate-private because only a retained `KeeperWriter` may establish that
    /// retrying the temp is serialized with every other directory mutation.
    pub(crate) fn write_temp_retryable(
        &self,
        directory: &Path,
    ) -> Result<PendingSegmentFile, QuillError> {
        let path = directory.join(format!(".tmp-segment-{:016x}", self.header.segment_id));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                file.write_all(&self.bytes)?;
                file.flush()?;
                file.sync_all()?;
            }
            Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {
                let metadata = fs::symlink_metadata(&path)?;
                if !metadata.file_type().is_file() || metadata.len() != self.file_len() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "existing segment temp is not the exact expected regular file",
                    )
                    .into());
                }
                let existing = fs::read(&path)?;
                if existing.as_slice() != self.bytes.as_slice() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::AlreadyExists,
                        "existing segment temp differs from the staged FSLX bytes",
                    )
                    .into());
                }
                OpenOptions::new().read(true).open(&path)?.sync_all()?;
            }
            Err(source) => return Err(source.into()),
        }
        Ok(self.pending_file(path))
    }

    fn pending_file(&self, path: PathBuf) -> PendingSegmentFile {
        PendingSegmentFile {
            path,
            segment_id: self.header.segment_id,
            file_len: self.file_len(),
            file_xxh3: self.file_xxh3,
            source_xxh3: self.source_xxh3,
        }
    }

    pub(crate) fn write_temp_with_observer<O>(
        &self,
        directory: &Path,
        mut observe: O,
    ) -> Result<PendingSegmentFile, QuillError>
    where
        O: FnMut(SegmentWriteCheckpoint, &Path) -> Result<(), QuillError>,
    {
        let path = directory.join(format!(".tmp-segment-{:016x}", self.header.segment_id));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.write_all(&self.bytes)?;
        file.flush()?;
        observe(SegmentWriteCheckpoint::TempWritten, &path)?;
        file.sync_all()?;
        observe(SegmentWriteCheckpoint::TempSynced, &path)?;
        Ok(self.pending_file(path))
    }
}

impl AsRef<[u8]> for EncodedSegment {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}
