#![allow(clippy::module_name_repetitions)]
//! FSLX segment container encoding and validation.
//!
//! This module deliberately treats section payloads as opaque bytes. Grimoire,
//! Quiver, and later codecs own their payload grammars; the segment container
//! owns only framing, alignment, schema-derived presence, and checksums.

use std::path::{Path, PathBuf};

use crate::schema::SchemaDescriptor;

mod codec;
mod reader;
mod writer;

#[cfg(test)]
mod tests;

pub use reader::SegmentReader;
pub use writer::EncodedSegment;
pub(crate) use writer::{PlannedSection, SegmentAssembler};

/// Magic bytes at the start of every FSLX segment.
pub const FSLX_SEGMENT_MAGIC: [u8; 8] = *b"FSLXSEG\0";
/// The only FSLX segment format version understood by this reader.
pub const FSLX_FORMAT_VERSION: u32 = 1;
/// Required byte alignment of every section start.
pub const FSLX_SECTION_ALIGNMENT: usize = 64;
/// Unknown section kinds are skippable only when this flag is the sole flag.
pub const OPTIONAL_SKIPPABLE: u16 = 1;

const FILE_PREFIX_LEN: usize = 16;
const FIXED_HEADER_LEN: usize = 56;
const SECTION_ENTRY_LEN: usize = 28;
const HEADER_CRC_LEN: usize = 4;
const TRAILER_LEN: usize = 12;
const DEFAULT_MAX_FILE_BYTES: u64 = u64::MAX;
const DEFAULT_MAX_HEADER_BYTES: usize = 2 * 1024 * 1024;
const DEFAULT_MAX_SECTIONS: usize = 65_535;
const MEMORY_PATH: &str = "<in-memory FSLX segment>";
const MAX_DOCID_EXCLUSIVE: u64 = 1_u64 << 32;

/// Stable numeric identity of one FSLX section kind.
///
/// A newtype, rather than a closed enum, preserves forward-compatible unknown
/// optional kinds.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SectionKind(u16);

impl SectionKind {
    /// Prefix-block term dictionary.
    pub const TERMDICT: Self = Self(1);
    /// Posting-list streams.
    pub const POSTINGS: Self = Self(2);
    /// Optional token-position streams.
    pub const POSITIONS: Self = Self(3);
    /// Block-max and skip metadata.
    pub const BLOCKMAX: Self = Self(4);
    /// Per-field fieldnorm columns.
    pub const DOCLEN: Self = Self(5);
    /// Global docid to external identifier mapping.
    pub const IDMAP: Self = Self(6);
    /// External identifier to global docid hash table.
    pub const IDHASH: Self = Self(7);
    /// Optional indexed or fast numeric columns.
    pub const NUMERIC: Self = Self(8);
    /// Optional stored-field bytes.
    pub const STOREDMETA: Self = Self(9);
    /// Per-field at-seal statistics.
    pub const STATS: Self = Self(10);

    /// Preserve a raw durable section kind, including future kinds.
    #[must_use]
    pub const fn from_raw(raw: u16) -> Self {
        Self(raw)
    }

    /// Return the durable `u16` section kind.
    #[must_use]
    pub const fn raw(self) -> u16 {
        self.0
    }

    const fn is_known(self) -> bool {
        self.0 >= Self::TERMDICT.0 && self.0 <= Self::STATS.0
    }
}

/// One opaque payload supplied to the canonical segment writer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SectionInput<'a> {
    /// Stable section kind.
    pub kind: SectionKind,
    /// Versioned section flags.
    pub flags: u16,
    /// Exact payload bytes, excluding alignment padding.
    pub bytes: &'a [u8],
}

impl<'a> SectionInput<'a> {
    /// Construct an ordinary known section with no flags.
    #[must_use]
    pub const fn new(kind: SectionKind, bytes: &'a [u8]) -> Self {
        Self {
            kind,
            flags: 0,
            bytes,
        }
    }
}

/// Caller-supplied immutable header fields for a fresh segment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SegmentHeaderInput {
    /// Random collision-checked segment identifier.
    pub segment_id: u64,
    /// Compile-time schema governing conditional sections.
    pub schema: SchemaDescriptor,
    /// Inclusive global docid lower bound.
    pub docid_lo: u64,
    /// Exclusive global docid upper bound.
    pub docid_hi: u64,
    /// Documents live when the segment was sealed.
    pub doc_count: u32,
    /// Informational creation timestamp.
    pub created_unix_s: i64,
    /// Informational packed engine version.
    pub engine_version: u32,
}

/// Parsed immutable fields from an FSLX segment header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SegmentHeader {
    /// Random collision-checked segment identifier.
    pub segment_id: u64,
    /// Stable schema descriptor hash.
    pub schema_id: u64,
    /// Inclusive global docid lower bound.
    pub docid_lo: u64,
    /// Exclusive global docid upper bound.
    pub docid_hi: u64,
    /// Documents live when the segment was sealed.
    pub doc_count: u32,
    /// Informational creation timestamp.
    pub created_unix_s: i64,
    /// Informational packed engine version.
    pub engine_version: u32,
    /// Number of entries in the section table.
    pub section_count: u16,
}

/// Validated section-table metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SectionEntry {
    /// Stable or forward-compatible section kind.
    pub kind: SectionKind,
    /// Versioned section flags.
    pub flags: u16,
    /// Absolute, 64-byte-aligned file offset.
    pub offset: u64,
    /// Exact payload length, excluding padding.
    pub len: u64,
    /// xxh3-64 over the exact payload bytes.
    pub xxh3: u64,
}

/// Explicit resource ceilings for hostile segment input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SegmentLimits {
    /// Maximum complete segment file size.
    pub max_file_bytes: u64,
    /// Maximum header block size.
    pub max_header_bytes: usize,
    /// Maximum retained section metadata entries.
    pub max_sections: usize,
}

impl Default for SegmentLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
            max_header_bytes: DEFAULT_MAX_HEADER_BYTES,
            max_sections: DEFAULT_MAX_SECTIONS,
        }
    }
}

/// Internal fault-injection points in the canonical segment temp writer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SegmentWriteCheckpoint {
    /// All bytes reached the file handle but have not been fsynced.
    TempWritten,
    /// The complete temp file has been fsynced.
    TempSynced,
}

/// A synced but unpublished `.tmp-segment-*` artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PendingSegmentFile {
    path: PathBuf,
    segment_id: u64,
    file_len: u64,
    file_xxh3: u64,
    source_xxh3: u64,
}

impl PendingSegmentFile {
    /// Temp path awaiting Keeper rename and manifest publication.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Random segment identifier.
    #[must_use]
    pub const fn segment_id(&self) -> u64 {
        self.segment_id
    }

    /// Exact file length.
    #[must_use]
    pub const fn file_len(&self) -> u64 {
        self.file_len
    }

    /// xxh3-64 over the file prefix before the trailer.
    #[must_use]
    pub const fn file_xxh3(&self) -> u64 {
        self.file_xxh3
    }

    /// xxh3-64 over every canonical FSLX byte, including the trailer.
    ///
    /// This derived witness is not persisted in FSLX.
    #[must_use]
    pub const fn source_xxh3(&self) -> u64 {
        self.source_xxh3
    }
}
