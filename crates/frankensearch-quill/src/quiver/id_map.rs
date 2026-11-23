use frankensearch_core::DocId;
use thiserror::Error;
use xxhash_rust::xxh3::xxh3_64;

use super::{read_u32, read_u64};

/// Packed size of the `{ entry_count: u32, blob_offset: u32 }` IDMAP header.
pub const ID_MAP_HEADER_LEN: usize = 8;
/// Packed size of one IDMAP offset.
pub const ID_MAP_OFFSET_LEN: usize = 4;
/// Packed size of one IDMAP canonical-content hash.
pub const ID_MAP_HASH_LEN: usize = 8;
/// Largest IDMAP span whose fixed header, offsets, and hashes fit `blob_offset: u32`.
pub const ID_MAP_MAX_SPAN: u64 = 357_913_940;
/// Packed size of the IDHASH capacity header.
pub const ID_HASH_HEADER_LEN: usize = 4;
/// Packed size of one `{ key_hash: u64, doc_ord_plus1: u32 }` IDHASH slot.
pub const ID_HASH_ENTRY_LEN: usize = 12;
/// Stable xxh3-64 seed for durable IDHASH keys (`"QUILL1"`).
pub const ID_HASH_SEED: u64 = 0x5155_494c_4c31;

/// Hash canonicalized document content for the durable IDMAP witness.
///
/// This is deliberately unseeded xxh3-64. It is distinct from both the
/// seeded IDHASH key and `DocumentFingerprint`'s in-memory FNV witness.
#[must_use]
pub fn id_map_content_hash(canonical_content: &[u8]) -> u64 {
    xxh3_64(canonical_content)
}

/// One present positional IDMAP row supplied to the deterministic writer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IdMapEntryInput<'a> {
    /// Non-empty external document identifier.
    pub document_id: &'a str,
    /// Unseeded xxh3-64 of the canonical document content.
    pub content_hash: u64,
}

impl<'a> IdMapEntryInput<'a> {
    /// Construct a row from an externally computed canonical-content witness.
    #[must_use]
    pub const fn new(document_id: &'a str, content_hash: u64) -> Self {
        Self {
            document_id,
            content_hash,
        }
    }

    /// Construct a row while hashing the supplied canonical content.
    #[must_use]
    pub fn from_canonical_content(document_id: &'a str, canonical_content: &[u8]) -> Self {
        Self::new(document_id, id_map_content_hash(canonical_content))
    }
}

/// Explicit resource ceilings for an FSLX IDMAP section.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IdMapLimits {
    /// Maximum half-open global docid span.
    pub max_docid_span: u64,
    /// Maximum complete section size.
    pub max_section_bytes: u64,
    /// Maximum concatenated document-identifier blob size.
    pub max_blob_bytes: u64,
    /// Maximum one-document identifier size.
    pub max_document_id_bytes: u64,
}

impl Default for IdMapLimits {
    fn default() -> Self {
        Self {
            max_docid_span: ID_MAP_MAX_SPAN,
            max_section_bytes: u64::from(u32::MAX),
            max_blob_bytes: u64::from(u32::MAX),
            max_document_id_bytes: u64::from(u32::MAX),
        }
    }
}

/// Typed failures from IDMAP encoding, validation, or concatenation.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum IdMapCodecError {
    /// A segment's half-open docid range was reversed or exceeded u32 payloads.
    #[error("invalid IDMAP docid range [{docid_lo}, {docid_hi})")]
    InvalidDocIdRange {
        /// Inclusive lower bound.
        docid_lo: u64,
        /// Exclusive upper bound.
        docid_hi: u64,
    },
    /// A caller or durable declaration exceeded an explicit resource ceiling.
    #[error("IDMAP {resource} {actual} exceeds limit {limit}")]
    ResourceLimit {
        /// Bounded resource name.
        resource: &'static str,
        /// Rejected amount.
        actual: u64,
        /// Configured ceiling.
        limit: u64,
    },
    /// Positional input must cover the complete segment span.
    #[error("IDMAP has {actual} input rows, expected {expected}")]
    EntryCountMismatch {
        /// Required span.
        expected: usize,
        /// Supplied or decoded count.
        actual: usize,
    },
    /// Equal adjacent offsets are reserved for holes, so IDs cannot be empty.
    #[error("IDMAP document id at ordinal {ordinal} is empty")]
    EmptyDocumentId {
        /// Rejected positional ordinal.
        ordinal: usize,
    },
    /// Checked layout arithmetic overflowed.
    #[error("IDMAP arithmetic overflow while computing {field}")]
    ArithmeticOverflow {
        /// Layout component being computed.
        field: &'static str,
    },
    /// A durable offset did not fit its u32 slot.
    #[error("IDMAP offset {offset} does not fit u32")]
    OffsetUnrepresentable {
        /// Computed offset.
        offset: usize,
    },
    /// The durable blob offset was not the one canonical packed layout.
    #[error("IDMAP blob offset {encoded} is non-canonical; expected {expected}")]
    NonCanonicalBlobOffset {
        /// Durable value.
        encoded: u32,
        /// Required value.
        expected: u32,
    },
    /// Every offset table begins at zero.
    #[error("IDMAP first offset is {actual}, expected 0")]
    NonZeroFirstOffset {
        /// Rejected first offset.
        actual: u32,
    },
    /// Offsets must be monotone.
    #[error("IDMAP offsets descend at ordinal {ordinal}: {start} to {end}")]
    DescendingOffsets {
        /// Rejected ordinal.
        ordinal: usize,
        /// Start offset.
        start: u32,
        /// End offset.
        end: u32,
    },
    /// An offset pointed beyond the identifier blob.
    #[error("IDMAP offset {offset} at ordinal {ordinal} exceeds blob length {blob_len}")]
    OffsetOutOfBounds {
        /// Rejected ordinal.
        ordinal: usize,
        /// Rejected offset.
        offset: u32,
        /// Exact available blob length.
        blob_len: usize,
    },
    /// Hole rows must carry the canonical zero content hash.
    #[error("IDMAP hole at ordinal {ordinal} has non-zero content hash {content_hash:#018x}")]
    NonZeroHoleHash {
        /// Rejected ordinal.
        ordinal: usize,
        /// Rejected witness.
        content_hash: u64,
    },
    /// Present identifier bytes must be valid UTF-8.
    #[error("IDMAP document id at ordinal {ordinal} is not valid UTF-8")]
    InvalidUtf8 {
        /// Rejected ordinal.
        ordinal: usize,
    },
    /// The terminal offset must consume the exact identifier blob.
    #[error("IDMAP terminal offset is {terminal}, but blob length is {blob_len}")]
    TerminalOffsetMismatch {
        /// Durable terminal offset.
        terminal: u32,
        /// Exact remaining bytes.
        blob_len: usize,
    },
    /// Durable bytes ended before the canonical layout did.
    #[error("truncated IDMAP section: expected at least {expected} bytes, got {actual}")]
    Truncated {
        /// Minimum required length.
        expected: usize,
        /// Available length.
        actual: usize,
    },
    /// Canonical sections end exactly after the identifier blob.
    #[error("IDMAP section has trailing bytes: expected {expected}, got {actual}")]
    TrailingBytes {
        /// Canonical complete size.
        expected: usize,
        /// Actual size.
        actual: usize,
    },
    /// Concatenation requires at least one validated source section.
    #[error("cannot concatenate an empty IDMAP section list")]
    EmptyConcat,
    /// Concat inputs must be ordered and non-overlapping.
    #[error("IDMAP concat section {index} begins at {current_lo}, before prior end {previous_hi}")]
    ConcatRangeOrder {
        /// Rejected source index.
        index: usize,
        /// Prior source's exclusive high bound.
        previous_hi: u64,
        /// Current source's inclusive low bound.
        current_lo: u64,
    },
    /// Fallible allocation failed without panicking.
    #[error("unable to reserve {bytes} bytes for IDMAP {resource}")]
    Allocation {
        /// Allocation purpose.
        resource: &'static str,
        /// Requested amount.
        bytes: usize,
    },
}

/// Owned canonical bytes produced by the fresh-seal IDMAP writer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedIdMapSection {
    pub(super) bytes: Vec<u8>,
    pub(super) docid_lo: u64,
    docid_hi: u64,
    pub(super) present_count: usize,
}

/// Validated exact layout for directly concatenating IDMAP source views.
pub struct IdMapConcatPlan {
    pub(super) docid_lo: u64,
    docid_hi: u64,
    pub(super) span: usize,
    blob_offset: usize,
    blob_len: usize,
    total_len: usize,
    pub(super) present_count: usize,
}

impl EncodedIdMapSection {
    /// Encode span-aligned document identifiers and canonical-content hashes.
    ///
    /// # Errors
    ///
    /// Rejects invalid ranges, row-count drift, empty or oversized identifiers,
    /// resource abuse, arithmetic overflow, and allocation failure.
    pub fn encode(
        docid_lo: u64,
        docid_hi: u64,
        entries: &[Option<IdMapEntryInput<'_>>],
    ) -> Result<Self, IdMapCodecError> {
        Self::encode_with_limits(docid_lo, docid_hi, entries, IdMapLimits::default())
    }

    /// Encode with caller-selected validation and allocation ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::encode`].
    pub fn encode_with_limits(
        docid_lo: u64,
        docid_hi: u64,
        entries: &[Option<IdMapEntryInput<'_>>],
        limits: IdMapLimits,
    ) -> Result<Self, IdMapCodecError> {
        let span = checked_id_map_span(docid_lo, docid_hi, limits)?;
        if entries.len() != span {
            return Err(IdMapCodecError::EntryCountMismatch {
                expected: span,
                actual: entries.len(),
            });
        }
        let mut blob_len = 0_usize;
        let mut present_count = 0_usize;
        for (ordinal, entry) in entries.iter().enumerate() {
            let Some(entry) = entry else {
                continue;
            };
            if entry.document_id.is_empty() {
                return Err(IdMapCodecError::EmptyDocumentId { ordinal });
            }
            let id_len = u64::try_from(entry.document_id.len()).unwrap_or(u64::MAX);
            if id_len > limits.max_document_id_bytes {
                return Err(IdMapCodecError::ResourceLimit {
                    resource: "document id bytes",
                    actual: id_len,
                    limit: limits.max_document_id_bytes,
                });
            }
            blob_len = blob_len.checked_add(entry.document_id.len()).ok_or(
                IdMapCodecError::ArithmeticOverflow {
                    field: "identifier blob length",
                },
            )?;
            present_count += 1;
        }
        validate_id_map_blob_len(blob_len, limits)?;
        let (blob_offset, total_len) = id_map_layout(span, blob_len, limits)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(total_len)
            .map_err(|_| IdMapCodecError::Allocation {
                resource: "section bytes",
                bytes: total_len,
            })?;
        let entry_count = u32::try_from(span).map_err(|_| IdMapCodecError::ResourceLimit {
            resource: "durable entry count",
            actual: u64::try_from(span).unwrap_or(u64::MAX),
            limit: u64::from(u32::MAX),
        })?;
        let blob_offset_u32 =
            u32::try_from(blob_offset).map_err(|_| IdMapCodecError::OffsetUnrepresentable {
                offset: blob_offset,
            })?;
        bytes.extend_from_slice(&entry_count.to_le_bytes());
        bytes.extend_from_slice(&blob_offset_u32.to_le_bytes());
        let mut offset = 0_u32;
        bytes.extend_from_slice(&offset.to_le_bytes());
        for entry in entries {
            if let Some(entry) = entry {
                let id_len = u32::try_from(entry.document_id.len()).map_err(|_| {
                    IdMapCodecError::OffsetUnrepresentable {
                        offset: entry.document_id.len(),
                    }
                })?;
                offset = offset
                    .checked_add(id_len)
                    .ok_or(IdMapCodecError::ArithmeticOverflow {
                        field: "identifier offset",
                    })?;
            }
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        for entry in entries {
            bytes.extend_from_slice(&entry.map_or(0, |entry| entry.content_hash).to_le_bytes());
        }
        debug_assert_eq!(bytes.len(), blob_offset);
        for entry in entries.iter().flatten() {
            bytes.extend_from_slice(entry.document_id.as_bytes());
        }
        debug_assert_eq!(bytes.len(), total_len);
        Ok(Self {
            bytes,
            docid_lo,
            docid_hi,
            present_count,
        })
    }

    /// Borrow the exact canonical durable bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the wrapper and return its durable bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Number of non-hole rows.
    #[must_use]
    pub const fn present_count(&self) -> usize {
        self.present_count
    }

    /// Re-open the owned bytes through the validating zero-copy reader.
    ///
    /// # Errors
    ///
    /// Returns an error if an internal invariant was violated.
    pub fn section(&self) -> Result<IdMapSection<'_>, IdMapCodecError> {
        IdMapSection::parse(&self.bytes, self.docid_lo, self.docid_hi)
    }
}

/// One borrowed, present IDMAP row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IdMapEntry<'a> {
    document_id: &'a str,
    content_hash: u64,
}

impl<'a> IdMapEntry<'a> {
    /// Borrow the exact external document identifier bytes as UTF-8.
    #[must_use]
    pub const fn document_id(self) -> &'a str {
        self.document_id
    }

    /// Unseeded xxh3-64 canonical-content witness.
    #[must_use]
    pub const fn content_hash(self) -> u64 {
        self.content_hash
    }

    /// Materialize the public compact document-identifier representation.
    #[must_use]
    pub fn materialize(self) -> DocId {
        DocId::new(self.document_id)
    }
}

/// Borrowed, fully validated FSLX IDMAP section.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IdMapSection<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) docid_lo: u64,
    docid_hi: u64,
    offsets: &'a [u8],
    hashes: &'a [u8],
    pub(super) blob: &'a str,
    pub(super) present_count: usize,
}

impl<'a> IdMapSection<'a> {
    /// Validate an IDMAP payload against its segment range.
    ///
    /// # Errors
    ///
    /// Rejects range/count drift, a noncanonical layout, malformed offsets,
    /// non-zero hole hashes, empty or invalid UTF-8 identifiers, truncation,
    /// trailing bytes, resource abuse, and allocation-independent overflow.
    pub fn parse(bytes: &'a [u8], docid_lo: u64, docid_hi: u64) -> Result<Self, IdMapCodecError> {
        Self::parse_with_limits(bytes, docid_lo, docid_hi, IdMapLimits::default())
    }

    /// Validate with caller-selected resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::parse`].
    pub fn parse_with_limits(
        bytes: &'a [u8],
        docid_lo: u64,
        docid_hi: u64,
        limits: IdMapLimits,
    ) -> Result<Self, IdMapCodecError> {
        let span = checked_id_map_span(docid_lo, docid_hi, limits)?;
        let section_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        if section_len > limits.max_section_bytes {
            return Err(IdMapCodecError::ResourceLimit {
                resource: "section bytes",
                actual: section_len,
                limit: limits.max_section_bytes,
            });
        }
        if bytes.len() < ID_MAP_HEADER_LEN {
            return Err(IdMapCodecError::Truncated {
                expected: ID_MAP_HEADER_LEN,
                actual: bytes.len(),
            });
        }
        let encoded_count =
            usize::try_from(read_u32(bytes, 0).ok_or(IdMapCodecError::Truncated {
                expected: std::mem::size_of::<u32>(),
                actual: bytes.len(),
            })?)
            .map_err(|_| IdMapCodecError::ResourceLimit {
                resource: "host entry count",
                actual: u64::from(u32::MAX),
                limit: u64::try_from(usize::MAX).unwrap_or(u64::MAX),
            })?;
        if encoded_count != span {
            return Err(IdMapCodecError::EntryCountMismatch {
                expected: span,
                actual: encoded_count,
            });
        }
        let expected_blob_offset = id_map_prefix_len(span)?;
        let expected_blob_offset_u32 = u32::try_from(expected_blob_offset).map_err(|_| {
            IdMapCodecError::OffsetUnrepresentable {
                offset: expected_blob_offset,
            }
        })?;
        let encoded_blob_offset = read_u32(bytes, 4).ok_or(IdMapCodecError::Truncated {
            expected: ID_MAP_HEADER_LEN,
            actual: bytes.len(),
        })?;
        if encoded_blob_offset != expected_blob_offset_u32 {
            return Err(IdMapCodecError::NonCanonicalBlobOffset {
                encoded: encoded_blob_offset,
                expected: expected_blob_offset_u32,
            });
        }
        if bytes.len() < expected_blob_offset {
            return Err(IdMapCodecError::Truncated {
                expected: expected_blob_offset,
                actual: bytes.len(),
            });
        }
        let offsets_len = span
            .checked_add(1)
            .and_then(|count| count.checked_mul(ID_MAP_OFFSET_LEN))
            .ok_or(IdMapCodecError::ArithmeticOverflow {
                field: "offset table length",
            })?;
        let offsets_start = ID_MAP_HEADER_LEN;
        let offsets_end =
            offsets_start
                .checked_add(offsets_len)
                .ok_or(IdMapCodecError::ArithmeticOverflow {
                    field: "offset table end",
                })?;
        let hashes_len =
            span.checked_mul(ID_MAP_HASH_LEN)
                .ok_or(IdMapCodecError::ArithmeticOverflow {
                    field: "hash table length",
                })?;
        let hashes_end =
            offsets_end
                .checked_add(hashes_len)
                .ok_or(IdMapCodecError::ArithmeticOverflow {
                    field: "hash table end",
                })?;
        debug_assert_eq!(hashes_end, expected_blob_offset);
        let offsets = bytes
            .get(offsets_start..offsets_end)
            .ok_or(IdMapCodecError::Truncated {
                expected: offsets_end,
                actual: bytes.len(),
            })?;
        let hashes = bytes
            .get(offsets_end..hashes_end)
            .ok_or(IdMapCodecError::Truncated {
                expected: hashes_end,
                actual: bytes.len(),
            })?;
        let blob_bytes = bytes.get(hashes_end..).ok_or(IdMapCodecError::Truncated {
            expected: hashes_end,
            actual: bytes.len(),
        })?;
        let first = id_map_offset(offsets, 0).ok_or(IdMapCodecError::Truncated {
            expected: offsets_start + ID_MAP_OFFSET_LEN,
            actual: bytes.len(),
        })?;
        if first != 0 {
            return Err(IdMapCodecError::NonZeroFirstOffset { actual: first });
        }
        for ordinal in 0..span {
            let start = id_map_offset(offsets, ordinal).ok_or(IdMapCodecError::Truncated {
                expected: offsets_start + (ordinal + 1) * ID_MAP_OFFSET_LEN,
                actual: bytes.len(),
            })?;
            let end = id_map_offset(offsets, ordinal + 1).ok_or(IdMapCodecError::Truncated {
                expected: offsets_start + (ordinal + 2) * ID_MAP_OFFSET_LEN,
                actual: bytes.len(),
            })?;
            if end < start {
                return Err(IdMapCodecError::DescendingOffsets {
                    ordinal,
                    start,
                    end,
                });
            }
            let end_usize =
                usize::try_from(end).map_err(|_| IdMapCodecError::OffsetOutOfBounds {
                    ordinal,
                    offset: end,
                    blob_len: blob_bytes.len(),
                })?;
            if end_usize > blob_bytes.len() {
                return Err(IdMapCodecError::OffsetOutOfBounds {
                    ordinal,
                    offset: end,
                    blob_len: blob_bytes.len(),
                });
            }
        }
        let terminal = id_map_offset(offsets, span).ok_or(IdMapCodecError::Truncated {
            expected: offsets_end,
            actual: bytes.len(),
        })?;
        let terminal_usize =
            usize::try_from(terminal).map_err(|_| IdMapCodecError::TerminalOffsetMismatch {
                terminal,
                blob_len: blob_bytes.len(),
            })?;
        if terminal_usize != blob_bytes.len() {
            if terminal_usize < blob_bytes.len() {
                return Err(IdMapCodecError::TrailingBytes {
                    expected: hashes_end + terminal_usize,
                    actual: bytes.len(),
                });
            }
            return Err(IdMapCodecError::TerminalOffsetMismatch {
                terminal,
                blob_len: blob_bytes.len(),
            });
        }
        validate_id_map_blob_len(blob_bytes.len(), limits)?;
        let blob = match std::str::from_utf8(blob_bytes) {
            Ok(blob) => blob,
            Err(error) => {
                let invalid_offset = error.valid_up_to();
                let mut invalid_ordinal = 0_usize;
                for ordinal in 0..span {
                    let start = id_map_offset(offsets, ordinal).unwrap_or(0);
                    let end = id_map_offset(offsets, ordinal + 1).unwrap_or(0);
                    if usize::try_from(start).is_ok_and(|start| start <= invalid_offset)
                        && usize::try_from(end).is_ok_and(|end| invalid_offset < end)
                    {
                        invalid_ordinal = ordinal;
                        break;
                    }
                }
                return Err(IdMapCodecError::InvalidUtf8 {
                    ordinal: invalid_ordinal,
                });
            }
        };
        let mut present_count = 0_usize;
        for ordinal in 0..span {
            let start = id_map_offset(offsets, ordinal).ok_or(IdMapCodecError::Truncated {
                expected: offsets_start + (ordinal + 1) * ID_MAP_OFFSET_LEN,
                actual: bytes.len(),
            })?;
            let end = id_map_offset(offsets, ordinal + 1).ok_or(IdMapCodecError::Truncated {
                expected: offsets_start + (ordinal + 2) * ID_MAP_OFFSET_LEN,
                actual: bytes.len(),
            })?;
            let end_usize =
                usize::try_from(end).map_err(|_| IdMapCodecError::OffsetOutOfBounds {
                    ordinal,
                    offset: end,
                    blob_len: blob.len(),
                })?;
            let content_hash = id_map_hash(hashes, ordinal).ok_or(IdMapCodecError::Truncated {
                expected: offsets_end + (ordinal + 1) * ID_MAP_HASH_LEN,
                actual: bytes.len(),
            })?;
            if start == end {
                if content_hash != 0 {
                    return Err(IdMapCodecError::NonZeroHoleHash {
                        ordinal,
                        content_hash,
                    });
                }
                continue;
            }
            let id_len = u64::from(end - start);
            if id_len > limits.max_document_id_bytes {
                return Err(IdMapCodecError::ResourceLimit {
                    resource: "document id bytes",
                    actual: id_len,
                    limit: limits.max_document_id_bytes,
                });
            }
            let start_usize =
                usize::try_from(start).map_err(|_| IdMapCodecError::OffsetOutOfBounds {
                    ordinal,
                    offset: start,
                    blob_len: blob.len(),
                })?;
            if blob.get(start_usize..end_usize).is_none() {
                return Err(IdMapCodecError::InvalidUtf8 { ordinal });
            }
            present_count += 1;
        }
        Ok(Self {
            bytes,
            docid_lo,
            docid_hi,
            offsets,
            hashes,
            blob,
            present_count,
        })
    }

    /// Exact durable bytes after canonical validation.
    #[must_use]
    pub const fn as_bytes(self) -> &'a [u8] {
        self.bytes
    }

    /// Inclusive lower global docid bound.
    #[must_use]
    pub const fn docid_lo(self) -> u64 {
        self.docid_lo
    }

    /// Exclusive upper global docid bound.
    #[must_use]
    pub const fn docid_hi(self) -> u64 {
        self.docid_hi
    }

    /// Half-open positional span, including holes.
    #[must_use]
    pub const fn span(self) -> u64 {
        self.docid_hi - self.docid_lo
    }

    /// Number of present rows.
    #[must_use]
    pub const fn present_count(self) -> usize {
        self.present_count
    }

    /// Exact concatenated identifier blob, useful for zero-copy/concat proofs.
    #[must_use]
    pub const fn blob(self) -> &'a [u8] {
        self.blob.as_bytes()
    }

    /// Borrow one positional row by global docid.
    #[must_use]
    pub fn get(self, global_docid: u64) -> Option<IdMapEntry<'a>> {
        let ordinal = global_docid.checked_sub(self.docid_lo)?;
        let ordinal = usize::try_from(ordinal).ok()?;
        self.entry_at_ordinal(ordinal)
    }

    /// Whether a global docid names a present row.
    #[must_use]
    pub fn contains(self, global_docid: u64) -> bool {
        let Some(ordinal) = global_docid
            .checked_sub(self.docid_lo)
            .and_then(|ordinal| usize::try_from(ordinal).ok())
        else {
            return false;
        };
        self.document_id_at_ordinal(ordinal).is_some()
    }

    /// Materialize one winner identifier without touching any other row.
    #[must_use]
    pub fn materialize(self, global_docid: u64) -> Option<DocId> {
        let ordinal = global_docid.checked_sub(self.docid_lo)?;
        let ordinal = usize::try_from(ordinal).ok()?;
        Some(DocId::new(self.document_id_at_ordinal(ordinal)?))
    }

    /// Iterate present rows in ascending global-docid order.
    pub fn entries(self) -> impl Iterator<Item = (u64, IdMapEntry<'a>)> + 'a {
        (0..self.offset_count()).filter_map(move |ordinal| {
            let entry = self.entry_at_ordinal(ordinal)?;
            let ordinal = u64::try_from(ordinal).ok()?;
            Some((self.docid_lo + ordinal, entry))
        })
    }

    pub(super) fn document_ids(self) -> impl Iterator<Item = (u64, &'a str)> + 'a {
        (0..self.offset_count()).filter_map(move |ordinal| {
            let document_id = self.document_id_at_ordinal(ordinal)?;
            let ordinal = u64::try_from(ordinal).ok()?;
            Some((self.docid_lo + ordinal, document_id))
        })
    }

    pub(super) fn offset_count(self) -> usize {
        self.offsets
            .len()
            .checked_div(ID_MAP_OFFSET_LEN)
            .unwrap_or(0)
            .saturating_sub(1)
    }

    fn entry_at_ordinal(self, ordinal: usize) -> Option<IdMapEntry<'a>> {
        let document_id = self.document_id_at_ordinal(ordinal)?;
        let content_hash = id_map_hash(self.hashes, ordinal)?;
        Some(IdMapEntry {
            document_id,
            content_hash,
        })
    }

    pub(super) fn document_id_at_ordinal(self, ordinal: usize) -> Option<&'a str> {
        if ordinal >= self.offset_count() {
            return None;
        }
        let start = usize::try_from(id_map_offset(self.offsets, ordinal)?).ok()?;
        let end = usize::try_from(id_map_offset(self.offsets, ordinal + 1)?).ok()?;
        if start == end {
            return None;
        }
        self.blob.get(start..end)
    }
}

impl EncodedIdMapSection {
    /// Concatenate ordered non-overlapping IDMAP sections, inserting holes in
    /// inter-segment gaps and copying each source identifier blob once.
    ///
    /// # Errors
    ///
    /// Rejects empty input, range overlap/reversal, target-limit violations,
    /// arithmetic overflow, or allocation failure.
    pub fn concatenate(sections: &[IdMapSection<'_>]) -> Result<Self, IdMapCodecError> {
        Self::concatenate_with_limits(sections, IdMapLimits::default())
    }

    /// Preflight an ordinary concat for direct emission into a larger buffer.
    pub(crate) fn plan_concatenate(
        sections: &[IdMapSection<'_>],
    ) -> Result<IdMapConcatPlan, IdMapCodecError> {
        IdMapConcatPlan::new(sections, IdMapLimits::default())
    }

    /// Concatenate with caller-selected validation and allocation ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::concatenate`].
    pub fn concatenate_with_limits(
        sections: &[IdMapSection<'_>],
        limits: IdMapLimits,
    ) -> Result<Self, IdMapCodecError> {
        let plan = IdMapConcatPlan::new(sections, limits)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(plan.total_len)
            .map_err(|_| IdMapCodecError::Allocation {
                resource: "concatenated section bytes",
                bytes: plan.total_len,
            })?;
        plan.append_to(sections, &mut bytes);
        Ok(Self {
            bytes,
            docid_lo: plan.docid_lo,
            docid_hi: plan.docid_hi,
            present_count: plan.present_count,
        })
    }
}

impl IdMapConcatPlan {
    fn new(sections: &[IdMapSection<'_>], limits: IdMapLimits) -> Result<Self, IdMapCodecError> {
        let Some(first) = sections.first().copied() else {
            return Err(IdMapCodecError::EmptyConcat);
        };
        for (index, pair) in sections.windows(2).enumerate() {
            if pair[1].docid_lo < pair[0].docid_hi {
                return Err(IdMapCodecError::ConcatRangeOrder {
                    index: index + 1,
                    previous_hi: pair[0].docid_hi,
                    current_lo: pair[1].docid_lo,
                });
            }
        }
        let docid_lo = first.docid_lo;
        let docid_hi = sections
            .last()
            .map(|section| section.docid_hi)
            .ok_or(IdMapCodecError::EmptyConcat)?;
        let span = checked_id_map_span(docid_lo, docid_hi, limits)?;
        let mut blob_len = 0_usize;
        let mut present_count = 0_usize;
        for section in sections {
            blob_len = blob_len.checked_add(section.blob.len()).ok_or(
                IdMapCodecError::ArithmeticOverflow {
                    field: "concatenated blob length",
                },
            )?;
            present_count = present_count.checked_add(section.present_count).ok_or(
                IdMapCodecError::ArithmeticOverflow {
                    field: "concatenated present count",
                },
            )?;
            for (_, entry) in section.entries() {
                let id_len = u64::try_from(entry.document_id.len()).unwrap_or(u64::MAX);
                if id_len > limits.max_document_id_bytes {
                    return Err(IdMapCodecError::ResourceLimit {
                        resource: "document id bytes",
                        actual: id_len,
                        limit: limits.max_document_id_bytes,
                    });
                }
            }
        }
        validate_id_map_blob_len(blob_len, limits)?;
        let (blob_offset, total_len) = id_map_layout(span, blob_len, limits)?;
        Ok(Self {
            docid_lo,
            docid_hi,
            span,
            blob_offset,
            blob_len,
            total_len,
            present_count,
        })
    }

    /// Exact canonical byte length of the merged IDMAP payload.
    pub const fn encoded_len(&self) -> usize {
        self.total_len
    }

    /// Inclusive lower global docid bound of the conceptual merged IDMAP.
    pub const fn docid_lo(&self) -> u64 {
        self.docid_lo
    }

    /// Append the canonical payload after [`Self::new`] validated the sources.
    pub fn append_to(&self, sections: &[IdMapSection<'_>], bytes: &mut Vec<u8>) {
        let section_start = bytes.len();
        let entry_count = u32::try_from(self.span).expect("validated IDMAP span fits u32");
        let blob_offset =
            u32::try_from(self.blob_offset).expect("validated IDMAP blob offset fits u32");
        bytes.extend_from_slice(&entry_count.to_le_bytes());
        bytes.extend_from_slice(&blob_offset.to_le_bytes());

        let mut current_offset = 0_u32;
        bytes.extend_from_slice(&current_offset.to_le_bytes());
        let mut cursor = self.docid_lo;
        for section in sections.iter().copied() {
            let gap =
                usize::try_from(section.docid_lo - cursor).expect("validated IDMAP gap fits usize");
            append_repeated_u32(bytes, current_offset, gap);
            for ordinal in 0..section.offset_count() {
                let start =
                    id_map_offset(section.offsets, ordinal).expect("validated IDMAP source offset");
                let end = id_map_offset(section.offsets, ordinal + 1)
                    .expect("validated IDMAP source offset end");
                current_offset = current_offset
                    .checked_add(end - start)
                    .expect("validated IDMAP blob length fits u32");
                bytes.extend_from_slice(&current_offset.to_le_bytes());
            }
            cursor = section.docid_hi;
        }
        debug_assert_eq!(usize::try_from(current_offset).ok(), Some(self.blob_len));

        let mut cursor = self.docid_lo;
        for section in sections.iter().copied() {
            let gap =
                usize::try_from(section.docid_lo - cursor).expect("validated IDMAP gap fits usize");
            let gap_bytes = gap
                .checked_mul(std::mem::size_of::<u64>())
                .expect("validated IDMAP gap hash bytes");
            bytes.resize(
                bytes
                    .len()
                    .checked_add(gap_bytes)
                    .expect("validated IDMAP gap hash end"),
                0,
            );
            for ordinal in 0..section.offset_count() {
                let content_hash =
                    id_map_hash(section.hashes, ordinal).expect("validated IDMAP source hash");
                bytes.extend_from_slice(&content_hash.to_le_bytes());
            }
            cursor = section.docid_hi;
        }
        debug_assert_eq!(bytes.len() - section_start, self.blob_offset);
        for section in sections {
            bytes.extend_from_slice(section.blob.as_bytes());
        }
        debug_assert_eq!(bytes.len() - section_start, self.total_len);
    }
}

pub(super) fn append_repeated_u32(bytes: &mut Vec<u8>, value: u32, count: usize) {
    if count == 0 {
        return;
    }
    let byte_count = count
        .checked_mul(std::mem::size_of::<u32>())
        .expect("validated repeated-u32 byte count");
    let start = bytes.len();
    let end = start
        .checked_add(byte_count)
        .expect("validated repeated-u32 end");
    if value == 0 {
        bytes.resize(end, 0);
        return;
    }
    bytes.extend_from_slice(&value.to_le_bytes());
    while bytes.len() < end {
        let initialized = bytes.len() - start;
        let copied = initialized.min(end - bytes.len());
        bytes.extend_from_within(start..start + copied);
    }
}

pub(super) fn checked_id_map_span(
    docid_lo: u64,
    docid_hi: u64,
    limits: IdMapLimits,
) -> Result<usize, IdMapCodecError> {
    let max_docid_hi = u64::from(u32::MAX) + 1;
    let span = docid_hi
        .checked_sub(docid_lo)
        .filter(|_| docid_hi <= max_docid_hi)
        .ok_or(IdMapCodecError::InvalidDocIdRange { docid_lo, docid_hi })?;
    if span > limits.max_docid_span {
        return Err(IdMapCodecError::ResourceLimit {
            resource: "docid span",
            actual: span,
            limit: limits.max_docid_span,
        });
    }
    if span > ID_MAP_MAX_SPAN {
        return Err(IdMapCodecError::ResourceLimit {
            resource: "durable IDMAP span",
            actual: span,
            limit: ID_MAP_MAX_SPAN,
        });
    }
    if span > u64::from(u32::MAX) {
        return Err(IdMapCodecError::ResourceLimit {
            resource: "durable entry count",
            actual: span,
            limit: u64::from(u32::MAX),
        });
    }
    usize::try_from(span).map_err(|_| IdMapCodecError::ResourceLimit {
        resource: "host docid span",
        actual: span,
        limit: u64::try_from(usize::MAX).unwrap_or(u64::MAX),
    })
}

fn validate_id_map_blob_len(blob_len: usize, limits: IdMapLimits) -> Result<(), IdMapCodecError> {
    let blob_len_u64 = u64::try_from(blob_len).unwrap_or(u64::MAX);
    let limit = limits.max_blob_bytes.min(u64::from(u32::MAX));
    if blob_len_u64 > limit {
        return Err(IdMapCodecError::ResourceLimit {
            resource: "identifier blob bytes",
            actual: blob_len_u64,
            limit,
        });
    }
    Ok(())
}

pub(super) fn id_map_prefix_len(span: usize) -> Result<usize, IdMapCodecError> {
    let offsets_len = span
        .checked_add(1)
        .and_then(|count| count.checked_mul(ID_MAP_OFFSET_LEN))
        .ok_or(IdMapCodecError::ArithmeticOverflow {
            field: "offset table length",
        })?;
    let hashes_len =
        span.checked_mul(ID_MAP_HASH_LEN)
            .ok_or(IdMapCodecError::ArithmeticOverflow {
                field: "hash table length",
            })?;
    ID_MAP_HEADER_LEN
        .checked_add(offsets_len)
        .and_then(|len| len.checked_add(hashes_len))
        .ok_or(IdMapCodecError::ArithmeticOverflow {
            field: "blob offset",
        })
}

fn id_map_layout(
    span: usize,
    blob_len: usize,
    limits: IdMapLimits,
) -> Result<(usize, usize), IdMapCodecError> {
    let blob_offset = id_map_prefix_len(span)?;
    if u32::try_from(blob_offset).is_err() {
        return Err(IdMapCodecError::OffsetUnrepresentable {
            offset: blob_offset,
        });
    }
    let total_len =
        blob_offset
            .checked_add(blob_len)
            .ok_or(IdMapCodecError::ArithmeticOverflow {
                field: "section length",
            })?;
    let total_len_u64 = u64::try_from(total_len).unwrap_or(u64::MAX);
    if total_len_u64 > limits.max_section_bytes {
        return Err(IdMapCodecError::ResourceLimit {
            resource: "section bytes",
            actual: total_len_u64,
            limit: limits.max_section_bytes,
        });
    }
    Ok((blob_offset, total_len))
}

pub(super) fn id_map_offset(offsets: &[u8], index: usize) -> Option<u32> {
    read_u32(offsets, index.checked_mul(ID_MAP_OFFSET_LEN)?)
}

pub(super) fn id_map_hash(hashes: &[u8], index: usize) -> Option<u64> {
    read_u64(hashes, index.checked_mul(ID_MAP_HASH_LEN)?)
}
