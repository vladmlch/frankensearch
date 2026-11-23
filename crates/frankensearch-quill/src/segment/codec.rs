use std::path::Path;

use crate::error::QuillError;
use crate::schema::{FieldKind, SchemaDescriptor};

use super::{
    FILE_PREFIX_LEN, FIXED_HEADER_LEN, FSLX_FORMAT_VERSION, FSLX_SECTION_ALIGNMENT,
    FSLX_SEGMENT_MAGIC, HEADER_CRC_LEN, MAX_DOCID_EXCLUSIVE, OPTIONAL_SKIPPABLE, SECTION_ENTRY_LEN,
    SectionEntry, SectionKind, SegmentHeader, SegmentLimits, TRAILER_LEN,
};

pub(super) struct ParsedContainer {
    pub(super) header: SegmentHeader,
    pub(super) sections: Vec<SectionEntry>,
    pub(super) file_xxh3: u64,
}

pub(super) fn parse_container(
    bytes: &[u8],
    path: &Path,
    schema: SchemaDescriptor,
    limits: SegmentLimits,
) -> Result<ParsedContainer, QuillError> {
    let file_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
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
    let trailer_start = bytes
        .len()
        .checked_sub(TRAILER_LEN)
        .ok_or_else(|| corrupted(path, "file is shorter than its trailer"))?;
    if bytes.get(..FSLX_SEGMENT_MAGIC.len()) != Some(FSLX_SEGMENT_MAGIC.as_slice()) {
        return Err(corrupted(path, "invalid FSLX segment magic"));
    }
    let version =
        read_u32_at(bytes, 8).ok_or_else(|| corrupted(path, "truncated format version"))?;
    if version != FSLX_FORMAT_VERSION {
        return Err(corrupted(
            path,
            format!("unsupported FSLX format version {version}"),
        ));
    }
    let declared_header_len = usize::try_from(
        read_u32_at(bytes, 12).ok_or_else(|| corrupted(path, "truncated header length"))?,
    )
    .map_err(|_| corrupted(path, "header length does not fit usize"))?;
    if declared_header_len < FIXED_HEADER_LEN {
        return Err(corrupted(
            path,
            format!("header length {declared_header_len} is shorter than {FIXED_HEADER_LEN}"),
        ));
    }
    if declared_header_len > limits.max_header_bytes {
        return Err(resource(
            "segment header",
            format!(
                "{} declares length {declared_header_len}, exceeding limit {}",
                path.display(),
                limits.max_header_bytes
            ),
        ));
    }
    let header_end = FILE_PREFIX_LEN
        .checked_add(declared_header_len)
        .ok_or_else(|| corrupted(path, "header end overflow"))?;
    let header_crc_end = header_end
        .checked_add(HEADER_CRC_LEN)
        .ok_or_else(|| corrupted(path, "header CRC end overflow"))?;
    if header_crc_end > trailer_start {
        return Err(corrupted(path, "header overlaps the file trailer"));
    }
    let header_bytes = bytes
        .get(FILE_PREFIX_LEN..header_end)
        .ok_or_else(|| corrupted(path, "truncated header block"))?;
    let expected_header_crc =
        read_u32_at(bytes, header_end).ok_or_else(|| corrupted(path, "truncated header CRC"))?;
    let actual_header_crc = crc32(header_bytes);
    if actual_header_crc != expected_header_crc {
        return Err(corrupted(
            path,
            format!(
                "header CRC mismatch: expected {expected_header_crc:#010x}, got {actual_header_crc:#010x}"
            ),
        ));
    }

    let mut reader = ByteReader::new(header_bytes);
    let segment_id = reader
        .read_u64()
        .map_err(|detail| corrupted(path, detail))?;
    let schema_id = reader
        .read_u64()
        .map_err(|detail| corrupted(path, detail))?;
    let docid_lo = reader
        .read_u64()
        .map_err(|detail| corrupted(path, detail))?;
    let docid_hi = reader
        .read_u64()
        .map_err(|detail| corrupted(path, detail))?;
    let doc_count = reader
        .read_u32()
        .map_err(|detail| corrupted(path, detail))?;
    let reserved = reader
        .read_u32()
        .map_err(|detail| corrupted(path, detail))?;
    if reserved != 0 {
        return Err(corrupted(
            path,
            format!("reserved header word is {reserved}"),
        ));
    }
    let created_unix_s = reader
        .read_i64()
        .map_err(|detail| corrupted(path, detail))?;
    let engine_version = reader
        .read_u32()
        .map_err(|detail| corrupted(path, detail))?;
    let section_count = reader
        .read_u16()
        .map_err(|detail| corrupted(path, detail))?;
    let reserved2 = reader
        .read_u16()
        .map_err(|detail| corrupted(path, detail))?;
    if reserved2 != 0 {
        return Err(corrupted(
            path,
            format!("reserved section word is {reserved2}"),
        ));
    }
    let section_count_usize = usize::from(section_count);
    if section_count_usize > limits.max_sections {
        return Err(resource(
            "section table",
            format!(
                "{} declares {section_count_usize} sections, exceeding limit {}",
                path.display(),
                limits.max_sections
            ),
        ));
    }
    let expected_header_len =
        header_len(section_count_usize).map_err(|detail| corrupted(path, detail))?;
    if declared_header_len != expected_header_len {
        return Err(corrupted(
            path,
            format!(
                "non-canonical header length {declared_header_len}; expected {expected_header_len}"
            ),
        ));
    }
    let expected_schema_id = schema.schema_id()?;
    if schema_id != expected_schema_id {
        return Err(QuillError::UnknownSchema { schema_id });
    }
    validate_header_range(docid_lo, docid_hi, doc_count)
        .map_err(|detail| corrupted(path, detail))?;

    let mut sections = Vec::new();
    sections
        .try_reserve_exact(section_count_usize)
        .map_err(|_| {
            resource(
                "section table",
                format!("allocation failed for {}", path.display()),
            )
        })?;
    for _ in 0..section_count_usize {
        sections.push(SectionEntry {
            kind: SectionKind::from_raw(
                reader
                    .read_u16()
                    .map_err(|detail| corrupted(path, detail))?,
            ),
            flags: reader
                .read_u16()
                .map_err(|detail| corrupted(path, detail))?,
            offset: reader
                .read_u64()
                .map_err(|detail| corrupted(path, detail))?,
            len: reader
                .read_u64()
                .map_err(|detail| corrupted(path, detail))?,
            xxh3: reader
                .read_u64()
                .map_err(|detail| corrupted(path, detail))?,
        });
    }
    if !reader.is_empty() {
        return Err(corrupted(path, "trailing bytes in header block"));
    }
    validate_section_sequence(
        schema,
        sections.len(),
        |index| {
            let entry = sections[index];
            (entry.kind, entry.flags)
        },
        true,
    )
    .map_err(|detail| corrupted(path, detail))?;
    validate_layout(bytes, header_crc_end, trailer_start, &sections)
        .map_err(|detail| corrupted(path, detail))?;

    let file_xxh3 = read_u64_at(bytes, trailer_start)
        .ok_or_else(|| corrupted(path, "truncated file checksum"))?;
    let trailer_crc = read_u32_at(bytes, trailer_start + 8)
        .ok_or_else(|| corrupted(path, "truncated trailer CRC"))?;
    let hash_bytes = file_xxh3.to_le_bytes();
    let actual_trailer_crc = crc32(&hash_bytes);
    if trailer_crc != actual_trailer_crc {
        return Err(corrupted(
            path,
            format!(
                "trailer CRC mismatch: expected {trailer_crc:#010x}, got {actual_trailer_crc:#010x}"
            ),
        ));
    }

    Ok(ParsedContainer {
        header: SegmentHeader {
            segment_id,
            schema_id,
            docid_lo,
            docid_hi,
            doc_count,
            created_unix_s,
            engine_version,
            section_count,
        },
        sections,
        file_xxh3,
    })
}

pub(super) fn validate_header_range(
    docid_lo: u64,
    docid_hi: u64,
    doc_count: u32,
) -> Result<(), String> {
    if docid_lo >= docid_hi {
        return Err(format!(
            "invalid empty or reversed docid range [{docid_lo}, {docid_hi})"
        ));
    }
    if docid_hi > MAX_DOCID_EXCLUSIVE {
        return Err(format!(
            "docid upper bound {docid_hi} exceeds the u32 payload domain {MAX_DOCID_EXCLUSIVE}"
        ));
    }
    let span = docid_hi
        .checked_sub(docid_lo)
        .ok_or_else(|| format!("invalid docid range [{docid_lo}, {docid_hi})"))?;
    if u64::from(doc_count) > span {
        return Err(format!("doc_count {doc_count} exceeds docid span {span}"));
    }
    Ok(())
}

pub(super) fn validate_section_sequence<F>(
    schema: SchemaDescriptor,
    count: usize,
    mut get: F,
    allow_unregistered_extensions: bool,
) -> Result<(), String>
where
    F: FnMut(usize) -> (SectionKind, u16),
{
    let mut seen = [false; 11];
    let mut previous = None;
    for index in 0..count {
        let (kind, flags) = get(index);
        if let Some(previous) = previous {
            if kind <= previous {
                return Err(format!(
                    "section kinds are not strictly ascending at index {index}: {} after {}",
                    kind.raw(),
                    previous.raw()
                ));
            }
        }
        previous = Some(kind);
        if kind.is_known() {
            if flags != 0 {
                return Err(format!(
                    "known section {} has unsupported flags {flags:#06x}",
                    kind.raw()
                ));
            }
            seen[usize::from(kind.raw())] = true;
        } else {
            if kind.raw() <= SectionKind::STATS.raw() || flags != OPTIONAL_SKIPPABLE {
                return Err(format!(
                    "unknown section {} is not canonically optional-skippable",
                    kind.raw()
                ));
            }
            if !allow_unregistered_extensions {
                return Err(format!(
                    "writer cannot emit unregistered optional section {}",
                    kind.raw()
                ));
            }
        }
    }

    let mut expected = [false; 11];
    for kind in [
        SectionKind::TERMDICT,
        SectionKind::POSTINGS,
        SectionKind::BLOCKMAX,
        SectionKind::DOCLEN,
        SectionKind::IDMAP,
        SectionKind::IDHASH,
        SectionKind::STATS,
    ] {
        expected[usize::from(kind.raw())] = true;
    }
    expected[usize::from(SectionKind::POSITIONS.raw())] = schema_has_positions(schema);
    expected[usize::from(SectionKind::NUMERIC.raw())] = schema_has_numeric_columns(schema);
    expected[usize::from(SectionKind::STOREDMETA.raw())] = schema_has_stored_fields(schema);
    for raw in usize::from(SectionKind::TERMDICT.raw())..=usize::from(SectionKind::STATS.raw()) {
        if seen[raw] != expected[raw] {
            let requirement = if expected[raw] {
                "missing"
            } else {
                "unexpected"
            };
            return Err(format!("{requirement} known section kind {raw}"));
        }
    }
    Ok(())
}

pub(super) fn published_segment_id(path: &Path) -> Result<u64, String> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "published segment name is not valid UTF-8".to_owned())?;
    let Some(hex) = name
        .strip_prefix("seg-")
        .and_then(|rest| rest.strip_suffix(".fslx"))
    else {
        return Err("published segment name must match seg-<hex16>.fslx".to_owned());
    };
    if hex.len() != 16
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err("published segment name must match seg-<hex16>.fslx".to_owned());
    }
    u64::from_str_radix(hex, 16).map_err(|_| "published segment id is not valid hex".to_owned())
}

pub(super) fn schema_has_positions(schema: SchemaDescriptor) -> bool {
    schema.fields.iter().any(|field| {
        matches!(
            field.kind,
            FieldKind::Text {
                positions: true,
                ..
            }
        )
    })
}

pub(super) fn schema_has_numeric_columns(schema: SchemaDescriptor) -> bool {
    schema
        .fields
        .iter()
        .any(|field| field.kind.has_numeric_column())
}

pub(super) fn schema_has_stored_fields(schema: SchemaDescriptor) -> bool {
    schema.fields.iter().any(|field| field.stored)
}

fn validate_layout(
    bytes: &[u8],
    header_crc_end: usize,
    trailer_start: usize,
    sections: &[SectionEntry],
) -> Result<(), String> {
    let mut cursor = header_crc_end;
    for entry in sections {
        let expected_offset = align_up(cursor, FSLX_SECTION_ALIGNMENT)
            .ok_or_else(|| "section alignment overflow".to_owned())?;
        let offset = usize::try_from(entry.offset)
            .map_err(|_| "section offset does not fit usize".to_owned())?;
        if offset != expected_offset {
            return Err(format!(
                "section {} has non-canonical offset {offset}; expected {expected_offset}",
                entry.kind.raw()
            ));
        }
        if offset % FSLX_SECTION_ALIGNMENT != 0 {
            return Err(format!(
                "section {} is not 64-byte aligned",
                entry.kind.raw()
            ));
        }
        let padding = bytes
            .get(cursor..offset)
            .ok_or_else(|| "alignment padding lies outside the file".to_owned())?;
        if let Some(relative) = padding.iter().position(|byte| *byte != 0) {
            return Err(format!(
                "non-zero alignment padding at byte {}",
                cursor + relative
            ));
        }
        let len = usize::try_from(entry.len)
            .map_err(|_| "section length does not fit usize".to_owned())?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| "section end overflow".to_owned())?;
        if end > trailer_start {
            return Err(format!(
                "section {} ends at {end}, beyond trailer start {trailer_start}",
                entry.kind.raw()
            ));
        }
        cursor = end;
    }
    if cursor != trailer_start {
        return Err(format!(
            "trailing bytes before file trailer: sections end at {cursor}, trailer starts at {trailer_start}"
        ));
    }
    Ok(())
}

pub(super) fn header_len(section_count: usize) -> Result<usize, String> {
    section_count
        .checked_mul(SECTION_ENTRY_LEN)
        .and_then(|table| FIXED_HEADER_LEN.checked_add(table))
        .ok_or_else(|| "header length overflow".to_owned())
}

pub(super) fn align_up(value: usize, alignment: usize) -> Option<usize> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
}

pub(super) fn write_header_fixed(output: &mut Vec<u8>, header: SegmentHeader) {
    output.extend_from_slice(&header.segment_id.to_le_bytes());
    output.extend_from_slice(&header.schema_id.to_le_bytes());
    output.extend_from_slice(&header.docid_lo.to_le_bytes());
    output.extend_from_slice(&header.docid_hi.to_le_bytes());
    output.extend_from_slice(&header.doc_count.to_le_bytes());
    output.extend_from_slice(&0_u32.to_le_bytes());
    output.extend_from_slice(&header.created_unix_s.to_le_bytes());
    output.extend_from_slice(&header.engine_version.to_le_bytes());
    output.extend_from_slice(&header.section_count.to_le_bytes());
    output.extend_from_slice(&0_u16.to_le_bytes());
}

pub(super) fn write_section_entry(output: &mut Vec<u8>, entry: SectionEntry) {
    output.extend_from_slice(&entry.kind.raw().to_le_bytes());
    output.extend_from_slice(&entry.flags.to_le_bytes());
    output.extend_from_slice(&entry.offset.to_le_bytes());
    output.extend_from_slice(&entry.len.to_le_bytes());
    output.extend_from_slice(&entry.xxh3.to_le_bytes());
}

pub(super) fn section_bytes(bytes: &[u8], entry: SectionEntry) -> Option<&[u8]> {
    let start = usize::try_from(entry.offset).ok()?;
    let len = usize::try_from(entry.len).ok()?;
    bytes.get(start..start.checked_add(len)?)
}

pub(super) fn crc32(bytes: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

pub(super) fn invalid_segment(detail: impl Into<String>) -> QuillError {
    QuillError::Invariant {
        detail: format!("invalid FSLX segment input: {}", detail.into()),
    }
}

pub(super) fn resource(resource: &'static str, detail: impl Into<String>) -> QuillError {
    QuillError::Resource {
        resource,
        detail: detail.into(),
    }
}

pub(super) fn corrupted(path: impl AsRef<Path>, detail: impl Into<String>) -> QuillError {
    QuillError::IndexCorrupted {
        path: path.as_ref().to_path_buf(),
        detail: detail.into(),
    }
}

pub(super) fn read_u32_at(bytes: &[u8], offset: usize) -> Option<u32> {
    let slice = bytes.get(offset..offset.checked_add(4)?)?;
    Some(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

pub(super) fn read_u64_at(bytes: &[u8], offset: usize) -> Option<u64> {
    let slice = bytes.get(offset..offset.checked_add(8)?)?;
    Some(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

struct ByteReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> ByteReader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self
            .position
            .checked_add(len)
            .ok_or_else(|| "header cursor overflow".to_owned())?;
        let result = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| format!("truncated header at byte {}", self.position))?;
        self.position = end;
        Ok(result)
    }

    fn read_u16(&mut self) -> Result<u16, String> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, String> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64, String> {
        let bytes = self.take(8)?;
        Ok(i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn is_empty(&self) -> bool {
        self.position == self.bytes.len()
    }
}
