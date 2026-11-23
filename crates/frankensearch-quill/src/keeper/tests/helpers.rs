use super::*;

pub(super) fn array_tombstone_bytes(chunk_id: u16, lows: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::new();
    put_u32(&mut bytes, 1);
    put_u16(&mut bytes, chunk_id);
    bytes.push(0);
    put_u16(
        &mut bytes,
        u16::try_from(lows.len()).expect("test tombstone count fits u16"),
    );
    for low in lows {
        put_u16(&mut bytes, *low);
    }
    bytes
}

pub(super) fn array_tombstones(chunk_id: u16, lows: &[u16]) -> TombstoneSet {
    TombstoneSet::from_bytes(&array_tombstone_bytes(chunk_id, lows))
        .expect("test array tombstones are canonical")
}

pub(super) fn bitmap_tombstone_bytes(chunk_id: u16, lows: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::new();
    put_u32(&mut bytes, 1);
    put_u16(&mut bytes, chunk_id);
    bytes.push(1);
    let encoded_count = if lows.len() == 65_536 {
        0
    } else {
        u16::try_from(lows.len()).expect("test bitmap count fits u16")
    };
    put_u16(&mut bytes, encoded_count);
    bytes.resize(bytes.len() + 8_192, 0);
    let payload_start = bytes.len() - 8_192;
    for low in lows {
        let low = usize::from(*low);
        bytes[payload_start + low / 8] |= 1 << (low % 8);
    }
    bytes
}

pub(super) fn bitmap_tombstones(chunk_id: u16, lows: &[u16]) -> TombstoneSet {
    TombstoneSet::from_bytes(&bitmap_tombstone_bytes(chunk_id, lows))
        .expect("test bitmap tombstones are canonical")
}

pub(super) fn sample_manifest(generation: u64) -> Manifest {
    Manifest {
        generation,
        docid_high_watermark: 70_000 + generation,
        schema_id: 0x1122_3344_5566_7788,
        engine_version: CURRENT_ENGINE_VERSION,
        flags: 0,
        last_publish_unix_s: 1_700_000_000,
        segments: vec![
            ManifestSegment {
                segment_id: 0x1001,
                seal_seq: 7,
                file_len: 4_096,
                file_xxh3: 0xa1a2_a3a4_a5a6_a7a8,
                docid_lo: 100,
                docid_hi: 200,
                doc_count: 3,
                tombstones: array_tombstones(0, &[101, 150]),
            },
            ManifestSegment {
                segment_id: 0x1002,
                seal_seq: 8,
                file_len: 8_192,
                file_xxh3: 0xb1b2_b3b4_b5b6_b7b8,
                docid_lo: 65_536,
                docid_hi: 65_600,
                doc_count: 2,
                tombstones: TombstoneSet::new(),
            },
        ],
        field_stats: vec![
            ManifestFieldStats {
                field_ord: 1,
                total_tokens: 12,
                doc_count: 5,
            },
            ManifestFieldStats {
                field_ord: 3,
                total_tokens: 42,
                doc_count: 5,
            },
        ],
    }
}

pub(super) fn rewrite_crc(bytes: &mut [u8]) {
    let body_len = bytes.len() - 4;
    let checksum = crc32fast::hash(&bytes[..body_len]);
    bytes[body_len..].copy_from_slice(&checksum.to_le_bytes());
}

pub(super) fn write_manifest(path: &Path, manifest: &Manifest) -> TestResult {
    let bytes = manifest.to_bytes()?;
    std::fs::write(path, bytes)?;
    Ok(())
}

/// Publish one two-document segment through a durable writer at `directory`
/// and return the published MANIFEST.
#[cfg(unix)]
pub(super) fn publish_fixture_segment(
    directory: &Path,
    segment_id: u64,
) -> Result<Manifest, Box<dyn std::error::Error>> {
    let encoded =
        encoded_identity_test_segment(segment_id, 0, &[Some("reuse-a"), Some("reuse-b")])?;
    let manifest = durable_test_manifest(2, vec![manifest_segment(&encoded, 1)]);
    let publish_directory = directory.to_path_buf();
    let published = manifest.clone();
    run_with_test_cx(move |cx| async move {
        let mut writer = KeeperWriter::create(&cx, &publish_directory, DEFAULT_SCHEMA)
            .await
            .map_err(|error| error.to_string())?;
        let pending = encoded
            .write_temp(&publish_directory)
            .map_err(|error| error.to_string())?;
        writer
            .publish_segment(&cx, pending)
            .await
            .map_err(|error| error.to_string())?;
        writer
            .publish(&cx, &published)
            .await
            .map_err(|error| error.to_string())?;
        Ok::<(), String>(())
    })
    .map_err(io::Error::other)?;
    Ok(manifest)
}

pub(super) fn encoded_test_segment(
    segment_id: u64,
    docid_lo: u64,
    docid_hi: u64,
    doc_count: u32,
) -> Result<EncodedSegment, Box<dyn std::error::Error>> {
    let span = usize::try_from(docid_hi.saturating_sub(docid_lo))?;
    let document_ids = (0..span)
        .map(|ordinal| format!("doc-{segment_id:016x}-{ordinal}"))
        .collect::<Vec<_>>();
    let identifiers = document_ids
        .iter()
        .enumerate()
        .map(|(ordinal, document_id)| {
            (ordinal < usize::try_from(doc_count).unwrap_or(usize::MAX))
                .then_some(document_id.as_str())
        })
        .collect::<Vec<_>>();
    encoded_identity_test_segment(segment_id, docid_lo, &identifiers)
}

pub(super) fn encoded_identity_test_segment(
    segment_id: u64,
    docid_lo: u64,
    document_ids: &[Option<&str>],
) -> Result<EncodedSegment, Box<dyn std::error::Error>> {
    encoded_identity_test_segment_with_dictionary(
        segment_id,
        docid_lo,
        document_ids,
        &EMPTY_TERM_DICTIONARY,
    )
}

pub(super) fn encoded_identity_test_segment_with_dictionary(
    segment_id: u64,
    docid_lo: u64,
    document_ids: &[Option<&str>],
    term_dictionary: &[u8],
) -> Result<EncodedSegment, Box<dyn std::error::Error>> {
    encoded_identity_test_segment_with_term_sections(
        segment_id,
        docid_lo,
        document_ids,
        term_dictionary,
        &[],
        &[],
        &[],
    )
}

pub(super) fn encoded_identity_test_segment_with_term_sections(
    segment_id: u64,
    docid_lo: u64,
    document_ids: &[Option<&str>],
    term_dictionary: &[u8],
    postings: &[u8],
    positions: &[u8],
    blockmax: &[u8],
) -> Result<EncodedSegment, Box<dyn std::error::Error>> {
    let docid_hi = docid_lo
        .checked_add(u64::try_from(document_ids.len())?)
        .ok_or_else(|| QuillError::Invariant {
            detail: "test IDMAP range overflow".to_owned(),
        })?;
    let entries = document_ids
        .iter()
        .enumerate()
        .map(|(ordinal, document_id)| {
            document_id.map(|document_id| {
                IdMapEntryInput::new(
                    document_id,
                    u64::try_from(ordinal).unwrap_or(u64::MAX).saturating_add(1),
                )
            })
        })
        .collect::<Vec<_>>();
    let doc_count = u32::try_from(entries.iter().flatten().count())?;
    let id_map = EncodedIdMapSection::encode(docid_lo, docid_hi, &entries)?;
    let id_hash = EncodedIdHashSection::encode(id_map.section()?)?;
    let mut numeric_entries = Vec::new();
    for (ordinal, document_id) in document_ids.iter().enumerate() {
        if document_id.is_none() {
            continue;
        }
        let global_docid = docid_lo
            .checked_add(u64::try_from(ordinal)?)
            .ok_or_else(|| QuillError::Invariant {
                detail: "test numeric docid overflow".to_owned(),
            })?;
        numeric_entries.push(NumericEntry::u64(
            global_docid,
            u32::try_from(global_docid)?,
        ));
    }
    let numeric_inputs = [NumericFieldInput::new(4, &numeric_entries)];
    let numeric =
        EncodedNumericSection::encode(DEFAULT_SCHEMA, docid_lo, docid_hi, &numeric_inputs)?;
    Ok(EncodedSegment::encode(
        SegmentHeaderInput {
            segment_id,
            schema: DEFAULT_SCHEMA,
            docid_lo,
            docid_hi,
            doc_count,
            created_unix_s: 1_700_000_000,
            engine_version: CURRENT_ENGINE_VERSION,
        },
        &[
            SectionInput::new(SectionKind::TERMDICT, term_dictionary),
            SectionInput::new(SectionKind::POSTINGS, postings),
            SectionInput::new(SectionKind::POSITIONS, positions),
            SectionInput::new(SectionKind::BLOCKMAX, blockmax),
            SectionInput::new(SectionKind::DOCLEN, b"doclen"),
            SectionInput::new(SectionKind::IDMAP, id_map.as_bytes()),
            SectionInput::new(SectionKind::IDHASH, id_hash.as_bytes()),
            SectionInput::new(SectionKind::NUMERIC, numeric.as_bytes()),
            SectionInput::new(SectionKind::STOREDMETA, b"storedmeta"),
            SectionInput::new(SectionKind::STATS, b"stats"),
        ],
    )?)
}

pub(super) fn reseal_test_segment_file_witness(
    bytes: &mut [u8],
) -> Result<u64, Box<dyn std::error::Error>> {
    const FILE_WITNESS_LEN: usize = std::mem::size_of::<u64>();
    const TRAILER_CRC_LEN: usize = std::mem::size_of::<u32>();
    const TRAILER_LEN: usize = FILE_WITNESS_LEN + TRAILER_CRC_LEN;

    let trailer_start =
        bytes
            .len()
            .checked_sub(TRAILER_LEN)
            .ok_or_else(|| QuillError::Invariant {
                detail: "test FSLX segment was shorter than its trailer".to_owned(),
            })?;
    let file_xxh3 = xxhash_rust::xxh3::xxh3_64(&bytes[..trailer_start]);
    bytes[trailer_start..trailer_start + FILE_WITNESS_LEN]
        .copy_from_slice(&file_xxh3.to_le_bytes());
    let trailer_crc = crc32fast::hash(&bytes[trailer_start..trailer_start + FILE_WITNESS_LEN]);
    bytes[trailer_start + FILE_WITNESS_LEN..].copy_from_slice(&trailer_crc.to_le_bytes());
    Ok(file_xxh3)
}

pub(super) fn write_identity_test_segment(
    directory: &Path,
    segment_id: u64,
    seal_seq: u64,
    docid_lo: u64,
    document_ids: &[Option<&str>],
) -> Result<ManifestSegment, Box<dyn std::error::Error>> {
    let encoded = encoded_identity_test_segment(segment_id, docid_lo, document_ids)?;
    std::fs::write(
        directory.join(canonical_segment_name(segment_id)),
        encoded.as_bytes(),
    )?;
    Ok(manifest_segment(&encoded, seal_seq))
}

pub(super) fn write_test_segment(
    directory: &Path,
    segment_id: u64,
    seal_seq: u64,
    docid_lo: u64,
    docid_hi: u64,
) -> Result<ManifestSegment, Box<dyn std::error::Error>> {
    let encoded = encoded_test_segment(segment_id, docid_lo, docid_hi, 1)?;
    std::fs::write(
        directory.join(canonical_segment_name(segment_id)),
        encoded.as_bytes(),
    )?;
    Ok(manifest_segment(&encoded, seal_seq))
}

/// Stamp the first-unreachable receipt a retiring publication would have
/// written for `segment_id`, so hand-built fixtures can model a directory
/// that really did retire a segment.
#[cfg(unix)]
pub(super) fn write_test_retirement_receipt(
    directory: &Path,
    segment_id: u64,
    retired_generation: u64,
    retired_at: SystemTime,
) -> Result<(), Box<dyn std::error::Error>> {
    let receipt = SegmentRetirementReceipt {
        segment_id,
        retired_unix_s: i64::try_from(
            retired_at.duration_since(SystemTime::UNIX_EPOCH)?.as_secs(),
        )?,
        retired_generation,
        witness: UnreachabilityWitness::RetiredByPublication,
    };
    std::fs::write(
        directory.join(retirement_receipt_name(segment_id)),
        receipt.encode(),
    )?;
    Ok(())
}

pub(super) fn manifest_segment(encoded: &EncodedSegment, seal_seq: u64) -> ManifestSegment {
    let header = encoded.header();
    ManifestSegment {
        segment_id: header.segment_id,
        seal_seq,
        file_len: encoded.file_len(),
        file_xxh3: encoded.file_xxh3(),
        docid_lo: header.docid_lo,
        docid_hi: header.docid_hi,
        doc_count: header.doc_count,
        tombstones: TombstoneSet::new(),
    }
}

pub(super) fn durable_test_manifest(generation: u64, segments: Vec<ManifestSegment>) -> Manifest {
    Manifest {
        generation,
        docid_high_watermark: segments
            .iter()
            .map(|segment| segment.docid_hi)
            .max()
            .unwrap_or(0),
        schema_id: DEFAULT_SCHEMA.schema_id().expect("valid test schema"),
        engine_version: CURRENT_ENGINE_VERSION,
        flags: 0,
        last_publish_unix_s: 1_700_000_000,
        segments,
        field_stats: Vec::new(),
    }
}

pub(super) fn directory_bytes(directory: &Path) -> Result<Vec<(OsString, Vec<u8>)>, io::Error> {
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            entries.push((entry.file_name(), std::fs::read(entry.path())?));
        }
    }
    entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    Ok(entries)
}

/// A directory whose generation 3 dropped `0xb` from both slots, with the
/// stranded segment's mtime backdated past any grace period. This is the
/// exact input a collector that infers unreachability from file age
/// deletes, and the only thing that may authorize deleting it is a
/// durable first-unreachable receipt.
#[cfg(unix)]
pub(super) fn stranded_segment_fixture()
-> Result<(tempfile::TempDir, ManifestSegment), Box<dyn std::error::Error>> {
    let directory = tempdir()?;
    let live = write_test_segment(directory.path(), 0xa, 1, 0, 2)?;
    let stranded = write_test_segment(directory.path(), 0xb, 2, 2, 4)?;
    File::options()
        .write(true)
        .open(directory.path().join(canonical_segment_name(0xb)))?
        .set_times(std::fs::FileTimes::new().set_modified(SystemTime::UNIX_EPOCH))?;
    let previous = durable_test_manifest(2, vec![live.clone()]);
    let mut current = durable_test_manifest(3, vec![live]);
    current.docid_high_watermark = previous.docid_high_watermark;
    write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
    write_manifest(&directory.path().join("MANIFEST"), &current)?;
    Ok((directory, stranded))
}

pub(super) fn run_with_test_cx<T, F, Fut>(operation: F) -> T
where
    T: Send + 'static,
    F: FnOnce(Cx) -> Fut + 'static,
    Fut: std::future::Future<Output = T> + 'static,
{
    let output = Arc::new(std::sync::Mutex::new(None));
    let task_output = Arc::clone(&output);
    asupersync::test_utils::run_test_with_cx(move |cx| async move {
        let value = operation(cx).await;
        let mut slot = task_output
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *slot = Some(value);
    });
    let output =
        Arc::try_unwrap(output).unwrap_or_else(|_| panic!("test runtime retained its result slot"));
    output
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .expect("test runtime produced a result")
}

#[cfg(any(unix, windows))]
pub(super) fn create_test_index(directory: &Path) -> Result<(), String> {
    let directory = directory.to_path_buf();
    run_with_test_cx(move |cx| async move {
        KeeperSnapshot::create(&cx, directory, DEFAULT_SCHEMA)
            .await
            .map(|_| ())
            .map_err(|error| error.to_string())
    })
}

pub(super) fn hex_bytes(source: &str) -> Vec<u8> {
    let compact = source
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect::<Vec<_>>();
    assert_eq!(compact.len() % 2, 0, "hex input must have even length");
    let (pairs, remainder) = compact.as_chunks::<2>();
    assert!(remainder.is_empty());
    pairs
        .iter()
        .map(|&[high, low]| {
            let high = hex_nibble(high);
            let low = hex_nibble(low);
            high << 4 | low
        })
        .collect()
}

fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        b'A'..=b'F' => byte - b'A' + 10,
        _ => panic!("invalid hex byte"),
    }
}
