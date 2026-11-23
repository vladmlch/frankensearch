use super::*;

#[test]
fn recovered_snapshot_eagerly_validates_termdict_and_keeps_unrelated_hashes_lazy() -> TestResult {
    let directory = tempdir()?;
    let encoded = encoded_test_segment(0xabc, 10, 20, 1)?;
    let doclen = encoded
        .section_entries()
        .iter()
        .find(|entry| entry.kind == SectionKind::DOCLEN)
        .expect("doclen entry");
    let doclen_offset = usize::try_from(doclen.offset)?;
    let doclen_len = usize::try_from(doclen.len)?;
    let mut bytes = encoded.as_bytes().to_vec();
    bytes[doclen_offset] ^= 0x80;
    assert_ne!(
        xxhash_rust::xxh3::xxh3_64(&bytes[doclen_offset..doclen_offset + doclen_len]),
        doclen.xxh3,
        "the DOCLEN payload mutation must deliberately leave its section witness stale"
    );
    let file_xxh3 = reseal_test_segment_file_witness(&mut bytes)?;
    assert_ne!(
        file_xxh3,
        encoded.file_xxh3(),
        "the payload mutation must rewrite the FSLX whole-file witness"
    );
    assert_eq!(
        SegmentReader::from_owned(bytes.clone(), DEFAULT_SCHEMA)?.verify_file_witness()?,
        file_xxh3,
        "the mutated fixture must retain a self-consistent FSLX trailer"
    );
    std::fs::write(directory.path().join(canonical_segment_name(0xabc)), bytes)?;
    let mut segment = manifest_segment(&encoded, 1);
    segment.file_xxh3 = file_xxh3;
    let manifest = durable_test_manifest(1, vec![segment]);
    write_manifest(&directory.path().join("MANIFEST"), &manifest)?;
    let before = directory_bytes(directory.path())?;

    let snapshot = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
    assert_eq!(directory_bytes(directory.path())?, before);
    assert_eq!(snapshot.segments().len(), 1);
    assert_eq!(
        snapshot.segments()[0].authenticated_file_witness_hash_count(),
        1,
        "durable mapped recovery must still recompute the exact backing's whole-file witness"
    );
    assert_eq!(
        snapshot.segments()[0].term_dictionary_cache_counts(),
        (1, 0)
    );
    for _ in 0..5 {
        assert_eq!(
            snapshot.segments()[0]
                .term_dictionary(DEFAULT_SCHEMA)?
                .term_count(),
            0
        );
    }
    assert_eq!(
        snapshot.segments()[0].term_dictionary_cache_counts(),
        (1, 5)
    );
    let reopened = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
    assert_eq!(
        reopened.segments()[0].term_dictionary_cache_counts(),
        (1, 0),
        "each durable reload performs one fresh validation"
    );
    assert_eq!(directory_bytes(directory.path())?, before);
    assert_eq!(
        snapshot.segments()[0].section(SectionKind::TERMDICT)?,
        Some(EMPTY_TERM_DICTIONARY.as_slice())
    );
    assert!(matches!(
        snapshot.segments()[0].section(SectionKind::DOCLEN),
        Err(QuillError::IndexCorrupted { .. })
    ));

    let mut mismatched = manifest;
    mismatched.segments[0].file_len += 1;
    write_manifest(&directory.path().join("MANIFEST"), &mismatched)?;
    assert!(matches!(
        KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA),
        Err(KeeperError::SegmentMetadataMismatch { .. })
    ));
    assert!(matches!(
        KeeperSnapshot::open(directory.path(), FSFS_CHUNK_SCHEMA),
        Err(KeeperError::SchemaMismatch { .. })
    ));
    Ok(())
}

#[test]
fn keeper_open_recomputes_actual_prefix_witness_before_lazy_postings_bind() -> TestResult {
    const FILE_PREFIX_LEN: usize = 16;
    const FIXED_HEADER_LEN: usize = 56;
    const SECTION_ENTRY_LEN: usize = 28;
    const SECTION_ENTRY_XXH3_OFFSET: usize = 20;
    const TRAILER_LEN: usize = 12;

    let directory = tempdir()?;
    let encoded_postings = EncodedPostingList::encode(&[Posting::new(0, 1), Posting::new(1, 1)])?;
    let replacement_postings =
        EncodedPostingList::encode(&[Posting::new(0, 1), Posting::new(2, 1)])?;
    assert_eq!(
        replacement_postings.as_bytes().len(),
        encoded_postings.as_bytes().len(),
        "the replacement must preserve the exact POSTINGS section length"
    );
    assert_ne!(
        replacement_postings.as_bytes(),
        encoded_postings.as_bytes(),
        "the replacement must change POSTINGS content while retaining its layout"
    );
    let posting_list = PostingList::parse(encoded_postings.as_bytes(), 2)?;
    let encoded_blockmax = EncodedBlockMax::encode(&posting_list, |_| Some(1))?;
    let term_dictionary = EncodedTermDictionary::encode_sorted(
        DEFAULT_SCHEMA,
        TermSectionLengths {
            postings: u64::try_from(encoded_postings.as_bytes().len())?,
            positions: Some(0),
            blockmax: u64::try_from(encoded_blockmax.as_bytes().len())?,
        },
        &[TermInput::new(
            0,
            b"witness",
            TermMetadata::without_positions(
                2,
                ByteSpan::new(0, u64::try_from(encoded_postings.as_bytes().len())?),
                ByteSpan::new(0, u64::try_from(encoded_blockmax.as_bytes().len())?),
            ),
        )],
    )?;
    let encoded = encoded_identity_test_segment_with_term_sections(
        0x6e11,
        0,
        &[
            Some("witness-doc-0"),
            Some("witness-doc-1"),
            Some("witness-doc-2"),
        ],
        term_dictionary.as_bytes(),
        encoded_postings.as_bytes(),
        &[],
        encoded_blockmax.as_bytes(),
    )?;
    let pending = encoded.write_temp(directory.path())?;
    let path = publish_pending_segment(pending)?;
    let manifest = durable_test_manifest(1, vec![manifest_segment(&encoded, 1)]);
    let manifest_path = directory.path().join("MANIFEST");
    write_manifest(&manifest_path, &manifest)?;
    let original_manifest = std::fs::read(&manifest_path)?;

    let baseline = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
    assert_eq!(
        baseline.segments().len(),
        1,
        "fixture publishes cleanly first"
    );
    drop(baseline);

    let postings_entry = encoded
        .section_entries()
        .iter()
        .position(|entry| entry.kind == SectionKind::POSTINGS)
        .expect("encoded fixture has POSTINGS");
    let postings_offset = usize::try_from(encoded.section_entries()[postings_entry].offset)?;
    let postings_len = usize::try_from(encoded.section_entries()[postings_entry].len)?;
    assert_eq!(
        postings_len,
        replacement_postings.as_bytes().len(),
        "fixture POSTINGS range matches the same-length replacement"
    );

    let mut bytes = std::fs::read(&path)?;
    let trailer_start = bytes.len() - TRAILER_LEN;
    let original_trailer = bytes[trailer_start..].to_vec();
    bytes[postings_offset..postings_offset + postings_len]
        .copy_from_slice(replacement_postings.as_bytes());
    let section_xxh3_offset = FILE_PREFIX_LEN
        + FIXED_HEADER_LEN
        + postings_entry * SECTION_ENTRY_LEN
        + SECTION_ENTRY_XXH3_OFFSET;
    let postings_xxh3 =
        xxhash_rust::xxh3::xxh3_64(&bytes[postings_offset..postings_offset + postings_len]);
    bytes[section_xxh3_offset..section_xxh3_offset + std::mem::size_of::<u64>()]
        .copy_from_slice(&postings_xxh3.to_le_bytes());
    let header_len = usize::try_from(u32::from_le_bytes(bytes[12..16].try_into()?))?;
    let header_crc_offset = FILE_PREFIX_LEN + header_len;
    let header_crc = crc32fast::hash(&bytes[FILE_PREFIX_LEN..header_crc_offset]);
    bytes[header_crc_offset..header_crc_offset + std::mem::size_of::<u32>()]
        .copy_from_slice(&header_crc.to_le_bytes());

    let lazy_reader = SegmentReader::from_owned(bytes.clone(), DEFAULT_SCHEMA)?;
    let replacement_postings = lazy_reader
        .section(SectionKind::POSTINGS)?
        .expect("fixture retains POSTINGS after valid replacement");
    assert_eq!(
        replacement_postings,
        &bytes[postings_offset..postings_offset + postings_len],
        "the recomputed section witness preserves lazy POSTINGS parsing"
    );
    let parsed_replacement = PostingList::parse(replacement_postings, 2)?;
    let blockmax = lazy_reader
        .section(SectionKind::BLOCKMAX)?
        .expect("fixture retains BLOCKMAX");
    let blockmax_relation = BlockMaxConcatList::parse(blockmax, &parsed_replacement)?;
    assert_eq!(
        blockmax_relation.entry_count(),
        parsed_replacement.block_count(),
        "the unchanged BLOCKMAX stream remains related to replacement POSTINGS"
    );
    let actual_prefix_xxh3 = xxhash_rust::xxh3::xxh3_64(&bytes[..trailer_start]);
    assert_ne!(actual_prefix_xxh3, encoded.file_xxh3());
    assert_eq!(lazy_reader.file_xxh3(), encoded.file_xxh3());
    assert_eq!(&bytes[trailer_start..], original_trailer.as_slice());
    assert!(matches!(
        lazy_reader.verify_file_witness(),
        Err(QuillError::IndexCorrupted { .. })
    ));

    std::fs::write(&path, &bytes)?;
    assert_eq!(std::fs::read(&manifest_path)?, original_manifest);
    assert!(matches!(
        KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA),
        Err(KeeperError::SegmentOpen {
            source: QuillError::IndexCorrupted { detail, .. },
            ..
        }) if detail.contains("file checksum mismatch")
    ));
    Ok(())
}

#[test]
fn term_dictionary_checksum_failure_is_rejected_during_read_only_open() -> TestResult {
    let directory = tempdir()?;
    let encoded = encoded_test_segment(0xacd, 10, 20, 1)?;
    let term_dictionary = encoded
        .section_entries()
        .iter()
        .find(|entry| entry.kind == SectionKind::TERMDICT)
        .expect("TERMDICT entry");
    let term_dictionary_offset = usize::try_from(term_dictionary.offset)?;
    let term_dictionary_len = usize::try_from(term_dictionary.len)?;
    let mut bytes = encoded.as_bytes().to_vec();
    bytes[term_dictionary_offset] ^= 0x80;
    assert_ne!(
        xxhash_rust::xxh3::xxh3_64(
            &bytes[term_dictionary_offset..term_dictionary_offset + term_dictionary_len]
        ),
        term_dictionary.xxh3,
        "the TERMDICT payload mutation must deliberately leave its section witness stale"
    );
    let file_xxh3 = reseal_test_segment_file_witness(&mut bytes)?;
    assert_ne!(
        file_xxh3,
        encoded.file_xxh3(),
        "the payload mutation must rewrite the FSLX whole-file witness"
    );
    assert_eq!(
        SegmentReader::from_owned(bytes.clone(), DEFAULT_SCHEMA)?.verify_file_witness()?,
        file_xxh3,
        "the mutated fixture must retain a self-consistent FSLX trailer"
    );
    std::fs::write(directory.path().join(canonical_segment_name(0xacd)), bytes)?;
    let mut segment = manifest_segment(&encoded, 1);
    segment.file_xxh3 = file_xxh3;
    let manifest = durable_test_manifest(1, vec![segment]);
    write_manifest(&directory.path().join("MANIFEST"), &manifest)?;
    let before = directory_bytes(directory.path())?;

    assert!(matches!(
        KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA),
        Err(KeeperError::SegmentOpen {
            source: QuillError::IndexCorrupted { detail, .. },
            ..
        }) if detail.contains("section 1 checksum mismatch")
    ));
    assert_eq!(
        directory_bytes(directory.path())?,
        before,
        "read-only recovery must not mutate after rejecting TERMDICT"
    );
    Ok(())
}

#[test]
fn recovery_rejects_manifest_length_before_corrupt_file_witness() -> TestResult {
    let directory = tempdir()?;
    let encoded = encoded_test_segment(0xacd, 10, 20, 1)?;
    let doclen = encoded
        .section_entries()
        .iter()
        .find(|entry| entry.kind == SectionKind::DOCLEN)
        .expect("doclen entry");
    let doclen_offset = usize::try_from(doclen.offset)?;
    let mut bytes = encoded.as_bytes().to_vec();
    bytes[doclen_offset] ^= 0x80;
    assert!(
        SegmentReader::from_owned(bytes.clone(), DEFAULT_SCHEMA)?
            .verify_file_witness()
            .is_err(),
        "fixture must fail a whole-file witness verification"
    );
    std::fs::write(directory.path().join(canonical_segment_name(0xacd)), bytes)?;

    let manifest_file_len = encoded.file_len() + 1;
    let manifest = durable_test_manifest(
        1,
        vec![ManifestSegment {
            segment_id: 0xacd,
            seal_seq: 1,
            file_len: manifest_file_len,
            file_xxh3: encoded.file_xxh3(),
            docid_lo: 10,
            docid_hi: 20,
            doc_count: 1,
            tombstones: TombstoneSet::new(),
        }],
    );
    write_manifest(&directory.path().join("MANIFEST"), &manifest)?;
    let before = directory_bytes(directory.path())?;

    assert!(matches!(
        KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA),
        Err(KeeperError::SegmentMetadataMismatch { detail, .. })
            if detail
                == format!(
                    "file length {} != manifest {manifest_file_len}",
                    encoded.file_len()
                )
    ));
    assert_eq!(
        directory_bytes(directory.path())?,
        before,
        "read-only recovery must not mutate after rejecting manifest metadata"
    );
    Ok(())
}

#[test]
fn recovery_witness_rejects_resealed_segment_with_stale_manifest() -> TestResult {
    let directory = tempdir()?;
    let encoded = encoded_test_segment(0xace, 10, 20, 1)?;
    let doclen = encoded
        .section_entries()
        .iter()
        .find(|entry| entry.kind == SectionKind::DOCLEN)
        .expect("doclen entry");
    let doclen_offset = usize::try_from(doclen.offset)?;
    let mut bytes = encoded.as_bytes().to_vec();
    bytes[doclen_offset] ^= 0x80;
    let resealed_file_xxh3 = reseal_test_segment_file_witness(&mut bytes)?;
    assert_ne!(resealed_file_xxh3, encoded.file_xxh3());
    assert_eq!(
        SegmentReader::from_owned(bytes.clone(), DEFAULT_SCHEMA)?.verify_file_witness()?,
        resealed_file_xxh3,
        "the hostile fixture must retain a self-consistent FSLX trailer"
    );
    std::fs::write(directory.path().join(canonical_segment_name(0xace)), bytes)?;

    let mut manifest = durable_test_manifest(
        1,
        vec![ManifestSegment {
            segment_id: 0xace,
            seal_seq: 1,
            file_len: encoded.file_len(),
            file_xxh3: encoded.file_xxh3(),
            docid_lo: 10,
            docid_hi: 20,
            doc_count: 1,
            tombstones: TombstoneSet::new(),
        }],
    );
    let manifest_path = directory.path().join("MANIFEST");
    write_manifest(&manifest_path, &manifest)?;
    let before_rejection = directory_bytes(directory.path())?;

    assert!(matches!(
        KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA),
        Err(KeeperError::SegmentMetadataMismatch { detail, .. })
            if detail
                == format!(
                    "actual file_xxh3 {resealed_file_xxh3:#018x} != manifest {:#018x}",
                    encoded.file_xxh3()
                )
    ));
    assert_eq!(
        directory_bytes(directory.path())?,
        before_rejection,
        "read-only rejection must not mutate the stale-manifest fixture"
    );

    manifest.segments[0].file_xxh3 = resealed_file_xxh3;
    write_manifest(&manifest_path, &manifest)?;
    let before_control_open = directory_bytes(directory.path())?;
    let snapshot = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
    assert_eq!(snapshot.segments().len(), 1);
    assert_eq!(
        snapshot.segments()[0].term_dictionary_cache_counts(),
        (1, 0)
    );
    assert_eq!(
        directory_bytes(directory.path())?,
        before_control_open,
        "read-only control open must not mutate the matching-manifest fixture"
    );
    Ok(())
}

#[test]
fn staged_delete_filters_before_commit_and_persists_after_publish() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempdir().expect("temp directory");
        let low = write_identity_test_segment(directory.path(), 0x201, 10, 0, &[Some("same")])
            .expect("write low segment");
        let mut high =
            write_identity_test_segment(directory.path(), 0x202, 30, 100, &[Some("same")])
                .expect("write high segment");
        assert!(
            high.tombstones
                .insert(100)
                .expect("tombstone newer representative")
        );
        let mut manifest = durable_test_manifest(1, vec![low, high]);
        manifest.field_stats.push(ManifestFieldStats {
            field_ord: 0,
            total_tokens: 17,
            doc_count: 2,
        });
        write_manifest(&directory.path().join("MANIFEST"), &manifest)
            .expect("write genesis manifest");

        let original =
            KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA).expect("open original snapshot");
        assert_eq!(original.at_seal_doc_count(), 2);
        assert_eq!(original.tombstone_count(), 1);
        assert_eq!(original.doc_count(), 1);
        assert!(original.is_live(0));
        assert!(!original.is_live(100));
        assert_eq!(
            original.materialize_document_id(0),
            Some(DocId::new("same"))
        );
        assert_eq!(original.materialize_document_id(100), None);
        assert_eq!(
            original
                .resolve_document_id("same")
                .expect("resolve through tombstoned newest segment"),
            Some(ResolvedDocumentId {
                segment_id: 0x201,
                seal_seq: 10,
                global_docid: 0,
                content_hash: 1,
            })
        );

        let mut proposed = original.next_manifest().expect("next generation");
        assert!(
            original
                .delete_document(&mut proposed, "same")
                .expect("stage delete")
        );
        assert!(original.is_live(0), "installed snapshot remains unchanged");
        assert!(!crate::argus::LiveDocs::is_live(&proposed, 0));
        assert_eq!(proposed.field_stats, manifest.field_stats);
        let idempotent = proposed.clone();
        assert!(
            !original
                .delete_document(&mut proposed, "same")
                .expect("already-deleted id is a no-op")
        );
        assert_eq!(proposed, idempotent);

        let mut writer = KeeperWriter::open(&cx, directory.path(), DEFAULT_SCHEMA)
            .await
            .expect("open writer");
        let published = writer
            .publish(&cx, &proposed)
            .await
            .expect("publish tombstone generation");
        assert_eq!(published.at_seal_doc_count(), 2);
        assert_eq!(published.tombstone_count(), 2);
        assert_eq!(published.doc_count(), 0);
        assert_eq!(published.materialize_document_id(0), None);
        assert_eq!(published.materialize_document_id(100), None);
        assert_eq!(
            published.loaded_manifest().manifest.field_stats,
            manifest.field_stats
        );
        assert_eq!(
            published
                .resolve_document_id("same")
                .expect("all-dead lookup"),
            None
        );
        let reopened = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
            .expect("reopen published tombstones");
        assert_eq!(reopened.doc_count(), 0);
        assert!(!reopened.is_live(0));
        assert!(original.is_live(0), "older mmap snapshot stays visible");

        let high_watermark = writer
            .retained_snapshot_for_bookkeeping()
            .loaded_manifest()
            .manifest
            .docid_high_watermark;
        let segment_paths = writer
            .retained_snapshot_for_bookkeeping()
            .segments()
            .iter()
            .map(|segment| segment.path().to_path_buf())
            .collect::<Vec<_>>();
        let mut empty = writer
            .retained_snapshot_for_bookkeeping()
            .next_manifest()
            .expect("empty successor");
        writer
            .retained_snapshot_for_bookkeeping()
            .delete_all(&mut empty)
            .expect("stage delete_all");
        assert!(empty.segments.is_empty());
        assert!(empty.field_stats.is_empty());
        assert_eq!(empty.docid_high_watermark, high_watermark);
        assert_eq!(empty.schema_id, manifest.schema_id);
        assert_eq!(empty.engine_version, manifest.engine_version);
        let empty_snapshot = writer
            .publish(&cx, &empty)
            .await
            .expect("publish delete_all generation");
        assert_eq!(empty_snapshot.at_seal_doc_count(), 0);
        assert_eq!(empty_snapshot.tombstone_count(), 0);
        assert_eq!(empty_snapshot.doc_count(), 0);
        assert!(
            segment_paths.iter().all(|path| path.exists()),
            "publication never synchronously reclaims segment files"
        );
    });
}

#[test]
fn identity_resolution_handles_repeated_upserts_and_duplicate_live_rows() -> TestResult {
    let directory = tempdir()?;
    let mut segments = Vec::new();
    for (ordinal, (docid_lo, seal_seq)) in [(0_u64, 40_u64), (10, 10), (20, 30), (30, 20)]
        .into_iter()
        .enumerate()
    {
        let segment_id = 0x300 + u64::try_from(ordinal)?;
        let mut segment = write_identity_test_segment(
            directory.path(),
            segment_id,
            seal_seq,
            docid_lo,
            &[Some("repeated")],
        )?;
        if docid_lo != 20 {
            assert!(segment.tombstones.insert(u32::try_from(docid_lo)?)?);
        }
        segments.push(segment);
    }
    let manifest = durable_test_manifest(1, segments);
    write_manifest(&directory.path().join("MANIFEST"), &manifest)?;
    let snapshot = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
    assert_eq!(snapshot.at_seal_doc_count(), 4);
    assert_eq!(snapshot.tombstone_count(), 3);
    assert_eq!(snapshot.doc_count(), 1);
    assert_eq!(
        snapshot.resolve_document_id("repeated")?,
        Some(ResolvedDocumentId {
            segment_id: 0x302,
            seal_seq: 30,
            global_docid: 20,
            content_hash: 1,
        })
    );

    let mut all_dead = manifest.clone();
    assert!(all_dead.segments[2].tombstones.insert(20)?);
    write_manifest(&directory.path().join("MANIFEST"), &all_dead)?;
    assert_eq!(
        KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?.resolve_document_id("repeated")?,
        None
    );

    let mut duplicate_live = manifest;
    assert!(duplicate_live.segments[0].tombstones.remove(0)?);
    write_manifest(&directory.path().join("MANIFEST"), &duplicate_live)?;
    let duplicate = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
    assert!(matches!(
        duplicate.resolve_document_id("repeated"),
        Err(KeeperError::MultipleLiveDocumentIds {
            first_global_docid: 0,
            duplicate_global_docid: 20,
            ..
        })
    ));
    Ok(())
}

#[test]
fn concat_representatives_prefer_the_only_live_duplicate_independent_of_row_age() -> TestResult {
    let encoded = EncodedIdMapSection::encode(
        0,
        3,
        &[
            Some(IdMapEntryInput::new("same", 11)),
            Some(IdMapEntryInput::new("other", 22)),
            Some(IdMapEntryInput::new("same", 33)),
        ],
    )?;

    assert_eq!(
        resolve_concat_representatives(encoded.section()?, &[0])?,
        [1, 2],
        "a live later row must replace a tombstoned earlier representative",
    );
    assert_eq!(
        resolve_concat_representatives(encoded.section()?, &[2])?,
        [0, 1],
        "a live earlier row must survive a tombstoned later representative",
    );
    assert_eq!(
        resolve_concat_representatives(encoded.section()?, &[0, 2])?,
        [0, 1],
        "an all-dead identity uses its lowest global docid as canonical hash input",
    );
    assert!(matches!(
        resolve_concat_representatives(encoded.section()?, &[]),
        Err(ConcatMergeError::MultipleLiveDocumentIds {
            first_global_docid: 0,
            duplicate_global_docid: 2,
            ..
        })
    ));
    Ok(())
}

#[test]
fn snapshot_rejects_tombstone_targeting_idmap_hole() -> TestResult {
    let directory = tempdir()?;
    let segment =
        write_identity_test_segment(directory.path(), 0x401, 1, 50, &[None, Some("present")])?;
    let clean_manifest = durable_test_manifest(1, vec![segment]);
    write_manifest(&directory.path().join("MANIFEST"), &clean_manifest)?;
    let clean = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
    assert_eq!(clean.doc_count(), 1);
    assert!(!clean.is_live(50));
    assert!(clean.is_live(51));
    assert_eq!(clean.materialize_document_id(50), None);
    assert_eq!(
        clean.materialize_document_id(51),
        Some(DocId::new("present"))
    );
    assert!(!crate::argus::LiveDocs::is_live(&clean.segments()[0], 50));
    assert!(crate::argus::LiveDocs::is_live(&clean.segments()[0], 51));

    let mut manifest = clean_manifest;
    assert!(manifest.segments[0].tombstones.insert(50)?);
    write_manifest(&directory.path().join("MANIFEST"), &manifest)?;
    assert!(matches!(
        KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA),
        Err(KeeperError::TombstoneReferencesHole {
            segment_id: 0x401,
            global_docid: 50,
            ..
        })
    ));
    Ok(())
}

#[test]
fn publish_preflight_rejects_tombstone_hole_without_replacing_manifest() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempdir().expect("temp directory");
        let segment =
            write_identity_test_segment(directory.path(), 0x402, 1, 50, &[None, Some("present")])
                .expect("write identity segment");
        let manifest = durable_test_manifest(1, vec![segment]);
        let manifest_path = directory.path().join("MANIFEST");
        write_manifest(&manifest_path, &manifest).expect("write initial manifest");

        let mut writer = KeeperWriter::open(&cx, directory.path(), DEFAULT_SCHEMA)
            .await
            .expect("open writer");
        let before = std::fs::read(&manifest_path).expect("read initial manifest");
        let mut proposed = writer
            .retained_snapshot_for_bookkeeping()
            .next_manifest()
            .expect("next generation");
        assert!(
            proposed.segments[0]
                .tombstones
                .insert(50)
                .expect("stage hole tombstone")
        );

        assert!(matches!(
            writer.publish(&cx, &proposed).await,
            Err(KeeperError::TombstoneReferencesHole {
                segment_id: 0x402,
                global_docid: 50,
                ..
            })
        ));
        assert_eq!(
            std::fs::read(&manifest_path).expect("read manifest after rejection"),
            before,
            "preflight rejection must leave the current MANIFEST untouched"
        );
        assert_eq!(
            writer
                .retained_snapshot_for_bookkeeping()
                .loaded_manifest()
                .manifest
                .generation,
            1
        );
        assert_eq!(
            KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
                .expect("reopen prior snapshot")
                .loaded_manifest()
                .manifest
                .generation,
            1
        );
    });
}

#[test]
fn fallback_ignores_manifest_temp_and_rejects_incompatible_claim() -> TestResult {
    let directory = tempdir()?;
    let segment = write_test_segment(directory.path(), 0x11, 1, 0, 2)?;
    let previous = durable_test_manifest(1, vec![segment]);
    write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
    let temp = durable_test_manifest(2, Vec::new());
    write_manifest(&directory.path().join(".tmp-manifest-2"), &temp)?;
    let before = directory_bytes(directory.path())?;

    let snapshot = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
    assert_eq!(
        snapshot.loaded_manifest().source,
        ManifestSource::PreviousAfterMissingCurrent
    );
    assert_eq!(snapshot.loaded_manifest().manifest.generation, 1);
    assert_eq!(directory_bytes(directory.path())?, before);

    std::fs::write(directory.path().join("gen-3.claim"), [])?;
    assert!(matches!(
        KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA),
        Err(KeeperError::InvalidRecoveryClaim {
            recovered: 1,
            claimed: 3,
            ..
        })
    ));
    Ok(())
}
