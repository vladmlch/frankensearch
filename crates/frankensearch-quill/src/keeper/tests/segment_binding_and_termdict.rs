use super::*;

/// The live-document floor refuses a successor that serves fewer live
/// documents than its predecessor unless the caller declared the deletion
/// (gh#45). Every transition shape that can hollow a generation while
/// passing the per-segment rules is covered: tombstone growth on a
/// retained segment, removal of a whole segment, and `delete_all`. A
/// one-for-one replacement keeps the count and passes.
#[test]
fn live_document_floor_refuses_undeclared_shrink_and_admits_declared_deletion() -> TestResult {
    let genesis = KeeperSnapshot::in_memory(DEFAULT_SCHEMA)?;
    let first = encoded_identity_test_segment(0xf01, 0, &[Some("floor-a"), Some("floor-b")])?;
    let second = encoded_identity_test_segment(0xf02, 65_536, &[Some("floor-c")])?;
    let mut proposed = genesis.next_manifest()?;
    proposed.docid_high_watermark = 131_072;
    proposed.segments = vec![manifest_segment(&first, 10), manifest_segment(&second, 20)];
    let published = genesis.publish_owned_segments_with_intent(
        &proposed,
        vec![first, second],
        PublishIntent::PreserveLiveDocuments,
    )?;
    assert_eq!(
        published.doc_count(),
        3,
        "additive publication passes the floor"
    );
    let published_generation = published.loaded_manifest().manifest.generation;

    // Tombstone growth on a retained segment: legal per segment, hollow
    // as a corpus.
    let mut tombstoned = published.next_manifest()?;
    assert!(published.delete_document(&mut tombstoned, "floor-a")?);
    match published.publish_owned_segments_with_intent(
        &tombstoned,
        Vec::new(),
        PublishIntent::PreserveLiveDocuments,
    ) {
        Err(KeeperError::LiveDocumentFloor {
            previous_generation,
            previous_live_docs,
            proposed_generation,
            proposed_live_docs,
        }) => {
            assert_eq!(previous_generation, published_generation);
            assert_eq!(proposed_generation, published_generation + 1);
            assert_eq!((previous_live_docs, proposed_live_docs), (3, 2));
        }
        Ok(_) => return Err("undeclared tombstone growth must be refused".into()),
        Err(other) => return Err(format!("expected LiveDocumentFloor, got {other:?}").into()),
    }
    assert_eq!(published.doc_count(), 3, "a refusal changes nothing");
    assert_eq!(
        published
            .publish_owned_segments_with_intent(
                &tombstoned,
                Vec::new(),
                PublishIntent::ExplicitDeletion,
            )?
            .doc_count(),
        2,
        "the same proposal is admitted once the deletion is declared"
    );
    assert_eq!(
        published
            .publish_owned_segments(&tombstoned, Vec::new())?
            .doc_count(),
        2,
        "the raw entry point takes the proposal's tombstones as the deletion intent"
    );

    // Removal of a whole segment.
    let mut removed = published.next_manifest()?;
    removed
        .segments
        .retain(|segment| segment.segment_id != 0xf02);
    assert!(matches!(
        published.publish_owned_segments_with_intent(
            &removed,
            Vec::new(),
            PublishIntent::PreserveLiveDocuments,
        ),
        Err(KeeperError::LiveDocumentFloor {
            previous_live_docs: 3,
            proposed_live_docs: 2,
            ..
        })
    ));
    assert_eq!(
        published
            .publish_owned_segments_with_intent(
                &removed,
                Vec::new(),
                PublishIntent::ExplicitDeletion,
            )?
            .doc_count(),
        2
    );

    // `delete_all`: the zero-segment generation.
    let mut emptied = published.next_manifest()?;
    published.delete_all(&mut emptied)?;
    assert!(matches!(
        published.publish_owned_segments_with_intent(
            &emptied,
            Vec::new(),
            PublishIntent::PreserveLiveDocuments,
        ),
        Err(KeeperError::LiveDocumentFloor {
            previous_live_docs: 3,
            proposed_live_docs: 0,
            ..
        })
    ));
    assert_eq!(
        published
            .publish_owned_segments_with_intent(
                &emptied,
                Vec::new(),
                PublishIntent::ExplicitDeletion,
            )?
            .doc_count(),
        0
    );

    // A one-for-one replacement — tombstone plus the row that supersedes
    // it — holds the live count and passes under the preserving intent.
    let replacement = encoded_identity_test_segment(0xf03, 131_072, &[Some("floor-a-v2")])?;
    let mut replaced = published.next_manifest()?;
    assert!(published.delete_document(&mut replaced, "floor-a")?);
    replaced.docid_high_watermark = 196_608;
    replaced.segments.push(manifest_segment(&replacement, 30));
    let replaced = published.publish_owned_segments_with_intent(
        &replaced,
        vec![replacement],
        PublishIntent::PreserveLiveDocuments,
    )?;
    assert_eq!(replaced.doc_count(), 3);
    assert_eq!(replaced.materialize_document_id(0), None);
    assert_eq!(
        replaced.materialize_document_id(131_072),
        Some(DocId::new("floor-a-v2"))
    );
    Ok(())
}

#[test]
fn owned_segment_publication_is_snapshot_isolated_and_uniform() -> TestResult {
    let original = KeeperSnapshot::in_memory(DEFAULT_SCHEMA)?;
    let first = encoded_identity_test_segment(0xb01, 0, &[Some("owned-a")])?;
    let second = encoded_identity_test_segment(0xb02, 65_536, &[None, Some("owned-b")])?;
    let mut proposed = original.next_manifest()?;
    proposed.docid_high_watermark = 131_072;
    proposed.segments = vec![manifest_segment(&first, 10), manifest_segment(&second, 20)];

    assert!(original.segments().is_empty());
    assert_eq!(original.doc_count(), 0);
    assert_eq!(original.materialize_document_id(0), None);

    let published = original.publish_owned_segments(&proposed, vec![first, second])?;
    assert!(
        original.segments().is_empty(),
        "old snapshot stays unchanged"
    );
    assert_eq!(original.doc_count(), 0);
    assert_eq!(published.segments().len(), 2);
    assert_eq!(published.at_seal_doc_count(), 2);
    assert_eq!(published.doc_count(), 2);
    assert_eq!(
        published.materialize_document_id(0),
        Some(DocId::new("owned-a"))
    );
    assert_eq!(published.materialize_document_id(65_536), None);
    assert_eq!(
        published.materialize_document_id(65_537),
        Some(DocId::new("owned-b"))
    );
    assert_eq!(published.segments()[0].header().segment_id, 0xb01);
    assert_eq!(
        published.segments()[1].section(SectionKind::TERMDICT)?,
        Some(EMPTY_TERM_DICTIONARY.as_slice())
    );
    published.segments()[0].verify()?;
    published.segments()[1].verify()?;

    let mut tombstoned = published.next_manifest()?;
    assert!(tombstoned.segments[0].insert_tombstone(0)?);
    let deleted = published.publish_owned_segments(&tombstoned, Vec::new())?;
    assert_eq!(deleted.doc_count(), 1);
    assert_eq!(deleted.materialize_document_id(0), None);
    assert_eq!(
        published.materialize_document_id(0),
        Some(DocId::new("owned-a")),
        "retained owned backing does not change the older snapshot"
    );
    Ok(())
}

#[test]
fn dense_idmap_liveness_preserves_owned_and_mapped_visibility() -> TestResult {
    let original = KeeperSnapshot::in_memory(DEFAULT_SCHEMA)?;
    let dense =
        encoded_identity_test_segment(0xb11, 10, &[Some("owned-dense-a"), Some("owned-dense-b")])?;
    let holes = encoded_identity_test_segment(0xb12, 20, &[None, Some("owned-hole")])?;
    let mut proposed = original.next_manifest()?;
    proposed.docid_high_watermark = 22;
    proposed.segments = vec![manifest_segment(&dense, 10), manifest_segment(&holes, 20)];
    let owned = original.publish_owned_segments(&proposed, vec![dense, holes])?;
    let [owned_dense, owned_holes] = owned.segments() else {
        return Err("fixture publishes two owned segments".into());
    };
    assert_eq!(
        owned_dense.id_lookup.identity_row_presence(10),
        IdMapPresence::Present
    );
    assert_eq!(
        owned_holes.id_lookup.identity_row_presence(20),
        IdMapPresence::RequiresIdMap
    );
    assert_eq!(
        owned_holes.id_lookup.identity_row_presence(21),
        IdMapPresence::RequiresIdMap
    );
    assert!(crate::argus::LiveDocs::is_live(owned_dense, 10));
    assert!(crate::argus::LiveDocs::is_live(owned_dense, 11));
    assert!(!crate::argus::LiveDocs::is_live(owned_holes, 20));
    assert!(crate::argus::LiveDocs::is_live(owned_holes, 21));
    let owned_dense_hits = crate::argus::ReferenceScorer::all(10, 12, 2)?
        .top_k(2, owned_dense)?
        .into_iter()
        .map(|hit| (hit.global_docid, hit.score.to_bits()))
        .collect::<Vec<_>>();
    assert_eq!(
        owned_dense_hits,
        vec![(10, 1.0_f32.to_bits()), (11, 1.0_f32.to_bits())],
        "dense owned rows retain the exact collector hit, order, and score contract"
    );
    assert_eq!(
        owned.materialize_document_id(10),
        Some(DocId::new("owned-dense-a"))
    );
    assert_eq!(owned.materialize_document_id(20), None);
    assert_eq!(
        owned.materialize_document_id(21),
        Some(DocId::new("owned-hole"))
    );

    let mut tombstoned_manifest = owned.next_manifest()?;
    assert!(tombstoned_manifest.segments[0].insert_tombstone(11)?);
    let tombstoned = owned.publish_owned_segments(&tombstoned_manifest, Vec::new())?;
    assert!(tombstoned.is_live(10));
    assert!(
        !tombstoned.is_live(11),
        "a dense physical row remains hidden by its manifest tombstone"
    );
    let tombstoned_dense = &tombstoned.segments()[0];
    let tombstoned_dense_hits = crate::argus::ReferenceScorer::all(10, 12, 2)?
        .top_k(2, tombstoned_dense)?
        .into_iter()
        .map(|hit| (hit.global_docid, hit.score.to_bits()))
        .collect::<Vec<_>>();
    assert_eq!(
        tombstoned_dense_hits,
        vec![(10, 1.0_f32.to_bits())],
        "dense rows retain the tombstone filter in collector results"
    );
    assert_eq!(tombstoned.materialize_document_id(11), None);
    assert!(tombstoned.is_live(21));

    let directory = tempdir()?;
    let mut mapped_dense = write_identity_test_segment(
        directory.path(),
        0xb13,
        1,
        50,
        &[Some("mapped-dense-a"), Some("mapped-dense-b")],
    )?;
    assert!(mapped_dense.insert_tombstone(51)?);
    let mapped_holes =
        write_identity_test_segment(directory.path(), 0xb14, 2, 60, &[None, Some("mapped-hole")])?;
    let mapped_manifest = durable_test_manifest(1, vec![mapped_dense, mapped_holes]);
    write_manifest(&directory.path().join("MANIFEST"), &mapped_manifest)?;
    let mapped = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
    let [mapped_dense, mapped_holes] = mapped.segments() else {
        return Err("fixture opens two mapped segments".into());
    };
    assert_eq!(
        mapped_dense.id_lookup.identity_row_presence(50),
        IdMapPresence::Present
    );
    assert_eq!(
        mapped_holes.id_lookup.identity_row_presence(60),
        IdMapPresence::RequiresIdMap
    );
    assert!(mapped.is_live(50));
    assert!(
        !mapped.is_live(51),
        "tombstones retain their visibility gate"
    );
    assert!(
        !mapped.is_live(60),
        "mapped holes retain byte-backed absence"
    );
    assert!(mapped.is_live(61));
    assert!(!mapped.is_live(62));
    let mapped_dense_hits = crate::argus::ReferenceScorer::all(50, 52, 2)?
        .top_k(2, mapped_dense)?
        .into_iter()
        .map(|hit| (hit.global_docid, hit.score.to_bits()))
        .collect::<Vec<_>>();
    assert_eq!(
        mapped_dense_hits,
        vec![(50, 1.0_f32.to_bits())],
        "mapped dense rows retain exact collector order, score, and tombstone parity"
    );
    assert_eq!(
        mapped.materialize_document_id(50),
        Some(DocId::new("mapped-dense-a"))
    );
    assert_eq!(mapped.materialize_document_id(51), None);
    assert_eq!(mapped.materialize_document_id(60), None);
    assert_eq!(
        mapped.materialize_document_id(61),
        Some(DocId::new("mapped-hole"))
    );
    Ok(())
}

#[test]
fn bound_identity_ranges_drive_lookup_across_owned_rebind() -> TestResult {
    let original = KeeperSnapshot::in_memory(DEFAULT_SCHEMA)?;
    let encoded =
        encoded_identity_test_segment(0xb21, 0, &[Some("range-a"), None, Some("range-b")])?;
    let mut proposed = original.next_manifest()?;
    proposed.docid_high_watermark = 3;
    proposed.segments = vec![manifest_segment(&encoded, 10)];
    let published = original.publish_owned_segments(&proposed, vec![encoded])?;
    let segment = &published.segments()[0];
    let expected_range = |kind| {
        let entry = segment
            .section_entries()
            .iter()
            .find(|entry| entry.kind == kind)
            .expect("identity section entry");
        let start = usize::try_from(entry.offset).expect("section offset fits usize");
        let len = usize::try_from(entry.len).expect("section length fits usize");
        start..start + len
    };
    assert_eq!(
        segment.identity_sections.id_map,
        expected_range(SectionKind::IDMAP)
    );
    assert_eq!(
        segment.identity_sections.id_hash,
        expected_range(SectionKind::IDHASH)
    );
    assert_eq!(
        published
            .resolve_document_id("range-a")?
            .map(|row| row.global_docid),
        Some(0)
    );
    assert_eq!(published.resolve_document_id("missing")?, None);

    let backing_len = segment.reader.source_bytes().len();
    let mut invalid_id_map_range = published.clone();
    invalid_id_map_range.segments[0].identity_sections.id_map =
        backing_len..backing_len.saturating_add(1);
    assert!(matches!(
        invalid_id_map_range.segments[0].lookup_document_id("range-a"),
        Err(KeeperError::SegmentMetadataMismatch { ref detail, .. })
            if detail == "validated IDMAP range exceeds the immutable backing"
    ));

    let mut invalid_id_hash_range = published.clone();
    invalid_id_hash_range.segments[0].identity_sections.id_hash =
        backing_len..backing_len.saturating_add(1);
    assert!(matches!(
        invalid_id_hash_range.segments[0].lookup_document_id("range-a"),
        Err(KeeperError::SegmentMetadataMismatch { ref detail, .. })
            if detail == "validated IDHASH range exceeds the immutable backing"
    ));

    let mut invalid = published.next_manifest()?;
    assert!(invalid.segments[0].insert_tombstone(1)?);
    assert!(matches!(
        published.publish_owned_segments(&invalid, Vec::new()),
        Err(KeeperError::TombstoneReferencesHole {
            global_docid: 1,
            ..
        })
    ));

    let mut tombstoned = published.next_manifest()?;
    assert!(tombstoned.segments[0].insert_tombstone(0)?);
    let rebound = published.publish_owned_segments(&tombstoned, Vec::new())?;
    assert_eq!(
        rebound.segments()[0].identity_sections,
        published.segments()[0].identity_sections,
        "an owned tombstone-only rebind must preserve the exact validated byte ranges"
    );
    assert_eq!(rebound.resolve_document_id("range-a")?, None);
    assert_eq!(
        rebound
            .resolve_document_id("range-b")?
            .map(|row| row.global_docid),
        Some(2)
    );
    Ok(())
}

#[test]
fn term_dictionary_metadata_is_validated_once_per_bound_segment_and_reused_concurrently()
-> TestResult {
    let original = KeeperSnapshot::in_memory(DEFAULT_SCHEMA)?;
    let first = encoded_identity_test_segment(0xc01, 0, &[Some("cache-a")])?;
    let second = encoded_identity_test_segment(0xc02, 65_536, &[Some("cache-b")])?;
    let mut proposed = original.next_manifest()?;
    proposed.docid_high_watermark = 65_537;
    proposed.segments = vec![manifest_segment(&first, 10), manifest_segment(&second, 20)];
    let published = original.publish_owned_segments(&proposed, vec![first, second])?;

    assert_eq!(
        published.segments()[0].term_dictionary_cache_counts(),
        (1, 0)
    );
    assert_eq!(
        published.segments()[1].term_dictionary_cache_counts(),
        (1, 0)
    );
    assert_eq!(
        published.segments()[0].authenticated_file_witness_hash_count(),
        0,
        "fresh owned binding must reuse the encoder's authenticated whole-file witness"
    );
    assert_eq!(
        published.segments()[1].authenticated_file_witness_hash_count(),
        0,
        "each fresh owned backing must avoid a redundant whole-prefix hash"
    );
    std::thread::scope(|scope| {
        let segment = &published.segments()[0];
        for _ in 0..4 {
            scope.spawn(move || {
                for _ in 0..32 {
                    let dictionary = segment
                        .term_dictionary(DEFAULT_SCHEMA)
                        .expect("borrow validated TERMDICT metadata");
                    assert_eq!(dictionary.term_count(), 0);
                }
            });
        }
    });
    assert_eq!(
        published.segments()[0].term_dictionary_cache_counts(),
        (1, 128)
    );
    assert_eq!(
        published.clone().segments()[0].term_dictionary_cache_counts(),
        (1, 128),
        "snapshot clones share the same immutable segment binding"
    );

    let mut tombstoned = published.next_manifest()?;
    assert!(tombstoned.segments[0].insert_tombstone(0)?);
    let rebound = published.publish_owned_segments(&tombstoned, Vec::new())?;
    assert!(
        Arc::ptr_eq(
            &rebound.segments()[0].term_dictionary_metadata,
            &published.segments()[0].term_dictionary_metadata,
        ),
        "a tombstone-only rebind must reuse the exact validated metadata \
         allocation for the identical immutable backing"
    );
    assert_eq!(
        rebound.segments()[0].term_dictionary_cache_counts(),
        (1, 128),
        "the rebound binding shares its backing's counters: still exactly \
         one complete validation for these bytes"
    );
    assert_eq!(
        rebound.segments()[0].term_dictionary_metadata_reuse_count(),
        1
    );
    assert_eq!(
        rebound.segments()[0].authenticated_file_witness_hash_count(),
        0,
        "an owned tombstone-only rebind must perform zero new whole-file hashes"
    );
    assert_eq!(
        published.segments()[0].term_dictionary_cache_counts(),
        (1, 128),
        "the prior snapshot observes the same shared backing counters"
    );

    let mut chained = rebound.next_manifest()?;
    assert!(chained.segments[1].insert_tombstone(65_536)?);
    let chained_snapshot = rebound.publish_owned_segments(&chained, Vec::new())?;
    assert!(Arc::ptr_eq(
        &chained_snapshot.segments()[0].term_dictionary_metadata,
        &published.segments()[0].term_dictionary_metadata,
    ));
    assert_eq!(
        chained_snapshot.segments()[0].term_dictionary_metadata_reuse_count(),
        2,
        "rebind chains keep reusing the original validation"
    );
    assert_eq!(
        chained_snapshot.segments()[0]
            .term_dictionary_cache_counts()
            .0,
        1,
        "no rebind in the chain may repeat a complete validation"
    );
    assert_eq!(
        chained_snapshot.segments()[0].authenticated_file_witness_hash_count(),
        0,
        "the retained owned backing must carry its original full-prefix receipt across rebinds"
    );
    Ok(())
}

#[test]
fn owned_rebind_rejects_manifest_witness_changes_and_backing_substitution() -> TestResult {
    let original = KeeperSnapshot::in_memory(DEFAULT_SCHEMA)?;
    let first = encoded_identity_test_segment(0xd01, 0, &[Some("wit-a")])?;
    let second = encoded_identity_test_segment(0xd02, 65_536, &[Some("wit-b")])?;
    let mut proposed = original.next_manifest()?;
    proposed.docid_high_watermark = 65_537;
    proposed.segments = vec![manifest_segment(&first, 10), manifest_segment(&second, 20)];
    let published = original.publish_owned_segments(&proposed, vec![first, second])?;
    let [seg_a, seg_b] = published.segments() else {
        panic!("fixture publishes exactly two segments");
    };

    let predecessor_witness_hashes = seg_a.authenticated_file_witness_hash_count();
    let substituted_backing_witness_hashes = seg_b.authenticated_file_witness_hash_count();

    // A hostile successor cannot retarget the predecessor's authenticated
    // whole-file receipt by changing its MANIFEST witness.
    let mut changed_witness = seg_a.manifest.clone();
    changed_witness.file_xxh3 ^= 1;
    assert!(
        matches!(
            seg_a.rebind(changed_witness),
            Err(KeeperError::InvalidTransition { detail })
                if detail.contains("immutable metadata changed")
        ),
        "owned rebind must reject a successor with a changed MANIFEST file witness"
    );
    assert_eq!(
        seg_a.authenticated_file_witness_hash_count(),
        predecessor_witness_hashes,
        "rejected manifest witness changes must not trigger a new whole-file hash"
    );

    // A predecessor naming a different owned Arc must reject rather than
    // borrowing either the backing or authenticated receipt from it.
    let mut successor = seg_a.manifest.clone();
    assert!(successor.tombstones.insert(0)?);
    assert!(
        matches!(
            seg_a.rebind_with_backing(successor, Arc::clone(&seg_b.reader)),
            Err(KeeperError::InvalidTransition { detail })
                if detail.contains("exact authenticated immutable backing")
        ),
        "owned rebind must reject backing substitution before metadata reuse"
    );
    assert_eq!(
        seg_a.authenticated_file_witness_hash_count(),
        predecessor_witness_hashes,
        "rejected backing substitution must not trigger a new whole-file hash"
    );
    assert_eq!(
        seg_b.authenticated_file_witness_hash_count(),
        substituted_backing_witness_hashes,
        "the substituted backing must not be rehashed while rejecting the transition"
    );
    assert_eq!(seg_a.term_dictionary_metadata_reuse_count(), 0);
    assert_eq!(seg_b.term_dictionary_metadata_reuse_count(), 0);
    Ok(())
}

#[test]
fn cached_termdict_views_do_not_re_read_bytes_while_fresh_open_fails_closed() -> TestResult {
    let directory = tempdir()?;
    let encoded = encoded_identity_test_segment(0xe01, 0, &[Some("live"), Some("spare")])?;
    let segment_path = directory.path().join(canonical_segment_name(0xe01));
    std::fs::write(&segment_path, encoded.as_bytes())?;
    let manifest = durable_test_manifest(1, vec![manifest_segment(&encoded, 1)]);
    write_manifest(&directory.path().join("MANIFEST"), &manifest)?;

    let snapshot = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
    assert_eq!(
        snapshot.segments()[0].term_dictionary_cache_counts(),
        (1, 0)
    );
    assert_eq!(
        snapshot.segments()[0]
            .term_dictionary(DEFAULT_SCHEMA)?
            .term_count(),
        0
    );

    // Corrupt the durable TERMDICT payload AFTER the snapshot validated
    // and cached its metadata. The empty dictionary's leading block_count
    // becomes non-zero, so ANY re-validation of these bytes must fail.
    //
    // Note the structural QG-3 guard this test leans on: mutable bytes
    // exist only on the durable path, and every durable successor
    // snapshot is a cold reopen that re-validates from scratch. The
    // in-memory rebind reuse path can never observe divergent bytes
    // because its backing is an immutable shared allocation.
    let termdict_offset = usize::try_from(
        encoded
            .section_entries()
            .iter()
            .find(|entry| entry.kind == SectionKind::TERMDICT)
            .expect("fixture TERMDICT entry")
            .offset,
    )?;
    let mut corrupted = std::fs::read(&segment_path)?;
    corrupted[termdict_offset] ^= 0x01;
    std::fs::write(&segment_path, corrupted)?;

    // The live snapshot keeps serving borrowed views from the cached
    // validated metadata without re-reading or re-hashing TERMDICT
    // content, so it must not detect the on-disk mutation.
    assert_eq!(
        snapshot.segments()[0]
            .term_dictionary(DEFAULT_SCHEMA)?
            .term_count(),
        0,
        "cached views must not re-read durable TERMDICT bytes"
    );
    assert_eq!(
        snapshot.segments()[0].term_dictionary_cache_counts().0,
        1,
        "the cached path must not run a second complete validation"
    );

    // A cold open of the same directory sees the corrupted bytes with no
    // cache to lean on and must fail closed before publication.
    let Err(error) = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA) else {
        panic!("fresh open must re-validate and reject corrupted TERMDICT bytes");
    };
    assert!(
        matches!(&error, KeeperError::SegmentOpen { path, .. } if path == &segment_path),
        "fresh open must fail closed on the corrupted segment: {error:?}"
    );
    Ok(())
}

/// Write `count` single-document segments with contiguous docid ranges
/// and a MANIFEST naming them in order; returns each path and encoding.
fn write_multi_segment_index(
    directory: &Path,
    count: u64,
) -> Result<Vec<(PathBuf, EncodedSegment)>, Box<dyn std::error::Error>> {
    let mut written = Vec::new();
    let mut manifest_segments = Vec::new();
    for ordinal in 0..count {
        let segment_id = 0xf00 + ordinal;
        let document = format!("doc-{ordinal}");
        let encoded =
            encoded_identity_test_segment(segment_id, ordinal, &[Some(document.as_str())])?;
        let path = directory.join(canonical_segment_name(segment_id));
        std::fs::write(&path, encoded.as_bytes())?;
        manifest_segments.push(manifest_segment(&encoded, ordinal + 1));
        written.push((path, encoded));
    }
    write_manifest(
        &directory.join("MANIFEST"),
        &durable_test_manifest(1, manifest_segments),
    )?;
    Ok(written)
}

fn flip_termdict_byte(path: &Path, encoded: &EncodedSegment) -> TestResult {
    let offset = usize::try_from(
        encoded
            .section_entries()
            .iter()
            .find(|entry| entry.kind == SectionKind::TERMDICT)
            .expect("fixture TERMDICT entry")
            .offset,
    )?;
    let mut bytes = std::fs::read(path)?;
    bytes[offset] ^= 0x01;
    std::fs::write(path, bytes)?;
    Ok(())
}

#[test]
fn concurrent_segment_open_keeps_manifest_order_and_hashes_each_segment_once() -> TestResult {
    let directory = tempdir()?;
    let written = write_multi_segment_index(directory.path(), 6)?;

    let snapshot = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
    assert_eq!(snapshot.segments().len(), written.len());
    for (segment, (path, encoded)) in snapshot.segments().iter().zip(&written) {
        assert_eq!(segment.manifest().segment_id, encoded.header().segment_id);
        assert_eq!(&segment.path, path);
        assert_eq!(
            segment.authenticated_file_witness_hash_count(),
            1,
            "every segment still gets exactly one full-prefix hash"
        );
    }
    assert_eq!(snapshot.live_doc_count, 6);
    Ok(())
}

#[test]
fn concurrent_segment_open_reports_the_first_corrupt_segment_in_manifest_order() -> TestResult {
    let directory = tempdir()?;
    let written = write_multi_segment_index(directory.path(), 6)?;
    // Corrupt a later segment as well, so a worker that reaches it first
    // cannot change which failure is reported.
    for index in [2, 5] {
        let (path, encoded) = &written[index];
        flip_termdict_byte(path, encoded)?;
    }

    for _ in 0..8 {
        let Err(error) = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA) else {
            panic!("an open over corrupted segments must fail closed");
        };
        assert!(
            matches!(&error, KeeperError::SegmentOpen { path, .. } if path == &written[2].0),
            "the first corrupt segment in MANIFEST order must be reported: {error:?}"
        );
    }
    Ok(())
}

#[test]
fn mapped_segment_rebind_is_rejected_with_reopen_required_transition_error() -> TestResult {
    let directory = tempdir()?;
    let encoded = encoded_identity_test_segment(0xe02, 0, &[Some("map-a"), Some("map-b")])?;
    std::fs::write(
        directory.path().join(canonical_segment_name(0xe02)),
        encoded.as_bytes(),
    )?;
    let manifest = durable_test_manifest(1, vec![manifest_segment(&encoded, 1)]);
    write_manifest(&directory.path().join("MANIFEST"), &manifest)?;

    let snapshot = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
    let segment = &snapshot.segments()[0];
    assert_eq!(segment.term_dictionary_cache_counts(), (1, 0));

    // A mapped backing must never rebind in place: neither metadata reuse
    // nor the fresh-validation fallback is honest over bytes that can
    // change on disk behind warm checksum gates. The typed transition
    // error tells the caller a durable reopen is required.
    let mut successor = snapshot.next_manifest()?;
    assert!(successor.segments[0].insert_tombstone(1)?);
    let error = segment.rebind(successor.segments[0].clone()).err();
    assert!(
        matches!(
            error.as_ref(),
            Some(KeeperError::InvalidTransition { detail })
                if detail.contains("cannot rebind in place")
                    && detail.contains("durable reopen")
                    && detail.contains("owned-only")
        ),
        "mapped rebind must fail with the typed reopen-required transition \
         error: {error:?}"
    );
    assert_eq!(
        segment.term_dictionary_cache_counts(),
        (1, 0),
        "the rejected rebind must not validate or reuse anything"
    );
    assert_eq!(segment.term_dictionary_metadata_reuse_count(), 0);
    Ok(())
}

#[test]
fn validated_termdict_metadata_bytes_are_accounted_and_shared_across_rebinds() -> TestResult {
    let original = KeeperSnapshot::in_memory(DEFAULT_SCHEMA)?;
    let first = encoded_identity_test_segment(0xf01, 0, &[Some("acct-a")])?;
    let second = encoded_identity_test_segment(0xf02, 65_536, &[Some("acct-b")])?;
    let mut proposed = original.next_manifest()?;
    proposed.docid_high_watermark = 65_537;
    proposed.segments = vec![manifest_segment(&first, 10), manifest_segment(&second, 20)];
    let published = original.publish_owned_segments(&proposed, vec![first, second])?;

    let per_segment = published
        .segments()
        .iter()
        .map(RecoveredSegment::term_dictionary_metadata_payload_bytes)
        .collect::<Vec<_>>();
    assert!(
        per_segment.iter().all(|&bytes| bytes > 0),
        "every binding must account its retained metadata object"
    );
    let expected_total = per_segment.iter().fold(0_u64, |total, &bytes| {
        total.saturating_add(u64::try_from(bytes).unwrap_or(u64::MAX))
    });
    assert_eq!(
        published.term_dictionary_metadata_payload_bytes(),
        expected_total
    );

    let mut tombstoned = published.next_manifest()?;
    assert!(tombstoned.segments[0].insert_tombstone(0)?);
    let rebound = published.publish_owned_segments(&tombstoned, Vec::new())?;
    assert_eq!(
        rebound.segments()[0].term_dictionary_metadata_payload_bytes(),
        per_segment[0],
        "a rebound binding reports the shared allocation, not a new one"
    );
    assert_eq!(
        rebound.term_dictionary_metadata_payload_bytes(),
        expected_total,
        "tombstone-only rebinds must not grow persistent metadata bytes"
    );
    Ok(())
}

#[test]
fn term_dictionary_admission_preserves_resource_and_corruption_taxonomy() {
    let path = Path::new("segment-0000000000000cab.fslx");
    let non_corruption = [
        TermDictionaryError::InvalidSchema {
            detail: "test schema rejection".to_owned(),
        },
        TermDictionaryError::ByteBudgetExceeded {
            limit: 8,
            actual: 9,
        },
        TermDictionaryError::BlockBudgetExceeded {
            limit: 8,
            actual: 9,
        },
        TermDictionaryError::TermBudgetExceeded {
            limit: 8,
            actual: 9,
        },
        TermDictionaryError::RestartBudgetExceeded {
            limit: 8,
            actual: 9,
        },
        TermDictionaryError::NonAscendingInput { index: 1 },
        TermDictionaryError::InvalidRange,
        TermDictionaryError::MaterializationLimitExceeded { limit: 8 },
        TermDictionaryError::GlobExpansionLimitExceeded {
            pattern: "test*".to_owned(),
            field_ord: 1,
            limit: 8,
            actual: 9,
        },
        TermDictionaryError::Allocation {
            context: "test metadata",
            count: 9,
        },
        TermDictionaryError::SizeOverflow {
            field: "test metadata",
        },
    ];
    for source in non_corruption {
        let error = term_dictionary_admission_error(path, source);
        assert!(
            matches!(error, KeeperError::SegmentTermDictionaryUnavailable { .. }),
            "non-corruption TERMDICT failure changed class: {error}",
        );
        assert!(
            !recovery_retryable(&error),
            "resource/schema/query-shape failures must not be retried as damaged segments",
        );
        let public: SearchError = error.into();
        assert!(
            matches!(
                public,
                SearchError::SubsystemError {
                    subsystem: "quill",
                    ..
                }
            ),
            "non-corruption TERMDICT failure became public corruption: {public}",
        );
    }

    let error = term_dictionary_admission_error(
        path,
        TermDictionaryError::Truncated {
            offset: 4,
            needed: 8,
            remaining: 1,
        },
    );
    assert!(
        matches!(error, KeeperError::SegmentMetadataMismatch { .. }),
        "malformed durable TERMDICT bytes must remain segment corruption: {error}",
    );
    assert!(
        recovery_retryable(&error),
        "durable TERMDICT mismatch retains the read-once recovery retry",
    );
    let public: SearchError = error.into();
    assert!(
        matches!(public, SearchError::IndexCorrupted { .. }),
        "malformed durable TERMDICT bytes must remain public corruption: {public}",
    );
}

#[test]
fn cached_term_dictionary_binding_failures_remain_public_subsystem_errors() -> TestResult {
    fn assert_public_invariant(error: KeeperError, expected_path: &Path, expected_detail: &str) {
        assert!(
            matches!(
                &error,
                KeeperError::SegmentTermDictionaryUnavailable {
                    path,
                    source: TermDictionaryError::InvalidSchema { detail },
                } if path == expected_path && detail.contains(expected_detail)
            ),
            "cached TERMDICT binding invariant changed class: {error}",
        );
        assert!(
            !recovery_retryable(&error),
            "cached TERMDICT binding invariants must not retry a prior generation",
        );
        let public: SearchError = error.into();
        assert!(
            matches!(
                public,
                SearchError::SubsystemError {
                    subsystem: "quill",
                    ..
                }
            ),
            "cached TERMDICT binding invariant became public corruption: {public}",
        );
    }

    let genesis = KeeperSnapshot::in_memory(DEFAULT_SCHEMA)?;
    let segment = encoded_identity_test_segment(0xcb1, 0, &[Some("live")])?;
    let mut manifest = genesis.next_manifest()?;
    manifest.docid_high_watermark = 1;
    manifest.segments = vec![manifest_segment(&segment, 1)];
    let published = genesis.publish_owned_segments(&manifest, vec![segment])?;
    let bound = &published.segments()[0];

    assert_ne!(DEFAULT_SCHEMA, FSFS_CHUNK_SCHEMA);
    assert_public_invariant(
        bound
            .term_dictionary(FSFS_CHUNK_SCHEMA)
            .expect_err("query schema disagreement must fail"),
        bound.path(),
        "query schema disagrees",
    );

    let validated_bytes = bound
        .reader
        .section(SectionKind::TERMDICT)?
        .expect("published identity fixture has TERMDICT");
    let copied_bytes = validated_bytes.to_vec();
    assert_public_invariant(
        bind_validated_term_dictionary(
            bound.path(),
            &copied_bytes,
            &bound.term_dictionary_metadata,
        )
        .expect_err("metadata must remain bound to its exact immutable bytes"),
        bound.path(),
        "different byte source",
    );
    assert_public_invariant(
        bind_validated_term_dictionary(
            bound.path(),
            &validated_bytes[..validated_bytes.len() - 1],
            &bound.term_dictionary_metadata,
        )
        .expect_err("metadata must remain bound to the exact validated byte length"),
        bound.path(),
        "different byte source",
    );
    Ok(())
}

#[test]
fn malformed_candidate_dictionary_fails_before_publication_and_preserves_live_snapshot()
-> TestResult {
    let genesis = KeeperSnapshot::in_memory(DEFAULT_SCHEMA)?;
    let live = encoded_identity_test_segment(0xc11, 0, &[Some("live")])?;
    let mut first_manifest = genesis.next_manifest()?;
    first_manifest.docid_high_watermark = 1;
    first_manifest.segments = vec![manifest_segment(&live, 10)];
    let published = genesis.publish_owned_segments(&first_manifest, vec![live])?;
    assert_eq!(published.doc_count(), 1);
    assert_eq!(
        published.materialize_document_id(0),
        Some(DocId::new("live"))
    );

    let implausible_block_header = 1_u32.to_le_bytes();
    let invalid = encoded_identity_test_segment_with_dictionary(
        0xc12,
        1,
        &[Some("must-not-publish")],
        &implausible_block_header,
    )?;
    let mut rejected_manifest = published.next_manifest()?;
    rejected_manifest.docid_high_watermark = 2;
    rejected_manifest
        .segments
        .push(manifest_segment(&invalid, 20));
    let Err(error) = published.publish_owned_segments(&rejected_manifest, vec![invalid]) else {
        panic!("malformed TERMDICT must fail before publication");
    };
    assert!(matches!(
        error,
        KeeperError::SegmentMetadataMismatch { detail, .. }
            if detail.contains("TERMDICT declares 1 blocks in only 4 bytes")
    ));
    assert_eq!(published.doc_count(), 1);
    assert_eq!(published.segments().len(), 1);
    assert_eq!(
        published.materialize_document_id(0),
        Some(DocId::new("live"))
    );
    assert_eq!(published.materialize_document_id(1), None);
    assert_eq!(
        published.segments()[0].term_dictionary_cache_counts(),
        (1, 0)
    );
    Ok(())
}
