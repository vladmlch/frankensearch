use crate::quiver::id_hash::{
    build_id_hash_entries, encode_id_hash_with_hasher, encode_resolved_id_hash_with_hasher,
    id_hash_slot, parse_id_hash_with_hasher, seeded_id_hash, write_id_hash_slot,
};
use crate::quiver::id_map::{checked_id_map_span, id_map_prefix_len};

use super::*;

#[test]
fn id_map_wire_roundtrip_holes_utf8_and_u32_boundary() -> TestResult {
    assert_eq!(id_map_content_hash(b"canonical"), 0x03f3_4438_b914_3253);
    assert_eq!(id_map_content_hash(b""), 0x2d06_8005_38d3_94c2);
    assert_eq!(
        IdMapEntryInput::from_canonical_content("id", b"canonical").content_hash,
        0x03f3_4438_b914_3253
    );
    let inputs = [
        Some(IdMapEntryInput::new("a", 0x0807_0605_0403_0201)),
        None,
        Some(IdMapEntryInput::new("β", 0)),
    ];
    let encoded = EncodedIdMapSection::encode(40, 43, &inputs)?;
    let expected = [
        3, 0, 0, 0, 48, 0, 0, 0, // directory
        0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, // offsets
        1, 2, 3, 4, 5, 6, 7, 8, // first content hash
        0, 0, 0, 0, 0, 0, 0, 0, // hole hash
        0, 0, 0, 0, 0, 0, 0, 0, // present zero hash
        b'a', 0xce, 0xb2,
    ];
    assert_eq!(encoded.as_bytes(), expected);
    assert_eq!(encoded.present_count(), 2);
    let section = encoded.section()?;
    assert_eq!(section.docid_lo(), 40);
    assert_eq!(section.docid_hi(), 43);
    assert_eq!(section.span(), 3);
    assert_eq!(section.present_count(), 2);
    assert_eq!(section.get(39), None);
    assert_eq!(section.get(40).map(IdMapEntry::document_id), Some("a"));
    assert_eq!(section.get(41), None);
    assert_eq!(section.get(42).map(IdMapEntry::document_id), Some("β"));
    assert_eq!(section.get(43), None);
    assert_eq!(section.materialize(42).as_deref(), Some("β"));
    let first = section.get(40).ok_or("first IDMAP row")?;
    assert_eq!(first.document_id().as_ptr(), section.blob().as_ptr());
    assert_eq!(
        section
            .entries()
            .map(|(docid, entry)| (docid, entry.document_id()))
            .collect::<Vec<_>>(),
        vec![(40, "a"), (42, "β")]
    );

    let max_input = [Some(IdMapEntryInput::new("max", 9))];
    let max =
        EncodedIdMapSection::encode(u64::from(u32::MAX), u64::from(u32::MAX) + 1, &max_input)?;
    assert_eq!(
        max.section()?
            .get(u64::from(u32::MAX))
            .map(IdMapEntry::document_id),
        Some("max")
    );

    let edge_holes = [None, Some(IdMapEntryInput::new("middle", 7)), None];
    let encoded_edge_holes = EncodedIdMapSection::encode(50, 53, &edge_holes)?;
    let edge_holes = encoded_edge_holes.section()?;
    assert_eq!(edge_holes.get(50), None);
    assert_eq!(
        edge_holes.get(51).map(IdMapEntry::document_id),
        Some("middle")
    );
    assert_eq!(edge_holes.get(52), None);

    let empty_map = EncodedIdMapSection::encode(0, 0, &[])?;
    assert_eq!(empty_map.as_bytes(), [0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0]);
    let empty_map = empty_map.section()?;
    let empty_hash = EncodedIdHashSection::encode(empty_map)?;
    assert_eq!(empty_hash.capacity(), 1);
    assert_eq!(empty_hash.occupied(), 0);
    assert_eq!(
        empty_hash.as_bytes(),
        [0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(empty_hash.section(empty_map)?.lookup("anything"), None);

    let inline_24 = "x".repeat(24);
    let spilled_25 = "y".repeat(25);
    let boundary_inputs = [
        Some(IdMapEntryInput::new(&inline_24, 1)),
        Some(IdMapEntryInput::new(&spilled_25, 2)),
    ];
    let encoded_boundary_map = EncodedIdMapSection::encode(60, 62, &boundary_inputs)?;
    let boundary_map = encoded_boundary_map.section()?;
    assert_eq!(
        boundary_map.materialize(60).as_deref(),
        Some(inline_24.as_str())
    );
    assert_eq!(
        boundary_map.materialize(61).as_deref(),
        Some(spilled_25.as_str())
    );
    Ok(())
}

#[test]
fn id_map_rejects_noncanonical_bytes_and_limits() -> TestResult {
    let max_span = usize::try_from(ID_MAP_MAX_SPAN)?;
    let max_prefix = u64::try_from(id_map_prefix_len(max_span)?)?;
    assert!(u32::try_from(max_prefix).is_ok());
    let next_span = ID_MAP_MAX_SPAN + 1;
    let next_prefix = u64::try_from(ID_MAP_HEADER_LEN)?
        + (next_span + 1) * u64::try_from(ID_MAP_OFFSET_LEN)?
        + next_span * u64::try_from(ID_MAP_HASH_LEN)?;
    assert!(u32::try_from(next_prefix).is_err());
    assert_eq!(
        checked_id_map_span(0, ID_MAP_MAX_SPAN, IdMapLimits::default())?,
        max_span
    );
    assert!(matches!(
        checked_id_map_span(
            0,
            ID_MAP_MAX_SPAN + 1,
            IdMapLimits {
                max_docid_span: ID_MAP_MAX_SPAN + 1,
                ..IdMapLimits::default()
            }
        ),
        Err(IdMapCodecError::ResourceLimit {
            resource: "durable IDMAP span",
            actual,
            limit: ID_MAP_MAX_SPAN,
        }) if actual == ID_MAP_MAX_SPAN + 1
    ));
    let inputs = [
        Some(IdMapEntryInput::new("a", 1)),
        None,
        Some(IdMapEntryInput::new("longer-than-inline-compact-id", 2)),
    ];
    assert!(matches!(
        EncodedIdMapSection::encode(0, 1, &[Some(IdMapEntryInput::new("", 1))]),
        Err(IdMapCodecError::EmptyDocumentId { ordinal: 0 })
    ));
    assert!(matches!(
        EncodedIdMapSection::encode(2, 1, &[]),
        Err(IdMapCodecError::InvalidDocIdRange { .. })
    ));
    assert!(matches!(
        EncodedIdMapSection::encode(u64::from(u32::MAX) + 1, u64::from(u32::MAX) + 2, &[]),
        Err(IdMapCodecError::InvalidDocIdRange { .. })
    ));
    assert!(matches!(
        EncodedIdMapSection::encode(0, 3, &inputs[..2]),
        Err(IdMapCodecError::EntryCountMismatch { .. })
    ));
    assert!(matches!(
        EncodedIdMapSection::encode_with_limits(
            0,
            3,
            &inputs,
            IdMapLimits {
                max_docid_span: 3,
                max_section_bytes: 1_000,
                max_blob_bytes: 1_000,
                max_document_id_bytes: 24,
            }
        ),
        Err(IdMapCodecError::ResourceLimit {
            resource: "document id bytes",
            ..
        })
    ));
    assert!(matches!(
        EncodedIdMapSection::encode_with_limits(
            0,
            3,
            &inputs,
            IdMapLimits {
                max_docid_span: 3,
                max_section_bytes: 1_000,
                max_blob_bytes: 4,
                max_document_id_bytes: 1_000,
            }
        ),
        Err(IdMapCodecError::ResourceLimit {
            resource: "identifier blob bytes",
            ..
        })
    ));
    assert!(matches!(
        EncodedIdMapSection::encode_with_limits(
            0,
            3,
            &inputs,
            IdMapLimits {
                max_docid_span: 3,
                max_section_bytes: 32,
                max_blob_bytes: 1_000,
                max_document_id_bytes: 1_000,
            }
        ),
        Err(IdMapCodecError::ResourceLimit {
            resource: "section bytes",
            ..
        })
    ));

    let encoded = EncodedIdMapSection::encode(0, 3, &inputs)?;
    for cut in 0..encoded.as_bytes().len() {
        assert!(
            IdMapSection::parse(&encoded.as_bytes()[..cut], 0, 3).is_err(),
            "cut={cut}"
        );
    }
    let mut count = encoded.as_bytes().to_vec();
    count[..4].copy_from_slice(&2_u32.to_le_bytes());
    assert!(matches!(
        IdMapSection::parse(&count, 0, 3),
        Err(IdMapCodecError::EntryCountMismatch { .. })
    ));
    let mut blob_offset = encoded.as_bytes().to_vec();
    blob_offset[4..8].copy_from_slice(&49_u32.to_le_bytes());
    assert!(matches!(
        IdMapSection::parse(&blob_offset, 0, 3),
        Err(IdMapCodecError::NonCanonicalBlobOffset { .. })
    ));
    let mut first = encoded.as_bytes().to_vec();
    first[8..12].copy_from_slice(&1_u32.to_le_bytes());
    assert!(matches!(
        IdMapSection::parse(&first, 0, 3),
        Err(IdMapCodecError::NonZeroFirstOffset { .. })
    ));
    let mut descending = encoded.as_bytes().to_vec();
    descending[16..20].copy_from_slice(&0_u32.to_le_bytes());
    assert!(matches!(
        IdMapSection::parse(&descending, 0, 3),
        Err(IdMapCodecError::DescendingOffsets { .. })
    ));
    let mut out_of_bounds = encoded.as_bytes().to_vec();
    out_of_bounds[20..24].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(matches!(
        IdMapSection::parse(&out_of_bounds, 0, 3),
        Err(IdMapCodecError::OffsetOutOfBounds { .. })
    ));
    let mut hole_hash = encoded.as_bytes().to_vec();
    hole_hash[32..40].copy_from_slice(&1_u64.to_le_bytes());
    assert!(matches!(
        IdMapSection::parse(&hole_hash, 0, 3),
        Err(IdMapCodecError::NonZeroHoleHash { ordinal: 1, .. })
    ));
    let mut invalid_utf8 = encoded.as_bytes().to_vec();
    let blob_start = usize::try_from(read_u32(&invalid_utf8, 4).ok_or("blob offset")?)?;
    invalid_utf8[blob_start] = 0xff;
    assert!(matches!(
        IdMapSection::parse(&invalid_utf8, 0, 3),
        Err(IdMapCodecError::InvalidUtf8 { ordinal: 0 })
    ));
    let split_inputs = [
        Some(IdMapEntryInput::new("aβ", 1)),
        Some(IdMapEntryInput::new("z", 2)),
    ];
    let mut split_scalar = EncodedIdMapSection::encode(0, 2, &split_inputs)?
        .as_bytes()
        .to_vec();
    split_scalar[12..16].copy_from_slice(&2_u32.to_le_bytes());
    assert!(matches!(
        IdMapSection::parse(&split_scalar, 0, 2),
        Err(IdMapCodecError::InvalidUtf8 { ordinal: 0 })
    ));
    let mut malformed_before_utf8 = invalid_utf8.clone();
    malformed_before_utf8[16..20].copy_from_slice(&0_u32.to_le_bytes());
    assert!(matches!(
        IdMapSection::parse(&malformed_before_utf8, 0, 3),
        Err(IdMapCodecError::DescendingOffsets { .. })
    ));
    let mut trailing = encoded.as_bytes().to_vec();
    trailing.push(0xff);
    assert!(matches!(
        IdMapSection::parse(&trailing, 0, 3),
        Err(IdMapCodecError::TrailingBytes { .. })
    ));
    assert!(matches!(
        IdMapSection::parse_with_limits(
            encoded.as_bytes(),
            0,
            3,
            IdMapLimits {
                max_docid_span: 2,
                ..IdMapLimits::default()
            }
        ),
        Err(IdMapCodecError::ResourceLimit {
            resource: "docid span",
            ..
        })
    ));
    assert!(matches!(
        IdMapSection::parse_with_limits(
            encoded.as_bytes(),
            0,
            3,
            IdMapLimits {
                max_blob_bytes: 1,
                ..IdMapLimits::default()
            }
        ),
        Err(IdMapCodecError::ResourceLimit {
            resource: "identifier blob bytes",
            ..
        })
    ));
    Ok(())
}

#[test]
fn id_map_concat_inserts_gaps_and_matches_monolithic_bytes() -> TestResult {
    let left_inputs = [Some(IdMapEntryInput::new("a", 1)), None];
    let middle_inputs = [Some(IdMapEntryInput::new("β", 2))];
    let right_inputs = [
        Some(IdMapEntryInput::new("longer-than-inline-compact-id", 3)),
        Some(IdMapEntryInput::new("z", 0)),
    ];
    let left = EncodedIdMapSection::encode(10, 12, &left_inputs)?;
    let middle = EncodedIdMapSection::encode(14, 15, &middle_inputs)?;
    let right = EncodedIdMapSection::encode(16, 18, &right_inputs)?;
    let sections = [left.section()?, middle.section()?, right.section()?];
    let merged = EncodedIdMapSection::concatenate(&sections)?;
    let monolithic_inputs = [
        Some(IdMapEntryInput::new("a", 1)),
        None,
        None,
        None,
        Some(IdMapEntryInput::new("β", 2)),
        None,
        Some(IdMapEntryInput::new("longer-than-inline-compact-id", 3)),
        Some(IdMapEntryInput::new("z", 0)),
    ];
    let monolithic = EncodedIdMapSection::encode(10, 18, &monolithic_inputs)?;
    assert_eq!(merged.as_bytes(), monolithic.as_bytes());
    assert_eq!(
        merged.section()?.blob(),
        b"a\xce\xb2longer-than-inline-compact-idz"
    );
    assert_eq!(merged.present_count(), 4);

    let left_associated = EncodedIdMapSection::concatenate(&sections[..2])?;
    let left_parts = [left_associated.section()?, right.section()?];
    let left_associated = EncodedIdMapSection::concatenate(&left_parts)?;
    assert_eq!(left_associated.as_bytes(), merged.as_bytes());
    let right_associated = EncodedIdMapSection::concatenate(&sections[1..])?;
    let right_parts = [left.section()?, right_associated.section()?];
    let right_associated = EncodedIdMapSection::concatenate(&right_parts)?;
    assert_eq!(right_associated.as_bytes(), merged.as_bytes());
    assert!(matches!(
        EncodedIdMapSection::concatenate(&[]),
        Err(IdMapCodecError::EmptyConcat)
    ));
    let reversed = [right.section()?, left.section()?];
    assert!(matches!(
        EncodedIdMapSection::concatenate(&reversed),
        Err(IdMapCodecError::ConcatRangeOrder { .. })
    ));
    assert!(matches!(
        EncodedIdMapSection::concatenate_with_limits(
            &sections,
            IdMapLimits {
                max_section_bytes: 32,
                ..IdMapLimits::default()
            }
        ),
        Err(IdMapCodecError::ResourceLimit {
            resource: "section bytes",
            ..
        })
    ));
    Ok(())
}

#[test]
fn resolved_id_hash_concat_domain_matches_monolithic_id_map_bytes() -> TestResult {
    let left_inputs = [Some(IdMapEntryInput::new("same", 1)), None];
    let middle_inputs = [Some(IdMapEntryInput::new("β", 2))];
    let right_inputs = [
        Some(IdMapEntryInput::new("same", 3)),
        Some(IdMapEntryInput::new("z", 4)),
    ];
    let left = EncodedIdMapSection::encode(10, 12, &left_inputs)?;
    let middle = EncodedIdMapSection::encode(14, 15, &middle_inputs)?;
    let right = EncodedIdMapSection::encode(16, 18, &right_inputs)?;
    let id_maps = [left.section()?, middle.section()?, right.section()?];
    let concat_plan = EncodedIdMapSection::plan_concatenate(&id_maps)?;

    let monolithic_inputs = [
        Some(IdMapEntryInput::new("same", 1)),
        None,
        None,
        None,
        Some(IdMapEntryInput::new("β", 2)),
        None,
        Some(IdMapEntryInput::new("same", 3)),
        Some(IdMapEntryInput::new("z", 4)),
    ];
    let monolithic = EncodedIdMapSection::encode(10, 18, &monolithic_inputs)?;
    let representative_ordinals = [4, 6, 7];
    let expected = EncodedIdHashSection::encode_resolved_representatives(
        monolithic.section()?,
        &representative_ordinals,
    )?;
    let actual = EncodedIdHashSection::encode_resolved_concat(
        &id_maps,
        &concat_plan,
        &representative_ordinals,
    )?;

    assert_eq!(actual.as_bytes(), expected.as_bytes());
    assert_eq!(actual.capacity(), expected.capacity());
    assert_eq!(actual.occupied(), expected.occupied());
    actual.section(monolithic.section()?)?;
    Ok(())
}

#[test]
fn id_hash_lookup_plan_presence_is_admission_derived() -> TestResult {
    let dense_map = EncodedIdMapSection::encode(
        10,
        12,
        &[
            Some(IdMapEntryInput::new("dense-a", 1)),
            Some(IdMapEntryInput::new("dense-b", 2)),
        ],
    )?;
    let dense_hash = EncodedIdHashSection::encode(dense_map.section()?)?;
    let dense = IdHashSection::parse(dense_hash.as_bytes(), dense_map.section()?)?.lookup_plan();
    assert_eq!(
        dense.identity_row_presence(9),
        IdMapPresence::Absent,
        "below the admitted domain is absent"
    );
    assert_eq!(
        dense.identity_row_presence(10),
        IdMapPresence::Present,
        "admission proves every dense in-range row"
    );
    assert_eq!(dense.identity_row_presence(11), IdMapPresence::Present);
    assert_eq!(
        dense.identity_row_presence(12),
        IdMapPresence::Absent,
        "the exclusive upper bound remains absent"
    );

    let hole_map = EncodedIdMapSection::encode(
        20,
        23,
        &[
            Some(IdMapEntryInput::new("left", 3)),
            None,
            Some(IdMapEntryInput::new("right", 4)),
        ],
    )?;
    let hole_hash = EncodedIdHashSection::encode(hole_map.section()?)?;
    let holes = IdHashSection::parse(hole_hash.as_bytes(), hole_map.section()?)?.lookup_plan();
    assert_eq!(holes.identity_row_presence(19), IdMapPresence::Absent);
    assert_eq!(
        holes.identity_row_presence(20),
        IdMapPresence::RequiresIdMap,
        "hole-bearing plans must not infer even a present row"
    );
    assert_eq!(
        holes.identity_row_presence(21),
        IdMapPresence::RequiresIdMap
    );
    assert_eq!(
        holes.identity_row_presence(22),
        IdMapPresence::RequiresIdMap
    );
    assert!(
        holes.contains_global_docid(hole_map.as_bytes(), 20),
        "the unchanged byte-backed path accepts a present row"
    );
    assert!(
        !holes.contains_global_docid(hole_map.as_bytes(), 21),
        "the unchanged byte-backed path rejects an IDMAP hole"
    );
    let mut malformed_hole_bytes = hole_map.as_bytes().to_vec();
    let first_present_end_offset = ID_MAP_HEADER_LEN + ID_MAP_OFFSET_LEN;
    malformed_hole_bytes[first_present_end_offset..first_present_end_offset + ID_MAP_OFFSET_LEN]
        .copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(
        !holes.contains_global_docid(&malformed_hole_bytes, 20),
        "the hole fallback fails closed on a same-length poisoned offset"
    );

    let empty_map = EncodedIdMapSection::encode(0, 0, &[])?;
    let empty_hash = EncodedIdHashSection::encode(empty_map.section()?)?;
    let empty = IdHashSection::parse(empty_hash.as_bytes(), empty_map.section()?)?.lookup_plan();
    assert_eq!(empty.identity_row_presence(0), IdMapPresence::Absent);

    let boundary_map = EncodedIdMapSection::encode(
        u64::from(u32::MAX),
        u64::from(u32::MAX) + 1,
        &[Some(IdMapEntryInput::new("max", 5))],
    )?;
    let boundary_hash = EncodedIdHashSection::encode(boundary_map.section()?)?;
    let boundary =
        IdHashSection::parse(boundary_hash.as_bytes(), boundary_map.section()?)?.lookup_plan();
    assert_eq!(
        boundary.identity_row_presence(u32::MAX - 1),
        IdMapPresence::Absent
    );
    assert_eq!(
        boundary.identity_row_presence(u32::MAX),
        IdMapPresence::Present
    );

    let mut corrupt = dense_map.as_bytes().to_vec();
    if corrupt.pop().is_none() {
        return Err("encoded IDMAP fixture is nonempty".into());
    }
    assert!(
        IdMapSection::parse(&corrupt, 10, 12).is_err(),
        "corrupt IDMAP bytes are rejected before a lookup plan can be admitted"
    );
    Ok(())
}

#[test]
fn id_hash_roundtrip_reopen_collisions_and_duplicate_rejection() -> TestResult {
    let inputs = [
        Some(IdMapEntryInput::new("alpha", 1)),
        None,
        Some(IdMapEntryInput::new("omega", 2)),
    ];
    let encoded_map = EncodedIdMapSection::encode(10, 13, &inputs)?;
    let id_map = encoded_map.section()?;
    let encoded_hash = EncodedIdHashSection::encode(id_map)?;
    assert_eq!(encoded_hash.capacity(), 4);
    assert_eq!(encoded_hash.occupied(), 2);
    let reopened_map = IdMapSection::parse(encoded_map.as_bytes(), 10, 13)?;
    let hash = IdHashSection::parse(encoded_hash.as_bytes(), reopened_map)?;
    assert_eq!(hash.lookup("alpha"), Some(10));
    assert_eq!(hash.lookup("omega"), Some(12));
    assert_eq!(hash.lookup("missing"), None);
    assert_eq!(
        hash.lookup_materialized("omega")
            .map(|(docid, id)| (docid, id.to_string())),
        Some((12, "omega".to_owned()))
    );
    let lookup = hash.lookup_plan();
    assert_eq!(
        lookup.lookup_with_content_hash(encoded_map.as_bytes(), encoded_hash.as_bytes(), "alpha"),
        Some((10, 1))
    );
    assert_eq!(
        lookup.lookup_with_content_hash(encoded_map.as_bytes(), encoded_hash.as_bytes(), "omega"),
        Some((12, 2))
    );
    assert_eq!(
        lookup.lookup_with_content_hash(encoded_map.as_bytes(), encoded_hash.as_bytes(), "missing"),
        None
    );

    let collision_inputs = [
        Some(IdMapEntryInput::new("a", 1)),
        Some(IdMapEntryInput::new("b", 2)),
        Some(IdMapEntryInput::new("c", 3)),
    ];
    let collision_map = EncodedIdMapSection::encode(20, 23, &collision_inputs)?;
    let collision_map = collision_map.section()?;
    let collision = encode_id_hash_with_hasher(collision_map, IdHashLimits::default(), |_| 3)?;
    let collision = parse_id_hash_with_hasher(
        collision.as_bytes(),
        collision_map,
        IdHashLimits::default(),
        |_| 3,
    )?;
    assert_eq!(collision.lookup_with_hash("a", 3), Some(20));
    assert_eq!(collision.lookup_with_hash("b", 3), Some(21));
    assert_eq!(collision.lookup_with_hash("c", 3), Some(22));
    assert_eq!(collision.lookup_with_hash("d", 3), None);

    let wrap = encode_id_hash_with_hasher(collision_map, IdHashLimits::default(), |_| u64::MAX)?;
    let wrap = parse_id_hash_with_hasher(
        wrap.as_bytes(),
        collision_map,
        IdHashLimits::default(),
        |_| u64::MAX,
    )?;
    assert_eq!(wrap.lookup_with_hash("a", u64::MAX), Some(20));
    assert_eq!(wrap.lookup_with_hash("c", u64::MAX), Some(22));

    let zero = encode_id_hash_with_hasher(collision_map, IdHashLimits::default(), |_| 0)?;
    let zero = parse_id_hash_with_hasher(
        zero.as_bytes(),
        collision_map,
        IdHashLimits::default(),
        |_| 0,
    )?;
    assert_eq!(zero.lookup_with_hash("b", 0), Some(21));

    let duplicate_inputs = [
        Some(IdMapEntryInput::new("same", 1)),
        Some(IdMapEntryInput::new("other", 2)),
        Some(IdMapEntryInput::new("same", 3)),
    ];
    let duplicate_map = EncodedIdMapSection::encode(30, 33, &duplicate_inputs)?;
    let duplicate_map = duplicate_map.section()?;
    assert!(matches!(
        EncodedIdHashSection::encode(duplicate_map),
        Err(IdHashCodecError::DuplicateDocumentId {
            first_ordinal: 0,
            duplicate_ordinal: 2,
        })
    ));

    let max_inputs = [Some(IdMapEntryInput::new("max", 1))];
    let max_map =
        EncodedIdMapSection::encode(u64::from(u32::MAX), u64::from(u32::MAX) + 1, &max_inputs)?;
    let max_map = max_map.section()?;
    let max_hash = EncodedIdHashSection::encode(max_map)?;
    assert_eq!(
        max_hash.section(max_map)?.lookup("max"),
        Some(u64::from(u32::MAX))
    );
    Ok(())
}

#[test]
fn randomized_id_map_and_hash_roundtrip_holes_lengths_and_collisions() -> TestResult {
    const BASE_SEED: u64 = 0x4f1b_bcdd_6a7e_2913;
    const BOUNDARY_SPANS: [usize; 8] = [0, 1, 2, 23, 24, 25, 127, 128];

    init_replay_tracing();
    for case in 0_u64..32 {
        let seed = BASE_SEED ^ case.wrapping_mul(0xd6e8_feb8_6659_fd93);
        tracing::info!(
            codec = "IDMAP_IDHASH",
            seed,
            case,
            replay = "randomized_id_map_and_hash_roundtrip_holes_lengths_and_collisions",
            "starting randomized roundtrip"
        );
        let mut state = seed;
        let span = if case < u64::try_from(BOUNDARY_SPANS.len())? {
            BOUNDARY_SPANS[usize::try_from(case)?]
        } else {
            usize::try_from(random_u32(&mut state) % 160)?
        };
        let docid_lo = u64::from(random_u32(&mut state) % 1_000_000);
        let docid_hi = docid_lo
            .checked_add(u64::try_from(span)?)
            .ok_or("randomized IDMAP docid range overflow")?;

        let mut document_ids = Vec::with_capacity(span);
        for ordinal in 0..span {
            if random_u32(&mut state).is_multiple_of(5) {
                document_ids.push(None);
                continue;
            }
            let mut document_id = format!("c{case:02x}-d{ordinal:03x}-");
            let requested_len = match ordinal % 7 {
                0 => 24,
                1 => 25,
                2 => 256,
                _ => usize::try_from(8 + random_u32(&mut state) % 120)?,
            };
            let target_len = requested_len.max(document_id.len());
            while document_id.len() < target_len {
                let letter = u8::try_from(random_u32(&mut state) % 26)? + b'a';
                document_id.push(char::from(letter));
            }
            document_ids.push(Some(document_id));
        }

        let inputs: Vec<_> = document_ids
            .iter()
            .enumerate()
            .map(|(ordinal, document_id)| {
                document_id.as_deref().map(|document_id| {
                    IdMapEntryInput::new(
                        document_id,
                        seed ^ u64::try_from(ordinal).unwrap_or(u64::MAX),
                    )
                })
            })
            .collect();
        let encoded_map = EncodedIdMapSection::encode(docid_lo, docid_hi, &inputs)?;
        assert_eq!(
            encoded_map,
            EncodedIdMapSection::encode(docid_lo, docid_hi, &inputs)?,
            "seed={seed:#x} case={case}: non-deterministic IDMAP bytes"
        );
        let id_map = encoded_map.section()?;
        let encoded_hash = EncodedIdHashSection::encode(id_map)?;
        let id_hash = encoded_hash.section(id_map)?;
        let collision_hash = seed.rotate_left(17);
        let encoded_collisions =
            encode_id_hash_with_hasher(id_map, IdHashLimits::default(), |_| collision_hash)?;
        let collisions = parse_id_hash_with_hasher(
            encoded_collisions.as_bytes(),
            id_map,
            IdHashLimits::default(),
            |_| collision_hash,
        )?;

        let mut present = 0_usize;
        for (ordinal, expected) in document_ids.iter().enumerate() {
            let docid = docid_lo + u64::try_from(ordinal)?;
            match expected {
                Some(document_id) => {
                    present += 1;
                    let expected_content_hash = seed ^ u64::try_from(ordinal)?;
                    assert_eq!(
                        id_map.get(docid).map(IdMapEntry::document_id),
                        Some(document_id.as_str()),
                        "seed={seed:#x} case={case} ordinal={ordinal}: IDMAP lookup"
                    );
                    assert_eq!(
                        id_map.get(docid).map(IdMapEntry::content_hash),
                        Some(expected_content_hash),
                        "seed={seed:#x} case={case} ordinal={ordinal}: IDMAP content hash"
                    );
                    assert_eq!(
                        id_hash.lookup(document_id),
                        Some(docid),
                        "seed={seed:#x} case={case} ordinal={ordinal}: IDHASH lookup"
                    );
                    assert_eq!(
                        collisions.lookup_with_hash(document_id, collision_hash),
                        Some(docid),
                        "seed={seed:#x} case={case} ordinal={ordinal}: collision lookup"
                    );
                }
                None => assert_eq!(
                    id_map.get(docid),
                    None,
                    "seed={seed:#x} case={case} ordinal={ordinal}: hole"
                ),
            }
        }
        assert_eq!(
            id_map.present_count(),
            present,
            "seed={seed:#x} case={case}: present count"
        );
    }
    Ok(())
}

#[test]
fn id_hash_resolved_representatives_require_external_choice_then_canonical_order() -> TestResult {
    let inputs = [
        Some(IdMapEntryInput::new("same", 1)),
        Some(IdMapEntryInput::new("other", 2)),
        None,
        Some(IdMapEntryInput::new("same", 3)),
    ];
    let encoded_map = EncodedIdMapSection::encode(30, 34, &inputs)?;
    let id_map = encoded_map.section()?;

    // The lower duplicate is the deterministic fallback when every `same`
    // row is tombstoned. The codec does not infer recency from docids.
    let lower = EncodedIdHashSection::encode_resolved_representatives(id_map, &[0, 1])?;
    let lower = lower.section(id_map)?;
    assert_eq!(lower.lookup("same"), Some(30));
    assert_eq!(lower.lookup("other"), Some(31));

    // Conversely, external tombstone state can identify the higher row as
    // the unique live representative even when the old row was republished
    // inside a container with a higher seal_seq.
    let higher = EncodedIdHashSection::encode_resolved_representatives(id_map, &[1, 3])?;
    let higher = higher.section(id_map)?;
    assert_eq!(higher.lookup("same"), Some(33));
    assert_eq!(higher.lookup("other"), Some(31));

    let collision =
        encode_resolved_id_hash_with_hasher(id_map, &[1, 3], IdHashLimits::default(), |_| {
            u64::MAX
        })?;
    let collision = parse_id_hash_with_hasher(
        collision.as_bytes(),
        id_map,
        IdHashLimits::default(),
        |_| u64::MAX,
    )?;
    assert_eq!(collision.lookup_with_hash("same", u64::MAX), Some(33));
    assert_eq!(collision.lookup_with_hash("other", u64::MAX), Some(31));
    assert!(matches!(
        encode_resolved_id_hash_with_hasher(
            id_map,
            &[1, 3],
            IdHashLimits {
                max_probe_steps: 4,
                ..IdHashLimits::default()
            },
            |_| 0,
        ),
        Err(IdHashCodecError::ResourceLimit {
            resource: "probe steps",
            ..
        })
    ));

    assert!(matches!(
        EncodedIdHashSection::encode_resolved_representatives(id_map, &[3, 1]),
        Err(IdHashCodecError::NonAscendingRepresentativeOrdinals {
            index: 1,
            previous: 3,
            current: 1,
        })
    ));
    assert!(matches!(
        EncodedIdHashSection::encode_resolved_representatives(id_map, &[1, 2]),
        Err(IdHashCodecError::RepresentativeOrdinalHole {
            index: 1,
            ordinal: 2,
        })
    ));
    assert!(matches!(
        EncodedIdHashSection::encode_resolved_representatives(id_map, &[1, 4]),
        Err(IdHashCodecError::RepresentativeOrdinalOutOfRange {
            index: 1,
            ordinal: 4,
        })
    ));
    assert!(matches!(
        EncodedIdHashSection::encode_resolved_representatives(id_map, &[0]),
        Err(IdHashCodecError::MissingMapping { ordinal: 1 })
    ));
    assert!(matches!(
        EncodedIdHashSection::encode_resolved_representatives(id_map, &[0, 1, 3]),
        Err(IdHashCodecError::DuplicateDocumentId {
            first_ordinal: 0,
            duplicate_ordinal: 3,
        })
    ));
    Ok(())
}

#[test]
fn id_hash_seed_wire_and_corruption_are_pinned() -> TestResult {
    // These constants are independent wire oracles for seeded xxh3-64.
    assert_eq!(seeded_id_hash(b"alpha"), 0x26fd_f19a_9d34_ddef);
    assert_eq!(seeded_id_hash(b"omega"), 0x2019_49b0_c7e7_214f);

    let inputs = [
        Some(IdMapEntryInput::new("alpha", 1)),
        None,
        Some(IdMapEntryInput::new("omega", 2)),
    ];
    let encoded_map = EncodedIdMapSection::encode(10, 13, &inputs)?;
    let id_map = encoded_map.section()?;
    let encoded = EncodedIdHashSection::encode(id_map)?;
    let expected = [
        4, 0, 0, 0, // capacity
        0x4f, 0x21, 0xe7, 0xc7, 0xb0, 0x49, 0x19, 0x20, 3, 0, 0, 0, // omega, ord 2
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // empty
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // empty
        0xef, 0xdd, 0x34, 0x9d, 0x9a, 0xf1, 0xfd, 0x26, 1, 0, 0, 0, // alpha, ord 0
    ];
    assert_eq!(encoded.as_bytes(), expected);
    let reparsed = IdHashSection::parse(encoded.as_bytes(), id_map)?;
    assert_eq!(reparsed.occupied(), 2);

    for cut in 0..encoded.as_bytes().len() {
        assert!(
            IdHashSection::parse(&encoded.as_bytes()[..cut], id_map).is_err(),
            "cut={cut}"
        );
    }
    let mut zero_capacity = encoded.as_bytes().to_vec();
    zero_capacity[..4].copy_from_slice(&0_u32.to_le_bytes());
    assert!(matches!(
        IdHashSection::parse(&zero_capacity, id_map),
        Err(IdHashCodecError::ZeroCapacity)
    ));
    let mut non_power = encoded.as_bytes().to_vec();
    non_power[..4].copy_from_slice(&3_u32.to_le_bytes());
    assert!(matches!(
        IdHashSection::parse(&non_power, id_map),
        Err(IdHashCodecError::NonPowerOfTwoCapacity { capacity: 3 })
    ));
    let mut trailing = encoded.as_bytes().to_vec();
    trailing.push(0);
    assert!(matches!(
        IdHashSection::parse(&trailing, id_map),
        Err(IdHashCodecError::TrailingBytes { .. })
    ));

    let empty_slot = (0..usize::try_from(encoded.capacity())?)
        .find(|&slot| id_hash_slot(&encoded.as_bytes()[4..], slot).is_some_and(|row| row.1 == 0))
        .ok_or("empty IDHASH slot")?;
    let mut nonzero_empty = encoded.as_bytes().to_vec();
    write_id_hash_slot(&mut nonzero_empty[4..], empty_slot, 1, 0)?;
    assert!(matches!(
        IdHashSection::parse(&nonzero_empty, id_map),
        Err(IdHashCodecError::NonZeroEmptyHash { .. })
    ));
    let occupied_slot = (0..usize::try_from(encoded.capacity())?)
        .find(|&slot| id_hash_slot(&encoded.as_bytes()[4..], slot).is_some_and(|row| row.1 != 0))
        .ok_or("occupied IDHASH slot")?;
    let (_, ordinal_plus1) =
        id_hash_slot(&encoded.as_bytes()[4..], occupied_slot).ok_or("occupied row")?;
    let mut wrong_hash = encoded.as_bytes().to_vec();
    write_id_hash_slot(&mut wrong_hash[4..], occupied_slot, 7, ordinal_plus1)?;
    assert!(matches!(
        IdHashSection::parse(&wrong_hash, id_map),
        Err(IdHashCodecError::KeyHashMismatch { .. })
    ));
    let mut hole = encoded.as_bytes().to_vec();
    write_id_hash_slot(&mut hole[4..], occupied_slot, seeded_id_hash(b"alpha"), 2)?;
    assert!(matches!(
        IdHashSection::parse(&hole, id_map),
        Err(IdHashCodecError::OrdinalHole { .. })
    ));

    let mut out_of_range = encoded.as_bytes().to_vec();
    write_id_hash_slot(
        &mut out_of_range[4..],
        occupied_slot,
        seeded_id_hash(b"alpha"),
        99,
    )?;
    assert!(matches!(
        IdHashSection::parse(&out_of_range, id_map),
        Err(IdHashCodecError::OrdinalOutOfRange { .. })
    ));

    let mut noncanonical_capacity = 8_u32.to_le_bytes().to_vec();
    noncanonical_capacity.extend_from_slice(&encoded.as_bytes()[4..]);
    noncanonical_capacity.resize(ID_HASH_HEADER_LEN + 8 * ID_HASH_ENTRY_LEN, 0);
    assert!(matches!(
        IdHashSection::parse(&noncanonical_capacity, id_map),
        Err(IdHashCodecError::NonCanonicalCapacity { .. })
    ));
    assert!(matches!(
        IdHashSection::parse_with_limits(
            encoded.as_bytes(),
            id_map,
            IdHashLimits {
                max_section_bytes: u64::try_from(encoded.as_bytes().len() - 1)?,
                ..IdHashLimits::default()
            }
        ),
        Err(IdHashCodecError::ResourceLimit {
            resource: "section bytes",
            ..
        })
    ));

    let mismatched_inputs = [
        Some(IdMapEntryInput::new("alpha", 1)),
        None,
        Some(IdMapEntryInput::new("different", 2)),
    ];
    let mismatched_map = EncodedIdMapSection::encode(10, 13, &mismatched_inputs)?;
    assert!(matches!(
        IdHashSection::parse(encoded.as_bytes(), mismatched_map.section()?),
        Err(IdHashCodecError::KeyHashMismatch { .. })
    ));
    Ok(())
}

#[test]
#[allow(clippy::cast_possible_truncation)]
fn id_hash_load_and_resource_limits_are_enforced() -> TestResult {
    let ids = ["a", "b", "c", "d", "e", "f"];
    let five_inputs = ids[..5]
        .iter()
        .enumerate()
        .map(|(index, id)| Some(IdMapEntryInput::new(id, index as u64)))
        .collect::<Vec<_>>();
    let five_map = EncodedIdMapSection::encode(0, 5, &five_inputs)?;
    let five_map = five_map.section()?;
    let five = encode_id_hash_with_hasher(five_map, IdHashLimits::default(), |bytes| {
        u64::from(bytes[0])
    })?;
    assert_eq!(five.capacity(), 8);
    parse_id_hash_with_hasher(
        five.as_bytes(),
        five_map,
        IdHashLimits::default(),
        |bytes| u64::from(bytes[0]),
    )?;

    let six_inputs = ids
        .iter()
        .enumerate()
        .map(|(index, id)| Some(IdMapEntryInput::new(id, index as u64)))
        .collect::<Vec<_>>();
    let six_map = EncodedIdMapSection::encode(0, 6, &six_inputs)?;
    let six_map = six_map.section()?;
    let (overloaded_entries, occupied) =
        build_id_hash_entries(six_map, 8, IdHashLimits::default(), |bytes| {
            u64::from(bytes[0])
        })?;
    assert_eq!(occupied, 6);
    let mut overloaded = 8_u32.to_le_bytes().to_vec();
    overloaded.extend_from_slice(&overloaded_entries);
    assert!(matches!(
        parse_id_hash_with_hasher(&overloaded, six_map, IdHashLimits::default(), |bytes| {
            u64::from(bytes[0])
        }),
        Err(IdHashCodecError::LoadFactorExceeded { .. })
    ));
    assert!(matches!(
        EncodedIdHashSection::encode_with_limits(
            six_map,
            IdHashLimits {
                max_entries: 5,
                ..IdHashLimits::default()
            }
        ),
        Err(IdHashCodecError::ResourceLimit {
            resource: "entry count",
            ..
        })
    ));
    assert!(matches!(
        EncodedIdHashSection::encode_with_limits(
            six_map,
            IdHashLimits {
                max_capacity: 8,
                ..IdHashLimits::default()
            }
        ),
        Err(IdHashCodecError::ResourceLimit {
            resource: "capacity",
            ..
        })
    ));
    assert!(matches!(
        encode_id_hash_with_hasher(
            five_map,
            IdHashLimits {
                max_probe_steps: 4,
                ..IdHashLimits::default()
            },
            |_| 0,
        ),
        Err(IdHashCodecError::ResourceLimit {
            resource: "probe steps",
            ..
        })
    ));
    let clustered = encode_id_hash_with_hasher(five_map, IdHashLimits::default(), |_| 0)?;
    assert!(matches!(
        parse_id_hash_with_hasher(
            clustered.as_bytes(),
            five_map,
            IdHashLimits {
                max_probe_steps: 4,
                ..IdHashLimits::default()
            },
            |_| 0,
        ),
        Err(IdHashCodecError::ResourceLimit {
            resource: "probe steps",
            ..
        })
    ));
    Ok(())
}

#[test]
fn id_hash_rejects_swapped_collision_slots_and_missing_rows_without_panic() -> TestResult {
    let inputs = [
        Some(IdMapEntryInput::new("a", 1)),
        Some(IdMapEntryInput::new("b", 2)),
    ];
    let encoded_map = EncodedIdMapSection::encode(0, 2, &inputs)?;
    let id_map = encoded_map.section()?;
    let collision = encode_id_hash_with_hasher(id_map, IdHashLimits::default(), |_| 0)?;
    let mut swapped = collision.as_bytes().to_vec();
    let first = swapped[4..16].to_vec();
    let second = swapped[16..28].to_vec();
    swapped[4..16].copy_from_slice(&second);
    swapped[16..28].copy_from_slice(&first);
    assert!(matches!(
        parse_id_hash_with_hasher(&swapped, id_map, IdHashLimits::default(), |_| 0),
        Err(IdHashCodecError::NonCanonicalSlot { .. })
    ));

    let mut missing = 2_u32.to_le_bytes().to_vec();
    missing.resize(ID_HASH_HEADER_LEN + 2 * ID_HASH_ENTRY_LEN, 0);
    write_id_hash_slot(&mut missing[4..], 1, u64::from(b'a'), 1)?;
    let parsed = std::panic::catch_unwind(|| {
        parse_id_hash_with_hasher(&missing, id_map, IdHashLimits::default(), |bytes| {
            u64::from(bytes[0])
        })
    });
    assert!(parsed.is_ok());
    assert!(matches!(
        parsed,
        Ok(Err(IdHashCodecError::MissingMapping { ordinal: 1 }))
    ));

    let single_input = [Some(IdMapEntryInput::new("a", 1))];
    let single_map = EncodedIdMapSection::encode(0, 1, &single_input)?;
    let single_map = single_map.section()?;
    let mut duplicate_ordinal = 4_u32.to_le_bytes().to_vec();
    duplicate_ordinal.resize(ID_HASH_HEADER_LEN + 4 * ID_HASH_ENTRY_LEN, 0);
    write_id_hash_slot(&mut duplicate_ordinal[4..], 0, 0, 1)?;
    write_id_hash_slot(&mut duplicate_ordinal[4..], 1, 0, 1)?;
    assert!(matches!(
        parse_id_hash_with_hasher(
            &duplicate_ordinal,
            single_map,
            IdHashLimits::default(),
            |_| 0
        ),
        Err(IdHashCodecError::SelectedCountMismatch {
            occupied: 2,
            selected: 1,
        })
    ));
    Ok(())
}

#[test]
fn arbitrary_id_map_and_hash_bytes_never_panic() -> TestResult {
    const SEED: u64 = 0x1d5a_91c3_e26f_5a17;

    init_replay_tracing();
    let inputs = [
        Some(IdMapEntryInput::new("alpha", 1)),
        None,
        Some(IdMapEntryInput::new("omega", 2)),
    ];
    let encoded_map = EncodedIdMapSection::encode(10, 13, &inputs)?;
    let id_map = encoded_map.section()?;
    let mut state = SEED;
    for case in 0..1_000 {
        tracing::info!(
            codec = "IDMAP_IDHASH",
            seed = SEED,
            case,
            replay = "arbitrary_id_map_and_hash_bytes_never_panic",
            "starting hostile-byte case"
        );
        let length = usize::try_from(random_u32(&mut state) % 257)?;
        let bytes = (0..length)
            .map(|_| random_u32(&mut state).to_le_bytes()[0])
            .collect::<Vec<_>>();
        let span = u64::from(random_u32(&mut state) % 16);
        let parsed_map = std::panic::catch_unwind(|| IdMapSection::parse(&bytes, 100, 100 + span));
        assert!(
            parsed_map.is_ok(),
            "seed={SEED:#x} IDMAP parser panic case={case} len={length}"
        );
        let parsed_hash = std::panic::catch_unwind(|| IdHashSection::parse(&bytes, id_map));
        assert!(
            parsed_hash.is_ok(),
            "seed={SEED:#x} IDHASH parser panic case={case} len={length}"
        );
        if let Ok(Ok(section)) = parsed_hash {
            let lookup = std::panic::catch_unwind(|| section.lookup("alpha"));
            assert!(
                lookup.is_ok(),
                "seed={SEED:#x} IDHASH lookup panic case={case}"
            );
        }
    }
    Ok(())
}

#[test]
#[allow(clippy::cast_possible_truncation)]
fn id_map_and_hash_roundtrip_sixty_five_thousand_mixed_ids() -> TestResult {
    const COUNT: usize = 65_536;
    let ids = (0..COUNT)
        .map(|index| {
            if index % 17 == 0 {
                format!("document-{index:08}-identifier-longer-than-inline")
            } else {
                format!("d{index:08}")
            }
        })
        .collect::<Vec<_>>();
    let entries = ids
        .iter()
        .enumerate()
        .map(|(index, id)| {
            Some(IdMapEntryInput::new(
                id,
                id_map_content_hash(&index.to_le_bytes()),
            ))
        })
        .collect::<Vec<_>>();
    let encoded_map = EncodedIdMapSection::encode(1_000, 1_000 + COUNT as u64, &entries)?;
    let id_map = encoded_map.section()?;
    let encoded_hash = EncodedIdHashSection::encode(id_map)?;
    let id_hash = encoded_hash.section(id_map)?;
    for &index in &[0, 1, 16, 17, COUNT / 2, COUNT - 1] {
        assert_eq!(
            id_hash.lookup(&ids[index]),
            Some(1_000 + u64::try_from(index)?)
        );
        assert_eq!(
            id_map.materialize(1_000 + u64::try_from(index)?).as_deref(),
            Some(ids[index].as_str())
        );
    }
    Ok(())
}

#[test]
#[ignore = "deterministic 1M-ID acceptance scale; run explicitly"]
#[allow(clippy::cast_possible_truncation)]
fn id_map_and_hash_roundtrip_one_million_ids() -> TestResult {
    const COUNT: usize = 1_000_000;
    let ids = (0..COUNT)
        .map(|index| {
            if index % 17 == 0 {
                format!("document-{index:08}-identifier-longer-than-inline")
            } else {
                format!("d{index:08}")
            }
        })
        .collect::<Vec<_>>();
    let entries = ids
        .iter()
        .enumerate()
        .map(|(index, id)| Some(IdMapEntryInput::new(id, index as u64)))
        .collect::<Vec<_>>();
    let encoded_map = EncodedIdMapSection::encode(0, COUNT as u64, &entries)?;
    let id_map = encoded_map.section()?;
    let encoded_hash = EncodedIdHashSection::encode(id_map)?;
    assert_eq!(encoded_hash.capacity(), 2_097_152);
    let id_hash = encoded_hash.section(id_map)?;
    for &index in &[0, 17, COUNT / 2, COUNT - 1] {
        assert_eq!(id_hash.lookup(&ids[index]), Some(index as u64));
    }
    Ok(())
}
