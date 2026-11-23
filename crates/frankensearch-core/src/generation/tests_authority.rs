use sha2::{Digest, Sha256};

use super::authority_core::{AUTHORITY_SLOT_BODY_BYTES, LOCK_FRAME_BODY_BYTES};
use super::codec::sha256_hex;
use super::tests_support::{
    activation_manifest, authority_lock, authority_reference, authority_slot, component_receipts,
    lock_frame,
};
use super::*;

#[test]
fn authority_slot_round_trips_and_authenticates_every_byte() {
    let authority = authority_reference(1, None);
    let slot = authority_slot(0, authority);
    let bytes = slot.encode().expect("encode slot");
    let decoded =
        AuthoritySlotV1::from_authenticated_bytes(&bytes, 0, [0x5a; 16]).expect("decode slot");
    assert_eq!(decoded, slot);

    let mut tampered = bytes;
    tampered[200] ^= 1;
    assert_eq!(
        AuthoritySlotV1::from_authenticated_bytes(&tampered, 0, [0x5a; 16]),
        Err(GenerationAuthorityErrorV1::ChecksumMismatch),
        "a byte outside the structured header must still be authenticated"
    );
}

#[test]
fn authority_slot_rejects_every_single_byte_mutation() {
    let encoded = authority_slot(0, authority_reference(1, None))
        .encode()
        .expect("encode slot");
    for byte_index in 0..GENERATION_AUTHORITY_SLOT_BYTES_V1 {
        let mut mutated = encoded;
        mutated[byte_index] ^= 0x80;
        assert!(
            AuthoritySlotV1::from_authenticated_bytes(&mutated, 0, [0x5a; 16]).is_err(),
            "single-byte mutation at offset {byte_index} must never decode"
        );
    }
}

#[test]
fn authority_slot_rejects_recomputed_noncanonical_padding_and_copied_slot() {
    let slot = authority_slot(0, authority_reference(1, None));
    let mut noncanonical = slot.encode().expect("encode slot");
    noncanonical[200] = 1;
    let digest = Sha256::digest(&noncanonical[..AUTHORITY_SLOT_BODY_BYTES]);
    noncanonical[AUTHORITY_SLOT_BODY_BYTES..].copy_from_slice(&digest);
    assert_eq!(
        AuthoritySlotV1::from_authenticated_bytes(&noncanonical, 0, [0x5a; 16]),
        Err(GenerationAuthorityErrorV1::NonCanonicalPadding),
        "a valid checksum does not permit noncanonical bytes"
    );
    assert_eq!(
        AuthoritySlotV1::from_authenticated_bytes(
            &slot.encode().expect("encode slot"),
            1,
            [0x5a; 16],
        ),
        Err(GenerationAuthorityErrorV1::SlotIndexMismatch),
        "a frame copied between physical slots must fail closed"
    );
}

#[test]
fn fixed_authority_frames_reject_resealed_future_schemas() {
    let slot = authority_slot(0, authority_reference(1, None));
    let mut future_slot = slot.encode().expect("encode authority slot");
    future_slot[8..10].copy_from_slice(&(GENERATION_AUTHORITY_SCHEMA_V1 + 1).to_be_bytes());
    let slot_digest = Sha256::digest(&future_slot[..AUTHORITY_SLOT_BODY_BYTES]);
    future_slot[AUTHORITY_SLOT_BODY_BYTES..].copy_from_slice(&slot_digest);
    assert_eq!(
        AuthoritySlotV1::from_authenticated_bytes(&future_slot, 0, [0x5a; 16]),
        Err(GenerationAuthorityErrorV1::InvalidField {
            field: "authority_slot.header"
        }),
        "a future AUTHORITY frame is rejected even with a valid checksum"
    );

    let mut future_lock = lock_frame(GenerationLockFrameKindV1::Owner)
        .encode()
        .expect("encode owner lock frame");
    future_lock[8..10].copy_from_slice(&(GENERATION_AUTHORITY_SCHEMA_V1 + 1).to_be_bytes());
    let lock_digest = Sha256::digest(&future_lock[..LOCK_FRAME_BODY_BYTES]);
    future_lock[LOCK_FRAME_BODY_BYTES..].copy_from_slice(&lock_digest);
    assert_eq!(
        GenerationLockFrameV1::from_authenticated_bytes(&future_lock, [0x71; 16]),
        Err(GenerationAuthorityErrorV1::InvalidField {
            field: "generation_lock.header"
        }),
        "a future LOCK frame is rejected even with a valid checksum"
    );
}

#[test]
fn lock_owner_and_attempt_frames_are_distinct_authenticated_contracts() {
    let owner = lock_frame(GenerationLockFrameKindV1::Owner);
    let owner_bytes = owner.encode().expect("encode owner lock frame");
    assert_eq!(
        GenerationLockFrameV1::from_authenticated_bytes(&owner_bytes, [0x71; 16])
            .expect("decode owner lock frame"),
        owner
    );

    let attempt = lock_frame(GenerationLockFrameKindV1::Attempt);
    assert_ne!(
        attempt.encode().expect("encode attempt lock frame"),
        owner_bytes,
        "an attempt must never serialize as an owner record"
    );

    let mut tampered = owner_bytes;
    tampered[300] ^= 1;
    assert_eq!(
        GenerationLockFrameV1::from_authenticated_bytes(&tampered, [0x71; 16]),
        Err(GenerationAuthorityErrorV1::ChecksumMismatch),
        "padding bytes remain authenticated"
    );
    assert_eq!(
        GenerationLockFrameV1::from_authenticated_bytes(&owner_bytes, [0x75; 16]),
        Err(GenerationAuthorityErrorV1::RootMismatch),
        "a lock frame copied across roots must fail closed"
    );
}

#[test]
fn lock_frame_rejects_every_single_byte_mutation() {
    let encoded = lock_frame(GenerationLockFrameKindV1::Owner)
        .encode()
        .expect("encode owner lock frame");
    for byte_index in 0..GENERATION_LOCK_FRAME_BYTES_V1 {
        let mut mutated = encoded;
        mutated[byte_index] ^= 0x80;
        assert!(
            GenerationLockFrameV1::from_authenticated_bytes(&mutated, [0x71; 16]).is_err(),
            "single-byte mutation at offset {byte_index} must never decode"
        );
    }
}

#[test]
fn lock_aware_resolver_requires_owner_agreement_and_attempt_reconciliation() {
    let genesis = authority_reference(1, None);
    let successor = authority_reference(2, Some(genesis.fingerprint()));
    let first = authority_slot(1, genesis);
    let second = authority_slot(0, successor);
    let owner = authority_lock(GenerationLockFrameKindV1::Owner, successor);
    assert_eq!(
        resolve_authority_slots_with_locks_v1(Some(first), Some(second), Some(owner), None),
        Ok(Some(second)),
        "a matching owner attests the resolved committed head"
    );

    let attempt = authority_lock(GenerationLockFrameKindV1::Attempt, successor);
    assert_eq!(
        resolve_authority_slots_with_locks_v1(
            Some(first),
            Some(second),
            Some(owner),
            Some(attempt),
        ),
        Err(GenerationAuthorityErrorV1::UnresolvedAttempt),
        "an authenticated attempt blocks fallback until publisher reconciliation"
    );

    let wrong_authority = GenerationLockFrameV1::new(
        GenerationLockFrameKindV1::Owner,
        [0x5a; 16],
        [0x72; 16],
        [0x73; 16],
        1,
        [0x91; 32],
    )
    .expect("well-formed mismatched owner lock");
    assert_eq!(
        resolve_authority_slots_with_locks_v1(
            Some(first),
            Some(second),
            Some(wrong_authority),
            None,
        ),
        Err(GenerationAuthorityErrorV1::LockAuthorityMismatch),
        "a valid lock for another authority must not select this head"
    );
}

#[test]
fn lock_aware_resolver_rejects_wrong_roles_and_cross_root_evidence() {
    let genesis = authority_reference(1, None);
    let successor = authority_reference(2, Some(genesis.fingerprint()));
    let first = authority_slot(1, genesis);
    let second = authority_slot(0, successor);
    let owner = authority_lock(GenerationLockFrameKindV1::Owner, successor);
    assert_eq!(
        resolve_authority_slots_with_locks_v1(Some(first), Some(second), None, Some(owner)),
        Err(GenerationAuthorityErrorV1::InvalidField {
            field: "generation_lock.attempt.kind"
        }),
        "owner frames cannot be reinterpreted as unresolved attempts"
    );

    let foreign_attempt = GenerationLockFrameV1::new(
        GenerationLockFrameKindV1::Attempt,
        [0x6b; 16],
        [0x72; 16],
        [0x73; 16],
        1,
        successor.fingerprint(),
    )
    .expect("well-formed foreign-root attempt");
    assert_eq!(
        resolve_authority_slots_with_locks_v1(
            Some(first),
            Some(second),
            None,
            Some(foreign_attempt),
        ),
        Err(GenerationAuthorityErrorV1::LockRootMismatch),
        "attempt evidence from another root cannot block or select this authority"
    );
}

#[test]
fn authority_resolver_requires_an_exact_consecutive_predecessor() {
    let first = authority_reference(1, None);
    let second = authority_reference(2, Some(first.fingerprint()));
    let resolved = resolve_authority_slots_v1(
        Some(authority_slot(1, first)),
        Some(authority_slot(0, second)),
    )
    .expect("linked authority slots resolve")
    .expect("at least one authority");
    assert_eq!(resolved.authority, second);

    let unlinked = authority_reference(2, Some([0x11; 32]));
    assert_eq!(
        resolve_authority_slots_v1(
            Some(authority_slot(1, first)),
            Some(authority_slot(0, unlinked)),
        ),
        Err(GenerationAuthorityErrorV1::BrokenPredecessorLink),
        "a newer authority without the exact predecessor fingerprint is not selectable"
    );
}

#[test]
fn authority_resolver_rejects_non_genesis_slot_parity_violation() {
    let genesis = authority_reference(1, None);
    let successor = authority_reference(2, Some(genesis.fingerprint()));
    assert_eq!(
        resolve_authority_slots_v1(
            Some(authority_slot(0, genesis)),
            Some(authority_slot(1, successor)),
        ),
        Err(GenerationAuthorityErrorV1::SlotSequenceParity),
        "a non-genesis authority copied into the opposite physical slot must fail"
    );
}

#[test]
fn authority_resolver_rejects_repeated_physical_slots_and_lone_parity_tears() {
    let genesis = authority_reference(1, None);
    let successor = authority_reference(2, Some(genesis.fingerprint()));
    assert_eq!(
        resolve_authority_slots_v1(
            Some(authority_slot(0, genesis)),
            Some(authority_slot(0, successor)),
        ),
        Err(GenerationAuthorityErrorV1::DuplicatePhysicalSlot),
        "two inputs claiming one physical slot cannot represent a recoverable pair"
    );
    assert_eq!(
        resolve_authority_slots_v1(Some(authority_slot(1, successor)), None),
        Err(GenerationAuthorityErrorV1::SlotSequenceParity),
        "one surviving non-genesis slot must still bind its publication parity"
    );
    assert_eq!(
        resolve_authority_slots_v1(
            Some(authority_slot(0, successor)),
            Some(authority_slot(1, genesis)),
        ),
        Ok(Some(authority_slot(0, successor))),
        "resolver input order never changes the selected consecutive head"
    );
}

#[test]
fn authority_resolver_accepts_consecutive_heads_across_slot_orientations() {
    let mut older = authority_reference(1, None);
    let mut older_slot_index = 1;
    for sequence in 2..=32 {
        let object_byte = u8::try_from(sequence).expect("bounded test sequence");
        let newer = AuthorityRefV1::new(
            sequence,
            [object_byte; 16],
            4_096 + sequence,
            [object_byte.wrapping_add(16); 32],
            Some(older.fingerprint()),
        )
        .expect("consecutive authority reference");
        let newer_slot_index = u8::try_from(sequence & 1).expect("slot parity fits u8");
        let older_slot = authority_slot(older_slot_index, older);
        let newer_slot = authority_slot(newer_slot_index, newer);
        assert_eq!(
            resolve_authority_slots_v1(Some(older_slot), Some(newer_slot)),
            Ok(Some(newer_slot)),
            "sequence {sequence} resolves in old/new input order"
        );
        assert_eq!(
            resolve_authority_slots_v1(Some(newer_slot), Some(older_slot)),
            Ok(Some(newer_slot)),
            "sequence {sequence} resolves in new/old input order"
        );
        older = newer;
        older_slot_index = newer_slot_index;
    }
}

#[test]
fn raw_authority_frame_resolver_preserves_corruption_as_an_error() {
    let genesis = authority_reference(1, None);
    let successor = authority_reference(2, Some(genesis.fingerprint()));
    let first = authority_slot(0, successor)
        .encode()
        .expect("encode successor frame");
    let second = authority_slot(1, genesis)
        .encode()
        .expect("encode genesis frame");
    assert_eq!(
        resolve_authority_slot_frames_v1(Some(&first), Some(&second), [0x5a; 16]),
        Ok(Some(authority_slot(0, successor))),
        "two valid raw physical frames resolve their linked head"
    );
    assert_eq!(
        resolve_authority_slot_frames_v1(Some(&first), None, [0x5a; 16]),
        Ok(Some(authority_slot(0, successor))),
        "only a genuinely absent second frame permits first-slot survival"
    );

    let mut corrupt_second = second;
    corrupt_second[200] ^= 1;
    assert_eq!(
        resolve_authority_slot_frames_v1(Some(&first), Some(&corrupt_second), [0x5a; 16]),
        Err(GenerationAuthorityErrorV1::ChecksumMismatch),
        "a present corrupt frame must not be silently treated as absent"
    );
}

#[test]
fn raw_authority_frame_resolver_rejects_length_and_order_tears() {
    let genesis = authority_reference(1, None);
    let successor = authority_reference(2, Some(genesis.fingerprint()));
    let first = authority_slot(0, successor)
        .encode()
        .expect("encode successor frame");
    let second = authority_slot(1, genesis)
        .encode()
        .expect("encode genesis frame");

    assert_eq!(
        resolve_authority_slot_frames_v1(Some(&first[..first.len() - 1]), None, [0x5a; 16]),
        Err(GenerationAuthorityErrorV1::InvalidSlotLength),
        "a truncated present frame is not one-slot survival"
    );
    let mut extended = first.to_vec();
    extended.push(0);
    assert_eq!(
        resolve_authority_slot_frames_v1(Some(&extended), None, [0x5a; 16]),
        Err(GenerationAuthorityErrorV1::InvalidSlotLength),
        "an extended present frame is not one-slot survival"
    );
    assert_eq!(
        resolve_authority_slot_frames_v1(Some(&second), Some(&first), [0x5a; 16]),
        Err(GenerationAuthorityErrorV1::SlotIndexMismatch),
        "reordered physical frames cannot swap old and new authorities"
    );
}

#[test]
fn authority_floor_resolver_rejects_stale_forked_and_gapped_heads() {
    let genesis = authority_reference(1, None);
    let floor = AuthorityFloorV1::new([0x5a; 16], genesis).expect("valid authority floor");
    assert_eq!(
        resolve_authority_slots_at_floor_v1(Some(authority_slot(0, genesis)), None, floor),
        Ok(authority_slot(0, genesis)),
        "the exact retained authority is selectable"
    );

    let successor = authority_reference(2, Some(genesis.fingerprint()));
    assert_eq!(
        resolve_authority_slots_at_floor_v1(Some(authority_slot(0, successor)), None, floor),
        Ok(authority_slot(0, successor)),
        "the immediate predecessor-linked successor is selectable"
    );

    let fork = AuthorityRefV1::new(1, [0x91; 16], 4_097, [0x92; 32], None)
        .expect("well-formed forked authority");
    assert_eq!(
        resolve_authority_slots_at_floor_v1(Some(authority_slot(1, fork)), None, floor),
        Err(GenerationAuthorityErrorV1::EqualSequenceFork),
        "equal floor sequence with another immutable object is a fork"
    );

    let gap = AuthorityRefV1::new(3, [0x93; 16], 4_099, [0x94; 32], Some([0x95; 32]))
        .expect("well-formed but unprovable authority gap");
    assert_eq!(
        resolve_authority_slots_at_floor_v1(Some(authority_slot(1, gap)), None, floor),
        Err(GenerationAuthorityErrorV1::BrokenPredecessorLink),
        "a head beyond the immediate anchored successor requires more evidence"
    );

    let mut foreign_root = authority_slot(0, successor);
    foreign_root.root_id = [0x96; 16];
    assert_eq!(
        resolve_authority_slots_at_floor_v1(Some(foreign_root), None, floor),
        Err(GenerationAuthorityErrorV1::RootMismatch),
        "an externally retained floor cannot be replayed across roots"
    );
}

#[test]
fn required_external_profile_never_silently_downgrades_to_local() {
    let genesis = authority_reference(1, None);
    let slot = authority_slot(0, genesis);
    assert_eq!(
        resolve_authority_slots_with_profile_v1(
            Some(slot),
            None,
            None,
            None,
            GenerationRootSecurityProfileV1::RequiredExternal,
            None,
        ),
        Err(GenerationAuthorityErrorV1::ExternalFloorRequired),
        "a valid local authority alone is insufficient for required-external admission"
    );

    let floor = AuthorityFloorV1::new([0x5a; 16], genesis).expect("valid external floor");
    assert_eq!(
        resolve_authority_slots_with_profile_v1(
            Some(slot),
            None,
            None,
            None,
            GenerationRootSecurityProfileV1::RequiredExternal,
            Some(floor),
        ),
        Ok(Some(slot)),
        "the exact retained external authority admits the root"
    );
}

#[test]
fn local_and_inspection_profiles_keep_lock_reconciliation_mandatory() {
    let genesis = authority_reference(1, None);
    let slot = authority_slot(0, genesis);
    let attempt = authority_lock(GenerationLockFrameKindV1::Attempt, genesis);
    for profile in [
        GenerationRootSecurityProfileV1::CooperativeLocal,
        GenerationRootSecurityProfileV1::ReadOnlyUnanchored,
    ] {
        assert_eq!(
            resolve_authority_slots_with_profile_v1(
                Some(slot),
                None,
                None,
                Some(attempt),
                profile,
                None,
            ),
            Err(GenerationAuthorityErrorV1::UnresolvedAttempt),
            "{profile:?} cannot bypass an unresolved publication attempt"
        );
    }
}

#[test]
fn root_security_profiles_explicitly_separate_mutation_authority() {
    assert!(GenerationRootSecurityProfileV1::RequiredExternal.permits_mutation());
    assert!(GenerationRootSecurityProfileV1::CooperativeLocal.permits_mutation());
    assert!(!GenerationRootSecurityProfileV1::ReadOnlyUnanchored.permits_mutation());
    assert_eq!(
        GenerationRootSecurityProfileV1::ReadOnlyUnanchored.require_mutation_authorized(),
        Err(GenerationAuthorityErrorV1::ReadOnlyProfile),
        "inspection-only admission cannot be reused for mutation"
    );
}

#[test]
fn authority_reference_fingerprint_uses_its_full_domain_separator() {
    let genesis = authority_reference(1, None);
    let bytes = genesis.canonical_bytes();
    assert_eq!(&bytes[..9], b"FSAUTHREF");
    assert_eq!(bytes.len(), 108);
    assert_ne!(genesis.fingerprint(), [0; 32]);
}

#[test]
fn authority_reference_codec_round_trips_and_rejects_noncanonical_forms() {
    let genesis = authority_reference(1, None);
    let successor = authority_reference(2, Some(genesis.fingerprint()));
    assert_eq!(
        AuthorityRefV1::from_canonical_bytes(&successor.canonical_bytes())
            .expect("decode successor authority reference"),
        successor
    );

    let mut noncanonical_genesis = genesis.canonical_bytes();
    noncanonical_genesis[76] = 1;
    assert_eq!(
        AuthorityRefV1::from_canonical_bytes(&noncanonical_genesis),
        Err(GenerationAuthorityErrorV1::NonCanonicalPadding),
        "absent predecessors have a single all-zero representation"
    );

    let mut future_schema = successor.canonical_bytes();
    future_schema[9..11].copy_from_slice(&(GENERATION_AUTHORITY_SCHEMA_V1 + 1).to_be_bytes());
    assert_eq!(
        AuthorityRefV1::from_canonical_bytes(&future_schema),
        Err(GenerationAuthorityErrorV1::InvalidField {
            field: "authority_ref.schema_version"
        }),
        "a self-consistent but unknown authority schema fails closed"
    );
    assert!(
        AuthorityRefV1::from_canonical_bytes(&successor.canonical_bytes()[..99]).is_err(),
        "truncated authority references never decode"
    );
}

#[test]
fn authority_reference_never_rolls_over_its_sequence() {
    assert_eq!(authority_reference(1, None).next_sequence(), Ok(2));
    let terminal = AuthorityRefV1::new(u64::MAX, [0x81; 16], 4_096, [0x82; 32], Some([0x83; 32]))
        .expect("terminal authority reference remains decodable");
    assert_eq!(
        terminal.next_sequence(),
        Err(GenerationAuthorityErrorV1::SequenceExhausted),
        "terminal authority state must not wrap into an earlier generation"
    );
}

#[test]
fn authority_resolver_accepts_only_duplicate_genesis() {
    let genesis = authority_reference(1, None);
    assert_eq!(
        resolve_authority_slots_v1(
            Some(authority_slot(0, genesis)),
            Some(authority_slot(1, genesis)),
        ),
        Ok(Some(authority_slot(0, genesis)))
    );

    let second = authority_reference(2, Some(genesis.fingerprint()));
    assert_eq!(
        resolve_authority_slots_v1(
            Some(authority_slot(0, second)),
            Some(authority_slot(1, second)),
        ),
        Err(GenerationAuthorityErrorV1::NonGenesisDuplicate)
    );
}

#[test]
fn authority_resolver_rejects_forks_gaps_and_mixed_roots_but_keeps_one_slot_survival() {
    let genesis = authority_reference(1, None);
    assert_eq!(
        resolve_authority_slots_v1(Some(authority_slot(0, genesis)), None),
        Ok(Some(authority_slot(0, genesis))),
        "one authenticated surviving slot remains selectable"
    );

    let fork = AuthorityRefV1::new(1, [0x61; 16], 4_097, [0x62; 32], None)
        .expect("valid conflicting genesis-shaped reference");
    assert_eq!(
        resolve_authority_slots_v1(
            Some(authority_slot(0, genesis)),
            Some(authority_slot(1, fork)),
        ),
        Err(GenerationAuthorityErrorV1::EqualSequenceFork),
        "equal sequence with distinct identities is a fork, never a tie-break"
    );

    let skipped = AuthorityRefV1::new(3, [0x63; 16], 4_099, [0x64; 32], Some([0x65; 32]))
        .expect("well-formed but nonconsecutive authority reference");
    assert_eq!(
        resolve_authority_slots_v1(
            Some(authority_slot(0, genesis)),
            Some(authority_slot(1, skipped)),
        ),
        Err(GenerationAuthorityErrorV1::BrokenPredecessorLink),
        "a structurally credible gap must not silently select the newest slot"
    );

    let mut other_root = authority_slot(1, genesis);
    other_root.root_id = [0x66; 16];
    assert_eq!(
        resolve_authority_slots_v1(Some(authority_slot(0, genesis)), Some(other_root)),
        Err(GenerationAuthorityErrorV1::InvalidField {
            field: "authority_slot.pair"
        }),
        "two slots from different roots cannot be combined"
    );
}

#[test]
fn activation_manifest_self_seals_all_component_and_transition_witnesses() {
    let manifest = activation_manifest(1, None);
    manifest.validate().expect("self-sealed manifest validates");
    let receipt = manifest.object_receipt();
    assert!(receipt.0 > 0);
    assert_ne!(receipt.1, [0; 32]);

    let mut substituted = manifest.clone();
    substituted.components.metadata.sha256[0] ^= 1;
    assert_eq!(
        substituted.validate(),
        Err(GenerationAuthorityErrorV1::ManifestSelfSealMismatch),
        "a component substitution must invalidate the activation-manifest self-seal"
    );
}

#[test]
fn activation_manifest_codec_round_trips_and_rejects_noncanonical_boundaries() {
    let manifest = activation_manifest(1, None);
    let bytes = manifest.canonical_bytes();
    assert_eq!(
        ActivationManifestV1::from_canonical_bytes(&bytes).expect("canonical decode"),
        manifest,
        "canonical decode must reproduce the exact structured manifest"
    );

    let mut extended = bytes.clone();
    extended.push(0);
    assert!(
        ActivationManifestV1::from_canonical_bytes(&extended).is_err(),
        "trailing bytes must not become a second representation"
    );
    assert!(
        ActivationManifestV1::from_canonical_bytes(&bytes[..bytes.len() - 1]).is_err(),
        "a truncated self-seal must fail before selecting the manifest"
    );
    let oversized = vec![0; GENERATION_ACTIVATION_MANIFEST_MAX_BYTES_V1 + 1];
    assert_eq!(
        ActivationManifestV1::from_canonical_bytes(&oversized),
        Err(GenerationAuthorityErrorV1::InvalidField {
            field: "activation_manifest.canonical_bytes"
        }),
        "the manifest ceiling is enforced before decode"
    );

    let mut future = manifest;
    future.schema_version = GENERATION_AUTHORITY_SCHEMA_V1 + 1;
    future.self_seal_sha256 = future.computed_self_seal();
    assert_eq!(
        ActivationManifestV1::from_canonical_bytes(&future.canonical_bytes()),
        Err(GenerationAuthorityErrorV1::InvalidField {
            field: "activation_manifest.schema_version"
        }),
        "a self-consistent future schema still fails closed"
    );
}

#[test]
fn activation_manifest_rejects_every_single_byte_mutation() {
    let encoded = activation_manifest(1, None).canonical_bytes();
    for byte_index in 0..encoded.len() {
        let mut mutated = encoded.clone();
        mutated[byte_index] ^= 0x80;
        assert!(
            ActivationManifestV1::from_canonical_bytes(&mutated).is_err(),
            "single-byte mutation at offset {byte_index} must never decode"
        );
    }
}

fn activation_manifest_v2(with_ann: bool) -> ActivationManifestV2 {
    let legacy = activation_manifest(1, None);
    ActivationManifestV2::new(
        legacy.authority_sequence,
        legacy.predecessor,
        legacy.action,
        legacy.generation,
        legacy.writer_fence_sha256,
        legacy.source_checkpoint_sha256,
        legacy.document_set_sha256,
        GenerationComponentReceiptsV2 {
            vector: legacy.components.vector,
            lexical: legacy.components.lexical,
            ann: with_ann.then_some(legacy.components.ann),
            metadata: legacy.components.metadata,
        },
    )
    .expect("schema-2 fixture")
}

#[test]
fn activation_manifest_versioned_codec_preserves_v1_and_exact_v2_authority() {
    let legacy = activation_manifest(1, None);
    let legacy_bytes = legacy.canonical_bytes();
    assert_eq!(legacy_bytes.len(), 434);
    assert_eq!(
        sha256_hex(&legacy_bytes),
        "21a7ce79d987cb20828889329624f016815dce363b7b26c836cc01c4c5f27d4e",
        "published schema-1 bytes must never be rewritten by the new codec"
    );
    let decoded = ActivationManifest::from_canonical_bytes(&legacy_bytes).expect("v1 decode");
    assert_eq!(decoded, ActivationManifest::V1(legacy));
    assert_eq!(decoded.canonical_bytes(), legacy_bytes);
    assert_eq!(decoded.components().iter().count(), 4);
    assert!(ActivationManifestV2::from_canonical_bytes(&legacy_bytes).is_err());

    for with_ann in [false, true] {
        let manifest = activation_manifest_v2(with_ann);
        let bytes = manifest.canonical_bytes();
        let decoded = ActivationManifest::from_canonical_bytes(&bytes).expect("v2 decode");
        assert_eq!(decoded, ActivationManifest::V2(manifest));
        assert_eq!(decoded.canonical_bytes(), bytes);
        assert_eq!(decoded.components().ann.is_some(), with_ann);
        assert_eq!(
            decoded.components().iter().count(),
            if with_ann { 4 } else { 3 }
        );
        assert!(ActivationManifestV1::from_canonical_bytes(&bytes).is_err());
        let (len, digest) = decoded.object_receipt();
        let authority = AuthorityRefV1::new(1, [0x41; 16], len, digest, None)
            .expect("unchanged authority schema");
        verify_authority_manifest_reference(&authority, &decoded).expect("exact binding");
        let other = ActivationManifest::V2(activation_manifest_v2(!with_ann));
        assert_eq!(
            verify_authority_manifest_reference(&authority, &other),
            Err(GenerationAuthorityErrorV1::ManifestReferenceMismatch),
            "present and absent ANN selections cannot share an authority"
        );
        for byte_index in 0..bytes.len() {
            let mut mutated = bytes.clone();
            mutated[byte_index] ^= 0x80;
            assert!(
                ActivationManifest::from_canonical_bytes(&mutated).is_err(),
                "schema-2 mutation at {byte_index}, ANN present={with_ann}"
            );
        }
    }
}

#[test]
fn activation_manifest_v2_rejects_resealed_presence_and_schema_ambiguity() {
    let reseal = |bytes: &mut Vec<u8>| {
        let unsealed_len = bytes.len() - 32;
        let seal: [u8; 32] = Sha256::digest(&bytes[..unsealed_len]).into();
        bytes[unsealed_len..].copy_from_slice(&seal);
    };
    // Each byte receipt is u64 length + length-prefixed 32-byte digest.
    const RECEIPT_BYTES: usize = 48;
    for with_ann in [false, true] {
        let manifest = activation_manifest_v2(with_ann);
        let bytes = manifest.canonical_bytes();
        let ann_offset =
            bytes.len() - 32 - RECEIPT_BYTES - if with_ann { 1 + RECEIPT_BYTES } else { 1 };
        for tag in [2, 255, u8::from(!with_ann)] {
            let mut changed = bytes.clone();
            changed[ann_offset] = tag;
            reseal(&mut changed);
            assert!(ActivationManifest::from_canonical_bytes(&changed).is_err());
        }
        let mut future = manifest.clone();
        future.schema_version = GENERATION_ACTIVATION_MANIFEST_SCHEMA_V2 + 1;
        future.self_seal_sha256 = Sha256::digest(future.canonical_unsealed_bytes()).into();
        assert!(ActivationManifest::from_canonical_bytes(&future.canonical_bytes()).is_err());
        let mut wrong_domain_version = manifest;
        wrong_domain_version.schema_version = GENERATION_AUTHORITY_SCHEMA_V1;
        wrong_domain_version.self_seal_sha256 =
            Sha256::digest(wrong_domain_version.canonical_unsealed_bytes()).into();
        assert!(
            ActivationManifest::from_canonical_bytes(&wrong_domain_version.canonical_bytes())
                .is_err()
        );
    }

    let present = activation_manifest_v2(true).canonical_bytes();
    let ann_start = present.len() - 32 - 2 * RECEIPT_BYTES;
    let mut absent_with_hidden_graph = activation_manifest_v2(false).canonical_bytes();
    let insertion = absent_with_hidden_graph.len() - 32 - RECEIPT_BYTES;
    absent_with_hidden_graph.splice(
        insertion..insertion,
        present[ann_start..ann_start + RECEIPT_BYTES]
            .iter()
            .copied(),
    );
    reseal(&mut absent_with_hidden_graph);
    assert!(
        ActivationManifest::from_canonical_bytes(&absent_with_hidden_graph).is_err(),
        "an absent tag cannot hide a supplied graph receipt before metadata"
    );
}

#[test]
fn activation_manifest_v2_validates_present_roles_and_transition_fields() {
    let mut manifest = activation_manifest_v2(true);
    manifest.components.ann.as_mut().expect("present").byte_len = 0;
    assert!(
        manifest.validate().is_err(),
        "invalid ANN cannot become absence"
    );
    for role in [
        GenerationComponentRole::Vector,
        GenerationComponentRole::Lexical,
        GenerationComponentRole::Metadata,
    ] {
        let mut manifest = activation_manifest_v2(false);
        match role {
            GenerationComponentRole::Vector => manifest.components.vector.sha256 = [0; 32],
            GenerationComponentRole::Lexical => manifest.components.lexical.sha256 = [0; 32],
            GenerationComponentRole::Metadata => manifest.components.metadata.sha256 = [0; 32],
            GenerationComponentRole::Ann => unreachable!("only mandatory roles"),
        }
        assert!(
            manifest.validate().is_err(),
            "mandatory {role:?} still validates without ANN"
        );
    }
    let genesis = activation_manifest_v2(false);
    let (len, digest) = genesis.object_receipt();
    let predecessor = AuthorityRefV1::new(1, [0x41; 16], len, digest, None).expect("predecessor");
    let mut broken = genesis.clone();
    broken.authority_sequence = 3;
    broken.predecessor = Some(predecessor);
    assert_eq!(
        broken.validate(),
        Err(GenerationAuthorityErrorV1::BrokenPredecessorLink)
    );
    let successor = ActivationManifestV2::new(
        2,
        Some(predecessor),
        GenerationAuthorityActionV1::Rollback,
        genesis.generation,
        genesis.writer_fence_sha256,
        genesis.source_checkpoint_sha256,
        genesis.document_set_sha256,
        genesis.components,
    )
    .expect("higher-sequence exact rollback");
    assert_eq!(successor.authority_sequence, 2);
    assert_eq!(successor.generation, genesis.generation);
    assert_eq!(
        ActivationManifestV2::from_canonical_bytes(&successor.canonical_bytes())
            .expect("successor"),
        successor
    );
    let oversized = vec![0; GENERATION_ACTIVATION_MANIFEST_MAX_BYTES_V2 + 1];
    assert!(ActivationManifest::from_canonical_bytes(&oversized).is_err());
}

#[test]
fn activation_manifest_requires_the_exact_preceding_authority_sequence() {
    let genesis = activation_manifest(1, None);
    let (manifest_len, manifest_sha256) = genesis.object_receipt();
    let predecessor = AuthorityRefV1::new(1, [0x41; 16], manifest_len, manifest_sha256, None)
        .expect("genesis authority reference");
    let successor = activation_manifest(2, Some(predecessor));
    successor
        .validate()
        .expect("consecutive predecessor validates");

    assert_eq!(
        ActivationManifestV1::new(
            3,
            Some(predecessor),
            GenerationAuthorityActionV1::Activate,
            ArtifactGenerationIdentityV1::new(7, [0x21; 16]).expect("test generation"),
            [0x31; 32],
            [0x32; 32],
            [0x33; 32],
            component_receipts(),
        ),
        Err(GenerationAuthorityErrorV1::BrokenPredecessorLink),
        "an activation manifest may not skip an authority sequence"
    );
}

#[test]
fn rollback_uses_a_new_authority_without_reusing_artifact_identity() {
    let genesis = activation_manifest(1, None);
    let (genesis_len, genesis_sha256) = genesis.object_receipt();
    let first = AuthorityRefV1::new(1, [0x41; 16], genesis_len, genesis_sha256, None)
        .expect("first authority reference");
    let second = AuthorityRefV1::new(
        2,
        [0x42; 16],
        genesis_len,
        genesis_sha256,
        Some(first.fingerprint()),
    )
    .expect("second authority reference");
    let rollback = ActivationManifestV1::new(
        3,
        Some(second),
        GenerationAuthorityActionV1::Rollback,
        ArtifactGenerationIdentityV1::new(7, [0x21; 16]).expect("reselected generation"),
        [0x31; 32],
        [0x32; 32],
        [0x33; 32],
        component_receipts(),
    )
    .expect("higher-authority rollback is canonical");
    assert_eq!(rollback.authority_sequence, 3);
    assert_eq!(rollback.generation.sequence, 7);
    assert_eq!(rollback.action, GenerationAuthorityActionV1::Rollback);
    rollback.validate().expect("rollback self-seal validates");
}

#[test]
fn authority_reference_requires_the_exact_self_sealed_manifest_object() {
    let manifest = activation_manifest(1, None);
    let (manifest_len, manifest_sha256) = manifest.object_receipt();
    let authority = AuthorityRefV1::new(1, [0x51; 16], manifest_len, manifest_sha256, None)
        .expect("authority names test manifest");
    verify_authority_manifest_reference_v1(&authority, &manifest)
        .expect("exact authority manifest binding");

    let wrong_length = AuthorityRefV1::new(1, [0x51; 16], manifest_len + 1, manifest_sha256, None)
        .expect("otherwise valid authority reference");
    assert_eq!(
        verify_authority_manifest_reference_v1(&wrong_length, &manifest),
        Err(GenerationAuthorityErrorV1::ManifestReferenceMismatch),
        "a length-mismatched external object must not be selected"
    );
}
