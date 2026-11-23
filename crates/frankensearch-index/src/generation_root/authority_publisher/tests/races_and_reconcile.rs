use super::super::{
    AttemptPermitV1, AttemptStageV1, ExpectedAuthorityPairV1, NotCommittedV1, PublicationOutcomeV1,
    ReconcileOutcomeV1, decode_pair,
};
use super::*;
use crate::generation_root::{
    GenerationRootError, GenerationRootErrorKind, GenerationRootStage, QualifiedGenerationRoot,
    platform,
};
use frankensearch_core::generation::{
    AuthorityRefV1, AuthoritySlotV1, GenerationAuthorityErrorV1, GenerationRootSecurityProfileV1,
    InMemoryAntiRollbackFloorStoreV1,
};
use std::fs::{self, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

/// Acceptance: fresh reconciliation returns `CommittedThenSuperseded`
/// (bd-ycng6).
///
/// Publisher A's genesis attempt goes unknown AFTER the slot write and
/// floor advance (owner frame never lands), so A holds a bounded
/// permit for a candidate that is in fact durable. Publisher B then
/// resolves the live head and publishes sequence 2, advancing the
/// floor past A's candidate. A's fresh reconcile must recognize its
/// own candidate in the slot, observe the floor beyond it, resolve
/// the pair, and report `CommittedThenSuperseded` carrying B's head —
/// never `StillUnknown`, never a fork, and never a mutation.
#[test]
fn unknown_attempt_reconciles_committed_then_superseded_after_a_successor() {
    use platform::TestBoundary as B;

    let root_path = fixture_root("superseded");
    let root = admit(&root_path);
    let floor = InMemoryAntiRollbackFloorStoreV1::new();
    let publisher_a = root
        .authority_publisher(
            ROOT_ID,
            WRITER_A,
            GenerationRootSecurityProfileV1::CooperativeLocal,
        )
        .expect("publisher A binds");
    let genesis = authority(1, None);
    let outcome = {
        let _hook = platform::install_test_hook(move |boundary| {
            if boundary == B::BeforePublishOwnerFrame {
                Err(GenerationRootError::new(
                    GenerationRootErrorKind::Io,
                    GenerationRootStage::SyncRegularFile,
                ))
            } else {
                Ok(())
            }
        });
        publisher_a
            .publish(
                genesis,
                ExpectedAuthorityPairV1::default(),
                Some(&floor),
                [0x51; 16],
            )
            .expect("interrupted genesis runs")
    };
    let PublicationOutcomeV1::CommitOutcomeUnknown(permit) = outcome else {
        panic!("owner-frame crash must be unknown: {outcome:?}");
    };
    assert_eq!(permit.stage, AttemptStageV1::OwnerFrame);

    // B resolves the durable genesis and publishes the successor.
    let root_b = admit(&root_path);
    let publisher_b = root_b
        .authority_publisher(
            ROOT_ID,
            WRITER_B,
            GenerationRootSecurityProfileV1::CooperativeLocal,
        )
        .expect("publisher B binds");
    let bytes = authority_bytes(&root_path);
    let pair = decode_pair(&bytes, ROOT_ID).expect("pair decodes");
    let genesis_slot = pair.second.expect("genesis slot is durable");
    assert_eq!(genesis_slot.authority, genesis);
    let successor = authority(2, Some(genesis.fingerprint()));
    let outcome_b = publisher_b
        .publish(successor, pair, Some(&floor), [0x52; 16])
        .expect("successor publication runs");
    let PublicationOutcomeV1::Committed { slot: head, .. } = outcome_b else {
        panic!("successor must commit: {outcome_b:?}");
    };

    // A's fresh reconcile: candidate durable, floor beyond it, head
    // resolved to B's successor.
    let before = authority_bytes(&root_path);
    let verdict = publisher_a
        .reconcile(&permit, Some(&floor))
        .expect("reconcile runs");
    let ReconcileOutcomeV1::CommittedThenSuperseded { head: observed } = verdict else {
        panic!("superseded candidate must reconcile as such: {verdict:?}");
    };
    assert_eq!(observed, head);
    assert_eq!(
        authority_bytes(&root_path),
        before,
        "reconciliation must not mutate the authority"
    );
}

/// Acceptance: two real publishers, one winner (bd-ycng6).
///
/// Two publisher handles over independently opened descriptors race
/// the SAME genesis sequence from two threads. The kernel flock
/// serializes them: exactly one commits; the other observes either
/// the contended lock or, after the winner, the no-longer-default
/// expected pair — both ordinary `NotCommitted` with ZERO authority
/// mutation from the loser. A retry with the stale default pair
/// stays refused, and the durable bytes are exactly the winner's.
#[test]
fn two_real_publishers_race_one_wins_and_the_loser_never_mutates() {
    let root_path = fixture_root("race");
    // Qualify both independent descriptors before starting the race.
    // A setup refusal must fail this test, not strand the other
    // publisher at a barrier that its failed peer can never reach.
    let root_a = admit(&root_path);
    let root_b = admit(&root_path);
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let barrier_a = std::sync::Arc::clone(&barrier);
    let barrier_b = std::sync::Arc::clone(&barrier);
    let run = |root: QualifiedGenerationRoot,
               writer: [u8; 16],
               key: u8,
               barrier: std::sync::Arc<std::sync::Barrier>| {
        move || {
            barrier.wait();
            let publisher = root
                .authority_publisher(
                    ROOT_ID,
                    writer,
                    GenerationRootSecurityProfileV1::CooperativeLocal,
                )
                .expect("racing publisher binds");
            publisher.publish(
                authority(1, None),
                ExpectedAuthorityPairV1::default(),
                None,
                [key; 16],
            )
        }
    };
    let thread_a = std::thread::spawn(run(root_a, WRITER_A, 0xA1, barrier_a));
    let thread_b = std::thread::spawn(run(root_b, WRITER_B, 0xB1, barrier_b));
    let outcome_a = thread_a
        .join()
        .expect("publisher A thread")
        .expect("publisher A runs");
    let outcome_b = thread_b
        .join()
        .expect("publisher B thread")
        .expect("publisher B runs");

    let (winner, loser) = match (&outcome_a, &outcome_b) {
        (PublicationOutcomeV1::Committed { slot, .. }, other) => (*slot, other),
        (other, PublicationOutcomeV1::Committed { slot, .. }) => (*slot, other),
        pair => panic!("exactly one publisher must commit: {pair:?}"),
    };
    assert!(
        matches!(loser, PublicationOutcomeV1::NotCommitted(_)),
        "the loser must be an ordinary refusal, never Unknown: {loser:?}"
    );

    // Durable bytes hold exactly the winner; sequence 1 is slot 1.
    let bytes = authority_bytes(&root_path);
    let pair = decode_pair(&bytes, ROOT_ID).expect("pair decodes");
    assert_eq!(
        pair,
        ExpectedAuthorityPairV1 {
            first: None,
            second: Some(winner)
        }
    );

    // A retry against the stale default pair is refused with zero
    // mutation: the bytes before and after are identical.
    let root = admit(&root_path);
    let retry = root
        .authority_publisher(
            ROOT_ID,
            WRITER_B,
            GenerationRootSecurityProfileV1::CooperativeLocal,
        )
        .expect("retry publisher binds");
    let before = authority_bytes(&root_path);
    let outcome = retry
        .publish(
            authority(1, None),
            ExpectedAuthorityPairV1::default(),
            None,
            [0xB2; 16],
        )
        .expect("stale retry runs");
    assert!(
        matches!(outcome, PublicationOutcomeV1::NotCommitted(_)),
        "a stale expected pair must refuse: {outcome:?}"
    );
    assert_eq!(authority_bytes(&root_path), before);
}

#[test]
fn many_publications_never_change_directory_entries_or_anchor_inodes() {
    use std::os::unix::fs::MetadataExt as _;

    // Each publication re-qualifies the full route and runs several
    // durability barriers (~2 s on the reference filesystem), so the
    // unit lane proves the invariant over 64 publications; the
    // thousands-scale soak the bead asks for belongs in a release-mode
    // lane, not here.
    const PUBLICATIONS: u64 = 64;
    let root_path = fixture_root("inodes");
    let root = admit(&root_path);
    let floor = InMemoryAntiRollbackFloorStoreV1::new();
    let publisher = root
        .authority_publisher(
            ROOT_ID,
            WRITER_A,
            GenerationRootSecurityProfileV1::CooperativeLocal,
        )
        .expect("publisher binds");
    let listing = |path: &Path| -> Vec<(String, u64)> {
        let mut entries: Vec<(String, u64)> = fs::read_dir(path)
            .expect("root readable")
            .map(|entry| {
                let entry = entry.expect("entry");
                (
                    entry.file_name().to_string_lossy().into_owned(),
                    entry.metadata().expect("metadata").ino(),
                )
            })
            .collect();
        entries.sort();
        entries
    };
    let before = listing(&root_path);
    assert_eq!(before.len(), 2, "exactly LOCK and AUTHORITY");

    let mut pair = ExpectedAuthorityPairV1::default();
    let mut previous: Option<AuthorityRefV1> = None;
    for sequence in 1..=PUBLICATIONS {
        let candidate = authority(sequence, previous.map(|p| p.fingerprint()));
        let outcome = publisher
            .publish(
                candidate,
                pair,
                Some(&floor),
                [u8::try_from(sequence % 251 + 1).expect("byte"); 16],
            )
            .expect("publication runs");
        let PublicationOutcomeV1::Committed { slot, .. } = outcome else {
            panic!("publication {sequence} must commit: {outcome:?}");
        };
        if slot.slot_index == 0 {
            pair.first = Some(slot);
        } else {
            pair.second = Some(slot);
        }
        previous = Some(candidate);
    }
    assert_eq!(listing(&root_path), before, "no dirent or inode changed");
    let head = floor.load(ROOT_ID).expect("floor").expect("advanced");
    assert_eq!(head.authority.sequence, PUBLICATIONS);
    assert_eq!(head.cas_version, PUBLICATIONS);
    // Both incumbents are retained: the pair holds the last two sequences.
    let bytes = authority_bytes(&root_path);
    let observed = decode_pair(&bytes, ROOT_ID).expect("pair decodes");
    let sequences: Vec<u64> = [observed.first, observed.second]
        .into_iter()
        .flatten()
        .map(|s| s.authority.sequence)
        .collect();
    assert_eq!(sequences.iter().max(), Some(&PUBLICATIONS));
    assert_eq!(sequences.iter().min(), Some(&(PUBLICATIONS - 1)));
}

#[test]
fn swapped_anchor_inodes_are_a_typed_refusal_with_zero_mutation() {
    use std::os::unix::fs::MetadataExt as _;

    for swapped in ["AUTHORITY", "LOCK"] {
        let root_path = fixture_root(&format!("swap-{}", swapped.to_lowercase()));
        let root = admit(&root_path);
        let publisher = root
            .authority_publisher(
                ROOT_ID,
                WRITER_A,
                GenerationRootSecurityProfileV1::CooperativeLocal,
            )
            .expect("publisher binds");

        // A same-length, same-mode replacement inode renamed over the
        // anchor: the classic path-substitution attack against a
        // retained descriptor set.
        let original_ino = fs::metadata(root_path.join(swapped)).expect("meta").ino();
        let replacement = root_path.join(format!("{swapped}.replacement"));
        {
            let len = fs::metadata(root_path.join(swapped)).expect("meta").len();
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o600)
                .open(&replacement)
                .expect("replacement creatable");
            file.write_all(&vec![0_u8; usize::try_from(len).expect("len")])
                .expect("replacement bytes");
            file.sync_all().expect("replacement durable");
        }
        fs::rename(&replacement, root_path.join(swapped)).expect("swap renamed");
        let swapped_ino = fs::metadata(root_path.join(swapped)).expect("meta").ino();
        assert_ne!(original_ino, swapped_ino);
        let before = fs::read(root_path.join(swapped)).expect("swapped readable");

        let outcome = publisher
            .publish(
                authority(1, None),
                ExpectedAuthorityPairV1::default(),
                None,
                [0x41; 16],
            )
            .expect("publication runs");
        assert_eq!(
            outcome,
            PublicationOutcomeV1::NotCommitted(NotCommittedV1::AnchorsChanged),
            "swapped {swapped}"
        );
        assert_eq!(
            fs::read(root_path.join(swapped)).expect("swapped readable"),
            before,
            "no byte reached the substituted {swapped}"
        );
        // The original inode (now unlinked from the tree) is untouched
        // too: the retained descriptor set refuses to write once the
        // route no longer resolves to it.
        let permit = AttemptPermitV1 {
            root_id: ROOT_ID,
            writer_id: WRITER_A,
            attempt_id: [0x70; 16],
            target_slot: 1,
            candidate: authority(1, None),
            expected: ExpectedAuthorityPairV1::default(),
            stage: AttemptStageV1::SlotWrite,
            idempotency_key: [0x41; 16],
        };
        assert_eq!(
            publisher.reconcile(&permit, None).expect("reconcile runs"),
            ReconcileOutcomeV1::StillUnknown(permit),
            "swapped {swapped}: reconcile cannot adjudicate on a substituted anchor"
        );
        // A freshly admitted root over the substituted anchors is a
        // different root set and publishes normally on its own inodes.
        let fresh = admit(&root_path);
        let fresh_publisher = fresh
            .authority_publisher(
                ROOT_ID,
                WRITER_B,
                GenerationRootSecurityProfileV1::CooperativeLocal,
            )
            .expect("publisher binds");
        assert!(matches!(
            fresh_publisher
                .publish(
                    authority(1, None),
                    ExpectedAuthorityPairV1::default(),
                    None,
                    [0x42; 16]
                )
                .expect("fresh publication runs"),
            PublicationOutcomeV1::Committed { .. }
        ));
    }
}

#[test]
fn short_slot_write_leaves_a_torn_slot_that_requires_repair() {
    let root_path = fixture_root("short-write");
    let root = admit(&root_path);
    let publisher = root
        .authority_publisher(
            ROOT_ID,
            WRITER_A,
            GenerationRootSecurityProfileV1::CooperativeLocal,
        )
        .expect("publisher binds");
    let genesis = authority(1, None);
    let before = authority_bytes(&root_path);

    // Fault after the first 1024-byte chunk of the 4096-byte slot.
    let outcome = {
        let _hook = platform::install_test_hook(|boundary| match boundary {
            platform::TestBoundary::DuringPublishSlotWrite { written: 1024 } => {
                Err(GenerationRootError::new(
                    GenerationRootErrorKind::Io,
                    GenerationRootStage::SyncRegularFile,
                ))
            }
            _ => Ok(()),
        });
        publisher
            .publish(
                genesis,
                ExpectedAuthorityPairV1::default(),
                None,
                [0x41; 16],
            )
            .expect("publication runs")
    };
    let PublicationOutcomeV1::CommitOutcomeUnknown(permit) = outcome else {
        panic!("a short write must be unknown: {outcome:?}");
    };
    assert_eq!(permit.stage, AttemptStageV1::SlotWrite);

    // Exactly the first chunk of slot 1 changed; the tail is still zero.
    let after = authority_bytes(&root_path);
    let range = slot_range(1);
    assert_ne!(
        &after[range.start..range.start + 1024],
        &before[range.start..range.start + 1024]
    );
    assert_eq!(
        &after[range.start + 1024..range.end],
        &before[range.start + 1024..range.end]
    );
    assert_eq!(
        &after[..range.start],
        &before[..range.start],
        "slot 0 untouched"
    );

    // The torn slot authenticates as nothing: reconcile demands repair,
    // and a fresh publication is blocked until it happens.
    assert!(matches!(
        publisher.reconcile(&permit, None).expect("reconcile runs"),
        ReconcileOutcomeV1::RepairRequired(GenerationAuthorityErrorV1::ChecksumMismatch)
    ));
    assert!(matches!(
        publisher
            .publish(
                genesis,
                ExpectedAuthorityPairV1::default(),
                None,
                [0x42; 16]
            )
            .expect("publication runs"),
        PublicationOutcomeV1::NotCommitted(NotCommittedV1::UnreadableSlot(
            GenerationAuthorityErrorV1::ChecksumMismatch
        ))
    ));
}

#[test]
fn rollback_is_a_higher_sequence_record_and_an_exhausted_head_is_read_only() {
    let root_path = fixture_root("rollback");
    let root = admit(&root_path);
    let floor = InMemoryAntiRollbackFloorStoreV1::new();
    let publisher = root
        .authority_publisher(
            ROOT_ID,
            WRITER_A,
            GenerationRootSecurityProfileV1::CooperativeLocal,
        )
        .expect("publisher binds");
    let genesis = authority(1, None);
    let PublicationOutcomeV1::Committed { slot: g_slot, .. } = publisher
        .publish(
            genesis,
            ExpectedAuthorityPairV1::default(),
            Some(&floor),
            [0x41; 16],
        )
        .expect("genesis runs")
    else {
        panic!("genesis must commit");
    };
    let successor = authority(2, Some(genesis.fingerprint()));
    let PublicationOutcomeV1::Committed { slot: s_slot, .. } = publisher
        .publish(
            successor,
            ExpectedAuthorityPairV1 {
                first: None,
                second: Some(g_slot),
            },
            Some(&floor),
            [0x42; 16],
        )
        .expect("successor runs")
    else {
        panic!("successor must commit");
    };

    // A rollback to the retained genesis is a NEW sequence-3 record
    // whose manifest is the genesis object; the sequence never moves
    // backward and the incumbent slot pair is what it fences against.
    let rollback = AuthorityRefV1::new(
        3,
        genesis.object_id,
        genesis.manifest_len,
        genesis.manifest_sha256,
        Some(successor.fingerprint()),
    )
    .expect("valid rollback reference");
    let outcome = publisher
        .publish(
            rollback,
            ExpectedAuthorityPairV1 {
                first: Some(s_slot),
                second: Some(g_slot),
            },
            Some(&floor),
            [0x43; 16],
        )
        .expect("rollback runs");
    let PublicationOutcomeV1::Committed {
        slot: r_slot,
        floor: receipt,
    } = outcome
    else {
        panic!("rollback must commit as sequence 3: {outcome:?}");
    };
    assert_eq!(r_slot.authority.sequence, 3);
    assert_eq!(
        r_slot.slot_index, 1,
        "sequence 3 overwrites the genesis slot"
    );
    assert_eq!(r_slot.authority.object_id, genesis.object_id);
    assert_eq!(receipt.map(|r| r.cas_version), Some(3));
    // Publishing the old sequence-2 successor again is a regression
    // and is refused before any write.
    let regress = publisher
        .publish(
            successor,
            ExpectedAuthorityPairV1 {
                first: Some(s_slot),
                second: Some(r_slot),
            },
            Some(&floor),
            [0x44; 16],
        )
        .expect("regression runs");
    assert!(
        matches!(
            regress,
            PublicationOutcomeV1::NotCommitted(NotCommittedV1::InvalidCandidate(_))
        ),
        "{regress:?}"
    );

    // A head at u64::MAX freezes the root read-only: the candidate
    // cannot be derived, nothing is written, the pair is unchanged.
    let exhausted_root = fixture_root("exhausted");
    let exhausted_head =
        AuthorityRefV1::new(u64::MAX, [0x7e; 16], 4_096, [0x7f; 32], Some([0x11; 32]))
            .expect("valid exhausted reference");
    let exhausted_slot = AuthoritySlotV1::new(1, ROOT_ID, exhausted_head)
        .expect("slot")
        .encode()
        .expect("slot encodes");
    {
        let file = OpenOptions::new()
            .write(true)
            .open(exhausted_root.join("AUTHORITY"))
            .expect("authority writable");
        std::os::unix::fs::FileExt::write_all_at(&file, &exhausted_slot, slot_offset(1))
            .expect("head written");
        file.sync_all().expect("head durable");
    }
    let exhausted = admit(&exhausted_root);
    let publisher = exhausted
        .authority_publisher(
            ROOT_ID,
            WRITER_A,
            GenerationRootSecurityProfileV1::CooperativeLocal,
        )
        .expect("publisher binds");
    let head = AuthoritySlotV1::new(1, ROOT_ID, exhausted_head).expect("slot");
    let before = authority_bytes(&exhausted_root);
    let attempt = publisher
        .publish(
            AuthorityRefV1::new(
                u64::MAX,
                [0x7e; 16],
                4_096,
                [0x7f; 32],
                Some(exhausted_head.fingerprint()),
            )
            .expect("valid reference"),
            ExpectedAuthorityPairV1 {
                first: None,
                second: Some(head),
            },
            None,
            [0x45; 16],
        )
        .expect("exhausted publication runs");
    assert_eq!(
        attempt,
        PublicationOutcomeV1::NotCommitted(NotCommittedV1::InvalidCandidate(
            GenerationAuthorityErrorV1::SequenceExhausted
        ))
    );
    assert_eq!(
        authority_bytes(&exhausted_root),
        before,
        "read-only: zero mutation"
    );
}

#[test]
fn reconcile_reports_fork_and_not_committed_from_exact_slot_state() {
    let root_path = fixture_root("fork");
    let root = admit(&root_path);
    let publisher = root
        .authority_publisher(
            ROOT_ID,
            WRITER_A,
            GenerationRootSecurityProfileV1::CooperativeLocal,
        )
        .expect("publisher binds");
    let genesis = authority(1, None);
    // A permit for a genesis that was never written: NotCommitted.
    let never = AttemptPermitV1 {
        root_id: ROOT_ID,
        writer_id: WRITER_A,
        attempt_id: [0x70; 16],
        target_slot: 1,
        candidate: genesis,
        expected: ExpectedAuthorityPairV1::default(),
        stage: AttemptStageV1::SlotWrite,
        idempotency_key: [0x41; 16],
    };
    assert_eq!(
        publisher.reconcile(&never, None).expect("reconcile runs"),
        ReconcileOutcomeV1::NotCommitted
    );
    // Another writer installs a different genesis into the same slot.
    let other_genesis =
        AuthorityRefV1::new(1, [0x99; 16], 5_000, [0x88; 32], None).expect("valid authority");
    let PublicationOutcomeV1::Committed { slot: observed, .. } = publisher
        .publish(
            other_genesis,
            ExpectedAuthorityPairV1::default(),
            None,
            [0x48; 16],
        )
        .expect("other genesis runs")
    else {
        panic!("other genesis must commit");
    };
    assert_eq!(
        publisher.reconcile(&never, None).expect("reconcile runs"),
        ReconcileOutcomeV1::Fork { observed }
    );
}
