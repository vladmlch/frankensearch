use super::*;
use crate::generation_root::GenerationRootAnchorLayout;
use frankensearch_core::generation::AuthorityRefV1;
use std::fs::{self, DirBuilder, OpenOptions};
use std::io::Write as _;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

static FIXTURE_BASE: OnceLock<PathBuf> = OnceLock::new();
pub(super) const ROOT_ID: [u8; 16] = [0x5a; 16];
pub(super) const WRITER_A: [u8; 16] = [0xa1; 16];
pub(super) const WRITER_B: [u8; 16] = [0xb2; 16];

// Route qualification requires an ancestor chain this user owns
// privately, so fixtures live under $HOME (a shared temp root fails
// qualification with WrongOwner); nothing here is ever deleted.
fn fixture_base() -> &'static Path {
    FIXTURE_BASE
        .get_or_init(|| {
            let home = PathBuf::from(
                std::env::var_os("HOME").expect("HOME identifies the private fixture ancestor"),
            );
            let parent = home.join(".frankensearch-generation-root-tests");
            match DirBuilder::new().mode(0o700).create(&parent) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => panic!("fixture parent: {error}"),
            }
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos());
            let base = parent.join(format!("publisher-{}-{nanos}", std::process::id()));
            DirBuilder::new()
                .mode(0o700)
                .create(&base)
                .expect("fixture base");
            base
        })
        .as_path()
}

pub(super) fn fixture_root(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let root = fixture_base().join(format!("{label}-{nanos}"));
    DirBuilder::new()
        .mode(0o700)
        .create(&root)
        .expect("fixture root");
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).expect("root mode");
    for (name, len) in [
        ("LOCK", AUTHORITY_PUBLISHER_LOCK_BYTES_V1),
        ("AUTHORITY", GENERATION_ROOT_AUTHORITY_FILE_BYTES),
    ] {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(root.join(name))
            .expect("anchor should be creatable");
        file.write_all(&vec![0_u8; usize::try_from(len).expect("len")])
            .expect("anchor bytes");
        file.sync_all().expect("anchor durable");
    }
    root
}

pub(super) fn admit(root: &Path) -> QualifiedGenerationRoot {
    QualifiedGenerationRoot::admit(
        root,
        GenerationRootAnchorLayout::new(AUTHORITY_PUBLISHER_LOCK_BYTES_V1)
            .expect("publisher lock layout"),
    )
    .expect("fixture root should qualify")
}

pub(super) fn authority(sequence: u64, predecessor: Option<[u8; 32]>) -> AuthorityRefV1 {
    // Sequence-wide (not u8-bound) so long publication runs stay valid;
    // every field is nonzero for any sequence >= 1.
    let word = sequence.to_be_bytes();
    let mut object_id = [0x01_u8; 16];
    object_id[8..].copy_from_slice(&word);
    let mut manifest_sha256 = [0x10_u8; 32];
    manifest_sha256[24..].copy_from_slice(&word);
    AuthorityRefV1::new(
        sequence,
        object_id,
        4_096 + sequence,
        manifest_sha256,
        predecessor,
    )
    .expect("valid authority reference")
}

pub(super) fn authority_bytes(root: &Path) -> Vec<u8> {
    fs::read(root.join("AUTHORITY")).expect("authority readable")
}

struct RefusingFloor {
    inner: InMemoryAntiRollbackFloorStoreV1,
    refuse: std::sync::atomic::AtomicBool,
}

impl AntiRollbackFloorProviderV1 for RefusingFloor {
    fn load(
        &self,
        root_id: [u8; 16],
    ) -> Result<Option<AntiRollbackFloorRecordV1>, GenerationAuthorityErrorV1> {
        self.inner.load(root_id)
    }

    fn compare_and_advance(
        &self,
        expected: Option<AntiRollbackFloorRecordV1>,
        next: AuthorityFloorV1,
        idempotency_key: [u8; 16],
    ) -> Result<AntiRollbackFloorRecordV1, GenerationAuthorityErrorV1> {
        if self.refuse.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(GenerationAuthorityErrorV1::FloorStoreUnavailable);
        }
        self.inner
            .compare_and_advance(expected, next, idempotency_key)
    }
}

#[test]
fn genesis_then_successor_publications_commit_durably_with_a_floor() {
    let root_path = fixture_root("commit");
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
    let first = publisher
        .publish(
            genesis,
            ExpectedAuthorityPairV1::default(),
            Some(&floor),
            [0x41; 16],
        )
        .expect("genesis publication runs");
    let PublicationOutcomeV1::Committed {
        slot: first_slot,
        floor: first_floor,
    } = first
    else {
        panic!("genesis must commit: {first:?}");
    };
    assert_eq!(
        first_slot.slot_index, 1,
        "sequence 1 selects physical slot 1"
    );
    assert_eq!(first_floor.map(|r| r.cas_version), Some(1));

    // Durable bytes: slot 1 holds the genesis, slot 0 is still empty.
    let bytes = authority_bytes(&root_path);
    assert_eq!(
        decode_pair(&bytes, ROOT_ID).expect("pair decodes"),
        ExpectedAuthorityPairV1 {
            first: None,
            second: Some(first_slot)
        }
    );
    let owner = publisher
        .owner_frame()
        .expect("owner frame reads")
        .expect("owner frame present");
    assert_eq!(owner.kind, GenerationLockFrameKindV1::Owner);
    assert_eq!(owner.fence, 1);
    assert_eq!(owner.authority_fingerprint, genesis.fingerprint());

    // Successor goes to slot 0 and advances the floor to CAS 2.
    let successor = authority(2, Some(genesis.fingerprint()));
    let second = publisher
        .publish(
            successor,
            ExpectedAuthorityPairV1 {
                first: None,
                second: Some(first_slot),
            },
            Some(&floor),
            [0x42; 16],
        )
        .expect("successor publication runs");
    let PublicationOutcomeV1::Committed {
        slot: second_slot,
        floor: second_floor,
    } = second
    else {
        panic!("successor must commit: {second:?}");
    };
    assert_eq!(second_slot.slot_index, 0);
    assert_eq!(second_floor.map(|r| r.cas_version), Some(2));
    let bytes = authority_bytes(&root_path);
    assert_eq!(
        decode_pair(&bytes, ROOT_ID).expect("pair decodes"),
        ExpectedAuthorityPairV1 {
            first: Some(second_slot),
            second: Some(first_slot)
        },
        "the incumbent slot is retained, never overwritten"
    );
    assert_eq!(
        floor
            .load(ROOT_ID)
            .expect("floor loads")
            .map(|r| r.authority),
        Some(successor)
    );
}

#[test]
fn stale_expected_pair_is_refused_with_zero_authority_mutation() {
    let root_path = fixture_root("stale");
    let root = admit(&root_path);
    let publisher = root
        .authority_publisher(
            ROOT_ID,
            WRITER_A,
            GenerationRootSecurityProfileV1::CooperativeLocal,
        )
        .expect("publisher binds");
    let genesis = authority(1, None);
    let PublicationOutcomeV1::Committed { slot, .. } = publisher
        .publish(
            genesis,
            ExpectedAuthorityPairV1::default(),
            None,
            [0x41; 16],
        )
        .expect("genesis runs")
    else {
        panic!("genesis must commit");
    };
    let before = authority_bytes(&root_path);
    let lock_before = fs::read(root_path.join("LOCK")).expect("lock readable");

    // A publisher that still believes the root is empty.
    let outcome = publisher
        .publish(
            authority(1, None),
            ExpectedAuthorityPairV1::default(),
            None,
            [0x43; 16],
        )
        .expect("stale publication runs");
    assert_eq!(
        outcome,
        PublicationOutcomeV1::NotCommitted(NotCommittedV1::ExpectedPairMismatch {
            observed: Box::new(ExpectedAuthorityPairV1 {
                first: None,
                second: Some(slot)
            })
        })
    );
    // A candidate that does not extend the head by exactly one.
    let outcome = publisher
        .publish(
            authority(3, Some(genesis.fingerprint())),
            ExpectedAuthorityPairV1 {
                first: None,
                second: Some(slot),
            },
            None,
            [0x44; 16],
        )
        .expect("bad candidate runs");
    assert_eq!(
        outcome,
        PublicationOutcomeV1::NotCommitted(NotCommittedV1::InvalidCandidate(
            GenerationAuthorityErrorV1::BrokenPredecessorLink
        ))
    );
    // Read-only profile never even takes the lock.
    let read_only = root
        .authority_publisher(
            ROOT_ID,
            WRITER_A,
            GenerationRootSecurityProfileV1::ReadOnlyUnanchored,
        )
        .expect("publisher binds");
    assert_eq!(
        read_only
            .publish(
                authority(2, Some(genesis.fingerprint())),
                ExpectedAuthorityPairV1 {
                    first: None,
                    second: Some(slot)
                },
                None,
                [0x45; 16]
            )
            .expect("read-only runs"),
        PublicationOutcomeV1::NotCommitted(NotCommittedV1::ReadOnlyProfile)
    );
    // RequiredExternal without a floor is refused before any write.
    let required = root
        .authority_publisher(
            ROOT_ID,
            WRITER_A,
            GenerationRootSecurityProfileV1::RequiredExternal,
        )
        .expect("publisher binds");
    assert_eq!(
        required
            .publish(
                authority(2, Some(genesis.fingerprint())),
                ExpectedAuthorityPairV1 {
                    first: None,
                    second: Some(slot)
                },
                None,
                [0x46; 16]
            )
            .expect("required-external runs"),
        PublicationOutcomeV1::NotCommitted(NotCommittedV1::FloorRefused(
            GenerationAuthorityErrorV1::ExternalFloorRequired
        ))
    );
    assert_eq!(
        authority_bytes(&root_path),
        before,
        "AUTHORITY bytes unchanged"
    );
    assert_eq!(
        fs::read(root_path.join("LOCK")).expect("lock readable"),
        lock_before,
        "no attempt frame is written for a refused publication"
    );
}

#[test]
fn two_real_publishers_on_one_base_admit_exactly_one_winner() {
    let root_path = fixture_root("race");
    let root_a = admit(&root_path);
    let root_b = admit(&root_path);
    let floor = InMemoryAntiRollbackFloorStoreV1::new();
    let genesis = authority(1, None);
    let seed = root_a
        .authority_publisher(
            ROOT_ID,
            WRITER_A,
            GenerationRootSecurityProfileV1::CooperativeLocal,
        )
        .expect("publisher binds")
        .publish(
            genesis,
            ExpectedAuthorityPairV1::default(),
            Some(&floor),
            [0x41; 16],
        )
        .expect("genesis runs");
    let PublicationOutcomeV1::Committed { slot, .. } = seed else {
        panic!("genesis must commit");
    };
    let base = ExpectedAuthorityPairV1 {
        first: None,
        second: Some(slot),
    };
    let successor = authority(2, Some(genesis.fingerprint()));

    let barrier = std::sync::Barrier::new(2);
    let publishers = [(&root_a, WRITER_A, 0x51_u8), (&root_b, WRITER_B, 0x52_u8)];
    let outcomes = std::thread::scope(|scope| {
        let handles: Vec<_> = publishers
            .iter()
            .map(|(root, writer, key)| {
                let barrier = &barrier;
                let floor = &floor;
                scope.spawn(move || {
                    let publisher = root
                        .authority_publisher(
                            ROOT_ID,
                            *writer,
                            GenerationRootSecurityProfileV1::CooperativeLocal,
                        )
                        .expect("publisher binds");
                    barrier.wait();
                    // Retry on lock contention only: a real writer would
                    // reload the pair and try again; here the reloaded pair
                    // must then mismatch, which is the typed loss.
                    loop {
                        match publisher
                            .publish(successor, base, Some(floor), [*key; 16])
                            .expect("publication runs")
                        {
                            PublicationOutcomeV1::NotCommitted(
                                NotCommittedV1::LockContended | NotCommittedV1::AnchorsChanged,
                            ) => {
                                std::thread::yield_now();
                            }
                            outcome => return outcome,
                        }
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("publisher thread"))
            .collect::<Vec<_>>()
    });
    let committed = outcomes
        .iter()
        .filter(|o| matches!(o, PublicationOutcomeV1::Committed { .. }))
        .count();
    assert_eq!(committed, 1, "exactly one winner: {outcomes:?}");
    assert!(
        outcomes.iter().any(|o| matches!(
            o,
            PublicationOutcomeV1::NotCommitted(NotCommittedV1::ExpectedPairMismatch { .. })
        )),
        "the loser sees the winner's slot as a typed pair mismatch: {outcomes:?}"
    );
    assert_eq!(
        floor.load(ROOT_ID).expect("floor").map(|r| r.cas_version),
        Some(2),
        "the floor advanced exactly once"
    );
}

#[test]
fn floor_refusal_after_the_slot_write_is_unknown_then_reconciles_committed() {
    let root_path = fixture_root("unknown");
    let root = admit(&root_path);
    let floor = RefusingFloor {
        inner: InMemoryAntiRollbackFloorStoreV1::new(),
        refuse: std::sync::atomic::AtomicBool::new(true),
    };
    let publisher = root
        .authority_publisher(
            ROOT_ID,
            WRITER_A,
            GenerationRootSecurityProfileV1::CooperativeLocal,
        )
        .expect("publisher binds");
    let genesis = authority(1, None);
    let outcome = publisher
        .publish(
            genesis,
            ExpectedAuthorityPairV1::default(),
            Some(&floor),
            [0x41; 16],
        )
        .expect("publication runs");
    let PublicationOutcomeV1::CommitOutcomeUnknown(permit) = outcome else {
        panic!("floor refusal after the slot write must be unknown: {outcome:?}");
    };
    assert_eq!(permit.stage, AttemptStageV1::FloorAdvance);
    assert_eq!(permit.target_slot, 1);
    // The slot is durable locally, the floor never advanced, and the
    // owner frame is absent: exactly the crash shape reconcile must read.
    let bytes = authority_bytes(&root_path);
    assert!(
        decode_slot(&bytes, 1, ROOT_ID)
            .expect("slot decodes")
            .is_some()
    );
    assert_eq!(floor.inner.load(ROOT_ID), Ok(None));
    assert!(publisher.owner_frame().expect("frame reads").is_none());

    // Still refusing: reconcile cannot prove the floor => StillUnknown.
    assert_eq!(
        publisher
            .reconcile(&permit, Some(&floor))
            .expect("reconcile runs"),
        ReconcileOutcomeV1::StillUnknown(permit)
    );
    // Floor back: reconcile advances it idempotently and completes.
    floor
        .refuse
        .store(false, std::sync::atomic::Ordering::SeqCst);
    let reconciled = publisher
        .reconcile(&permit, Some(&floor))
        .expect("reconcile runs");
    let ReconcileOutcomeV1::Committed {
        slot,
        floor: receipt,
    } = reconciled
    else {
        panic!("must reconcile committed: {reconciled:?}");
    };
    assert_eq!(slot.authority, genesis);
    assert_eq!(receipt.map(|r| r.cas_version), Some(1));
    assert_eq!(
        publisher
            .owner_frame()
            .expect("frame reads")
            .map(|f| f.fence),
        Some(1)
    );
    // Reconciling again is idempotent.
    assert!(matches!(
        publisher
            .reconcile(&permit, Some(&floor))
            .expect("reconcile runs"),
        ReconcileOutcomeV1::Committed { .. }
    ));
}

#[test]
fn torn_target_slot_requires_repair_and_blocks_publication() {
    let root_path = fixture_root("torn");
    let root = admit(&root_path);
    let publisher = root
        .authority_publisher(
            ROOT_ID,
            WRITER_A,
            GenerationRootSecurityProfileV1::CooperativeLocal,
        )
        .expect("publisher binds");
    let genesis = authority(1, None);
    let PublicationOutcomeV1::Committed { slot, .. } = publisher
        .publish(
            genesis,
            ExpectedAuthorityPairV1::default(),
            None,
            [0x41; 16],
        )
        .expect("genesis runs")
    else {
        panic!("genesis must commit");
    };
    // Tear the tail of slot 1 in place (same physical length).
    let mut bytes = authority_bytes(&root_path);
    let range = slot_range(1);
    for byte in &mut bytes[range.end - 64..range.end] {
        *byte ^= 0xff;
    }
    let file = OpenOptions::new()
        .write(true)
        .open(root_path.join("AUTHORITY"))
        .expect("authority writable");
    std::os::unix::fs::FileExt::write_all_at(&file, &bytes, 0).expect("tear written");
    file.sync_all().expect("tear durable");

    let permit = AttemptPermitV1 {
        root_id: ROOT_ID,
        writer_id: WRITER_A,
        attempt_id: [0x77; 16],
        target_slot: 1,
        candidate: genesis,
        expected: ExpectedAuthorityPairV1::default(),
        stage: AttemptStageV1::SlotReread,
        idempotency_key: [0x41; 16],
    };
    assert!(matches!(
        publisher.reconcile(&permit, None).expect("reconcile runs"),
        ReconcileOutcomeV1::RepairRequired(GenerationAuthorityErrorV1::ChecksumMismatch)
    ));
    assert!(matches!(
        publisher
            .publish(
                authority(2, Some(genesis.fingerprint())),
                ExpectedAuthorityPairV1 {
                    first: None,
                    second: Some(slot)
                },
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
fn every_publish_crash_point_reconciles_to_its_exact_verdict() {
    use platform::TestBoundary as B;
    // (crash point, stage the permit must report, verdict a fresh
    // reconcile must reach, whether the floor is expected to hold the
    // candidate afterwards).
    let matrix: [(B, AttemptStageV1, bool); 8] = [
        (
            B::BeforePublishAttemptFrame,
            AttemptStageV1::AttemptFrame,
            false,
        ),
        (
            B::AfterPublishAttemptFrame,
            AttemptStageV1::AttemptFrame,
            false,
        ),
        (B::BeforePublishSlotWrite, AttemptStageV1::SlotWrite, false),
        (B::AfterPublishSlotWrite, AttemptStageV1::SlotWrite, true),
        (B::BeforePublishSlotReread, AttemptStageV1::SlotReread, true),
        (
            B::BeforePublishFloorAdvance,
            AttemptStageV1::FloorAdvance,
            true,
        ),
        (
            B::AfterPublishFloorAdvance,
            AttemptStageV1::FloorAdvance,
            true,
        ),
        (B::BeforePublishOwnerFrame, AttemptStageV1::OwnerFrame, true),
    ];
    for (index, (crash_at, stage, slot_written)) in matrix.into_iter().enumerate() {
        let root_path = fixture_root(&format!("crash-{index}"));
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
        let before = authority_bytes(&root_path);

        let outcome = {
            let _hook = platform::install_test_hook(move |boundary| {
                if boundary == crash_at {
                    Err(GenerationRootError::new(
                        GenerationRootErrorKind::Io,
                        GenerationRootStage::SyncRegularFile,
                    ))
                } else {
                    Ok(())
                }
            });
            publisher
                .publish(
                    genesis,
                    ExpectedAuthorityPairV1::default(),
                    Some(&floor),
                    [0x41; 16],
                )
                .expect("publication runs")
        };
        let PublicationOutcomeV1::CommitOutcomeUnknown(permit) = outcome else {
            panic!("crash at {crash_at:?} must be unknown, got {outcome:?}");
        };
        assert_eq!(permit.stage, stage, "crash at {crash_at:?}");

        // Local slot state is exactly what the crash point implies.
        let after = authority_bytes(&root_path);
        if slot_written {
            assert_ne!(after, before, "crash at {crash_at:?} left the slot durable");
            assert_eq!(
                decode_slot(&after, 1, ROOT_ID)
                    .expect("slot decodes")
                    .map(|s| s.authority),
                Some(genesis)
            );
        } else {
            assert_eq!(
                after, before,
                "crash at {crash_at:?} must not touch AUTHORITY"
            );
        }
        let floor_advanced_before_reconcile = matches!(
            crash_at,
            B::AfterPublishFloorAdvance | B::BeforePublishOwnerFrame
        );
        assert_eq!(
            floor.load(ROOT_ID).expect("floor loads").is_some(),
            floor_advanced_before_reconcile,
            "crash at {crash_at:?}"
        );

        // A fresh reconcile (hook gone) reaches the exact verdict.
        let verdict = publisher
            .reconcile(&permit, Some(&floor))
            .expect("reconcile runs");
        if slot_written {
            let ReconcileOutcomeV1::Committed {
                slot,
                floor: receipt,
            } = verdict
            else {
                panic!("crash at {crash_at:?} must reconcile committed: {verdict:?}");
            };
            assert_eq!(slot.authority, genesis);
            assert_eq!(receipt.map(|r| r.cas_version), Some(1));
            assert_eq!(
                floor.load(ROOT_ID).expect("floor").map(|r| r.authority),
                Some(genesis),
                "reconcile advanced the floor exactly once for {crash_at:?}"
            );
            assert_eq!(
                publisher.owner_frame().expect("frame").map(|f| f.fence),
                Some(1),
                "reconcile completed the owner frame for {crash_at:?}"
            );
            // The next publication extends the reconciled head normally.
            let next = publisher
                .publish(
                    authority(2, Some(genesis.fingerprint())),
                    ExpectedAuthorityPairV1 {
                        first: None,
                        second: Some(slot),
                    },
                    Some(&floor),
                    [0x42; 16],
                )
                .expect("successor runs");
            assert!(
                matches!(next, PublicationOutcomeV1::Committed { .. }),
                "{next:?}"
            );
        } else {
            assert_eq!(
                verdict,
                ReconcileOutcomeV1::NotCommitted,
                "crash at {crash_at:?}"
            );
            assert_eq!(floor.load(ROOT_ID), Ok(None));
            // Nothing was published: a fresh genesis publication succeeds.
            let retry = publisher
                .publish(
                    genesis,
                    ExpectedAuthorityPairV1::default(),
                    Some(&floor),
                    [0x43; 16],
                )
                .expect("retry runs");
            assert!(
                matches!(retry, PublicationOutcomeV1::Committed { .. }),
                "{retry:?}"
            );
        }
    }
}

#[test]
fn writer_identity_is_bound_to_host_boot_process_and_nonce() {
    let first = WriterIdentityV1::for_current_process().expect("identity derives");
    let second = WriterIdentityV1::for_current_process().expect("identity derives");
    assert_ne!(first.writer_id, [0; 16]);
    assert_ne!(
        first.nonce, second.nonce,
        "every derivation carries a fresh nonce"
    );
    assert_ne!(
        first.writer_id, second.writer_id,
        "the nonce is bound into the id"
    );
    assert_eq!(first.pid, std::process::id());
    assert!(first.same_host_and_boot(&second));
    assert_eq!(first.host_fingerprint, second.host_fingerprint);
    assert_eq!(first.process_start_ticks, second.process_start_ticks);
    assert!(first.boot_id.is_some(), "Linux exposes boot_id");
    assert!(
        first.process_start_ticks.is_some(),
        "Linux exposes /proc/self/stat"
    );

    // The derived id is accepted by the publisher and lands in the
    // owner frame verbatim, so a frame can be attributed to exactly
    // this process incarnation.
    let root_path = fixture_root("writer-identity");
    let root = admit(&root_path);
    let publisher = root
        .authority_publisher(
            ROOT_ID,
            first.writer_id,
            GenerationRootSecurityProfileV1::CooperativeLocal,
        )
        .expect("publisher binds");
    let outcome = publisher
        .publish(
            authority(1, None),
            ExpectedAuthorityPairV1::default(),
            None,
            [0x41; 16],
        )
        .expect("genesis runs");
    assert!(matches!(outcome, PublicationOutcomeV1::Committed { .. }));
    assert_eq!(
        publisher.owner_frame().expect("frame").map(|f| f.writer_id),
        Some(first.writer_id)
    );
}

const LOCK_HOLDER_CHILD_ENV: &str = "FRANKENSEARCH_PUBLISHER_LOCK_HOLDER_ROOT";

/// Child-process body for the SIGKILL test: holds the exclusive anchor
/// lock until killed. Selected only through the env var, so it is a
/// no-op when the test filter runs it directly.
#[test]
fn lock_holder_child() {
    let Some(root) = std::env::var_os(LOCK_HOLDER_CHILD_ENV) else {
        return;
    };
    let root = admit(Path::new(&root));
    let guard = root
        .try_exclusive_anchor_guard()
        .expect("child takes the exclusive lock");
    // With one test thread, libtest leaves its test-name prefix on
    // stdout without a newline. Keep the readiness marker standalone.
    println!("\nLOCKED");
    std::io::Write::flush(&mut std::io::stdout()).expect("flush");
    std::thread::sleep(std::time::Duration::from_secs(120));
    drop(guard);
}

#[test]
fn sigkill_of_the_live_owner_releases_the_lock_without_any_frame_write() {
    use std::io::BufRead as _;
    use std::process::{Command, Stdio};

    let root_path = fixture_root("sigkill");
    let mut child = Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "--exact",
            "generation_root::authority_publisher::tests::lock_holder_child",
            "--nocapture",
        ])
        .env(LOCK_HOLDER_CHILD_ENV, &root_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn lock-holder child");
    let stdout = child.stdout.take().expect("child stdout");
    let mut lines = std::io::BufReader::new(stdout).lines();
    let locked = std::iter::from_fn(|| lines.next().and_then(Result::ok))
        .take(50)
        .any(|line| line.trim() == "LOCKED");
    assert!(locked, "child never reported holding the lock");

    // While the child lives, the lock is contended: a typed refusal,
    // no frame, no slot.
    let root = admit(&root_path);
    let publisher = root
        .authority_publisher(
            ROOT_ID,
            WRITER_A,
            GenerationRootSecurityProfileV1::CooperativeLocal,
        )
        .expect("publisher binds");
    let before = authority_bytes(&root_path);
    assert_eq!(
        publisher
            .publish(
                authority(1, None),
                ExpectedAuthorityPairV1::default(),
                None,
                [0x41; 16]
            )
            .expect("contended publication runs"),
        PublicationOutcomeV1::NotCommitted(NotCommittedV1::LockContended)
    );

    // SIGKILL: no unlock code runs in the child. The kernel releases
    // the flock at descriptor close; nothing else is needed and no
    // owner frame is consulted for liveness.
    child.kill().expect("kill -9 the holder");
    child.wait().expect("reap the holder");
    let mut outcome = None;
    for _ in 0..200 {
        let attempt = publisher
            .publish(
                authority(1, None),
                ExpectedAuthorityPairV1::default(),
                None,
                [0x42; 16],
            )
            .expect("post-kill publication runs");
        if matches!(
            attempt,
            PublicationOutcomeV1::NotCommitted(NotCommittedV1::LockContended)
        ) {
            std::thread::sleep(std::time::Duration::from_millis(10));
            continue;
        }
        outcome = Some(attempt);
        break;
    }
    assert!(
        matches!(outcome, Some(PublicationOutcomeV1::Committed { .. })),
        "the lock must be free after SIGKILL: {outcome:?}"
    );
    assert_ne!(authority_bytes(&root_path), before);
    assert_eq!(
        publisher.owner_frame().expect("frame").map(|f| f.fence),
        Some(1)
    );
}

mod races_and_reconcile;
