use super::super::super::authority_publisher::{
    ExpectedAuthorityPairV1, PublicationOutcomeV1, ReconcileOutcomeV1,
};
use super::super::{
    ACTIVATION_MANIFEST_SUFFIX_V1, ClosureViolationV1, GenerationSnapshotCellV1, SnapshotHealthV1,
    SnapshotOpenOutcomeV1, SnapshotRefusalV1,
};
use super::*;
use crate::generation_root::{
    GenerationRootError, GenerationRootErrorKind, GenerationRootStage, platform,
};
use frankensearch_core::generation::{
    AuthorityRefV1, GenerationAuthorityErrorV1, GenerationComponentRole,
    GenerationRootSecurityProfileV1, InMemoryAntiRollbackFloorStoreV1,
};
use std::sync::Arc;

#[test]
fn cell_installs_only_on_success_and_keeps_the_old_arc_on_any_refusal() {
    let (root_path, root, slot) = genesis_root("cell-install");
    let cell = GenerationSnapshotCellV1::new();
    assert!(cell.load().is_none());

    let outcome = cell
        .refresh(
            &root,
            ROOT_ID,
            GenerationRootSecurityProfileV1::CooperativeLocal,
            None,
        )
        .expect("refresh runs");
    let installed = opened(outcome);
    let loaded = cell.load().expect("installed");
    assert!(Arc::ptr_eq(&installed, &loaded));
    assert_eq!(loaded.head(), slot);
    let bytes_before = loaded
        .closure()
        .bytes(GenerationComponentRole::Vector)
        .expect("mandatory vector");

    // Plant a violation: the refresh is refused and the cell is untouched.
    write_sealed_object(&root_path, "stray.wal", b"journal");
    let refusal = refused(
        cell.refresh(
            &root,
            ROOT_ID,
            GenerationRootSecurityProfileV1::CooperativeLocal,
            None,
        )
        .expect("refresh runs"),
    );
    assert_eq!(
        refusal,
        SnapshotRefusalV1::ClosureViolation(ClosureViolationV1::WriteAheadLog)
    );
    let still = cell.load().expect("still installed");
    assert!(Arc::ptr_eq(&still, &loaded));
    assert_eq!(
        &*still
            .closure()
            .bytes(GenerationComponentRole::Vector)
            .expect("mandatory vector"),
        &*bytes_before
    );
    assert!(format!("{cell:?}").contains("installed_head_sequence: Some(1)"));
}

#[test]
fn cell_refuses_a_regressed_head_from_a_stale_root() {
    // Root A advances to sequence 2; root B (same root id, e.g. a
    // replica restored from an older copy) only has genesis.
    let root_path = fixture_root("cell-regress-a");
    let root_a = admit(&root_path);
    let publisher = publisher(&root_a);
    let (genesis, _, genesis_slot) = publish_with_manifest(
        &root_path,
        &publisher,
        1,
        None,
        ExpectedAuthorityPairV1::default(),
        None,
    );
    publish_with_manifest(
        &root_path,
        &publisher,
        2,
        Some(genesis),
        ExpectedAuthorityPairV1 {
            first: None,
            second: Some(genesis_slot),
        },
        None,
    );
    let (_, root_b, _) = genesis_root("cell-regress-b");

    let cell = GenerationSnapshotCellV1::new();
    let current = opened(
        cell.refresh(
            &root_a,
            ROOT_ID,
            GenerationRootSecurityProfileV1::CooperativeLocal,
            None,
        )
        .expect("refresh runs"),
    );
    assert_eq!(current.head().authority.sequence, 2);
    assert_eq!(
        refused(
            cell.refresh(
                &root_b,
                ROOT_ID,
                GenerationRootSecurityProfileV1::CooperativeLocal,
                None,
            )
            .expect("refresh runs")
        ),
        SnapshotRefusalV1::RegressedHead {
            installed: 2,
            observed: 1
        }
    );
    assert!(Arc::ptr_eq(&cell.load().expect("kept"), &current));
    // Explicit install is the caller's decision and is not gated.
    let stale = opened(open(
        &root_b,
        GenerationRootSecurityProfileV1::CooperativeLocal,
        None,
    ));
    let previous = cell.install(Arc::clone(&stale)).expect("previous");
    assert!(Arc::ptr_eq(&previous, &current));
    assert!(Arc::ptr_eq(&cell.load().expect("stale"), &stale));
}

#[test]
fn concurrent_refresh_cannot_overwrite_a_newer_installed_head() {
    let (_, stale_root, _) = genesis_root("cell-concurrent-stale");
    let (current_path, current_root, genesis_slot) = genesis_root("cell-concurrent-current");
    let publisher = publisher(&current_root);
    publish_with_manifest(
        &current_path,
        &publisher,
        2,
        Some(genesis_slot.authority),
        ExpectedAuthorityPairV1 {
            first: None,
            second: Some(genesis_slot),
        },
        None,
    );
    let old = opened(open(
        &stale_root,
        GenerationRootSecurityProfileV1::CooperativeLocal,
        None,
    ));
    let old_bytes = old
        .closure()
        .bytes(GenerationComponentRole::Vector)
        .expect("mandatory vector");

    // Exercise a race from both None and an installed generation. In
    // either case the slow thread has already observed the cell before
    // the newer refresh wins, so a load/check/store would regress it.
    for initially_populated in [false, true] {
        let cell = GenerationSnapshotCellV1::new();
        if initially_populated {
            assert!(cell.install(Arc::clone(&old)).is_none());
        }
        let (observed_tx, observed_rx) = std::sync::mpsc::channel();
        let (resume_tx, resume_rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let cell = &cell;
            let stale_root = &stale_root;
            let slow = scope.spawn(move || {
                let mut paused = false;
                let _hook = platform::install_test_hook(move |boundary| {
                    if boundary == platform::TestBoundary::BeforeSnapshotCellCompareExchange
                        && !paused
                    {
                        paused = true;
                        observed_tx.send(()).expect("announce observation");
                        resume_rx
                            .recv_timeout(std::time::Duration::from_secs(30))
                            .expect("newer refresh completes");
                    }
                    Ok(())
                });
                cell.refresh(
                    stale_root,
                    ROOT_ID,
                    GenerationRootSecurityProfileV1::CooperativeLocal,
                    None,
                )
                .expect("slow refresh runs")
            });
            observed_rx
                .recv_timeout(std::time::Duration::from_secs(30))
                .expect("slow refresh observed the cell");
            let newer = opened(
                cell.refresh(
                    &current_root,
                    ROOT_ID,
                    GenerationRootSecurityProfileV1::CooperativeLocal,
                    None,
                )
                .expect("newer refresh runs without waiting for slow admission"),
            );
            resume_tx.send(()).expect("release slow refresher");
            assert_eq!(
                refused(slow.join().expect("slow refresher completes")),
                SnapshotRefusalV1::RegressedHead {
                    installed: 2,
                    observed: 1,
                }
            );
            assert!(Arc::ptr_eq(&cell.load().expect("newer stays"), &newer));
        });
    }
    assert_eq!(old.head().authority.sequence, 1);
    assert_eq!(
        &*old
            .closure()
            .bytes(GenerationComponentRole::Vector)
            .expect("mandatory vector"),
        &*old_bytes
    );
}

#[test]
fn cell_refuses_an_equal_sequence_fork_but_accepts_the_same_authority() {
    let (_, root, _) = genesis_root("cell-equal-head");
    let cell = GenerationSnapshotCellV1::new();
    let first = opened(
        cell.refresh(
            &root,
            ROOT_ID,
            GenerationRootSecurityProfileV1::CooperativeLocal,
            None,
        )
        .expect("initial refresh"),
    );
    let same = opened(
        cell.refresh(
            &root,
            ROOT_ID,
            GenerationRootSecurityProfileV1::CooperativeLocal,
            None,
        )
        .expect("same exact authority remains admissible"),
    );
    assert_eq!(same.head(), first.head());

    // A separately admitted root has identical manifest/component
    // bytes at the same sequence, but a different exact authority
    // reference. Matching sequence/content alone cannot join the fork.
    let fork_path = fixture_root("cell-equal-fork");
    let fork_root = admit(&fork_path);
    write_component_objects(&fork_path, 1);
    let id = [0xef; 16];
    write_manifest_object(&fork_path, id, &same.manifest_bytes());
    let (len, sha256) = same.manifest().object_receipt();
    let authority = AuthorityRefV1::new(1, id, len, sha256, None).expect("fork authority");
    assert!(matches!(
        publisher(&fork_root)
            .publish(authority, ExpectedAuthorityPairV1::default(), None, WRITER)
            .expect("fork publication"),
        PublicationOutcomeV1::Committed { .. }
    ));
    assert_eq!(
        refused(
            cell.refresh(
                &fork_root,
                ROOT_ID,
                GenerationRootSecurityProfileV1::CooperativeLocal,
                None,
            )
            .expect("fork refresh runs")
        ),
        SnapshotRefusalV1::Authority(GenerationAuthorityErrorV1::EqualSequenceFork)
    );
    assert!(Arc::ptr_eq(&cell.load().expect("original stays"), &same));
}

#[test]
fn long_lived_reader_stays_byte_exact_across_many_publications() {
    const PUBLICATIONS: u64 = 12;
    let root_path = fixture_root("cell-long-lived");
    let root = admit(&root_path);
    let floor = InMemoryAntiRollbackFloorStoreV1::new();
    let publisher = publisher(&root);
    let (mut previous, genesis_manifest, mut previous_slot) = publish_with_manifest(
        &root_path,
        &publisher,
        1,
        None,
        ExpectedAuthorityPairV1::default(),
        Some(&floor),
    );
    let cell = GenerationSnapshotCellV1::new();
    let old = opened(
        cell.refresh(
            &root,
            ROOT_ID,
            GenerationRootSecurityProfileV1::RequiredExternal,
            Some(&floor),
        )
        .expect("refresh runs"),
    );
    let old_closure: Vec<Arc<[u8]>> = ROLES
        .iter()
        .map(|role| old.closure().bytes(*role).expect("v1 component"))
        .collect();
    let mut pair = ExpectedAuthorityPairV1 {
        first: None,
        second: Some(previous_slot),
    };

    for sequence in 2..=PUBLICATIONS {
        let (authority, _, slot) = publish_with_manifest(
            &root_path,
            &publisher,
            sequence,
            Some(previous),
            pair,
            Some(&floor),
        );
        if slot.slot_index == 0 {
            pair.first = Some(slot);
        } else {
            pair.second = Some(slot);
        }
        previous = authority;
        previous_slot = slot;
        // Refresh every other publication; the cell must follow.
        if sequence % 2 == 0 {
            let fresh = opened(
                cell.refresh(
                    &root,
                    ROOT_ID,
                    GenerationRootSecurityProfileV1::RequiredExternal,
                    Some(&floor),
                )
                .expect("refresh runs"),
            );
            assert_eq!(fresh.head(), slot);
            assert_eq!(
                fresh.health(),
                SnapshotHealthV1::ExternallyAnchored {
                    cas_version: sequence
                }
            );
        }
    }

    // The first reader never noticed: same authority, manifest, and
    // component bytes; its descriptors still open.
    assert_eq!(old.head().authority.sequence, 1);
    assert_eq!(old.manifest(), &genesis_manifest);
    for (role, bytes) in ROLES.iter().zip(&old_closure) {
        assert_eq!(
            &*old.closure().bytes(*role).expect("v1 component"),
            &**bytes
        );
        assert_eq!(
            old.closure()
                .file(*role)
                .expect("v1 component")
                .witness()
                .hard_links(),
            1
        );
    }
    assert_eq!(
        &*old.manifest_bytes(),
        genesis_manifest.canonical_bytes().as_slice()
    );
    let latest = cell.load().expect("installed");
    assert_eq!(latest.head(), previous_slot);
    assert!(!Arc::ptr_eq(&latest, &old));
    // Every generation's objects are still retained beside each other.
    let inventory = root.inventory().expect("inventory");
    let manifests = inventory
        .entries()
        .iter()
        .filter(|entry| {
            entry
                .name()
                .to_str()
                .is_some_and(|name| name.ends_with(ACTIVATION_MANIFEST_SUFFIX_V1))
        })
        .count();
    assert_eq!(u64::try_from(manifests).expect("count"), PUBLICATIONS);
}

#[test]
fn a_live_exclusive_holder_yields_a_typed_lock_contended_refusal() {
    use std::io::BufRead as _;
    use std::process::{Command, Stdio};

    let (root_path, root, slot) = genesis_root("cell-contended");
    let mut child = Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "--exact",
            "generation_root::authority_publisher::tests::lock_holder_child",
            "--nocapture",
        ])
        .env("FRANKENSEARCH_PUBLISHER_LOCK_HOLDER_ROOT", &root_path)
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

    let cell = GenerationSnapshotCellV1::new();
    assert_eq!(
        refused(
            cell.refresh(
                &root,
                ROOT_ID,
                GenerationRootSecurityProfileV1::CooperativeLocal,
                None,
            )
            .expect("contended refresh runs")
        ),
        SnapshotRefusalV1::LockContended
    );
    assert!(cell.load().is_none(), "nothing installed while contended");

    child.kill().expect("kill the holder");
    child.wait().expect("reap the holder");
    let mut admitted = None;
    for _ in 0..200 {
        match cell
            .refresh(
                &root,
                ROOT_ID,
                GenerationRootSecurityProfileV1::CooperativeLocal,
                None,
            )
            .expect("post-kill refresh runs")
        {
            SnapshotOpenOutcomeV1::Refused(SnapshotRefusalV1::LockContended) => {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            other => {
                admitted = Some(other);
                break;
            }
        }
    }
    let snapshot = opened(admitted.expect("the lock frees after the holder dies"));
    assert_eq!(snapshot.head(), slot);
    assert!(Arc::ptr_eq(&cell.load().expect("installed"), &snapshot));
}

#[test]
fn readers_see_old_or_complete_new_at_every_publisher_crash_point() {
    use platform::TestBoundary as B;
    // (crash point, the successor slot is durable afterwards, the torn
    // slot needs repair rather than reconciliation).
    let matrix: [(B, bool, bool); 9] = [
        (B::BeforePublishAttemptFrame, false, false),
        (B::AfterPublishAttemptFrame, false, false),
        (B::BeforePublishSlotWrite, false, false),
        (B::DuringPublishSlotWrite { written: 1024 }, false, true),
        (B::AfterPublishSlotWrite, true, false),
        (B::BeforePublishSlotReread, true, false),
        (B::BeforePublishFloorAdvance, true, false),
        (B::AfterPublishFloorAdvance, true, false),
        (B::BeforePublishOwnerFrame, true, false),
    ];
    for (index, (crash_at, slot_written, torn)) in matrix.into_iter().enumerate() {
        let root_path = fixture_root(&format!("reader-crash-{index}"));
        let root = admit(&root_path);
        let floor = InMemoryAntiRollbackFloorStoreV1::new();
        let publisher = publisher(&root);
        let (genesis, genesis_manifest, genesis_slot) = publish_with_manifest(
            &root_path,
            &publisher,
            1,
            None,
            ExpectedAuthorityPairV1::default(),
            Some(&floor),
        );
        let old = opened(open(
            &root,
            GenerationRootSecurityProfileV1::RequiredExternal,
            Some(&floor),
        ));
        let old_vector = old
            .closure()
            .bytes(GenerationComponentRole::Vector)
            .expect("mandatory vector");

        // Successor objects are sealed first (outside the lock), then
        // the publication crashes at the boundary under test.
        let successor_manifest = manifest(2, Some(genesis));
        let (len, sha256) = successor_manifest.object_receipt();
        write_component_objects(&root_path, 2);
        write_manifest_object(
            &root_path,
            object_id(2),
            &successor_manifest.canonical_bytes(),
        );
        let successor =
            AuthorityRefV1::new(2, object_id(2), len, sha256, Some(genesis.fingerprint()))
                .expect("authority");
        let expected = ExpectedAuthorityPairV1 {
            first: None,
            second: Some(genesis_slot),
        };
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
                .publish(successor, expected, Some(&floor), [0x71; 16])
                .expect("publication runs")
        };
        let PublicationOutcomeV1::CommitOutcomeUnknown(permit) = outcome else {
            panic!("crash at {crash_at:?} must be unknown, got {outcome:?}");
        };

        // Fresh reader between crash and reconcile: either the old
        // head, complete, or a typed fail-closed refusal — never the
        // successor and never a mix.
        let fresh = open(
            &root,
            GenerationRootSecurityProfileV1::RequiredExternal,
            Some(&floor),
        );
        if crash_at == B::BeforePublishAttemptFrame {
            let snapshot = opened(fresh);
            assert_eq!(snapshot.head(), genesis_slot, "{crash_at:?}");
            assert_eq!(
                snapshot.health(),
                SnapshotHealthV1::ExternallyAnchored { cas_version: 1 }
            );
        } else {
            match fresh {
                SnapshotOpenOutcomeV1::Refused(refusal) => assert_eq!(
                    refusal,
                    SnapshotRefusalV1::PendingReconciliation,
                    "{crash_at:?}"
                ),
                SnapshotOpenOutcomeV1::Opened(snapshot) => panic!(
                    "{crash_at:?}: served head {} ({:?}) while an attempt was unresolved (permit stage {:?})",
                    snapshot.head().authority.sequence,
                    snapshot.health(),
                    permit.stage
                ),
            }
        }
        // The retained reader is untouched by any of it.
        assert_eq!(old.head(), genesis_slot);
        assert_eq!(old.manifest(), &genesis_manifest);
        assert_eq!(
            &*old
                .closure()
                .bytes(GenerationComponentRole::Vector)
                .expect("mandatory vector"),
            &*old_vector
        );

        // Reconcile, then the reader converges to exactly one verdict.
        let verdict = publisher
            .reconcile(&permit, Some(&floor))
            .expect("reconcile runs");
        if torn {
            assert!(
                matches!(verdict, ReconcileOutcomeV1::RepairRequired(_)),
                "{crash_at:?}: {verdict:?}"
            );
            // Blocked until repair: still fail-closed, still typed.
            assert_eq!(
                refused(open(
                    &root,
                    GenerationRootSecurityProfileV1::RequiredExternal,
                    Some(&floor)
                )),
                SnapshotRefusalV1::PendingReconciliation,
                "{crash_at:?}"
            );
            assert_eq!(old.head(), genesis_slot);
            continue;
        }
        if slot_written {
            assert!(
                matches!(verdict, ReconcileOutcomeV1::Committed { .. }),
                "{crash_at:?}: {verdict:?}"
            );
        } else {
            assert_eq!(verdict, ReconcileOutcomeV1::NotCommitted, "{crash_at:?}");
            let after_reconcile = open(
                &root,
                GenerationRootSecurityProfileV1::RequiredExternal,
                Some(&floor),
            );
            if crash_at == B::BeforePublishAttemptFrame {
                // Nothing ever reached the LOCK: the old head serves.
                assert_eq!(opened(after_reconcile).head(), genesis_slot);
            } else {
                // The attempt frame still has no completing owner frame.
                assert_eq!(
                    refused(after_reconcile),
                    SnapshotRefusalV1::PendingReconciliation,
                    "{crash_at:?}"
                );
            }
            let retry = publisher
                .publish(successor, expected, Some(&floor), [0x72; 16])
                .expect("retry runs");
            assert!(
                matches!(retry, PublicationOutcomeV1::Committed { .. }),
                "{crash_at:?}: {retry:?}"
            );
        }
        let converged = opened(open(
            &root,
            GenerationRootSecurityProfileV1::RequiredExternal,
            Some(&floor),
        ));
        assert_eq!(converged.head().authority, successor, "{crash_at:?}");
        assert_eq!(converged.manifest(), &successor_manifest);
        assert_eq!(
            converged.health(),
            SnapshotHealthV1::ExternallyAnchored { cas_version: 2 },
            "{crash_at:?}"
        );
        assert_eq!(
            &*converged
                .closure()
                .bytes(GenerationComponentRole::Vector)
                .expect("mandatory vector"),
            component_bytes(GenerationComponentRole::Vector, 2).as_slice()
        );
        assert_eq!(old.head(), genesis_slot, "{crash_at:?}");
    }
}
