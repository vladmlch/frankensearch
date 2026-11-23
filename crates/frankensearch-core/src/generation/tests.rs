use super::authority_local_floor::{
    LOCAL_FLOOR_FILE_BYTES_V1, LOCAL_FLOOR_FILE_SUFFIX_V1, LocalFloorFileV1,
};
use super::tests_support::authority_reference;
use super::*;

#[test]
fn local_anti_rollback_floor_is_durable_monotone_and_idempotent_across_reopen() {
    let directory = tempfile::tempdir().expect("temp floor directory");
    let store = LocalAntiRollbackFloorStoreV1::open(directory.path()).expect("open store");
    let root_id = [0x5a; 16];
    assert_eq!(store.load(root_id), Ok(None));

    let genesis = authority_reference(1, None);
    let genesis_floor = AuthorityFloorV1::new(root_id, genesis).expect("valid genesis floor");
    let first = store
        .compare_and_advance(None, genesis_floor, [0x41; 16])
        .expect("first durable advance");
    assert_eq!(first.cas_version, 1);
    assert_eq!(store.load(root_id), Ok(Some(first)));
    assert_eq!(
        store.compare_and_advance(None, genesis_floor, [0x41; 16]),
        Ok(first),
        "replaying the exact request returns its original receipt"
    );

    let successor = authority_reference(2, Some(genesis.fingerprint()));
    let successor_floor = AuthorityFloorV1::new(root_id, successor).expect("valid successor floor");
    let second = store
        .compare_and_advance(Some(first), successor_floor, [0x42; 16])
        .expect("monotone successor advance");
    assert_eq!(second.cas_version, 2);

    // A fresh store instance over the same directory (a restart) sees the
    // same floor and honours the same idempotency journal.
    let reopened = LocalAntiRollbackFloorStoreV1::open(directory.path()).expect("reopen store");
    assert_eq!(reopened.load(root_id), Ok(Some(second)));
    assert_eq!(
        reopened.compare_and_advance(None, genesis_floor, [0x41; 16]),
        Ok(first),
        "idempotent replay survives restart"
    );
    assert_eq!(
        reopened.compare_and_advance(Some(first), successor_floor, [0x43; 16]),
        Err(GenerationAuthorityErrorV1::FloorCompareAndAdvanceConflict),
        "a stale exact CAS receipt cannot overwrite a newer floor"
    );
    assert_eq!(
        reopened.compare_and_advance(Some(second), genesis_floor, [0x44; 16]),
        Err(GenerationAuthorityErrorV1::FloorSequenceRegression),
        "a valid older authority cannot roll the durable floor backward"
    );
    assert_eq!(
        reopened.compare_and_advance(Some(second), successor_floor, [0x41; 16]),
        Err(GenerationAuthorityErrorV1::FloorIdempotencyConflict),
        "a completed idempotency key cannot be rebound to another CAS"
    );
    assert_eq!(
        reopened.compare_and_advance(
            Some(second),
            AuthorityFloorV1::new(
                root_id,
                authority_reference(4, Some(successor.fingerprint()))
            )
            .expect("valid floor"),
            [0x45; 16]
        ),
        Err(GenerationAuthorityErrorV1::BrokenPredecessorLink),
        "the durable floor requires the exact next sequence"
    );
    assert_eq!(reopened.load(root_id), Ok(Some(second)));
    assert_eq!(reopened.load([0x5b; 16]), Ok(None), "roots are isolated");
    assert_eq!(
        reopened.load([0; 16]),
        Err(GenerationAuthorityErrorV1::InvalidField {
            field: "anti_rollback_floor.root_id"
        })
    );
}

#[test]
fn local_anti_rollback_floor_torn_or_tampered_head_is_an_unresolved_attempt() {
    let directory = tempfile::tempdir().expect("temp floor directory");
    let store = LocalAntiRollbackFloorStoreV1::open(directory.path()).expect("open store");
    let root_id = [0x5a; 16];
    let genesis = authority_reference(1, None);
    let genesis_floor = AuthorityFloorV1::new(root_id, genesis).expect("valid genesis floor");
    let first = store
        .compare_and_advance(None, genesis_floor, [0x41; 16])
        .expect("first durable advance");
    let successor = authority_reference(2, Some(genesis.fingerprint()));
    let successor_floor = AuthorityFloorV1::new(root_id, successor).expect("valid successor floor");
    let second = store
        .compare_and_advance(Some(first), successor_floor, [0x42; 16])
        .expect("second durable advance");

    let head = store.root_directory(root_id).join(format!(
        "v{:020}{LOCAL_FLOOR_FILE_SUFFIX_V1}",
        second.cas_version
    ));
    let original = std::fs::read(&head).expect("read head file");
    assert_eq!(original.len(), LOCAL_FLOOR_FILE_BYTES_V1);

    // Torn tail: a crash mid-write. Neither the floor nor an advance may
    // silently fall back to the previous version.
    std::fs::write(&head, &original[..original.len() / 2]).expect("truncate head");
    assert_eq!(
        store.load(root_id),
        Err(GenerationAuthorityErrorV1::UnresolvedAttempt)
    );
    let third = authority_reference(3, Some(successor.fingerprint()));
    let third_floor = AuthorityFloorV1::new(root_id, third).expect("valid floor");
    assert_eq!(
        store.compare_and_advance(Some(second), third_floor, [0x46; 16]),
        Err(GenerationAuthorityErrorV1::UnresolvedAttempt)
    );
    assert_eq!(
        store.compare_and_advance(Some(first), successor_floor, [0x47; 16]),
        Err(GenerationAuthorityErrorV1::UnresolvedAttempt),
        "a lower valid base cannot bypass the unresolved head"
    );

    // Single-bit tamper of a full-length file fails the trailing digest.
    let mut tampered = original.clone();
    tampered[40] ^= 0x01;
    std::fs::write(&head, &tampered).expect("tamper head");
    assert_eq!(
        store.load(root_id),
        Err(GenerationAuthorityErrorV1::UnresolvedAttempt)
    );

    // Restoring the exact bytes reconciles it.
    std::fs::write(&head, &original).expect("restore head");
    assert_eq!(store.load(root_id), Ok(Some(second)));
    assert_eq!(
        store
            .compare_and_advance(Some(second), third_floor, [0x46; 16])
            .map(|record| record.cas_version),
        Ok(3)
    );
}

/// One request in the floor property model.
#[derive(Debug, Clone, Copy)]
enum FloorOp {
    /// Valid successor on the current head with a fresh key.
    Advance,
    /// Same-base race loser: an exact receipt that is no longer the head.
    StaleBase,
    /// Byte-exact replay of an earlier successful request.
    Replay,
    /// Reuse an earlier key with a different next authority.
    Rebind,
    /// Valid base, but a next authority whose sequence does not exceed the head.
    Regress,
}

/// Drive `load`/`compare_and_advance` through a random request stream and
/// check, after every request, that the floor never decreases, that CAS
/// versions are dense and monotone, that replays return the original
/// receipt, and that every refusal leaves the head byte-identical.
fn check_floor_never_decreases<L, C>(ops: &[FloorOp], load: L, cas: C)
where
    L: Fn([u8; 16]) -> Result<Option<AntiRollbackFloorRecordV1>, GenerationAuthorityErrorV1>,
    C: Fn(
        Option<AntiRollbackFloorRecordV1>,
        AuthorityFloorV1,
        [u8; 16],
    ) -> Result<AntiRollbackFloorRecordV1, GenerationAuthorityErrorV1>,
{
    let root_id = [0x5a; 16];
    let mut head: Option<AntiRollbackFloorRecordV1> = None;
    // (key, expected, next, receipt) of every successful request.
    let mut successes: Vec<(
        [u8; 16],
        Option<AntiRollbackFloorRecordV1>,
        AuthorityFloorV1,
        AntiRollbackFloorRecordV1,
    )> = Vec::new();
    let mut next_key = 0x40_u8;
    let mut fresh_key = || {
        next_key = next_key.wrapping_add(1).max(1);
        [next_key; 16]
    };
    let successor_of = |head: Option<AntiRollbackFloorRecordV1>| {
        head.map_or_else(
            || authority_reference(1, None),
            |record| {
                authority_reference(
                    record.authority.sequence + 1,
                    Some(record.authority.fingerprint()),
                )
            },
        )
    };

    for (step, op) in ops.iter().enumerate() {
        let before = head;
        match op {
            FloorOp::Advance => {
                let next = AuthorityFloorV1::new(root_id, successor_of(head)).expect("floor");
                let key = fresh_key();
                let receipt = cas(head, next, key).unwrap_or_else(|error| {
                    panic!("step {step}: valid advance refused: {error:?}")
                });
                assert_eq!(
                    receipt.cas_version,
                    head.map_or(0, |record| record.cas_version) + 1,
                    "step {step}: CAS versions are dense"
                );
                assert_eq!(receipt.authority, next.authority);
                successes.push((key, head, next, receipt));
                head = Some(receipt);
            }
            FloorOp::StaleBase => {
                // The most recent superseded receipt, or None while a head exists.
                let stale = if successes.len() >= 2 {
                    Some(successes[successes.len() - 2].3)
                } else if head.is_some() {
                    None
                } else {
                    continue;
                };
                let next = AuthorityFloorV1::new(root_id, successor_of(stale)).expect("floor");
                assert_eq!(
                    cas(stale, next, fresh_key()),
                    Err(GenerationAuthorityErrorV1::FloorCompareAndAdvanceConflict),
                    "step {step}: a stale base cannot advance"
                );
            }
            FloorOp::Replay => {
                let Some((key, expected, next, receipt)) = successes.last().copied() else {
                    continue;
                };
                assert_eq!(
                    cas(expected, next, key),
                    Ok(receipt),
                    "step {step}: replay returns the original receipt"
                );
            }
            FloorOp::Rebind => {
                let Some((key, expected, _, _)) = successes.last().copied() else {
                    continue;
                };
                let other = AuthorityFloorV1::new(
                    root_id,
                    AuthorityRefV1::new(
                        head.map_or(1, |record| record.authority.sequence + 1),
                        [0xee; 16],
                        9_000,
                        [0xdd; 32],
                        head.map(|record| record.authority.fingerprint()),
                    )
                    .expect("valid rebind authority"),
                )
                .expect("floor");
                assert_eq!(
                    cas(expected, other, key),
                    Err(GenerationAuthorityErrorV1::FloorIdempotencyConflict),
                    "step {step}: a completed key cannot be rebound"
                );
            }
            FloorOp::Regress => {
                let Some(record) = head else {
                    continue;
                };
                let regress = AuthorityFloorV1::new(root_id, record.authority).expect("floor");
                assert_eq!(
                    cas(head, regress, fresh_key()),
                    Err(GenerationAuthorityErrorV1::FloorSequenceRegression),
                    "step {step}: the floor never moves backward"
                );
            }
        }
        let observed = load(root_id).expect("load");
        assert_eq!(observed, head, "step {step}: store head matches the model");
        if let (Some(before), Some(after)) = (before, head) {
            assert!(
                after.authority.sequence >= before.authority.sequence
                    && after.cas_version >= before.cas_version,
                "step {step}: monotone"
            );
        }
    }
}

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(48))]

    #[test]
    fn in_memory_anti_rollback_floor_never_decreases_under_random_requests(
        raw in proptest::collection::vec(0_u8..5, 1..40)
    ) {
        let ops: Vec<FloorOp> = raw
            .into_iter()
            .map(|byte| match byte {
                0 => FloorOp::Advance,
                1 => FloorOp::StaleBase,
                2 => FloorOp::Replay,
                3 => FloorOp::Rebind,
                _ => FloorOp::Regress,
            })
            .collect();
        let store = InMemoryAntiRollbackFloorStoreV1::new();
        check_floor_never_decreases(
            &ops,
            |root| store.load(root),
            |expected, next, key| store.compare_and_advance(expected, next, key),
        );
    }

    #[test]
    fn local_anti_rollback_floor_never_decreases_under_random_requests(
        raw in proptest::collection::vec(0_u8..5, 1..24)
    ) {
        let ops: Vec<FloorOp> = raw
            .into_iter()
            .map(|byte| match byte {
                0 => FloorOp::Advance,
                1 => FloorOp::StaleBase,
                2 => FloorOp::Replay,
                3 => FloorOp::Rebind,
                _ => FloorOp::Regress,
            })
            .collect();
        let directory = tempfile::tempdir().expect("temp floor directory");
        let store =
            LocalAntiRollbackFloorStoreV1::open(directory.path()).expect("open store");
        check_floor_never_decreases(
            &ops,
            |root| store.load(root),
            |expected, next, key| store.compare_and_advance(expected, next, key),
        );
    }
}

#[test]
fn local_anti_rollback_floor_allows_exactly_one_same_base_publisher() {
    let directory = tempfile::tempdir().expect("temp floor directory");
    let store = LocalAntiRollbackFloorStoreV1::open(directory.path()).expect("open store");
    let root_id = [0x5a; 16];
    let genesis = authority_reference(1, None);
    let genesis_floor = AuthorityFloorV1::new(root_id, genesis).expect("valid genesis floor");
    let base = store
        .compare_and_advance(None, genesis_floor, [0x41; 16])
        .expect("genesis advance");
    let successor_floor =
        AuthorityFloorV1::new(root_id, authority_reference(2, Some(genesis.fingerprint())))
            .expect("valid successor floor");

    const PUBLISHERS: usize = 8;
    let barrier = std::sync::Barrier::new(PUBLISHERS);
    let outcomes = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..PUBLISHERS)
            .map(|index| {
                let store = &store;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    let key = [u8::try_from(0x60 + index).expect("small index"); 16];
                    store.compare_and_advance(Some(base), successor_floor, key)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("publisher thread"))
            .collect::<Vec<_>>()
    });
    let winners = outcomes.iter().filter(|outcome| outcome.is_ok()).count();
    assert_eq!(winners, 1, "create_new elects exactly one same-base winner");
    assert!(
        outcomes.iter().all(|outcome| matches!(
            outcome,
            Ok(_) | Err(GenerationAuthorityErrorV1::FloorCompareAndAdvanceConflict)
        )),
        "every loser sees a typed conflict, never an I/O error: {outcomes:?}"
    );
    let head = store.load(root_id).expect("load head").expect("advanced");
    assert_eq!(head.cas_version, 2);
    assert_eq!(
        std::fs::read_dir(store.root_directory(root_id))
            .expect("root directory")
            .count(),
        2,
        "losers never leave version files behind"
    );
}

struct FloorProbeChild(std::process::Child);

impl FloorProbeChild {
    fn spawn(directory: &std::path::Path, role: &str) -> Self {
        Self(
            std::process::Command::new(std::env::current_exe().expect("test executable"))
                .args([
                    "--ignored",
                    "--exact",
                    "generation::tests::local_floor_process_probe",
                    "--nocapture",
                ])
                .env("FRANKENSEARCH_FLOOR_PROBE_DIR", directory)
                .env("FRANKENSEARCH_FLOOR_PROBE_ROLE", role)
                .spawn()
                .expect("spawn floor probe"),
        )
    }

    fn await_ready(&mut self, directory: &std::path::Path, role: &str) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !directory.join(format!("{role}.ready")).exists() {
            assert!(self.0.try_wait().expect("probe status").is_none());
            assert!(
                std::time::Instant::now() < deadline,
                "probe startup timed out"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    fn finish(&mut self) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if let Some(status) = self.0.try_wait().expect("probe status") {
                assert!(status.success(), "floor probe failed: {status}");
                return;
            }
            assert!(std::time::Instant::now() < deadline, "probe did not finish");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
}

impl Drop for FloorProbeChild {
    fn drop(&mut self) {
        // Also runs during assertion unwinding: no blocked subprocess escapes.
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
#[ignore = "subprocess helper executed by the bounded local-floor overlap tests"]
fn local_floor_process_probe() {
    let directory = std::path::PathBuf::from(
        std::env::var_os("FRANKENSEARCH_FLOOR_PROBE_DIR").expect("probe directory"),
    );
    let role = std::env::var("FRANKENSEARCH_FLOOR_PROBE_ROLE").expect("probe role");
    let store = LocalAntiRollbackFloorStoreV1::open(&directory).expect("independent store");
    let root_id = [0x5a; 16];
    let genesis = authority_reference(1, None);
    let base = AntiRollbackFloorRecordV1::new(AuthorityFloorV1::new(root_id, genesis).unwrap(), 1)
        .unwrap();
    let successor =
        AuthorityFloorV1::new(root_id, authority_reference(2, Some(genesis.fingerprint())))
            .unwrap();
    if role == "interrupted" {
        let _guard = store.lock_root(root_id).expect("child writer lock");
        let path = LocalAntiRollbackFloorStoreV1::version_path(&store.root_directory(root_id), 2);
        std::fs::write(path, b"partial publication").expect("partial writer bytes");
        std::fs::write(directory.join("interrupted.ready"), b"ready").unwrap();
        // Parent kills this process while it owns the lock. Bounded even
        // if the parent disappears; unwinding then releases the kernel lock.
        std::thread::sleep(std::time::Duration::from_secs(30));
        panic!("parent failed to interrupt writer");
    }
    std::fs::write(directory.join(format!("{role}.ready")), b"ready").unwrap();
    match role.as_str() {
        "reader" => assert_eq!(
            store.load(root_id).unwrap().unwrap(),
            AntiRollbackFloorRecordV1::new(successor, 2).unwrap()
        ),
        "competitor" => assert_eq!(
            store.compare_and_advance(Some(base), successor, [0x63; 16]),
            Err(GenerationAuthorityErrorV1::FloorCompareAndAdvanceConflict)
        ),
        "recovery" => {
            assert_eq!(
                store.load(root_id),
                Err(GenerationAuthorityErrorV1::UnresolvedAttempt)
            );
            assert_eq!(
                store.compare_and_advance(Some(base), successor, [0x63; 16]),
                Err(GenerationAuthorityErrorV1::UnresolvedAttempt)
            );
        }
        _ => panic!("unknown floor probe role"),
    }
}

#[test]
fn local_floor_processes_wait_for_complete_durable_publication() {
    let directory = tempfile::tempdir().unwrap();
    let store = LocalAntiRollbackFloorStoreV1::open(directory.path()).unwrap();
    let root_id = [0x5a; 16];
    let genesis = authority_reference(1, None);
    let base = store
        .compare_and_advance(
            None,
            AuthorityFloorV1::new(root_id, genesis).unwrap(),
            [0x41; 16],
        )
        .unwrap();
    let record = AntiRollbackFloorRecordV1::new(
        AuthorityFloorV1::new(root_id, authority_reference(2, Some(genesis.fingerprint())))
            .unwrap(),
        2,
    )
    .unwrap();
    let bytes = LocalFloorFileV1 {
        record,
        idempotency_key: [0x42; 16],
        expected_record_sha256: Some(base.record_sha256),
    }
    .encode();
    let guard = store.lock_root(root_id).unwrap();
    let path = LocalAntiRollbackFloorStoreV1::version_path(&store.root_directory(root_id), 2);
    let mut writer = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    std::io::Write::write_all(&mut writer, &bytes[..bytes.len() / 2]).unwrap();
    let mut reader = FloorProbeChild::spawn(directory.path(), "reader");
    let mut competitor = FloorProbeChild::spawn(directory.path(), "competitor");
    reader.await_ready(directory.path(), "reader");
    competitor.await_ready(directory.path(), "competitor");
    std::thread::sleep(std::time::Duration::from_millis(200));
    assert!(
        reader.0.try_wait().unwrap().is_none(),
        "reader observed partial publication"
    );
    assert!(
        competitor.0.try_wait().unwrap().is_none(),
        "competitor observed partial publication"
    );
    std::io::Write::write_all(&mut writer, &bytes[bytes.len() / 2..]).unwrap();
    writer.sync_all().unwrap();
    store
        .sync_directories(&store.root_directory(root_id))
        .unwrap();
    drop(guard);
    reader.finish();
    competitor.finish();
}

#[test]
fn local_floor_interrupted_process_releases_lock_but_preserves_unresolved_head() {
    let directory = tempfile::tempdir().unwrap();
    let store = LocalAntiRollbackFloorStoreV1::open(directory.path()).unwrap();
    store
        .compare_and_advance(
            None,
            AuthorityFloorV1::new([0x5a; 16], authority_reference(1, None)).unwrap(),
            [0x41; 16],
        )
        .unwrap();
    let mut interrupted = FloorProbeChild::spawn(directory.path(), "interrupted");
    interrupted.await_ready(directory.path(), "interrupted");
    interrupted.0.kill().expect("interrupt writer");
    assert!(!interrupted.0.wait().unwrap().success());
    let mut recovery = FloorProbeChild::spawn(directory.path(), "recovery");
    recovery.finish();
}

#[test]
fn in_memory_anti_rollback_floor_is_exact_monotone_and_idempotent() {
    let store = InMemoryAntiRollbackFloorStoreV1::new();
    let genesis = authority_reference(1, None);
    let genesis_floor = AuthorityFloorV1::new([0x5a; 16], genesis).expect("valid genesis floor");
    let first = store
        .compare_and_advance(None, genesis_floor, [0x41; 16])
        .expect("first floor advance");
    assert_eq!(first.cas_version, 1);
    assert_eq!(store.load([0x5a; 16]), Ok(Some(first)));
    assert_eq!(
        store.compare_and_advance(None, genesis_floor, [0x41; 16]),
        Ok(first),
        "repeating the exact request returns its original receipt"
    );

    let successor = authority_reference(2, Some(genesis.fingerprint()));
    let successor_floor =
        AuthorityFloorV1::new([0x5a; 16], successor).expect("valid successor floor");
    let second = store
        .compare_and_advance(Some(first), successor_floor, [0x42; 16])
        .expect("monotone successor advance");
    assert_eq!(second.cas_version, 2);
    assert_eq!(store.load([0x5a; 16]), Ok(Some(second)));

    assert_eq!(
        store.compare_and_advance(Some(first), successor_floor, [0x43; 16]),
        Err(GenerationAuthorityErrorV1::FloorCompareAndAdvanceConflict),
        "a stale exact CAS receipt cannot overwrite a newer floor"
    );
    assert_eq!(
        store.compare_and_advance(Some(second), genesis_floor, [0x44; 16]),
        Err(GenerationAuthorityErrorV1::FloorSequenceRegression),
        "a valid older authority cannot roll the retained floor backward"
    );
    assert_eq!(
        store.compare_and_advance(Some(second), successor_floor, [0x41; 16]),
        Err(GenerationAuthorityErrorV1::FloorIdempotencyConflict),
        "a completed idempotency key cannot be rebound to another CAS"
    );
    assert_eq!(store.load([0x5a; 16]), Ok(Some(second)));
}

#[test]
fn in_memory_anti_rollback_floor_requires_genesis_and_exact_successor_linkage() {
    let store = InMemoryAntiRollbackFloorStoreV1::new();
    let root_id = [0x5a; 16];
    let fabricated_prior = authority_reference(1, None);
    let unanchored_successor = AuthorityFloorV1::new(
        root_id,
        authority_reference(2, Some(fabricated_prior.fingerprint())),
    )
    .expect("self-consistent non-genesis authority is structurally valid");
    assert_eq!(
        store.compare_and_advance(None, unanchored_successor, [0x45; 16]),
        Err(GenerationAuthorityErrorV1::BrokenPredecessorLink),
        "an unanchored root cannot claim an unobserved predecessor"
    );

    let genesis = authority_reference(1, None);
    let first = store
        .compare_and_advance(
            None,
            AuthorityFloorV1::new(root_id, genesis).expect("valid genesis floor"),
            [0x46; 16],
        )
        .expect("genesis advance must succeed");

    let skipped =
        AuthorityFloorV1::new(root_id, authority_reference(3, Some(genesis.fingerprint())))
            .expect("self-consistent skipped authority is structurally valid");
    assert_eq!(
        store.compare_and_advance(Some(first), skipped, [0x47; 16]),
        Err(GenerationAuthorityErrorV1::BrokenPredecessorLink),
        "a floor CAS cannot skip an unobserved authority sequence"
    );

    let wrong_predecessor = AuthorityFloorV1::new(
        root_id,
        AuthorityRefV1::new(2, [0x91; 16], 4_098, [0x92; 32], Some([0x93; 32]))
            .expect("wrongly linked successor remains structurally valid"),
    )
    .expect("floor validation defers predecessor identity to CAS state");
    assert_eq!(
        store.compare_and_advance(Some(first), wrong_predecessor, [0x48; 16]),
        Err(GenerationAuthorityErrorV1::BrokenPredecessorLink),
        "a successor must bind the exact retained authority fingerprint"
    );
    assert_eq!(store.load(root_id), Ok(Some(first)));
}

#[test]
fn in_memory_anti_rollback_floor_reports_terminal_authority_exhaustion() {
    let store = InMemoryAntiRollbackFloorStoreV1::new();
    let root_id = [0x5a; 16];
    let terminal = AuthorityRefV1::new(u64::MAX, [0x81; 16], 4_096, [0x82; 32], Some([0x83; 32]))
        .expect("terminal authority remains structurally valid");
    let current = AntiRollbackFloorRecordV1::new(
        AuthorityFloorV1::new(root_id, terminal).expect("valid terminal floor"),
        7,
    )
    .expect("valid terminal record");
    store
        .state
        .lock()
        .expect("test reference store mutex must be available")
        .records
        .insert(root_id, current);

    assert_eq!(
        store.compare_and_advance(
            Some(current),
            AuthorityFloorV1::new(root_id, terminal).expect("valid terminal floor"),
            [0x49; 16],
        ),
        Err(GenerationAuthorityErrorV1::SequenceExhausted),
        "terminal authority cannot be reinterpreted as a stale or wrapped advance"
    );
    assert_eq!(store.load(root_id), Ok(Some(current)));
}

#[test]
fn in_memory_anti_rollback_floor_rejects_invalid_root_and_request_identity() {
    let store = InMemoryAntiRollbackFloorStoreV1::new();
    let authority = authority_reference(1, None);
    let floor = AuthorityFloorV1::new([0x5a; 16], authority).expect("valid floor");
    assert_eq!(
        store.load([0; 16]),
        Err(GenerationAuthorityErrorV1::InvalidField {
            field: "anti_rollback_floor.root_id"
        })
    );
    assert_eq!(
        store.compare_and_advance(None, floor, [0; 16]),
        Err(GenerationAuthorityErrorV1::InvalidField {
            field: "anti_rollback_floor.idempotency_key"
        })
    );
    assert_eq!(store.load([0x5a; 16]), Ok(None));
}

#[test]
fn in_memory_anti_rollback_floor_allows_exactly_one_same_base_publisher() {
    let store = std::sync::Arc::new(InMemoryAntiRollbackFloorStoreV1::new());
    let genesis = authority_reference(1, None);
    let floor = AuthorityFloorV1::new([0x5a; 16], genesis).expect("valid floor");
    let results = std::thread::scope(|scope| {
        // Every publisher must be spawned BEFORE the first join, otherwise
        // the eight threads run one at a time and "exactly one winner"
        // holds trivially without ever racing a same-base CAS. The spawn
        // loop is deliberately eager for that reason; a lazy
        // `.map(spawn).map(join)` chain would serialize them.
        let mut handles = Vec::with_capacity(8);
        for key_byte in 1..=8_u8 {
            let store = std::sync::Arc::clone(&store);
            handles
                .push(scope.spawn(move || store.compare_and_advance(None, floor, [key_byte; 16])));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("publisher thread must not panic"))
            .collect::<Vec<_>>()
    });
    let successful = results
        .into_iter()
        .filter_map(Result::ok)
        .collect::<Vec<_>>();
    assert_eq!(successful.len(), 1, "exactly one empty-base CAS may win");
    assert_eq!(successful[0].cas_version, 1);
    assert_eq!(store.load([0x5a; 16]), Ok(Some(successful[0])));
}
