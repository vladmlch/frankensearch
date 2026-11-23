use super::super::authority_publisher::{
    AUTHORITY_PUBLISHER_LOCK_BYTES_V1, AuthorityPublisherV1, PublicationOutcomeV1,
    ReconcileOutcomeV1,
};
use super::*;
use crate::generation_root::{
    GENERATION_ROOT_AUTHORITY_FILE_BYTES, GenerationRootAnchorLayout, GenerationRootError,
    GenerationRootStage, platform,
};
use frankensearch_core::generation::{
    ActivationManifestV1, ActivationManifestV2, ArtifactGenerationIdentityV1,
    GenerationAuthorityActionV1, GenerationComponentReceiptV1, GenerationComponentReceiptsV1,
    GenerationComponentReceiptsV2, InMemoryAntiRollbackFloorStoreV1,
};
use sha2::{Digest as _, Sha256};
use std::fs::{self, DirBuilder, OpenOptions};
use std::io::Write as _;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

static FIXTURE_BASE: OnceLock<PathBuf> = OnceLock::new();
pub(super) const ROOT_ID: [u8; 16] = [0x7c; 16];
const WRITER: [u8; 16] = [0xc3; 16];

// Same discipline as the publisher fixtures: a privately owned
// ancestor under $HOME, and nothing is ever deleted.
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
            let base = parent.join(format!("reader-{}-{nanos}", std::process::id()));
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

pub(super) fn publisher(root: &QualifiedGenerationRoot) -> AuthorityPublisherV1<'_> {
    root.authority_publisher(
        ROOT_ID,
        WRITER,
        GenerationRootSecurityProfileV1::CooperativeLocal,
    )
    .expect("publisher binds")
}

pub(super) const ROLES: [GenerationComponentRole; 4] = [
    GenerationComponentRole::Vector,
    GenerationComponentRole::Lexical,
    GenerationComponentRole::Ann,
    GenerationComponentRole::Metadata,
];

/// Deterministic per-role, per-generation component bytes.
pub(super) fn component_bytes(role: GenerationComponentRole, sequence: u64) -> Vec<u8> {
    let tag = match role {
        GenerationComponentRole::Vector => 0x11,
        GenerationComponentRole::Lexical => 0x12,
        GenerationComponentRole::Ann => 0x13,
        GenerationComponentRole::Metadata => 0x14,
    };
    let len = 100 + usize::try_from(sequence).expect("small sequence");
    let mut bytes = vec![tag; len];
    bytes[..8].copy_from_slice(&sequence.to_be_bytes());
    bytes
}

pub(super) fn component_receipt(
    role: GenerationComponentRole,
    sequence: u64,
) -> GenerationComponentReceiptV1 {
    let bytes = component_bytes(role, sequence);
    GenerationComponentReceiptV1 {
        byte_len: u64::try_from(bytes.len()).expect("len"),
        sha256: Sha256::digest(&bytes).into(),
    }
}

fn component_receipts(sequence: u64) -> GenerationComponentReceiptsV1 {
    GenerationComponentReceiptsV1 {
        vector: component_receipt(GenerationComponentRole::Vector, sequence),
        lexical: component_receipt(GenerationComponentRole::Lexical, sequence),
        ann: component_receipt(GenerationComponentRole::Ann, sequence),
        metadata: component_receipt(GenerationComponentRole::Metadata, sequence),
    }
}

pub(super) fn manifest(sequence: u64, predecessor: Option<AuthorityRefV1>) -> ActivationManifest {
    ActivationManifest::V1(
        ActivationManifestV1::new(
            sequence,
            predecessor,
            GenerationAuthorityActionV1::Activate,
            ArtifactGenerationIdentityV1::new(sequence, [0x21; 16]).expect("generation"),
            [0x31; 32],
            [0x32; 32],
            [0x33; 32],
            component_receipts(sequence),
        )
        .expect("valid v1 activation manifest"),
    )
}

pub(super) fn manifest_v2(
    sequence: u64,
    predecessor: Option<AuthorityRefV1>,
    with_ann: bool,
) -> ActivationManifest {
    let receipts = component_receipts(sequence);
    ActivationManifest::V2(
        ActivationManifestV2::new(
            sequence,
            predecessor,
            GenerationAuthorityActionV1::Activate,
            ArtifactGenerationIdentityV1::new(sequence, [0x21; 16]).expect("generation"),
            [0x31; 32],
            [0x32; 32],
            [0x33; 32],
            GenerationComponentReceiptsV2 {
                vector: receipts.vector,
                lexical: receipts.lexical,
                ann: with_ann.then_some(receipts.ann),
                metadata: receipts.metadata,
            },
        )
        .expect("valid v2 activation manifest"),
    )
}

/// Write one sealed immutable object (`create_new`, owner read-only,
/// synced) under `name`; returns its path.
pub(super) fn write_sealed_object(root: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let path = root.join(name);
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&path)
        .expect("object is create_new");
    file.write_all(bytes).expect("object bytes");
    file.sync_all().expect("object durable");
    drop(file);
    fs::set_permissions(&path, fs::Permissions::from_mode(0o400)).expect("seal mode");
    path
}

/// Write the four content-addressed component objects of `sequence`
/// (skipping any that an earlier generation already retained).
pub(super) fn write_component_objects(root: &Path, sequence: u64) {
    for role in ROLES {
        let bytes = component_bytes(role, sequence);
        let name = component_object_name_v1(role, Sha256::digest(&bytes).into());
        if !root.join(&name).exists() {
            write_sealed_object(root, &name, &bytes);
        }
    }
}

pub(super) fn object_id(sequence: u64) -> [u8; 16] {
    let mut id = [0x0f_u8; 16];
    id[8..].copy_from_slice(&sequence.to_be_bytes());
    id
}

/// Write the immutable manifest object exactly as a publisher lane
/// would: `create_new`, sealed read-only for its owner, fully synced.
pub(super) fn write_manifest_object(root: &Path, object_id: [u8; 16], bytes: &[u8]) -> PathBuf {
    let confined = activation_manifest_path_v1(object_id).expect("confined manifest name");
    assert_eq!(confined.component_count(), 1);
    write_sealed_object(root, &activation_manifest_name_v1(object_id), bytes)
}

/// Publish `sequence` with a real manifest object; returns the
/// authority reference and the manifest it addresses.
pub(super) fn publish_with_manifest(
    root_path: &Path,
    publisher: &AuthorityPublisherV1<'_>,
    sequence: u64,
    predecessor: Option<AuthorityRefV1>,
    expected: ExpectedAuthorityPairV1,
    floor: Option<&dyn AntiRollbackFloorProviderV1>,
) -> (AuthorityRefV1, ActivationManifest, AuthoritySlotV1) {
    let manifest = manifest(sequence, predecessor);
    let (authority, slot) = publish_manifest(root_path, publisher, &manifest, expected, floor);
    (authority, manifest, slot)
}

/// Stage precisely the selected manifest's roles. Preexisting objects
/// remain untouched so corrupt candidates reach reader-refusal tests.
pub(super) fn publish_manifest(
    root_path: &Path,
    publisher: &AuthorityPublisherV1<'_>,
    manifest: &ActivationManifest,
    expected: ExpectedAuthorityPairV1,
    floor: Option<&dyn AntiRollbackFloorProviderV1>,
) -> (AuthorityRefV1, AuthoritySlotV1) {
    let sequence = manifest.authority_sequence();
    let bytes = manifest.canonical_bytes();
    let (len, sha256) = manifest.object_receipt();
    let id = object_id(sequence);
    for (role, receipt) in manifest.components().iter() {
        let name = component_object_name_v1(role, receipt.sha256);
        if !root_path.join(&name).exists() {
            write_sealed_object(root_path, &name, &component_bytes(role, sequence));
        }
    }
    write_manifest_object(root_path, id, &bytes);
    let authority = AuthorityRefV1::new(
        sequence,
        id,
        len,
        sha256,
        manifest.predecessor().map(|p| p.fingerprint()),
    )
    .expect("authority reference");
    let outcome = publisher
        .publish(authority, expected, floor, object_id(sequence ^ 0xa5))
        .expect("publication runs");
    let PublicationOutcomeV1::Committed { slot, .. } = outcome else {
        panic!("sequence {sequence} must commit: {outcome:?}");
    };
    (authority, slot)
}

pub(super) fn open(
    root: &QualifiedGenerationRoot,
    profile: GenerationRootSecurityProfileV1,
    floor: Option<&dyn AntiRollbackFloorProviderV1>,
) -> SnapshotOpenOutcomeV1 {
    root.open_generation_snapshot(ROOT_ID, profile, floor)
        .expect("open runs without infrastructure faults")
}

pub(super) fn refused(outcome: SnapshotOpenOutcomeV1) -> SnapshotRefusalV1 {
    match outcome {
        SnapshotOpenOutcomeV1::Refused(refusal) => refusal,
        SnapshotOpenOutcomeV1::Opened(snapshot) => {
            panic!("expected a typed refusal, got {snapshot:?}")
        }
    }
}

pub(super) fn opened(outcome: SnapshotOpenOutcomeV1) -> Arc<OpenedGenerationSnapshotV1> {
    match outcome {
        SnapshotOpenOutcomeV1::Opened(snapshot) => snapshot,
        SnapshotOpenOutcomeV1::Refused(refusal) => {
            panic!("expected an admitted snapshot, got {refusal:?}")
        }
    }
}

#[test]
fn activation_manifest_names_are_confined_hex_with_suffix() {
    let path = activation_manifest_path_v1([0xab; 16]).expect("confined");
    assert_eq!(path.component_count(), 1);
    let path = activation_manifest_path_v1(object_id(1)).expect("confined");
    assert_eq!(path.component_count(), 1);
}

#[test]
fn fresh_open_serves_the_published_head_and_retains_the_exact_manifest() {
    let root_path = fixture_root("reader-genesis");
    let root = admit(&root_path);
    let floor = InMemoryAntiRollbackFloorStoreV1::new();
    let publisher = publisher(&root);

    // Empty root: nothing to serve, typed.
    assert_eq!(
        refused(open(
            &root,
            GenerationRootSecurityProfileV1::CooperativeLocal,
            None
        )),
        SnapshotRefusalV1::NoAuthority
    );

    let (genesis, genesis_manifest, genesis_slot) = publish_with_manifest(
        &root_path,
        &publisher,
        1,
        None,
        ExpectedAuthorityPairV1::default(),
        Some(&floor),
    );

    let snapshot = opened(open(
        &root,
        GenerationRootSecurityProfileV1::RequiredExternal,
        Some(&floor),
    ));
    assert_eq!(snapshot.head(), genesis_slot);
    assert_eq!(snapshot.head().authority, genesis);
    assert_eq!(snapshot.manifest(), &genesis_manifest);
    assert_eq!(
        &*snapshot.manifest_bytes(),
        genesis_manifest.canonical_bytes().as_slice()
    );
    assert_eq!(
        snapshot.health(),
        SnapshotHealthV1::ExternallyAnchored { cas_version: 1 }
    );
    assert_eq!(snapshot.floor().map(|r| r.authority), Some(genesis));
    assert_eq!(snapshot.root_id(), ROOT_ID);
    assert_eq!(snapshot.authority_pair().second, Some(genesis_slot));
    assert_eq!(snapshot.root_witness(), root.witness());
    // Anchor witnesses are captured under the final shared lock; the
    // reads themselves advance atime, so compare the identity, not
    // the full timestamp tuple.
    assert_eq!(
        snapshot.anchor_witnesses().lock().inode(),
        root.anchor_witnesses().lock().inode()
    );

    // The same root under a local profile is served, but explicitly unanchored.
    let local = opened(open(
        &root,
        GenerationRootSecurityProfileV1::CooperativeLocal,
        None,
    ));
    assert_eq!(local.head(), genesis_slot);
    assert_eq!(
        local.health(),
        SnapshotHealthV1::Unanchored {
            profile: GenerationRootSecurityProfileV1::CooperativeLocal
        }
    );
    let inspect = opened(open(
        &root,
        GenerationRootSecurityProfileV1::ReadOnlyUnanchored,
        Some(&floor),
    ));
    assert_eq!(inspect.floor().map(|r| r.cas_version), Some(1));
    assert_eq!(
        inspect.health(),
        SnapshotHealthV1::Unanchored {
            profile: GenerationRootSecurityProfileV1::ReadOnlyUnanchored
        }
    );

    // RequiredExternal never downgrades to local when no floor is injected.
    assert_eq!(
        refused(open(
            &root,
            GenerationRootSecurityProfileV1::RequiredExternal,
            None
        )),
        SnapshotRefusalV1::Authority(GenerationAuthorityErrorV1::ExternalFloorRequired)
    );
    // A zero root id is refused before any I/O; a foreign root id
    // cannot authenticate the slot.
    assert!(matches!(
        root.open_generation_snapshot(
            [0; 16],
            GenerationRootSecurityProfileV1::CooperativeLocal,
            None
        )
        .expect("runs"),
        SnapshotOpenOutcomeV1::Refused(SnapshotRefusalV1::Authority(
            GenerationAuthorityErrorV1::InvalidField { .. }
        ))
    ));
    assert!(matches!(
        root.open_generation_snapshot(
            [0x99; 16],
            GenerationRootSecurityProfileV1::CooperativeLocal,
            None
        )
        .expect("runs"),
        SnapshotOpenOutcomeV1::Refused(SnapshotRefusalV1::Authority(_))
    ));
}

#[test]
fn retained_snapshot_is_byte_exact_across_a_successor_publication() {
    let root_path = fixture_root("reader-successor");
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
    let old_bytes = old.manifest_bytes();
    let old_witness = old.manifest_file().witness();

    let (successor, successor_manifest, successor_slot) = publish_with_manifest(
        &root_path,
        &publisher,
        2,
        Some(genesis),
        ExpectedAuthorityPairV1 {
            first: None,
            second: Some(genesis_slot),
        },
        Some(&floor),
    );

    // The retained snapshot is untouched by the publication.
    assert_eq!(old.head(), genesis_slot);
    assert_eq!(old.manifest(), &genesis_manifest);
    assert_eq!(&*old.manifest_bytes(), &*old_bytes);
    assert_eq!(old.manifest_file().witness(), old_witness);
    assert_eq!(
        old.health(),
        SnapshotHealthV1::ExternallyAnchored { cas_version: 1 }
    );

    // A fresh open serves the successor with both slots retained.
    let new = opened(open(
        &root,
        GenerationRootSecurityProfileV1::RequiredExternal,
        Some(&floor),
    ));
    assert_eq!(new.head(), successor_slot);
    assert_eq!(new.head().authority, successor);
    assert_eq!(new.manifest(), &successor_manifest);
    assert_eq!(
        new.health(),
        SnapshotHealthV1::ExternallyAnchored { cas_version: 2 }
    );
    assert_eq!(
        new.authority_pair(),
        ExpectedAuthorityPairV1 {
            first: Some(successor_slot),
            second: Some(genesis_slot),
        }
    );
    // Old and new coexist; neither borrows the other's bytes.
    assert_ne!(&*old.manifest_bytes(), &*new.manifest_bytes());
}

#[test]
fn closure_faults_are_typed_and_never_invalidate_an_earlier_snapshot() {
    let root_path = fixture_root("reader-closure");
    let root = admit(&root_path);
    let publisher = publisher(&root);
    let (genesis, _, genesis_slot) = publish_with_manifest(
        &root_path,
        &publisher,
        1,
        None,
        ExpectedAuthorityPairV1::default(),
        None,
    );
    let retained = opened(open(
        &root,
        GenerationRootSecurityProfileV1::CooperativeLocal,
        None,
    ));

    // Successor whose manifest object was never written: the slot
    // commits (the publisher does not open objects), the reader
    // refuses on the missing closure and keeps zero state.
    let missing_manifest = manifest(2, Some(genesis));
    let (len, sha256) = missing_manifest.object_receipt();
    let missing = AuthorityRefV1::new(2, object_id(2), len, sha256, Some(genesis.fingerprint()))
        .expect("authority");
    let outcome = publisher
        .publish(
            missing,
            ExpectedAuthorityPairV1 {
                first: None,
                second: Some(genesis_slot),
            },
            None,
            [0x52; 16],
        )
        .expect("publication runs");
    assert!(matches!(outcome, PublicationOutcomeV1::Committed { .. }));
    assert!(matches!(
        refused(open(
            &root,
            GenerationRootSecurityProfileV1::CooperativeLocal,
            None
        )),
        SnapshotRefusalV1::Closure(_)
    ));

    // Object present but one byte longer than the authority says.
    let mut longer = missing_manifest.canonical_bytes();
    longer.push(0);
    write_manifest_object(&root_path, object_id(2), &longer);
    assert!(matches!(
        refused(open(
            &root,
            GenerationRootSecurityProfileV1::CooperativeLocal,
            None
        )),
        SnapshotRefusalV1::Closure(_)
    ));

    // Exact bytes (length and digest match the reference) of a
    // manifest sealed for a different authority: the object holds a
    // sequence-4 manifest while the reference claims sequence 3, so
    // the closure admits and the reference check refuses.
    let three = AuthorityRefV1::new(
        3,
        [0x3c; 16],
        4_096,
        [0x3d; 32],
        Some(missing.fingerprint()),
    )
    .expect("authority");
    let wrong_manifest = manifest(4, Some(three));
    let (wrong_len, wrong_sha256) = wrong_manifest.object_receipt();
    write_manifest_object(&root_path, object_id(3), &wrong_manifest.canonical_bytes());
    let mismatched = AuthorityRefV1::new(
        3,
        object_id(3),
        wrong_len,
        wrong_sha256,
        Some(missing.fingerprint()),
    )
    .expect("authority");
    let outcome = publisher
        .publish(
            mismatched,
            ExpectedAuthorityPairV1 {
                first: Some(AuthoritySlotV1 {
                    slot_index: 0,
                    root_id: ROOT_ID,
                    authority: missing,
                }),
                second: Some(genesis_slot),
            },
            None,
            [0x53; 16],
        )
        .expect("publication runs");
    assert!(matches!(outcome, PublicationOutcomeV1::Committed { .. }));
    assert_eq!(
        refused(open(
            &root,
            GenerationRootSecurityProfileV1::CooperativeLocal,
            None
        )),
        SnapshotRefusalV1::Manifest(GenerationAuthorityErrorV1::ManifestReferenceMismatch)
    );

    // Through all of it the earlier snapshot is exactly what it was.
    assert_eq!(retained.head(), genesis_slot);
    assert_eq!(retained.manifest().authority_sequence(), 1);
}

#[test]
fn pending_attempt_without_owner_frame_fails_closed_until_a_commit() {
    let root_path = fixture_root("reader-pending-attempt");
    let root = admit(&root_path);
    let floor = InMemoryAntiRollbackFloorStoreV1::new();
    let publisher = publisher(&root);
    let (genesis, _, genesis_slot) = publish_with_manifest(
        &root_path,
        &publisher,
        1,
        None,
        ExpectedAuthorityPairV1::default(),
        Some(&floor),
    );

    // A publisher dies right after persisting its attempt frame.
    let successor_manifest = manifest(2, Some(genesis));
    let (len, sha256) = successor_manifest.object_receipt();
    write_component_objects(&root_path, 2);
    write_manifest_object(
        &root_path,
        object_id(2),
        &successor_manifest.canonical_bytes(),
    );
    let successor = AuthorityRefV1::new(2, object_id(2), len, sha256, Some(genesis.fingerprint()))
        .expect("authority");
    let expected = ExpectedAuthorityPairV1 {
        first: None,
        second: Some(genesis_slot),
    };
    let permit = {
        let _hook = platform::install_test_hook(|boundary| {
            if boundary == platform::TestBoundary::AfterPublishAttemptFrame {
                Err(GenerationRootError::new(
                    GenerationRootErrorKind::Io,
                    GenerationRootStage::SyncRegularFile,
                ))
            } else {
                Ok(())
            }
        });
        match publisher
            .publish(successor, expected, Some(&floor), [0x61; 16])
            .expect("publication runs")
        {
            PublicationOutcomeV1::CommitOutcomeUnknown(permit) => permit,
            other => panic!("crash must be unknown: {other:?}"),
        }
    };

    // Readers refuse: the old pair is intact but an attempt is unresolved.
    assert_eq!(
        refused(open(
            &root,
            GenerationRootSecurityProfileV1::RequiredExternal,
            Some(&floor)
        )),
        SnapshotRefusalV1::PendingReconciliation
    );
    assert_eq!(
        refused(open(
            &root,
            GenerationRootSecurityProfileV1::ReadOnlyUnanchored,
            None
        )),
        SnapshotRefusalV1::PendingReconciliation
    );

    // Reconciliation says NotCommitted; the attempt frame stays until
    // the next completed publication, so readers keep failing closed.
    assert_eq!(
        publisher
            .reconcile(&permit, Some(&floor))
            .expect("reconcile runs"),
        ReconcileOutcomeV1::NotCommitted
    );
    assert_eq!(
        refused(open(
            &root,
            GenerationRootSecurityProfileV1::RequiredExternal,
            Some(&floor)
        )),
        SnapshotRefusalV1::PendingReconciliation
    );

    // The retried publication commits and unblocks fresh readers.
    let outcome = publisher
        .publish(successor, expected, Some(&floor), [0x62; 16])
        .expect("publication runs");
    let PublicationOutcomeV1::Committed { slot, .. } = outcome else {
        panic!("retry must commit: {outcome:?}");
    };
    let snapshot = opened(open(
        &root,
        GenerationRootSecurityProfileV1::RequiredExternal,
        Some(&floor),
    ));
    assert_eq!(snapshot.head(), slot);
    assert_eq!(snapshot.manifest(), &successor_manifest);
    assert_eq!(
        snapshot.health(),
        SnapshotHealthV1::ExternallyAnchored { cas_version: 2 }
    );
}

/// A floor provider that answers `load` with an older retained record
/// (external store restored from backup) while the local anchors have
/// already advanced.
struct BehindFloor {
    live: InMemoryAntiRollbackFloorStoreV1,
    report: std::sync::Mutex<Option<AntiRollbackFloorRecordV1>>,
}

impl AntiRollbackFloorProviderV1 for BehindFloor {
    fn load(
        &self,
        root_id: [u8; 16],
    ) -> Result<Option<AntiRollbackFloorRecordV1>, GenerationAuthorityErrorV1> {
        let pinned = *self.report.lock().expect("report lock");
        pinned.map_or_else(|| self.live.load(root_id), |record| Ok(Some(record)))
    }

    fn compare_and_advance(
        &self,
        expected: Option<AntiRollbackFloorRecordV1>,
        next: frankensearch_core::generation::AuthorityFloorV1,
        idempotency_key: [u8; 16],
    ) -> Result<AntiRollbackFloorRecordV1, GenerationAuthorityErrorV1> {
        self.live
            .compare_and_advance(expected, next, idempotency_key)
    }
}

#[test]
fn external_floor_behind_the_local_head_serves_the_anchored_predecessor() {
    let root_path = fixture_root("reader-floor-behind");
    let root = admit(&root_path);
    let floor = BehindFloor {
        live: InMemoryAntiRollbackFloorStoreV1::new(),
        report: std::sync::Mutex::new(None),
    };
    let publisher = publisher(&root);
    let (genesis, genesis_manifest, genesis_slot) = publish_with_manifest(
        &root_path,
        &publisher,
        1,
        None,
        ExpectedAuthorityPairV1::default(),
        Some(&floor),
    );
    let genesis_record = floor
        .live
        .load(ROOT_ID)
        .expect("floor loads")
        .expect("floor holds genesis");
    let (successor, _, successor_slot) = publish_with_manifest(
        &root_path,
        &publisher,
        2,
        Some(genesis),
        ExpectedAuthorityPairV1 {
            first: None,
            second: Some(genesis_slot),
        },
        Some(&floor),
    );

    // External store rolls back to the genesis record.
    *floor.report.lock().expect("report lock") = Some(genesis_record);
    let snapshot = opened(open(
        &root,
        GenerationRootSecurityProfileV1::RequiredExternal,
        Some(&floor),
    ));
    assert_eq!(
        snapshot.head(),
        genesis_slot,
        "the externally proven head is served, not the unproven successor"
    );
    assert_eq!(snapshot.manifest(), &genesis_manifest);
    assert_eq!(
        snapshot.health(),
        SnapshotHealthV1::PendingLocalSuccessor { pending: successor }
    );
    assert_eq!(snapshot.authority_pair().first, Some(successor_slot));

    // Once the external store catches up again, the successor is anchored.
    *floor.report.lock().expect("report lock") = None;
    let caught_up = opened(open(
        &root,
        GenerationRootSecurityProfileV1::RequiredExternal,
        Some(&floor),
    ));
    assert_eq!(caught_up.head(), successor_slot);
    assert_eq!(
        caught_up.health(),
        SnapshotHealthV1::ExternallyAnchored { cas_version: 2 }
    );

    // A floor naming a head that neither slot retains fails closed.
    let foreign = AuthorityRefV1::new(1, [0x77; 16], 4_096, [0x78; 32], None).expect("authority");
    let foreign_floor = InMemoryAntiRollbackFloorStoreV1::new();
    foreign_floor
        .compare_and_advance(
            None,
            frankensearch_core::generation::AuthorityFloorV1::new(ROOT_ID, foreign).expect("floor"),
            [0x79; 16],
        )
        .expect("foreign floor advances");
    assert!(matches!(
        refused(open(
            &root,
            GenerationRootSecurityProfileV1::RequiredExternal,
            Some(&foreign_floor)
        )),
        SnapshotRefusalV1::Authority(_)
    ));
}

#[test]
fn inventory_lists_exact_entries_sorted_with_identity_and_no_pathname_adoption() {
    use crate::generation_root::GenerationRootEntryKind;
    use std::ffi::OsStr;

    let root_path = fixture_root("inventory-exact");
    let root = admit(&root_path);
    let manifest = manifest(1, None);
    let bytes = manifest.canonical_bytes();
    let manifest_path = write_manifest_object(&root_path, object_id(1), &bytes);
    let manifest_name = manifest_path
        .file_name()
        .expect("manifest name")
        .to_os_string();

    let inventory = root.inventory().expect("inventory scans");
    let names: Vec<&OsStr> = inventory.entries().iter().map(|e| e.name()).collect();
    let mut expected = vec![
        OsStr::new("AUTHORITY"),
        OsStr::new("LOCK"),
        manifest_name.as_os_str(),
    ];
    expected.sort();
    assert_eq!(names, expected, "sorted, exact, nothing else");
    for entry in inventory.entries() {
        assert_eq!(entry.kind(), GenerationRootEntryKind::RegularFile);
        assert_eq!(entry.hard_links(), 1);
        assert_eq!(entry.device(), root.witness().device());
    }
    let authority = inventory
        .entry(OsStr::new("AUTHORITY"))
        .expect("AUTHORITY listed");
    assert_eq!(authority.byte_len(), GENERATION_ROOT_AUTHORITY_FILE_BYTES);
    assert_eq!(
        authority.inode(),
        root.anchor_witnesses().authority().inode()
    );
    assert_eq!(
        inventory
            .entry(OsStr::new("LOCK"))
            .expect("LOCK listed")
            .inode(),
        root.anchor_witnesses().lock().inode()
    );
    assert_eq!(
        inventory
            .entry(&manifest_name)
            .expect("manifest listed")
            .byte_len(),
        u64::try_from(bytes.len()).expect("len")
    );
    assert!(inventory.entry(OsStr::new("missing")).is_none());

    // Closure helpers: the manifest is the only undeclared residue
    // relative to the anchors; no aliases; nothing off-device.
    let undeclared = inventory.undeclared(&[OsStr::new("LOCK"), OsStr::new("AUTHORITY")]);
    assert_eq!(undeclared.len(), 1);
    assert_eq!(undeclared[0].name(), manifest_name.as_os_str());
    assert!(inventory.aliases().is_empty());
    assert!(inventory.foreign_device(root.witness().device()).is_empty());

    // Stable across repeats and identical under the shared lock.
    assert_eq!(root.inventory().expect("second scan"), inventory);
    let guard = root.read_guard().expect("shared guard");
    assert_eq!(guard.inventory().expect("locked scan"), inventory);
    guard.release().expect("release");
}

/// A committed genesis with every object written, ready for one
/// planted negative.
pub(super) fn genesis_root(label: &str) -> (PathBuf, QualifiedGenerationRoot, AuthoritySlotV1) {
    let root_path = fixture_root(label);
    let root = admit(&root_path);
    let slot = {
        let publisher = publisher(&root);
        let (_, _, slot) = publish_with_manifest(
            &root_path,
            &publisher,
            1,
            None,
            ExpectedAuthorityPairV1::default(),
            None,
        );
        slot
    };
    (root_path, root, slot)
}

mod cell_and_soak;
mod closure;
