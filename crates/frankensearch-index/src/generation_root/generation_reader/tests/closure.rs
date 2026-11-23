use super::super::super::authority_publisher::ExpectedAuthorityPairV1;
use super::super::{
    ClosureViolationV1, GenerationSnapshotCellV1, SnapshotHealthV1, SnapshotRefusalV1,
    activation_manifest_name_v1, component_object_name_v1, is_retained_object_name_v1,
};
use super::*;
use crate::generation_root::{GenerationRootEntryKind, GenerationRootErrorKind};
use frankensearch_core::generation::{
    ActivationManifest, GenerationComponentRole, GenerationRootSecurityProfileV1,
    InMemoryAntiRollbackFloorStoreV1,
};
use sha2::{Digest as _, Sha256};
use std::fs::{self, DirBuilder};
use std::os::unix::fs::DirBuilderExt;
use std::sync::Arc;

#[test]
fn inventory_reports_aliases_symlinks_and_subdirectories_without_following() {
    use crate::generation_root::GenerationRootEntryKind;
    use std::ffi::OsStr;

    let root_path = fixture_root("inventory-planted");
    let root = admit(&root_path);
    let manifest = manifest(1, None);
    let manifest_path =
        write_manifest_object(&root_path, object_id(1), &manifest.canonical_bytes());
    let manifest_name = manifest_path
        .file_name()
        .expect("manifest name")
        .to_os_string();
    let clean = root.inventory().expect("control scan");
    assert!(clean.aliases().is_empty());

    // Planted alone, each with the clean control above.
    fs::hard_link(&manifest_path, root_path.join("alias")).expect("hard link");
    std::os::unix::fs::symlink("LOCK", root_path.join("link")).expect("symlink");
    DirBuilder::new()
        .mode(0o700)
        .create(root_path.join("nested"))
        .expect("subdirectory");

    let planted = root.inventory().expect("planted scan");
    assert_eq!(planted.entries().len(), clean.entries().len() + 3);

    let aliases = planted.aliases();
    assert_eq!(aliases.len(), 1, "exactly one alias group");
    let mut alias_names: Vec<&OsStr> = aliases[0].iter().map(|e| e.name()).collect();
    alias_names.sort();
    assert_eq!(
        alias_names,
        vec![manifest_name.as_os_str(), OsStr::new("alias")]
    );
    assert_eq!(
        planted
            .entry(&manifest_name)
            .expect("manifest")
            .hard_links(),
        2
    );

    let link = planted.entry(OsStr::new("link")).expect("symlink listed");
    assert_eq!(link.kind(), GenerationRootEntryKind::Symlink);
    assert_ne!(
        link.inode(),
        planted.entry(OsStr::new("LOCK")).expect("LOCK").inode(),
        "the link is identified, never followed"
    );
    assert_eq!(
        planted.entry(OsStr::new("nested")).expect("dir").kind(),
        GenerationRootEntryKind::Directory
    );

    // The undeclared residue names every planted entry.
    let mut undeclared: Vec<&OsStr> = planted
        .undeclared(&[
            OsStr::new("LOCK"),
            OsStr::new("AUTHORITY"),
            manifest_name.as_os_str(),
        ])
        .into_iter()
        .map(|e| e.name())
        .collect();
    undeclared.sort();
    assert_eq!(
        undeclared,
        vec![
            OsStr::new("alias"),
            OsStr::new("link"),
            OsStr::new("nested")
        ]
    );
}
#[test]
fn closure_admits_every_declared_component_at_its_exact_receipt() {
    let (_, root, slot) = genesis_root("closure-clean");
    let snapshot = opened(open(
        &root,
        GenerationRootSecurityProfileV1::CooperativeLocal,
        None,
    ));
    assert_eq!(snapshot.head(), slot);
    let receipts = snapshot.manifest().components();
    assert!(matches!(snapshot.manifest(), ActivationManifest::V1(_)));
    for (role, receipt) in receipts.iter() {
        let file = snapshot.closure().file(role).expect("declared component");
        assert_eq!(file.sha256(), receipt.sha256, "{role:?} digest");
        assert_eq!(
            u64::try_from(file.as_bytes().len()).expect("len"),
            receipt.byte_len,
            "{role:?} length"
        );
        assert_eq!(
            &*snapshot.closure().bytes(role).expect("declared bytes"),
            &*component_bytes(role, 1)
        );
        assert_eq!(file.witness().hard_links(), 1);
    }
    assert!(is_retained_object_name_v1(
        component_object_name_v1(
            GenerationComponentRole::Ann,
            receipts.ann.expect("v1 requires ANN").sha256
        )
        .as_bytes()
    ));
    assert!(is_retained_object_name_v1(
        activation_manifest_name_v1(object_id(1)).as_bytes()
    ));
    for bad in [
        "vector",
        ".vector",
        "abc.vector",
        "0123456789abcdef0123456789abcdef.vector",
        "0123456789ABCDEF0123456789ABCDEF.activation",
        "LOCK",
    ] {
        assert!(!is_retained_object_name_v1(bad.as_bytes()), "{bad}");
    }
}

#[test]
fn ann_and_exact_transitions_preserve_versioned_manifests_and_old_closures() {
    let root_path = fixture_root("closure-optional-ann");
    let root = admit(&root_path);
    let floor = InMemoryAntiRollbackFloorStoreV1::new();
    let publisher = publisher(&root);
    let mut predecessor = None;
    let mut expected = ExpectedAuthorityPairV1::default();
    let mut retained = Vec::new();

    // V1 ANN -> V2 exact -> V2 ANN -> V2 exact exercises both
    // directions without deleting any previous graph or manifest.
    for sequence in 1..=4 {
        let selected = if sequence == 1 {
            manifest(sequence, predecessor)
        } else {
            manifest_v2(sequence, predecessor, sequence == 3)
        };
        let (authority, slot) =
            publish_manifest(&root_path, &publisher, &selected, expected, Some(&floor));
        if slot.slot_index == 0 {
            expected.first = Some(slot);
        } else {
            expected.second = Some(slot);
        }
        predecessor = Some(authority);
        let floor_before = floor.load(ROOT_ID).expect("floor before read");
        let authority_before = fs::read(root_path.join("AUTHORITY")).expect("authority");
        let lock_before = fs::read(root_path.join("LOCK")).expect("lock");

        // Every admission uses independently opened root descriptors.
        let fresh_root = admit(&root_path);
        let snapshot = opened(open(
            &fresh_root,
            GenerationRootSecurityProfileV1::RequiredExternal,
            Some(&floor),
        ));
        assert_eq!(snapshot.head(), slot);
        assert_eq!(snapshot.manifest(), &selected);
        assert_eq!(
            snapshot.manifest_bytes().as_ref(),
            selected.canonical_bytes().as_slice()
        );
        assert_eq!(
            snapshot.health(),
            SnapshotHealthV1::ExternallyAnchored {
                cas_version: sequence
            }
        );
        assert_eq!(floor.load(ROOT_ID).expect("floor after read"), floor_before);
        assert_eq!(
            fs::read(root_path.join("AUTHORITY")).unwrap(),
            authority_before
        );
        assert_eq!(fs::read(root_path.join("LOCK")).unwrap(), lock_before);
        retained.push(snapshot);

        for snapshot in &retained {
            let sequence = snapshot.head().authority.sequence;
            for role in ROLES {
                match snapshot.manifest().components().get(role) {
                    Some(receipt) => {
                        let file = snapshot.closure().file(role).expect("declared role");
                        assert_eq!(file.sha256(), receipt.sha256);
                        assert_eq!(
                            snapshot.closure().bytes(role).unwrap().as_ref(),
                            component_bytes(role, sequence).as_slice()
                        );
                    }
                    None => {
                        assert_eq!(role, GenerationComponentRole::Ann);
                        assert!(snapshot.closure().file(role).is_none());
                        assert!(snapshot.closure().bytes(role).is_none());
                    }
                }
            }
        }
    }

    for sequence in [1, 3] {
        assert!(
            root_path
                .join(component_object_name_v1(
                    GenerationComponentRole::Ann,
                    component_receipt(GenerationComponentRole::Ann, sequence).sha256,
                ))
                .exists(),
            "old ANN objects remain retained while exact generations serve"
        );
    }
}

#[test]
fn declared_v2_ann_failure_never_degrades_or_replaces_an_exact_reader() {
    for missing in [false, true] {
        let root_path = fixture_root("closure-declared-v2-ann-failure");
        let root = admit(&root_path);
        let floor = InMemoryAntiRollbackFloorStoreV1::new();
        let publisher = publisher(&root);
        let exact = manifest_v2(1, None, false);
        let (genesis, genesis_slot) = publish_manifest(
            &root_path,
            &publisher,
            &exact,
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
            .expect("admit exact genesis"),
        );
        assert!(old.closure().file(GenerationComponentRole::Ann).is_none());
        assert_eq!(
            root.inventory().expect("exact inventory").entries().len(),
            6,
            "exact genesis has two anchors, its manifest, and three components"
        );

        let selected = manifest_v2(2, Some(genesis), true);
        let receipt = selected.components().ann.expect("explicit ANN selection");
        let name = component_object_name_v1(GenerationComponentRole::Ann, receipt.sha256);
        if !missing {
            let mut bytes = component_bytes(GenerationComponentRole::Ann, 2);
            bytes[50] ^= 1;
            write_sealed_object(&root_path, &name, &bytes);
        }
        // The low-level authority publisher does not parse engines;
        // plant a committed bad closure to exercise fresh admission.
        publish_manifest(
            &root_path,
            &publisher,
            &selected,
            ExpectedAuthorityPairV1 {
                first: None,
                second: Some(genesis_slot),
            },
            Some(&floor),
        );
        if missing {
            fs::rename(
                root_path.join(&name),
                root_path.join(component_object_name_v1(
                    GenerationComponentRole::Ann,
                    [0xe5; 32],
                )),
            )
            .expect("retain graph under a different name");
        }
        let floor_before = floor.load(ROOT_ID).expect("committed floor");
        let authority_before = fs::read(root_path.join("AUTHORITY")).expect("authority");
        let lock_before = fs::read(root_path.join("LOCK")).expect("lock");
        let fresh_root = admit(&root_path);
        let refusal = refused(
            cell.refresh(
                &fresh_root,
                ROOT_ID,
                GenerationRootSecurityProfileV1::RequiredExternal,
                Some(&floor),
            )
            .expect("refresh returns a typed closure refusal"),
        );
        // A declared name that is absent never reaches open(2): the
        // exact-name enumeration (verify_exact_directory_entry) finds
        // zero entries for it and reports ObjectChanged, 1 expected, 0
        // observed.
        assert_eq!(
            refusal,
            SnapshotRefusalV1::Closure(if missing {
                GenerationRootErrorKind::ObjectChanged
            } else {
                GenerationRootErrorKind::HashMismatch
            })
        );
        assert!(Arc::ptr_eq(
            &cell.load().expect("prior snapshot remains"),
            &old
        ));
        assert_eq!(old.manifest(), &exact);
        assert!(old.closure().bytes(GenerationComponentRole::Ann).is_none());
        assert_eq!(
            old.closure()
                .bytes(GenerationComponentRole::Vector)
                .unwrap()
                .as_ref(),
            component_bytes(GenerationComponentRole::Vector, 1).as_slice()
        );
        assert_eq!(floor.load(ROOT_ID).expect("unchanged floor"), floor_before);
        assert_eq!(
            fs::read(root_path.join("AUTHORITY")).unwrap(),
            authority_before
        );
        assert_eq!(fs::read(root_path.join("LOCK")).unwrap(), lock_before);
    }
}

#[test]
fn tampered_component_object_is_refused_by_role_before_any_use() {
    let root_path = fixture_root("closure-tamper");
    let root = admit(&root_path);
    let publisher = publisher(&root);
    // Write a lexical object with the right name but one flipped byte
    // BEFORE the generation writes its real objects, so the
    // content-addressed name is taken by the wrong bytes.
    let mut wrong = component_bytes(GenerationComponentRole::Lexical, 1);
    let name = component_object_name_v1(
        GenerationComponentRole::Lexical,
        Sha256::digest(&wrong).into(),
    );
    wrong[50] ^= 1;
    write_sealed_object(&root_path, &name, &wrong);
    publish_with_manifest(
        &root_path,
        &publisher,
        1,
        None,
        ExpectedAuthorityPairV1::default(),
        None,
    );
    assert_eq!(
        refused(open(
            &root,
            GenerationRootSecurityProfileV1::CooperativeLocal,
            None
        )),
        SnapshotRefusalV1::Closure(GenerationRootErrorKind::HashMismatch)
    );
}

#[test]
fn a_write_ahead_log_beside_a_sealed_object_is_refused() {
    let (root_path, root, _) = genesis_root("closure-wal");
    opened(open(
        &root,
        GenerationRootSecurityProfileV1::CooperativeLocal,
        None,
    ));
    let vector = component_object_name_v1(
        GenerationComponentRole::Vector,
        Sha256::digest(component_bytes(GenerationComponentRole::Vector, 1)).into(),
    );
    write_sealed_object(&root_path, &format!("{vector}.wal"), b"journal");
    assert_eq!(
        refused(open(
            &root,
            GenerationRootSecurityProfileV1::CooperativeLocal,
            None
        )),
        SnapshotRefusalV1::ClosureViolation(ClosureViolationV1::WriteAheadLog)
    );
}

#[test]
fn an_undeclared_entry_is_refused_and_named_by_kind() {
    let (root_path, root, _) = genesis_root("closure-undeclared-file");
    opened(open(
        &root,
        GenerationRootSecurityProfileV1::CooperativeLocal,
        None,
    ));
    write_sealed_object(&root_path, "notes.txt", b"not an object");
    assert_eq!(
        refused(open(
            &root,
            GenerationRootSecurityProfileV1::CooperativeLocal,
            None
        )),
        SnapshotRefusalV1::ClosureViolation(ClosureViolationV1::UndeclaredEntry {
            kind: GenerationRootEntryKind::RegularFile
        })
    );

    let (root_path, root, _) = genesis_root("closure-undeclared-symlink");
    std::os::unix::fs::symlink("LOCK", root_path.join("link")).expect("symlink");
    assert_eq!(
        refused(open(
            &root,
            GenerationRootSecurityProfileV1::CooperativeLocal,
            None
        )),
        SnapshotRefusalV1::ClosureViolation(ClosureViolationV1::UndeclaredEntry {
            kind: GenerationRootEntryKind::Symlink
        })
    );

    let (root_path, root, _) = genesis_root("closure-undeclared-dir");
    DirBuilder::new()
        .mode(0o700)
        .create(root_path.join("nested"))
        .expect("subdirectory");
    assert_eq!(
        refused(open(
            &root,
            GenerationRootSecurityProfileV1::CooperativeLocal,
            None
        )),
        SnapshotRefusalV1::ClosureViolation(ClosureViolationV1::UndeclaredEntry {
            kind: GenerationRootEntryKind::Directory
        })
    );
}

#[test]
fn aliases_are_refused_whether_declared_or_retained() {
    // A second name for a declared component: admission sees the
    // extra link before the closure does.
    let (root_path, root, _) = genesis_root("closure-alias-declared");
    let vector = component_object_name_v1(
        GenerationComponentRole::Vector,
        Sha256::digest(component_bytes(GenerationComponentRole::Vector, 1)).into(),
    );
    let alias = component_object_name_v1(GenerationComponentRole::Vector, [0xee; 32]);
    fs::hard_link(root_path.join(&vector), root_path.join(&alias)).expect("hard link");
    assert_eq!(
        refused(open(
            &root,
            GenerationRootSecurityProfileV1::CooperativeLocal,
            None
        )),
        SnapshotRefusalV1::Closure(GenerationRootErrorKind::HardLinked)
    );

    // Two retained-shaped names for one undeclared inode.
    let (root_path, root, _) = genesis_root("closure-alias-retained");
    let first = component_object_name_v1(GenerationComponentRole::Ann, [0xe1; 32]);
    let second = component_object_name_v1(GenerationComponentRole::Ann, [0xe2; 32]);
    write_sealed_object(&root_path, &first, b"retained");
    fs::hard_link(root_path.join(&first), root_path.join(&second)).expect("hard link");
    assert_eq!(
        refused(open(
            &root,
            GenerationRootSecurityProfileV1::CooperativeLocal,
            None
        )),
        SnapshotRefusalV1::ClosureViolation(ClosureViolationV1::Alias)
    );
}

#[test]
fn retained_objects_of_other_generations_are_not_violations() {
    let root_path = fixture_root("closure-retained-history");
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
    // Genesis objects stay beside the successor's; a well-formed
    // retained object nobody references any more is fine too.
    let orphan = component_object_name_v1(GenerationComponentRole::Metadata, [0xd1; 32]);
    write_sealed_object(&root_path, &orphan, b"orphaned but sealed");
    let snapshot = opened(open(
        &root,
        GenerationRootSecurityProfileV1::CooperativeLocal,
        None,
    ));
    assert_eq!(snapshot.head().authority.sequence, 2);
    assert_eq!(
        root.inventory().expect("inventory").entries().len(),
        2 + 2 + 8 + 1,
        "anchors + two manifests + eight components + orphan"
    );
}
