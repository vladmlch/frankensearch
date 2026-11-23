use super::*;

// ─── Attested vs declared identity discriminator (bd-9xuj T2-C4-write) ──

#[test]
fn admitted_v2_open_reports_header_attested_identity() {
    // Isolated dir: exact admission snapshots the containing directory
    // and fails closed when sibling files churn it, so the shared test
    // temp dir is not a legal admission site (see 58726e26).
    let dir = temp_index_dir("admitted-v2-attested");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
    let (fast_binding, fast_identity) = fsvi_v2_binding("attested-fast-model", 4, 3);
    let (quality_binding, quality_identity) = fsvi_v2_binding("attested-quality-model", 4, 3);
    let rows: [(&str, &[f32]); 2] = [
        ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
        ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
    ];
    write_v2_tier(&fast_path, &fast_binding, &rows);
    write_v2_tier(&quality_path, &quality_binding, &rows);

    // Plain discovery/open can never reach a v2 tier: VectorIndex::open
    // is strictly v1, so the attested state is unreachable except through
    // exact admission.
    let plain = TwoTierIndex::open(&dir, TwoTierConfig::default());
    assert!(
        matches!(plain, Err(SearchError::IndexVersionMismatch { .. })),
        "plain open must reject v2 bytes, got {plain:?}"
    );

    let paths = TwoTierIndexPaths::new(&fast_path).with_quality_index(&quality_path);
    let index = TwoTierIndex::open_admitted_v2_with_paths(
        &paths,
        TwoTierConfig::default(),
        &fast_binding,
        Some(&quality_binding),
    )
    .expect("admit both v2 tiers");

    assert!(
        index.fast_identity_is_attested(),
        "identity parsed from the artifact's validated v2 header is ATTESTED"
    );
    assert!(index.quality_identity_is_attested());
    assert_eq!(
        index.fast_space_fingerprint_hex(),
        Some(fast_identity.space.fingerprint().as_str()),
        "the join key must be the header's space fingerprint, bit-for-bit"
    );
    assert_eq!(
        index.quality_space_fingerprint_hex(),
        Some(quality_identity.space.fingerprint().as_str())
    );
    assert!(
        index.fast_declared_identity().is_none(),
        "attested identity is not a builder declaration"
    );
    assert!(index.quality_declared_identity().is_none());
    assert_eq!(index.doc_count(), 2);
    let hits = index
        .search_fast(&[0.0, 1.0, 0.0, 0.0], 1)
        .expect("search admitted v2 fast tier");
    assert_eq!(hits[0].doc_id, "doc-b");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn builder_declared_identity_is_never_attested() {
    // The C4-write discriminator half of red proof (c): a builder-time
    // declaration retains the identity (C2) but must NOT read as
    // attested — the persisted artifacts are v1 and their headers attest
    // nothing.
    let dir = temp_index_dir("declared-not-attested");
    let fast_bundle = EmbeddingIdentityBundleV1::explicit_test_model("declared-fast", 4);
    let quality_bundle = EmbeddingIdentityBundleV1::explicit_test_model("declared-quality", 4);
    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
    builder
        .set_fast_identity(&fast_bundle)
        .expect("declare fast identity");
    builder
        .set_quality_identity(&quality_bundle)
        .expect("declare quality identity");
    builder
        .add_record("doc-a", &[1.0, 0.0, 0.0, 0.0], Some(&[0.0, 1.0, 0.0, 0.0]))
        .expect("add record");
    let index = builder.finish().expect("finish");

    assert!(
        index.fast_space_fingerprint_hex().is_some() && index.fast_declared_identity().is_some(),
        "the declaration is retained (C2)…"
    );
    assert!(
        !index.fast_identity_is_attested(),
        "…but a declaration is a process-local claim, never a header attestation"
    );
    assert!(!index.quality_identity_is_attested());

    drop(index);
    let reopened = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("reopen");
    assert!(!reopened.fast_identity_is_attested());
    assert!(!reopened.quality_identity_is_attested());
    assert_eq!(reopened.fast_space_fingerprint_hex(), None);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn admitted_v2_open_requires_binding_for_quality_path() {
    let dir = temp_index_dir("admitted-v2-missing-binding");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
    let (fast_binding, _) = fsvi_v2_binding("binding-gap-fast", 4, 1);
    let (quality_binding, _) = fsvi_v2_binding("binding-gap-quality", 4, 1);
    let rows: [(&str, &[f32]); 1] = [("doc-a", &[1.0, 0.0, 0.0, 0.0])];
    write_v2_tier(&fast_path, &fast_binding, &rows);
    write_v2_tier(&quality_path, &quality_binding, &rows);

    let paths = TwoTierIndexPaths::new(&fast_path).with_quality_index(&quality_path);
    let error = TwoTierIndex::open_admitted_v2_with_paths(
        &paths,
        TwoTierConfig::default(),
        &fast_binding,
        None,
    )
    .expect_err("a quality path without its binding must be refused");
    assert!(
        matches!(
            error,
            SearchError::InvalidConfig { ref field, .. }
                if field == "two_tier.quality_v2_admission"
        ),
        "got {error:?}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn admitted_v2_open_rejects_foreign_binding() {
    let dir = temp_index_dir("admitted-v2-foreign-binding");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let (fast_binding, _) = fsvi_v2_binding("real-artifact-model", 4, 1);
    let rows: [(&str, &[f32]); 1] = [("doc-a", &[1.0, 0.0, 0.0, 0.0])];
    write_v2_tier(&fast_path, &fast_binding, &rows);

    let (foreign_binding, _) = fsvi_v2_binding("foreign-expectation-model", 4, 1);
    let paths = TwoTierIndexPaths::new(&fast_path);
    let error = TwoTierIndex::open_admitted_v2_with_paths(
        &paths,
        TwoTierConfig::default(),
        &foreign_binding,
        None,
    )
    .expect_err("admission against a foreign identity binding must be refused");
    assert!(
        matches!(
            error,
            SearchError::InvalidConfig { ref field, .. }
                if field == "two_tier.fast_v2_admission"
        ),
        "got {error:?}"
    );

    let _ = fs::remove_dir_all(&dir);
}

// ─── C4-write r2: retained sealed owners + read-only observation ─────

/// NO-GO item 3 repair: exact v2 admission must retain the complete
/// `ValidatedFsviBytes` owner per tier — byte capability, witness, and
/// publication state — instead of peeling `validated.index` and dropping
/// the rest. Red on 868c0801: `fast_admitted_owner` does not exist there
/// (the owner was destructured away), so this test cannot compile.
#[test]
fn admitted_v2_open_retains_sealed_owners_in_full() {
    let dir = temp_index_dir("admitted-v2-retained-owners");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
    let (fast_binding, _) = fsvi_v2_binding("retained-fast-model", 4, 5);
    let (quality_binding, _) = fsvi_v2_binding("retained-quality-model", 4, 5);
    let rows: [(&str, &[f32]); 2] = [
        ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
        ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
    ];
    write_v2_tier(&fast_path, &fast_binding, &rows);
    write_v2_tier(&quality_path, &quality_binding, &rows);

    let paths = TwoTierIndexPaths::new(&fast_path).with_quality_index(&quality_path);
    let index = TwoTierIndex::open_admitted_v2_with_paths(
        &paths,
        TwoTierConfig::default(),
        &fast_binding,
        Some(&quality_binding),
    )
    .expect("admit both v2 tiers");

    let fast_owner = index
        .fast_admitted_owner()
        .expect("fast admission owner must be retained");
    let quality_owner = index
        .quality_admitted_owner()
        .expect("quality admission owner must be retained");
    assert_eq!(fast_owner.witness().record_count, 2);
    assert_eq!(quality_owner.witness().record_count, 2);
    assert!(
        fast_owner.published_wal_absent(),
        "canonical pathname admission proves WAL absence; the publication \
         state must survive into the retained owner"
    );
    assert!(quality_owner.published_wal_absent());
    assert_eq!(
        fast_owner.owned_byte_len(),
        usize::try_from(fs::metadata(&fast_path).expect("stat fast").len())
            .expect("length fits usize"),
        "the retained owner holds the complete admitted byte image"
    );
    // The retained owner and the served tier are the same admission, not
    // a re-open: identity metadata must agree bit-for-bit.
    assert!(index.fast_identity_is_attested());
    assert_eq!(
        index.fast_space_fingerprint_hex(),
        Some(crate::fingerprint_hex(&fast_owner.identity_v2().space_fingerprint).as_str())
    );

    let _ = fs::remove_dir_all(&dir);
}

/// Required test (iv): after admission, replacing the underlying file
/// does not affect the retained owner's reads — the `Arc`'d bytes are
/// the authority (owner contract, lib.rs).
#[test]
fn admitted_owner_reads_survive_path_replacement() {
    let dir = temp_index_dir("admitted-v2-path-replacement");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let (fast_binding, _) = fsvi_v2_binding("replacement-proof-model", 4, 9);
    let rows: [(&str, &[f32]); 2] = [
        ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
        ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
    ];
    write_v2_tier(&fast_path, &fast_binding, &rows);

    let paths = TwoTierIndexPaths::new(&fast_path);
    let index = TwoTierIndex::open_admitted_v2_with_paths(
        &paths,
        TwoTierConfig::default(),
        &fast_binding,
        None,
    )
    .expect("admit fast v2 tier");
    let witness_before = index
        .fast_admitted_owner()
        .expect("retained owner")
        .witness()
        .clone();
    let hits_before = index
        .search_fast(&[0.0, 1.0, 0.0, 0.0], 1)
        .expect("search before replacement");
    assert_eq!(hits_before[0].doc_id, "doc-b");

    // Rename the source away, then plant garbage at the original path.
    let renamed = dir.join("vector.fast.idx.renamed-away");
    fs::rename(&fast_path, &renamed).expect("rename admitted source");
    fs::write(&fast_path, b"garbage-not-an-index").expect("plant garbage");

    let owner = index.fast_admitted_owner().expect("retained owner");
    assert_eq!(
        owner.witness(),
        &witness_before,
        "witness is part of the sealed owner, not re-derived from the path"
    );
    let hits_after = index
        .search_fast(&[0.0, 1.0, 0.0, 0.0], 1)
        .expect("search after replacement");
    assert_eq!(
        hits_after[0].doc_id, "doc-b",
        "reads must serve the admitted Arc'd bytes, never the pathname"
    );
    assert_eq!(owner.row(0).expect("row 0").doc_id(), "doc-a");
    assert_eq!(index.doc_count(), 2);

    let _ = fs::remove_dir_all(&dir);
}

/// C2, the mutation classes `admitted_owner_reads_survive_path_replacement`
/// does not reach. Rename-then-plant only proves the owner ignores the
/// PATHNAME. These three mutate the admitted inode itself -- truncated in
/// place, overwritten in place with a same-length foreign image, and
/// replaced by a fresh file that reuses the freed descriptor slot -- which
/// is what a mapping-backed or lazily re-read owner would actually
/// observe. Every borrowed view, the witness, and the served rows must be
/// unchanged, because the admitted bytes are the authority.
#[test]
fn admitted_owner_survives_in_place_truncation_swap_and_descriptor_reuse() {
    let dir = temp_index_dir("admitted-v2-inode-mutation");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let (fast_binding, _) = fsvi_v2_binding("inode-mutation-model", 4, 11);
    let rows: [(&str, &[f32]); 2] = [
        ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
        ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
    ];
    write_v2_tier(&fast_path, &fast_binding, &rows);
    let admitted_bytes = fs::read(&fast_path).expect("read admitted image");

    let paths = TwoTierIndexPaths::new(&fast_path);
    let index = TwoTierIndex::open_admitted_v2_with_paths(
        &paths,
        TwoTierConfig::default(),
        &fast_binding,
        None,
    )
    .expect("admit fast v2 tier");
    let witness_before = index
        .fast_admitted_owner()
        .expect("retained owner")
        .witness()
        .clone();
    let fingerprint_before = index.fast_space_fingerprint_hex().map(str::to_owned);

    let assert_owner_intact = |label: &str| {
        let owner = index.fast_admitted_owner().expect("retained owner");
        assert_eq!(owner.witness(), &witness_before, "{label}: witness");
        assert_eq!(
            owner.owned_byte_len(),
            admitted_bytes.len(),
            "{label}: owned byte length"
        );
        assert_eq!(
            owner.row(0).expect("row 0").doc_id(),
            "doc-a",
            "{label}: borrowed row view"
        );
        assert_eq!(
            index.fast_space_fingerprint_hex().map(str::to_owned),
            fingerprint_before,
            "{label}: space fingerprint"
        );
        assert_eq!(index.doc_count(), 2, "{label}: doc count");
        let hits = index
            .search_fast(&[0.0, 1.0, 0.0, 0.0], 1)
            .unwrap_or_else(|error| panic!("{label}: search failed: {error}"));
        assert_eq!(hits[0].doc_id, "doc-b", "{label}: served rows");
    };
    assert_owner_intact("before any mutation");

    // (i) truncate the admitted inode to nothing, in place.
    fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(&fast_path)
        .expect("truncate admitted inode");
    assert_eq!(
        fs::metadata(&fast_path).expect("stat truncated").len(),
        0,
        "the fixture must really have truncated the source"
    );
    assert_owner_intact("after in-place truncation");

    // (ii) overwrite the same inode with a same-length foreign image, so
    // no length check could notice the substitution.
    let foreign = vec![0x5a_u8; admitted_bytes.len()];
    assert_ne!(foreign, admitted_bytes);
    fs::write(&fast_path, &foreign).expect("overwrite admitted inode in place");
    assert_owner_intact("after same-length in-place byte swap");

    // (iii) unlink and recreate, so the pathname resolves to a brand new
    // inode and the freed descriptor slot is reused by an unrelated open.
    fs::remove_file(&fast_path).expect("unlink admitted source");
    let decoy = dir.join("descriptor-reuse-decoy");
    let decoy_handle = fs::File::create(&decoy).expect("claim the freed descriptor slot");
    write_v2_tier(
        &fast_path,
        &fsvi_v2_binding("descriptor-reuse-impostor", 4, 12).0,
        &[("doc-impostor", &[0.0, 0.0, 1.0, 0.0])],
    );
    assert_owner_intact("after unlink, descriptor reuse, and re-creation");
    drop(decoy_handle);

    let _ = fs::remove_dir_all(&dir);
}

/// C3: fast and quality owners are independent even when their identities
/// agree on every field a subset comparison would look at. Both tiers here
/// carry the same model id, dimension, storage identity and generation
/// sequence, so anything that keyed off "identities compare equal" would
/// happily serve one owner for both; the owners must still be two distinct
/// admissions over two distinct byte images, and each tier must serve its
/// own rows.
#[test]
fn same_identity_fast_and_quality_tiers_retain_independent_owners() {
    let dir = temp_index_dir("admitted-v2-independent-owners");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
    let (fast_binding, _) = fsvi_v2_binding("indistinguishable-model", 4, 21);
    let (quality_binding, _) = fsvi_v2_binding("indistinguishable-model", 4, 21);
    let fast_rows: [(&str, &[f32]); 2] = [
        ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
        ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
    ];
    let quality_rows: [(&str, &[f32]); 2] = [
        ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
        ("doc-b", &[0.0, 0.0, 1.0, 0.0]),
    ];
    write_v2_tier(&fast_path, &fast_binding, &fast_rows);
    write_v2_tier(&quality_path, &quality_binding, &quality_rows);

    let paths = TwoTierIndexPaths::new(&fast_path).with_quality_index(&quality_path);
    let index = TwoTierIndex::open_admitted_v2_with_paths(
        &paths,
        TwoTierConfig::default(),
        &fast_binding,
        Some(&quality_binding),
    )
    .expect("admit both v2 tiers");

    let fast_owner = index.fast_admitted_owner().expect("fast owner");
    let quality_owner = index.quality_admitted_owner().expect("quality owner");
    assert_eq!(
        fast_owner.identity_v2().space_fingerprint,
        quality_owner.identity_v2().space_fingerprint,
        "the fixture is only meaningful while the two identities are indistinguishable"
    );
    assert!(
        !std::ptr::eq(fast_owner, quality_owner),
        "each tier retains its own admission owner"
    );
    // Distinct byte images: doc-b's vector differs between the tiers, so
    // an aliased owner would serve the fast vectors for quality reads.
    assert_ne!(
        fast_owner.witness().whole_image_sha256,
        quality_owner.witness().whole_image_sha256,
        "two independent admissions over two artifacts must witness \
         different images"
    );
    assert_eq!(
        index
            .quality_vector_for_doc_id("doc-b")
            .expect("quality vector lookup")
            .expect("doc-b is present in the quality tier"),
        vec![0.0, 0.0, 1.0, 0.0],
        "the quality tier must serve its own vectors, not the fast tier's"
    );
    assert_eq!(
        index
            .search_fast(&[0.0, 1.0, 0.0, 0.0], 1)
            .expect("fast search")[0]
            .doc_id,
        "doc-b",
        "the fast tier keeps serving its own vectors"
    );

    let _ = fs::remove_dir_all(&dir);
}

/// C5 + C2 refresh: a refused candidate leaves the retained owner
/// serving byte-identically, an accepted one installs, and a reader that
/// opened its own index over the same artifacts is unaffected by either
/// outcome. The failed and successful halves run against the SAME index
/// in the same test, so neither can pass by the seam being inert.
#[test]
fn refused_candidate_leaves_the_retained_owner_serving_and_accepted_one_installs() {
    let dir = temp_index_dir("admitted-v2-candidate-refresh");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let (generation_one, _) = fsvi_v2_binding("refresh-model", 4, 31);
    let incumbent_rows: [(&str, &[f32]); 2] = [
        ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
        ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
    ];
    write_v2_tier(&fast_path, &generation_one, &incumbent_rows);

    let paths = TwoTierIndexPaths::new(&fast_path);
    let mut index = TwoTierIndex::open_admitted_v2_with_paths(
        &paths,
        TwoTierConfig::default(),
        &generation_one,
        None,
    )
    .expect("admit generation one");
    // An independent reader over the same artifact, held across both
    // refresh attempts.
    let reader = TwoTierIndex::open_admitted_v2_with_paths(
        &paths,
        TwoTierConfig::default(),
        &generation_one,
        None,
    )
    .expect("second reader admits generation one");

    let incumbent_witness = index
        .fast_admitted_owner()
        .expect("retained owner")
        .witness()
        .clone();
    let reader_witness = reader
        .fast_admitted_owner()
        .expect("reader owner")
        .witness()
        .clone();

    // (i) REFUSED: the artifact on disk is still generation one, but the
    // caller presents generation two's binding. Admission must reject it.
    let (generation_two, _) = fsvi_v2_binding("refresh-model", 4, 32);
    let refusal = index
        .try_replace_admitted_v2(&paths, &generation_two, None)
        .expect_err("a candidate whose identity does not match the artifact is refused");
    assert!(
        matches!(
            refusal,
            SearchError::InvalidConfig { ref field, .. } if field == "two_tier.fast_v2_admission"
        ),
        "got {refusal:?}"
    );
    assert_eq!(
        index
            .fast_admitted_owner()
            .expect("incumbent still retained")
            .witness(),
        &incumbent_witness,
        "a refused candidate must not modify or evict the retained owner"
    );
    assert_eq!(
        index
            .search_fast(&[0.0, 1.0, 0.0, 0.0], 1)
            .expect("incumbent still serves")[0]
            .doc_id,
        "doc-b"
    );
    assert_eq!(index.doc_count(), 2);

    // (ii) ACCEPTED: publish a real generation two and refresh into it.
    let successor_rows: [(&str, &[f32]); 3] = [
        ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
        ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
        ("doc-c", &[0.0, 0.0, 1.0, 0.0]),
    ];
    write_v2_tier(&fast_path, &generation_two, &successor_rows);
    index
        .try_replace_admitted_v2(&paths, &generation_two, None)
        .expect("a matching candidate installs");
    let installed = index.fast_admitted_owner().expect("successor retained");
    assert_eq!(installed.witness().generation.sequence, 32);
    assert_ne!(
        installed.witness(),
        &incumbent_witness,
        "the successor is a different admission"
    );
    assert_eq!(index.doc_count(), 3);
    assert_eq!(
        index
            .search_fast(&[0.0, 0.0, 1.0, 0.0], 1)
            .expect("successor serves")[0]
            .doc_id,
        "doc-c"
    );

    // (iii) The independent reader is untouched by either outcome: it
    // still serves generation one from its own retained bytes, even
    // though the pathname now holds generation two.
    assert_eq!(
        reader
            .fast_admitted_owner()
            .expect("reader owner")
            .witness(),
        &reader_witness,
        "installing a successor must not disturb a live predecessor reader"
    );
    assert_eq!(reader.doc_count(), 2);
    assert_eq!(
        reader
            .search_fast(&[0.0, 1.0, 0.0, 0.0], 1)
            .expect("reader still serves generation one")[0]
            .doc_id,
        "doc-b"
    );
    assert!(
        !reader
            .search_fast(&[0.0, 0.0, 1.0, 0.0], 3)
            .expect("reader exhaustive search")
            .iter()
            .any(|hit| hit.doc_id == "doc-c"),
        "generation one has no doc-c; the reader must not observe the successor"
    );

    let _ = fs::remove_dir_all(&dir);
}

/// C2 residual (bd-r6lwt): TRUE concurrency, not two sequentially opened
/// indexes.
///
/// `refused_candidate_leaves_the_retained_owner_serving_and_accepted_one_installs`
/// holds a second reader across a refresh, but everything in it happens on
/// one thread in a fixed order, so it cannot distinguish "the owner is
/// immune to the pathname changing" from "nothing ran at the same time".
/// Here real OS threads search a retained generation-one owner while
/// another thread rewrites the SAME pathname to generation two and admits
/// and installs the successor on its own handle.
///
/// The property under test is that a retained owner serves from its own
/// sealed `Arc<[u8]>` and never re-reads the path it was opened from. A
/// reader must therefore observe generation one on EVERY iteration: not a
/// torn read, not an empty result, and never `doc-c`, which exists only in
/// the successor.
///
/// This is an OS-level contract (a real file being overwritten under live
/// readers), so it uses real threads rather than a `LabRuntime` schedule --
/// a deterministic scheduler would be simulating the very interleaving that
/// is the thing in question.
#[test]
fn retained_owners_serve_concurrent_readers_while_a_successor_installs() {
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    const READERS: usize = 4;
    // Retain repeated-read coverage in addition to the overlap handshake.
    const MIN_ITERATIONS_PER_READER: usize = 200;

    struct WriterCompletion<'a>(&'a AtomicBool);

    impl Drop for WriterCompletion<'_> {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }

    let dir = temp_index_dir("admitted-v2-concurrent-refresh");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);

    let (generation_one, _) = fsvi_v2_binding("concurrent-model", 4, 41);
    let incumbent_rows: [(&str, &[f32]); 2] = [
        ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
        ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
    ];
    write_v2_tier(&fast_path, &generation_one, &incumbent_rows);

    let paths = TwoTierIndexPaths::new(&fast_path);
    let reader_index = TwoTierIndex::open_admitted_v2_with_paths(
        &paths,
        TwoTierConfig::default(),
        &generation_one,
        None,
    )
    .expect("admit generation one for the reader handle");
    let witness_before = reader_index
        .fast_admitted_owner()
        .expect("retained owner")
        .witness()
        .clone();

    // Release all threads together; this alone does not guarantee that a
    // reader runs before the writer finishes, so also handshake below.
    let barrier = Barrier::new(READERS + 1);
    let writer_done = AtomicBool::new(false);
    let pathname_rewritten = AtomicBool::new(false);
    let (checked_read_tx, checked_read_rx) = mpsc::channel();
    let observed = AtomicUsize::new(0);
    // Reads completed strictly BETWEEN the writer starting its rewrite and
    // finishing its install. This is the overlap proof: a run where the
    // writer finished before any reader looped would leave it at zero, and
    // every assertion in the reader loop would then be about a file that
    // was never concurrently rewritten.
    let overlapped = AtomicUsize::new(0);
    let index_ref = &reader_index;
    let barrier_ref = &barrier;
    let writer_done_ref = &writer_done;
    let pathname_rewritten_ref = &pathname_rewritten;
    let observed_ref = &observed;
    let overlapped_ref = &overlapped;
    let fast_path_ref = fast_path.as_path();

    thread::scope(|scope| {
        for _ in 0..READERS {
            let checked_read_tx = checked_read_tx.clone();
            scope.spawn(move || {
                barrier_ref.wait();
                let mut iterations = 0_usize;
                let mut acknowledged_rewrite = false;
                // Keep going until the writer is finished AND this reader
                // has done enough passes for the overlap to be real.
                while !writer_done_ref.load(Ordering::Acquire)
                    || iterations < MIN_ITERATIONS_PER_READER
                {
                    // Sample BEFORE searching: the acknowledgement must
                    // prove a complete read of the retained owner after
                    // the pathname holds the successor's bytes.
                    let after_rewrite = pathname_rewritten_ref.load(Ordering::Acquire);
                    let hits = index_ref
                        .search_fast(&[0.0, 1.0, 0.0, 0.0], 3)
                        .expect("a retained owner never fails a concurrent read");
                    assert_eq!(
                        hits.len(),
                        2,
                        "generation one has exactly two rows; a short read means the \
                         owner re-read the pathname"
                    );
                    assert_eq!(
                        hits[0].doc_id, "doc-b",
                        "generation one's nearest neighbour must not change under a \
                         concurrent install"
                    );
                    assert!(
                        !hits.iter().any(|hit| hit.doc_id == "doc-c"),
                        "doc-c exists only in the successor; a retained owner must \
                         never observe it"
                    );
                    assert_eq!(index_ref.doc_count(), 2);
                    iterations = iterations.saturating_add(1);
                    observed_ref.fetch_add(1, Ordering::Release);
                    if after_rewrite && !acknowledged_rewrite {
                        // The writer may already have received another
                        // reader's acknowledgement and dropped its end.
                        let _ = checked_read_tx.send(());
                        acknowledged_rewrite = true;
                    }
                }
            });
        }

        scope.spawn(move || {
            // A writer assertion or I/O panic must also release readers
            // before the scope propagates that failure.
            let _completion = WriterCompletion(writer_done_ref);
            barrier_ref.wait();
            let (generation_two, _) = fsvi_v2_binding("concurrent-model", 4, 42);
            let successor_rows: [(&str, &[f32]); 3] = [
                ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
                ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
                ("doc-c", &[0.0, 0.0, 1.0, 0.0]),
            ];
            // Overwrite the pathname the readers' owner was opened from,
            // then admit and install the successor on a separate handle.
            let reads_before_rewrite = observed_ref.load(Ordering::Acquire);
            write_v2_tier(fast_path_ref, &generation_two, &successor_rows);
            pathname_rewritten_ref.store(true, Ordering::Release);
            // Require a fully checked retained-owner read before install,
            // independent of which OS thread ran first after the barrier.
            // Fail boundedly and let readers exit if no acknowledgement
            // arrives, rather than hanging the scoped thread join.
            checked_read_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("a checked retained-owner read after pathname rewrite");
            let successor_paths = TwoTierIndexPaths::new(fast_path_ref);
            let mut successor = TwoTierIndex::open_admitted_v2_with_paths(
                &successor_paths,
                TwoTierConfig::default(),
                &generation_two,
                None,
            )
            .expect("admit generation two on an independent handle");
            successor
                .try_replace_admitted_v2(&successor_paths, &generation_two, None)
                .expect("re-admitting the installed generation succeeds");
            assert_eq!(successor.doc_count(), 3);
            overlapped_ref.store(
                observed_ref
                    .load(Ordering::Acquire)
                    .saturating_sub(reads_before_rewrite),
                Ordering::Release,
            );
            writer_done_ref.store(true, Ordering::Release);
        });
    });

    // Vacuity guard, two parts. Both must hold or the assertions inside
    // the reader loop proved nothing.
    assert!(
        observed.load(Ordering::Acquire) >= READERS * MIN_ITERATIONS_PER_READER,
        "readers did not complete the minimum number of passes"
    );
    assert!(
        overlapped.load(Ordering::Acquire) > 0,
        "no read completed between the pathname rewrite and the successor install; \
         the reads never actually raced the writer, so this run is vacuous"
    );

    // The owner is bit-for-bit what it was before the pathname changed.
    assert_eq!(
        reader_index
            .fast_admitted_owner()
            .expect("retained owner survives the concurrent install")
            .witness(),
        &witness_before,
        "a concurrent successor install must not mutate a live predecessor's owner"
    );
    assert_eq!(reader_index.doc_count(), 2);

    let _ = fs::remove_dir_all(&dir);
}

/// C2 residual (bd-r6lwt): DROP ORDER between the owner's byte image and
/// the index that serves from it.
///
/// The borrow checker covers one half of this: a borrowed view cannot be
/// held across a `&mut self` refresh, pinned by the `compile_fail` doctest
/// on [`crate::ValidatedFsviBytes`]. It does NOT cover the other half.
/// `ValidatedFsviBytes` declares `bytes: Arc<[u8]>` BEFORE `index`, and
/// Rust drops fields in declaration order, so the byte image is released
/// first while the index that serves from it is still alive.
///
/// That is sound only because `index.data` is
/// `VectorIndexData::Immutable(Arc::clone(&bytes))` -- the SAME allocation
/// behind a second refcount, not a borrow and not a second copy. A naive
/// owner that stored a borrow, a raw pointer, or an independent copy would
/// be either unsound or silently double-resident under exactly this
/// declaration order. `owner_and_search_share_allocation` is the assertion
/// that distinguishes them, and it goes red if construction is ever changed
/// to copy the image rather than share it.
#[test]
fn owner_drop_order_is_not_load_bearing_because_the_image_is_shared_not_borrowed() {
    use std::sync::Arc;

    let dir = temp_index_dir("admitted-v2-drop-order");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let (generation, _) = fsvi_v2_binding("drop-order-model", 4, 51);
    let rows: [(&str, &[f32]); 2] = [
        ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
        ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
    ];
    write_v2_tier(&fast_path, &generation, &rows);

    let paths = TwoTierIndexPaths::new(&fast_path);
    let index = TwoTierIndex::open_admitted_v2_with_paths(
        &paths,
        TwoTierConfig::default(),
        &generation,
        None,
    )
    .expect("admit the owner");

    let owner = index.fast_admitted_owner().expect("retained owner");

    // (i) The serving index and the owner name ONE allocation. This is the
    // whole reason field drop order is not load-bearing.
    assert!(
        owner.owner_and_search_share_allocation(),
        "the retained owner and its serving index must share one refcounted \
         image; a separate copy or a borrow would make declaration order \
         load-bearing"
    );

    // (ii) Two live handles to that one allocation: the owner's field and
    // the index's `Immutable` data.
    let external = Arc::clone(&owner.bytes);
    let image_before: Vec<u8> = external.to_vec();
    assert_eq!(
        Arc::strong_count(&external),
        3,
        "owner field + serving index + this test's clone"
    );

    // (iii) Drop the ENTIRE index, and with it the owner, while holding
    // only the external clone. If the image were borrowed rather than
    // shared, this is where it would become dangling.
    drop(index);

    assert_eq!(
        Arc::strong_count(&external),
        1,
        "dropping the owner must release exactly its own two references"
    );
    assert_eq!(
        external.as_ref(),
        image_before.as_slice(),
        "the byte image must be unchanged and fully readable after the owner \
         that admitted it is gone"
    );

    let _ = fs::remove_dir_all(&dir);
}

/// C2 residual (bd-r6lwt): ANN sidecar validation against a RETAINED
/// OWNER, where the discriminator is identity rather than content.
///
/// The `ann` feature was not exercised by any owner test at bd-3t52d's
/// closure. It matters here because `plan_load_or_build_ann` validates the
/// sidecar against `TierSource::index()`, which for an admitted tier is a
/// borrow INSIDE the sealed owner -- an index whose `path` is the synthetic
/// `<owned-fsvi-v2>` and whose data is the owner's own `Arc<[u8]>`. If
/// [`HnswSourceIdentityV1::capture`] could not read a retained owner's v2
/// identity, the generation gate would silently degrade to `(None, None)`,
/// which `admits` treats as legacy-v1 and lets through on content alone.
///
/// So this test holds the rows BYTE-IDENTICAL across generations and varies
/// only the generation binding. Every content check in
/// `matches_vector_index` -- dimension, doc ids, ordering, source positions
/// -- passes by construction, which leaves the identity gate as the only
/// thing that can reject. Content equality is not identity equality
/// (bd-r65a).
///
/// Planted negative: the generation-31 sidecar must NOT admit the
/// generation-32 owner. Reducing the gate to a content fingerprint turns
/// that assertion red. The paired positive control -- the same sidecar DOES
/// admit its own generation-31 owner -- means a blanket-reject regression
/// cannot pass either.
#[cfg(feature = "ann")]
#[test]
fn ann_sidecar_validation_binds_the_retained_owners_generation_not_its_content() {
    let dir = temp_index_dir("admitted-v2-ann-identity");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let ann_path = dir.join("fast.ann");

    // Identical in every observable content dimension across generations.
    let rows: [(&str, &[f32]); 3] = [
        ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
        ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
        ("doc-c", &[0.0, 0.0, 1.0, 0.0]),
    ];
    let config = TwoTierConfig {
        hnsw_threshold: 1,
        hnsw_ef_search: 64,
        ..TwoTierConfig::default()
    };

    let (generation_one, _) = fsvi_v2_binding("ann-identity-model", 4, 31);
    write_v2_tier(&fast_path, &generation_one, &rows);
    let paths = TwoTierIndexPaths::new(&fast_path).with_fast_ann(&ann_path);

    let first =
        TwoTierIndex::open_admitted_v2_with_paths(&paths, config.clone(), &generation_one, None)
            .expect("admit generation 31");
    assert!(
        first.has_fast_ann(),
        "an admitted owner must be able to carry an ANN sidecar"
    );
    // The owner is intact after ANN build and persistence ran against it.
    let owner_one = first.fast_admitted_owner().expect("retained owner");
    assert!(
        owner_one.owner_and_search_share_allocation(),
        "building an ANN sidecar must not detach the owner from its image"
    );
    let witness_one = owner_one.witness().clone();
    assert_eq!(
        first
            .search_fast(&[0.0, 0.0, 1.0, 0.0], 1)
            .expect("generation 31 serves")[0]
            .doc_id,
        "doc-c"
    );
    assert!(ann_path.exists(), "the sidecar was persisted");

    // POSITIVE CONTROL: the persisted sidecar admits the generation it was
    // built against, borrowed from inside the retained owner.
    let sidecar = load_native_ann_sidecar(&ann_path, first.fast_tier());
    assert!(
        sidecar
            .matches_vector_index(first.fast_tier())
            .expect("validate sidecar against its own generation"),
        "a sidecar must admit the retained owner it was built from; if this fails \
         the identity capture cannot see an owner-borrowed index at all"
    );

    // Generation 32: same model, same rows, same order -- only the
    // generation binding differs.
    let (generation_two, _) = fsvi_v2_binding("ann-identity-model", 4, 32);
    write_v2_tier(&fast_path, &generation_two, &rows);
    let successor =
        TwoTierIndex::open_admitted_v2_with_paths(&paths, config, &generation_two, None)
            .expect("admit generation 32");
    assert_ne!(
        successor
            .fast_admitted_owner()
            .expect("successor owner")
            .witness(),
        &witness_one,
        "the fixture must actually change generation, or this test is vacuous"
    );

    // PLANTED NEGATIVE: byte-identical content, different generation. Only
    // the identity gate can tell these apart.
    assert!(
        !sidecar
            .matches_vector_index(successor.fast_tier())
            .expect("validate the generation-31 sidecar against generation 32"),
        "an ANN sidecar bound to generation 31 must not be served against a \
         generation-32 retained owner, even though their content is identical"
    );

    // The successor still serves correctly and its owner is intact, so the
    // rejection produced a rebuild rather than a broken tier.
    assert!(successor.has_fast_ann());
    assert!(
        successor
            .fast_admitted_owner()
            .expect("successor owner")
            .owner_and_search_share_allocation()
    );
    assert_eq!(
        successor
            .search_fast(&[0.0, 0.0, 1.0, 0.0], 1)
            .expect("generation 32 serves")[0]
            .doc_id,
        "doc-c"
    );

    let _ = fs::remove_dir_all(&dir);
}

/// C5, the partial-install case: a refresh whose FAST tier admits but
/// whose QUALITY tier is refused must leave BOTH incumbent tiers exactly
/// as they were. This is the shape a half-installed generation would
/// take, and it is the only shape that distinguishes "admit everything,
/// then install" from "install each tier as it admits".
#[test]
fn a_refused_quality_tier_leaves_both_incumbent_tiers_installed() {
    let dir = temp_index_dir("admitted-v2-partial-install");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
    let (generation_one, _) = fsvi_v2_binding("partial-install-model", 4, 61);
    let rows: [(&str, &[f32]); 2] = [
        ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
        ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
    ];
    write_v2_tier(&fast_path, &generation_one, &rows);
    write_v2_tier(&quality_path, &generation_one, &rows);

    let paths = TwoTierIndexPaths::new(&fast_path).with_quality_index(&quality_path);
    let mut index = TwoTierIndex::open_admitted_v2_with_paths(
        &paths,
        TwoTierConfig::default(),
        &generation_one,
        Some(&generation_one),
    )
    .expect("admit generation one on both tiers");
    let fast_witness = index
        .fast_admitted_owner()
        .expect("fast owner")
        .witness()
        .clone();
    let quality_witness = index
        .quality_admitted_owner()
        .expect("quality owner")
        .witness()
        .clone();

    // Publish a real generation two for the FAST tier only, and leave the
    // quality artifact at generation one. The candidate's fast binding
    // therefore admits and its quality binding cannot.
    let (generation_two, _) = fsvi_v2_binding("partial-install-model", 4, 62);
    let successor_rows: [(&str, &[f32]); 3] = [
        ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
        ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
        ("doc-c", &[0.0, 0.0, 1.0, 0.0]),
    ];
    write_v2_tier(&fast_path, &generation_two, &successor_rows);
    let refusal = index
        .try_replace_admitted_v2(&paths, &generation_two, Some(&generation_two))
        .expect_err("a candidate whose quality tier cannot be admitted is refused whole");
    assert!(
        matches!(
            refusal,
            SearchError::InvalidConfig { ref field, .. }
                if field == "two_tier.quality_v2_admission"
        ),
        "the refusal must name the quality tier: got {refusal:?}"
    );

    assert_eq!(
        index.fast_admitted_owner().expect("fast owner").witness(),
        &fast_witness,
        "the incumbent FAST owner must survive a quality-tier refusal"
    );
    assert_eq!(
        index
            .quality_admitted_owner()
            .expect("quality owner")
            .witness(),
        &quality_witness,
        "the incumbent QUALITY owner must survive too"
    );
    assert_eq!(index.doc_count(), 2, "no half-installed successor");
    assert!(
        !index
            .search_fast(&[0.0, 0.0, 1.0, 0.0], 3)
            .expect("incumbent still serves")
            .iter()
            .any(|hit| hit.doc_id == "doc-c"),
        "the successor's doc-c must not be observable after a refused refresh"
    );

    let _ = fs::remove_dir_all(&dir);
}

/// C3, the enumerated mismatch classes, driven through the two-tier open
/// rather than only through the admission layer. Each candidate differs
/// from the artifact in exactly ONE identity component, and every one
/// must be refused; the unmodified binding is the positive control in the
/// same test, so a seam that rejected everything could not pass.
#[test]
fn each_single_component_identity_mismatch_is_refused_at_two_tier_open() {
    let dir = temp_index_dir("admitted-v2-mismatch-table");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let (binding, identity) = fsvi_v2_binding("mismatch-table-model", 4, 41);
    let rows: [(&str, &[f32]); 2] = [
        ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
        ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
    ];
    write_v2_tier(&fast_path, &binding, &rows);
    let paths = TwoTierIndexPaths::new(&fast_path);

    // Positive control first: the exact binding admits.
    let index =
        TwoTierIndex::open_admitted_v2_with_paths(&paths, TwoTierConfig::default(), &binding, None)
            .expect("the exact binding admits");
    assert_eq!(index.doc_count(), 2);
    drop(index);

    let generation =
        ArtifactGenerationIdentityV1::new(41, [0x4d; 16]).expect("valid test generation");
    // The defence is TWO layers, and this test measures both rather than
    // assuming one. The bundle is SELF-BINDING: producer.space_fingerprint
    // must bind the bundled space identity and space.input_contract_
    // fingerprint must bind the bundled input contract, so a candidate
    // whose model, tokenizer, dimension or input contract disagrees with
    // the rest of its own bundle cannot even be FROZEN -- it is refused
    // before an admission call is reachable. Components that are legally
    // independent freeze fine and must then be refused by admission.
    for (label, mutation) in [
        (
            "tokenizer",
            Box::new(|bundle: &mut EmbeddingIdentityBundleV1| {
                "a-different-tokenizer".clone_into(&mut bundle.space.tokenizer_fingerprint);
            }) as Box<dyn Fn(&mut EmbeddingIdentityBundleV1)>,
        ),
        (
            "input contract",
            Box::new(|bundle: &mut EmbeddingIdentityBundleV1| {
                "a-different-canonicalization".clone_into(&mut bundle.input.canonicalization);
            }),
        ),
        (
            "storage endianness",
            Box::new(|bundle: &mut EmbeddingIdentityBundleV1| {
                "big-endian".clone_into(&mut bundle.storage.endianness);
            }),
        ),
    ] {
        let mut mutated = identity.clone();
        mutation(&mut mutated);
        assert!(
            mutated.freeze().is_err(),
            "{label}: an identity the format cannot represent must not be constructible"
        );
    }

    let mutate = |mutation: &dyn Fn(&mut EmbeddingIdentityBundleV1)| {
        let mut mutated = identity.clone();
        mutation(&mut mutated);
        FsviV2IdentityBinding::new(
            generation,
            mutated.freeze().expect("freeze mutated identity"),
        )
        .expect("valid binding over a mutated identity")
    };

    type IdentityMutation = Box<dyn Fn(&mut EmbeddingIdentityBundleV1)>;
    let cases: [(&str, IdentityMutation); 2] = [
        (
            "producer attestation",
            Box::new(|bundle: &mut EmbeddingIdentityBundleV1| {
                "a-different-backend".clone_into(&mut bundle.producer.backend);
            }),
        ),
        (
            "quantization",
            Box::new(|bundle: &mut EmbeddingIdentityBundleV1| {
                bundle.storage.quantization = QuantizationFormat::F32;
            }),
        ),
    ];

    // Space-level classes, as coherent bundles: a different model and a
    // different output dimension each produce a valid but different
    // embedding space, which is what a caller would actually present
    // after a re-embed.
    for (label, candidate) in [
        ("model", fsvi_v2_binding("a-different-model", 4, 41).0),
        (
            "dimension",
            fsvi_v2_binding("mismatch-table-model", 8, 41).0,
        ),
    ] {
        let error = TwoTierIndex::open_admitted_v2_with_paths(
            &paths,
            TwoTierConfig::default(),
            &candidate,
            None,
        )
        .err()
        .unwrap_or_else(|| panic!("{label} mismatch must be refused"));
        assert!(
            matches!(
                error,
                SearchError::InvalidConfig { ref field, .. }
                    if field == "two_tier.fast_v2_admission"
            ),
            "{label}: got {error:?}"
        );
    }

    for (label, mutation) in &cases {
        let candidate = mutate(mutation.as_ref());
        let error = TwoTierIndex::open_admitted_v2_with_paths(
            &paths,
            TwoTierConfig::default(),
            &candidate,
            None,
        )
        .expect_err(&format!("{label} mismatch must be refused"));
        assert!(
            matches!(
                error,
                SearchError::InvalidConfig { ref field, .. }
                    if field == "two_tier.fast_v2_admission"
            ),
            "{label}: got {error:?}"
        );
    }

    // A wrong GENERATION over an otherwise identical identity is refused
    // too, so "same space, newer generation" cannot be laundered in.
    let wrong_generation = FsviV2IdentityBinding::new(
        ArtifactGenerationIdentityV1::new(42, [0x4d; 16]).expect("valid generation"),
        identity.clone().freeze().expect("freeze identity"),
    )
    .expect("valid binding");
    assert!(
        TwoTierIndex::open_admitted_v2_with_paths(
            &paths,
            TwoTierConfig::default(),
            &wrong_generation,
            None,
        )
        .is_err(),
        "generation mismatch must be refused"
    );

    let _ = fs::remove_dir_all(&dir);
}

/// C6, discharged structurally rather than with a compile-fail harness
/// (adding one would mean a new dev-dependency, which this bead is not
/// the place to decide). The property is that a retained owner is only
/// reachable through admission: `TierSource` is private, `TwoTierIndex`
/// exposes owners by reference only, and the only public constructors of
/// an owner-backed index are the admitting ones. A v1 open — the sole
/// public path that takes no binding — yields no owner at all, so no
/// public API returns an owner that was not admitted.
#[test]
fn owners_are_reachable_only_through_admission() {
    let dir = temp_index_dir("owner-reachability");
    fs::create_dir_all(&dir).expect("create temp dir");
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])]).expect("write v1 fixture");

    // Every non-admitting public constructor: no owner, and no fabricated
    // identity to stand in for one.
    let mut opened = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("v1 open");
    assert!(opened.fast_admitted_owner().is_none());
    assert!(opened.quality_admitted_owner().is_none());
    assert!(!opened.fast_identity_is_attested());
    assert!(opened.fast_space_fingerprint_hex().is_none());
    assert!(matches!(
        opened.enable_native_quality_hnsw(HnswParams::default(), 7),
        Err(SearchError::InvalidConfig { field, .. })
            if field == "two_tier.native_quality_hnsw.owner"
    ));
    assert!(!opened.has_native_quality_hnsw());
    assert!(matches!(
        opened.enable_native_fast_hnsw(HnswParams::default(), 7),
        Err(SearchError::InvalidConfig { field, .. })
            if field == "two_tier.native_fast_hnsw.owner"
    ));
    assert!(!opened.has_native_fast_hnsw());

    let by_paths = TwoTierIndex::open_with_paths(
        &TwoTierIndexPaths::new(&fast_path),
        TwoTierConfig::default(),
    )
    .expect("v1 open_with_paths");
    assert!(by_paths.fast_admitted_owner().is_none());
    assert!(!by_paths.fast_identity_is_attested());

    // And v2 bytes cannot reach either of them: the v1 opener rejects the
    // image outright, so there is no non-admitting route to an owner.
    let v2_dir = temp_index_dir("owner-reachability-v2");
    fs::create_dir_all(&v2_dir).expect("create temp dir");
    let v2_path = v2_dir.join(VECTOR_INDEX_FAST_FILENAME);
    let (binding, _) = fsvi_v2_binding("reachability-model", 4, 51);
    write_v2_tier(&v2_path, &binding, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])]);
    assert!(
        TwoTierIndex::open(&v2_dir, TwoTierConfig::default()).is_err(),
        "a v2 artifact must not be reachable through the v1 opener"
    );

    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&v2_dir);
}
