#[cfg(any(target_os = "linux", target_os = "android"))]
use std::fs::{self, OpenOptions};
#[cfg(any(target_os = "linux", target_os = "android"))]
use std::io::Read;
#[cfg(any(target_os = "linux", target_os = "android"))]
use std::time::UNIX_EPOCH;

#[cfg(any(target_os = "linux", target_os = "android"))]
use frankensearch_core::SearchError;
use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, QuantizationFormat,
};
#[cfg(any(target_os = "linux", target_os = "android"))]
use sha2::{Digest, Sha256};

#[cfg(any(target_os = "linux", target_os = "android"))]
use crate::{
    FsviAdmissionError, open_readonly_noatime_nofollow, snapshot_parent_or_current,
    stable_file_identity,
};

#[cfg(any(target_os = "linux", target_os = "android"))]
use super::super::io::{acquire_wal_writer_lock, encode_vector, write_wal_header};
use super::super::*;
use super::{identity_binding_v2, refresh_identity_header_crc};
#[cfg(any(target_os = "linux", target_os = "android"))]
use super::{make_entry, temp_wal_path};

#[cfg(any(target_os = "linux", target_os = "android"))]
fn encoded_test_batch(entries: &[WalEntry], quantization: Quantization) -> Vec<u8> {
    let mut batch = Vec::new();
    batch.extend_from_slice(&BATCH_MAGIC);
    batch.extend_from_slice(
        &u32::try_from(entries.len())
            .expect("test batch entry count fits")
            .to_le_bytes(),
    );
    for entry in entries {
        let doc_id = entry.doc_id.as_bytes();
        batch.extend_from_slice(
            &u16::try_from(doc_id.len())
                .expect("test document id fits")
                .to_le_bytes(),
        );
        batch.extend_from_slice(doc_id);
        encode_vector(&mut batch, &entry.embedding, quantization);
    }
    let crc = crc32_of(&batch);
    batch.extend_from_slice(&crc.to_le_bytes());
    batch
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn directory_entry_names(path: &Path) -> Vec<std::ffi::OsString> {
    let mut names: Vec<_> = fs::read_dir(path)
        .expect("read private WAL fixture directory")
        .map(|entry| entry.expect("read WAL fixture entry").file_name())
        .collect();
    names.sort_unstable();
    names
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn read_noatime(path: &Path) -> Vec<u8> {
    let mut file = open_readonly_noatime_nofollow(path).expect("open test WAL without atime");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .expect("read test WAL without atime");
    bytes
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn freeze_test_timestamps(path: &Path) {
    let timestamp = UNIX_EPOCH + std::time::Duration::from_secs(1_600_000_321);
    let file = OpenOptions::new()
        .read(true)
        .open(path)
        .expect("open WAL to freeze timestamps");
    file.set_times(
        std::fs::FileTimes::new()
            .set_accessed(timestamp)
            .set_modified(timestamp),
    )
    .expect("freeze WAL timestamps");
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn assert_strict_inspection_preserved_snapshot(
    directory: &Path,
    path: &Path,
    expected_bytes: &[u8],
    file_before: crate::StableFileIdentity,
    parent_before: crate::StableFileIdentity,
    entries_before: &[std::ffi::OsString],
) {
    assert_eq!(
        stable_file_identity(&fs::symlink_metadata(path).expect("WAL metadata after")),
        file_before
    );
    assert_eq!(
        stable_file_identity(&fs::symlink_metadata(directory).expect("WAL parent metadata after")),
        parent_before
    );
    assert_eq!(directory_entry_names(directory), entries_before);
    assert_eq!(read_noatime(path), expected_bytes);
    assert_eq!(
        stable_file_identity(&fs::symlink_metadata(path).expect("WAL metadata after reread")),
        file_before
    );
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn strict_inspection_classifies_absent_empty_legacy_and_identity_bound_images() {
    let directory = tempfile::tempdir().expect("private strict WAL directory");

    let absent = directory.path().join("absent.fsvi.wal");
    let absent_entries = directory_entry_names(directory.path());
    let absent_parent = stable_file_identity(
        &fs::symlink_metadata(directory.path()).expect("absent parent before"),
    );
    assert_eq!(
        inspect_wal_strict(&absent).expect("inspect absent WAL"),
        StrictWalInspection::Absent
    );
    assert_eq!(
        stable_file_identity(&fs::symlink_metadata(directory.path()).expect("absent parent after")),
        absent_parent
    );
    assert_eq!(directory_entry_names(directory.path()), absent_entries);
    assert!(!absent.exists());

    let empty = directory.path().join("empty.fsvi.wal");
    fs::write(&empty, []).expect("write empty WAL");
    freeze_test_timestamps(&empty);
    let empty_bytes = Vec::new();
    let empty_entries = directory_entry_names(directory.path());
    let empty_file =
        stable_file_identity(&fs::symlink_metadata(&empty).expect("empty metadata before"));
    let empty_parent =
        stable_file_identity(&fs::symlink_metadata(directory.path()).expect("empty parent before"));
    assert_eq!(
        inspect_wal_strict(&empty).expect("inspect empty WAL"),
        StrictWalInspection::Empty {
            whole_image_sha256: Sha256::digest([]).into(),
        }
    );
    assert_strict_inspection_preserved_snapshot(
        directory.path(),
        &empty,
        &empty_bytes,
        empty_file,
        empty_parent,
        &empty_entries,
    );

    let header_only = directory.path().join("header-only.fsvi.wal");
    let mut header_bytes = Vec::new();
    write_wal_header(&mut header_bytes, 4, Quantization::F32, 7).expect("encode legacy header");
    fs::write(&header_only, &header_bytes).expect("write header-only WAL");
    freeze_test_timestamps(&header_only);
    let header_entries = directory_entry_names(directory.path());
    let header_file =
        stable_file_identity(&fs::symlink_metadata(&header_only).expect("header metadata before"));
    let header_parent = stable_file_identity(
        &fs::symlink_metadata(directory.path()).expect("header parent before"),
    );
    let StrictWalInspection::LegacyV1(header_image) =
        inspect_wal_strict(&header_only).expect("inspect header-only WAL")
    else {
        panic!("header-only WAL was not classified as legacy v1");
    };
    assert_eq!(header_image.byte_len, WAL_HEADER_SIZE as u64);
    let expected_header_sha256: [u8; 32] = Sha256::digest(&header_bytes).into();
    assert_eq!(header_image.whole_image_sha256, expected_header_sha256);
    assert_eq!(header_image.dimension, 4);
    assert_eq!(header_image.quantization, Quantization::F32);
    assert_eq!(header_image.batch_count, 0);
    assert_eq!(header_image.entry_count, 0);
    assert_strict_inspection_preserved_snapshot(
        directory.path(),
        &header_only,
        &header_bytes,
        header_file,
        header_parent,
        &header_entries,
    );

    let multi_batch = directory.path().join("multi-batch.fsvi.wal");
    append_wal_batch(
        &multi_batch,
        &[make_entry("first", 1.0, 4)],
        4,
        Quantization::F16,
        0,
        false,
    )
    .expect("write first strict batch");
    append_wal_batch(
        &multi_batch,
        &[make_entry("second", 2.0, 4), make_entry("third", 3.0, 4)],
        4,
        Quantization::F16,
        0,
        false,
    )
    .expect("write second strict batch");
    let multi_bytes = fs::read(&multi_batch).expect("read multi-batch bytes");
    freeze_test_timestamps(&multi_batch);
    let multi_entries = directory_entry_names(directory.path());
    let multi_file =
        stable_file_identity(&fs::symlink_metadata(&multi_batch).expect("multi metadata before"));
    let multi_parent =
        stable_file_identity(&fs::symlink_metadata(directory.path()).expect("multi parent before"));
    let StrictWalInspection::LegacyV1(multi_image) =
        inspect_wal_strict(&multi_batch).expect("inspect multi-batch WAL")
    else {
        panic!("multi-batch WAL was not classified as legacy v1");
    };
    assert_eq!(
        multi_image.byte_len,
        u64::try_from(multi_bytes.len()).expect("multi byte length fits")
    );
    let expected_multi_sha256: [u8; 32] = Sha256::digest(&multi_bytes).into();
    assert_eq!(multi_image.whole_image_sha256, expected_multi_sha256);
    assert_eq!(multi_image.dimension, 4);
    assert_eq!(multi_image.quantization, Quantization::F16);
    assert_eq!(multi_image.batch_count, 2);
    assert_eq!(multi_image.entry_count, 3);
    assert_strict_inspection_preserved_snapshot(
        directory.path(),
        &multi_batch,
        &multi_bytes,
        multi_file,
        multi_parent,
        &multi_entries,
    );

    let identity_bound = directory.path().join("identity-bound.fsvi.wal");
    let identity_binding = identity_binding_v2(4, QuantizationFormat::F16);
    let identity_header = identity_binding.encode_header();
    let identity_entries = [make_entry("identity-doc", 4.0, 4)];
    let mut identity_bytes = identity_header.to_vec();
    identity_bytes.extend_from_slice(&encoded_test_batch(&identity_entries, Quantization::F16));
    fs::write(&identity_bound, &identity_bytes).expect("write identity-bound WAL");
    freeze_test_timestamps(&identity_bound);
    let bound_entries = directory_entry_names(directory.path());
    let bound_file = stable_file_identity(
        &fs::symlink_metadata(&identity_bound).expect("identity metadata before"),
    );
    let bound_parent = stable_file_identity(
        &fs::symlink_metadata(directory.path()).expect("identity parent before"),
    );
    let StrictWalInspection::IdentityBoundV2 { header, image } =
        inspect_wal_strict(&identity_bound).expect("inspect identity-bound WAL")
    else {
        panic!("identity-bound WAL was not classified as v2");
    };
    identity_binding
        .validate_header(&header.encode(), &identity_bound)
        .expect("strict inspector retained exact identity header");
    assert_eq!(
        image.byte_len,
        u64::try_from(identity_bytes.len()).expect("identity byte length fits")
    );
    let expected_identity_sha256: [u8; 32] = Sha256::digest(&identity_bytes).into();
    assert_eq!(image.whole_image_sha256, expected_identity_sha256);
    assert_eq!(image.dimension, 4);
    assert_eq!(image.quantization, Quantization::F16);
    assert_eq!(image.batch_count, 1);
    assert_eq!(image.entry_count, 1);
    assert_strict_inspection_preserved_snapshot(
        directory.path(),
        &identity_bound,
        &identity_bytes,
        bound_file,
        bound_parent,
        &bound_entries,
    );
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn strict_inspection_rejects_corrupt_or_truncated_suffix_without_recovery_mutation() {
    let directory = tempfile::tempdir().expect("private corrupt WAL directory");
    let source = directory.path().join("source.fsvi.wal");
    append_wal_batch(
        &source,
        &[make_entry("complete", 1.0, 4)],
        4,
        Quantization::F16,
        0,
        false,
    )
    .expect("write complete source WAL");
    let complete = fs::read(&source).expect("read complete source WAL");

    let mut bad_crc = complete.clone();
    *bad_crc.last_mut().expect("batch CRC byte") ^= 1;
    let mut truncated = complete.clone();
    truncated.pop();
    let mut corrupt_suffix = complete;
    corrupt_suffix.extend_from_slice(b"FWB");
    for (name, bytes) in [
        ("bad-crc", bad_crc),
        ("truncated", truncated),
        ("corrupt-suffix", corrupt_suffix),
    ] {
        let path = directory.path().join(format!("{name}.fsvi.wal"));
        fs::write(&path, &bytes).expect("write malformed strict WAL");
        freeze_test_timestamps(&path);
        let entries_before = directory_entry_names(directory.path());
        let file_before =
            stable_file_identity(&fs::symlink_metadata(&path).expect("malformed metadata before"));
        let parent_before = stable_file_identity(
            &fs::symlink_metadata(directory.path()).expect("malformed parent before"),
        );

        assert!(matches!(
            inspect_wal_strict(&path),
            Err(FsviAdmissionError::Index(
                SearchError::IndexCorrupted { .. }
            ))
        ));
        assert_strict_inspection_preserved_snapshot(
            directory.path(),
            &path,
            &bytes,
            file_before,
            parent_before,
            &entries_before,
        );
    }
}

#[test]
fn identity_bound_header_round_trips_full_width_generation() {
    let path = Path::new("/test/identity-bound.fsvi.wal");
    let binding = identity_binding_v2(37, QuantizationFormat::F16);
    let encoded = binding.encode_header();
    let decoded = WalIdentityHeaderV2::decode(&encoded, path).expect("decode v2 header");

    binding
        .validate_header(&encoded, path)
        .expect("exact identity binding");
    assert_eq!(decoded.generation().sequence, u64::MAX);
    assert_eq!(decoded.generation().nonce, [0xa5; 16]);
    assert_eq!(decoded.dimension(), 37);
    assert_eq!(decoded.quantization(), Quantization::F16);
    assert_ne!(
        *decoded.identity_bundle_fingerprint(),
        [0; SHA256_FINGERPRINT_BYTES]
    );
    assert_ne!(*decoded.space_fingerprint(), [0; SHA256_FINGERPRINT_BYTES]);
    assert_ne!(
        *decoded.producer_fingerprint(),
        [0; SHA256_FINGERPRINT_BYTES]
    );
    assert_ne!(*decoded.input_fingerprint(), [0; SHA256_FINGERPRINT_BYTES]);
    assert_ne!(
        *decoded.storage_fingerprint(),
        [0; SHA256_FINGERPRINT_BYTES]
    );
    assert_eq!(decoded.encode(), encoded);
}

#[test]
fn identity_bound_header_rejects_every_component_fingerprint_drift() {
    let path = Path::new("/test/identity-bound-fingerprint.fsvi.wal");
    let binding = identity_binding_v2(8, QuantizationFormat::F16);
    let original = binding.encode_header();

    for (field, offset) in [
        (
            "embedding identity bundle fingerprint",
            IDENTITY_BUNDLE_FINGERPRINT_OFFSET,
        ),
        ("embedding space fingerprint", SPACE_FINGERPRINT_OFFSET),
        (
            "embedding producer fingerprint",
            PRODUCER_FINGERPRINT_OFFSET,
        ),
        ("embedding input fingerprint", INPUT_FINGERPRINT_OFFSET),
        ("vector storage fingerprint", STORAGE_FINGERPRINT_OFFSET),
    ] {
        let mut drifted = original;
        drifted[offset] ^= 1;
        refresh_identity_header_crc(&mut drifted);
        let error = binding
            .validate_header(&drifted, path)
            .expect_err("fingerprint drift must fail closed");
        assert!(
            error.to_string().contains(field),
            "expected {field} diagnostic, got {error}"
        );
    }
}

#[test]
fn identity_bound_header_rejects_generation_dimension_and_quantization_drift() {
    let path = Path::new("/test/identity-bound-storage.fsvi.wal");
    let binding = identity_binding_v2(8, QuantizationFormat::F16);
    let original = binding.encode_header();

    let mut sequence_drift = original;
    sequence_drift[20] ^= 1;
    refresh_identity_header_crc(&mut sequence_drift);
    assert!(
        binding
            .validate_header(&sequence_drift, path)
            .expect_err("generation sequence drift")
            .to_string()
            .contains("artifact generation mismatch")
    );

    let mut nonce_drift = original;
    nonce_drift[28] ^= 1;
    refresh_identity_header_crc(&mut nonce_drift);
    assert!(
        binding
            .validate_header(&nonce_drift, path)
            .expect_err("generation nonce drift")
            .to_string()
            .contains("artifact generation mismatch")
    );

    let mut schema_drift = original;
    schema_drift[16..18].copy_from_slice(&2_u16.to_le_bytes());
    refresh_identity_header_crc(&mut schema_drift);
    assert!(
        binding
            .validate_header(&schema_drift, path)
            .expect_err("generation schema drift")
            .to_string()
            .contains("artifact generation is invalid")
    );

    let mut dimension_drift = original;
    dimension_drift[8..12].copy_from_slice(&9_u32.to_le_bytes());
    refresh_identity_header_crc(&mut dimension_drift);
    assert!(
        binding
            .validate_header(&dimension_drift, path)
            .expect_err("dimension drift")
            .to_string()
            .contains("storage dimension mismatch")
    );

    let mut quantization_drift = original;
    quantization_drift[12] = quantization_to_wire(Quantization::F32);
    refresh_identity_header_crc(&mut quantization_drift);
    assert!(
        binding
            .validate_header(&quantization_drift, path)
            .expect_err("quantization drift")
            .to_string()
            .contains("storage quantization mismatch")
    );
}

#[test]
fn identity_bound_header_rejects_structural_corruption() {
    let path = Path::new("/test/identity-bound-corrupt.fsvi.wal");
    let binding = identity_binding_v2(8, QuantizationFormat::F16);
    let original = binding.encode_header();

    let mut bad_version = original;
    bad_version[4..6].copy_from_slice(&99_u16.to_le_bytes());
    assert!(WalIdentityHeaderV2::decode(&bad_version, path).is_err());

    let mut bad_size = original;
    bad_size[6..8].copy_from_slice(&207_u16.to_le_bytes());
    assert!(WalIdentityHeaderV2::decode(&bad_size, path).is_err());

    let mut reserved = original;
    reserved[13] = 1;
    refresh_identity_header_crc(&mut reserved);
    assert!(
        WalIdentityHeaderV2::decode(&reserved, path)
            .expect_err("reserved bytes")
            .to_string()
            .contains("reserved")
    );

    let mut bad_crc = original;
    bad_crc[IDENTITY_BOUND_WAL_CRC_OFFSET] ^= 1;
    assert!(
        WalIdentityHeaderV2::decode(&bad_crc, path)
            .expect_err("CRC drift")
            .to_string()
            .contains("CRC")
    );

    let mut zero_fingerprint = original;
    zero_fingerprint[IDENTITY_BUNDLE_FINGERPRINT_OFFSET..SPACE_FINGERPRINT_OFFSET].fill(0);
    refresh_identity_header_crc(&mut zero_fingerprint);
    assert!(
        WalIdentityHeaderV2::decode(&zero_fingerprint, path)
            .expect_err("zero fingerprint")
            .to_string()
            .contains("must not be all zero")
    );

    assert!(
        WalIdentityHeaderV2::decode(&original[..IDENTITY_BOUND_WAL_HEADER_SIZE - 1], path).is_err()
    );
}

#[test]
fn identity_binding_requires_exact_supported_fsvi_v2_storage() {
    let generation = ArtifactGenerationIdentityV1::new(1, [1; 16]).expect("valid generation");
    let mut legacy = EmbeddingIdentityBundleV1::explicit_test_model("wal-legacy", 8);
    legacy.storage.format = "fsvi-v1".to_owned();
    legacy.storage.quantization = QuantizationFormat::F16;
    legacy.storage.endianness = "little-endian".to_owned();
    let legacy = legacy.freeze().expect("valid legacy identity");
    assert!(
        WalIdentityBindingV2::new(generation, &legacy)
            .expect_err("legacy storage must not bind")
            .to_string()
            .contains("storage.format")
    );

    let mut unsupported = EmbeddingIdentityBundleV1::explicit_test_model("wal-unsupported", 8);
    unsupported.storage.format = "fsvi-v2".to_owned();
    unsupported.storage.quantization = QuantizationFormat::Int8;
    unsupported.storage.endianness = "little-endian".to_owned();
    let unsupported = unsupported.freeze().expect("valid unsupported identity");
    assert!(
        WalIdentityBindingV2::new(generation, &unsupported)
            .expect_err("unsupported quantization must not bind")
            .to_string()
            .contains("storage.quantization")
    );
}

// ─── bd-xx286: WAL torn-header policy and single-writer arbitration ───

/// A second appender must be refused, not allowed to resolve its own
/// append offset behind the first one's back.
///
/// The planted negative is the exact interleaving the unarbitrated path
/// permits: a foreign writer holds the sidecar while this call resolves
/// `SeekFrom::End(0)`. Without an arbiter both writers derive the same
/// end offset and the second `write_all` lands on top of the first
/// batch, so an acknowledged append is silently lost.
#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn a_second_wal_appender_is_refused_while_a_foreign_writer_holds_the_sidecar() {
    let path = temp_wal_path("single-writer-arbitration");
    let dimension = 4;
    let quantization = Quantization::F16;

    append_wal_batch(
        &path,
        &[make_entry("doc-first", 1.0, dimension)],
        dimension,
        quantization,
        0,
        false,
    )
    .expect("first append establishes the sidecar");
    let committed_len = fs::metadata(&path)
        .expect("sidecar metadata after the first append")
        .len();

    // A foreign writer — another process under the single-writer
    // invariant — holds the sidecar for the duration of its own append.
    let foreign = OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("foreign writer opens the sidecar");
    foreign
        .try_lock()
        .expect("foreign writer takes the sidecar writer lock");

    let refused = append_wal_batch(
        &path,
        &[make_entry("doc-second", 2.0, dimension)],
        dimension,
        quantization,
        0,
        false,
    );
    assert!(
        matches!(
            &refused,
            Err(SearchError::InvalidConfig { field, .. }) if field == "wal.writer_lock"
        ),
        "a second appender must be refused while another writer holds the sidecar, observed {refused:?}"
    );
    assert_eq!(
        fs::metadata(&path)
            .expect("sidecar metadata after the refusal")
            .len(),
        committed_len,
        "a refused append must not have written any bytes"
    );

    // Once the foreign writer is done, the same append succeeds and the
    // already-committed batch is still there.
    foreign
        .unlock()
        .expect("foreign writer explicitly releases its completed mutation");
    drop(foreign);
    append_wal_batch(
        &path,
        &[make_entry("doc-second", 2.0, dimension)],
        dimension,
        quantization,
        0,
        false,
    )
    .expect("append succeeds once the sidecar is free");

    let (entries, _, _) =
        read_wal(&path, dimension, quantization).expect("read the arbitrated sidecar");
    let ids: Vec<&str> = entries.iter().map(|entry| entry.doc_id.as_str()).collect();
    assert_eq!(
        ids,
        vec!["doc-first", "doc-second"],
        "both acknowledged batches must survive arbitration"
    );

    std::fs::remove_file(&path).ok();
}

/// Exercise the kernel's real duplicated-open-description behavior, the
/// same lifetime issue an inherited descriptor can cause across fork/exec.
/// This is descriptor-level evidence, not a claim to have executed fork.
#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn wal_writer_release_survives_a_retained_duplicate_without_losing_batches() {
    let path = temp_wal_path("writer-retained-duplicate");
    let dimension = 4;
    let quantization = Quantization::F16;
    append_wal_batch(
        &path,
        &[make_entry("doc-first", 1.0, dimension)],
        dimension,
        quantization,
        0,
        false,
    )
    .expect("first committed batch");
    let committed = fs::read(&path).expect("original WAL bytes");
    let owner = acquire_wal_writer_lock(&path, OpenOptions::new().write(true).open(&path).unwrap())
        .expect("take writer ownership");
    let retained = owner.file.try_clone().expect("retain the same description");

    let refused = append_wal_batch(
        &path,
        &[make_entry("doc-second", 2.0, dimension)],
        dimension,
        quantization,
        0,
        false,
    );
    assert!(matches!(
        refused,
        Err(SearchError::InvalidConfig { field, .. }) if field == "wal.writer_lock"
    ));
    assert_eq!(fs::read(&path).unwrap(), committed);

    drop(owner);
    append_wal_batch(
        &path,
        &[make_entry("doc-second", 2.0, dimension)],
        dimension,
        quantization,
        0,
        false,
    )
    .expect("owner release frees the lock while its duplicate stays open");
    let after = fs::read(&path).unwrap();
    assert!(
        after.starts_with(&committed),
        "original batch bytes survive"
    );
    assert_eq!(
        retained.metadata().unwrap().len(),
        fs::metadata(&path).unwrap().len(),
        "the duplicate remains open through the successful append"
    );
    let (entries, _, _) = read_wal(&path, dimension, quantization).unwrap();
    let ids = entries
        .iter()
        .map(|entry| entry.doc_id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(ids, vec!["doc-first", "doc-second"]);
}

/// Simulate the PID guard's child-side branch while retaining a real
/// duplicate for the creator. No unsafe fork is needed for this boundary.
#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn wal_writer_noncreator_drop_cannot_unlock_the_creators_description() {
    let path = temp_wal_path("writer-noncreator-drop");
    let dimension = 4;
    let quantization = Quantization::F16;
    append_wal_batch(
        &path,
        &[make_entry("doc-first", 1.0, dimension)],
        dimension,
        quantization,
        0,
        false,
    )
    .unwrap();
    let committed = fs::read(&path).unwrap();
    let mut child_copy =
        acquire_wal_writer_lock(&path, OpenOptions::new().write(true).open(&path).unwrap())
            .unwrap();
    let creator = child_copy.file.try_clone().unwrap();
    child_copy.creator_pid = std::process::id().wrapping_add(1);
    drop(child_copy);

    let refused = append_wal_batch(
        &path,
        &[make_entry("doc-second", 2.0, dimension)],
        dimension,
        quantization,
        0,
        false,
    );
    assert!(matches!(
        refused,
        Err(SearchError::InvalidConfig { field, .. }) if field == "wal.writer_lock"
    ));
    assert_eq!(fs::read(&path).unwrap(), committed);
    creator
        .unlock()
        .expect("only the creator releases its lock");
    append_wal_batch(
        &path,
        &[make_entry("doc-second", 2.0, dimension)],
        dimension,
        quantization,
        0,
        false,
    )
    .expect("append succeeds after creator release");
    let (entries, _, _) = read_wal(&path, dimension, quantization).unwrap();
    let ids = entries
        .iter()
        .map(|entry| entry.doc_id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(ids, vec!["doc-first", "doc-second"]);
}

/// A torn header at or above `WAL_HEADER_SIZE` must be quarantined, not
/// left to brick every future open of the index.
///
/// The planted negative is the asymmetry the bead names: a sidecar
/// shorter than the header is already ignored, and a torn batch TAIL is
/// already tolerated, but a torn header of exactly this length is a hard
/// error forever. The corrupt bytes must survive the recovery — a
/// quarantine that deletes or truncates the evidence would make this
/// test pass while destroying the only copy of the lost appends.
#[cfg(any(target_os = "linux", target_os = "android"))]
#[test]
fn a_torn_wal_header_is_quarantined_instead_of_bricking_every_read() {
    let path = temp_wal_path("torn-header-quarantine");
    let dimension = 4;
    let quantization = Quantization::F16;

    append_wal_batch(
        &path,
        &[make_entry("doc-live", 1.0, dimension)],
        dimension,
        quantization,
        0,
        false,
    )
    .expect("establish a valid sidecar");
    let valid = fs::read(&path).expect("read the valid sidecar");
    assert!(valid.len() > WAL_HEADER_SIZE);

    // A torn write that landed a full-length header with the wrong magic.
    // Shorter than this is already ignored; a torn batch tail is already
    // tolerated; only this window is fatal today.
    let mut torn = valid.clone();
    torn[0] ^= 0xff;
    fs::write(&path, &torn).expect("write the torn header");

    // The pure reader must still fail, and must still leave the bytes
    // alone: `observe_tier` documents that classification is
    // side-effect-free and shares this exact function.
    assert!(
        matches!(
            read_wal(&path, dimension, quantization),
            Err(SearchError::IndexCorrupted { .. })
        ),
        "the side-effect-free reader must keep reporting the corruption"
    );
    assert_eq!(
        fs::read(&path).expect("sidecar after the pure read"),
        torn,
        "the side-effect-free reader must not touch the sidecar"
    );

    let recovered = read_wal_recovering_torn_header(&path, dimension, quantization)
        .expect("a torn header must not brick a write-capable open");
    assert!(
        recovered.0.is_empty(),
        "a quarantined sidecar must yield no entries, observed {:?}",
        recovered.0
    );
    assert!(
        !path.exists(),
        "the torn sidecar must be moved out of the live pathname"
    );

    let directory = snapshot_parent_or_current(&path);
    let quarantined: Vec<_> = fs::read_dir(directory)
        .expect("read the sidecar directory")
        .filter_map(|entry| {
            let entry = entry.expect("read a directory entry");
            let name = entry.file_name();
            name.to_string_lossy()
                .starts_with(&format!(
                    "{}.corrupt",
                    path.file_name()
                        .expect("sidecar file name")
                        .to_string_lossy()
                ))
                .then(|| entry.path())
        })
        .collect();
    assert_eq!(
        quarantined.len(),
        1,
        "exactly one quarantine copy must exist, observed {quarantined:?}"
    );
    assert_eq!(
        fs::read(&quarantined[0]).expect("read the quarantined bytes"),
        torn,
        "the quarantine must preserve the corrupt bytes verbatim"
    );

    for stale in quarantined {
        std::fs::remove_file(stale).ok();
    }
    std::fs::remove_file(&path).ok();
}
