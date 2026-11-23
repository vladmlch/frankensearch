use super::super::platform::{stat_device_as_u64_for_test, validate_macos_object_flags_for_test};
use super::super::{
    ConfinedGenerationPath, GENERATION_ROOT_AUTHORITY_FILE_BYTES, GenerationFileExpectation,
    GenerationRootAnchorLayout, GenerationRootError, GenerationRootErrorKind, GenerationRootStage,
    PreopenedGenerationRootAnchors, QualifiedGenerationRoot as NamedGenerationRoot,
};
use crate::fd_acl::{ExtendedAclPresence, extended_acl_presence};
use sha2::{Digest, Sha256};
use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static FIXTURE_SERIAL: AtomicU64 = AtomicU64::new(0);
static FIXTURE_SLOT_SERIAL: AtomicU64 = AtomicU64::new(0);
static FIXTURE_BASE: OnceLock<PathBuf> = OnceLock::new();
const FIXTURE_SLOT_COUNT: u64 = 512;

#[test]
fn apfs_object_flag_profile_accepts_only_zero() {
    validate_macos_object_flags_for_test(0)
        .expect("the narrow APFS object-flag profile accepts zero");
    for flags in [libc::UF_IMMUTABLE, libc::UF_APPEND, libc::UF_NODUMP] {
        let error = validate_macos_object_flags_for_test(flags)
            .expect_err("every nonzero APFS object flag is outside the profile");
        assert_eq!(
            error.kind(),
            GenerationRootErrorKind::UnsupportedObjectFlags
        );
        assert_eq!(error.stage(), GenerationRootStage::OpenRegularFile);
        assert_eq!(error.expected(), Some(0));
        assert_eq!(error.observed(), Some(u64::from(flags)));
    }
}

#[test]
fn macos_signed_device_ids_are_normalized_by_sign_extension() {
    assert_eq!(stat_device_as_u64_for_test(0), 0);
    assert_eq!(stat_device_as_u64_for_test(1), 1);
    assert_eq!(stat_device_as_u64_for_test(-1), u64::MAX);
    assert_eq!(
        stat_device_as_u64_for_test(i32::MIN),
        u64::from_ne_bytes(i64::from(i32::MIN).to_ne_bytes())
    );
}

pub(super) fn set_empty_acl(path: &Path) {
    let output = Command::new("/bin/chmod")
        .arg("-N")
        .arg(path)
        .output()
        .expect("the physical APFS ACL receipt requires /bin/chmod");
    assert!(
        output.status.success(),
        "chmod -N failed: status={:?}, stdout={}, stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let retained =
        File::open(path).expect("the cleared APFS fixture must remain openable for ACL proof");
    assert_eq!(
        extended_acl_presence(retained.as_fd())
            .expect("the descriptor-bound ACL absence probe must succeed"),
        ExtendedAclPresence::Absent,
        "an admissible APFS fixture must have no extended ACL entries"
    );
}

pub(super) fn set_non_empty_acl(path: &Path, allow: bool) {
    let user = std::env::var("USER").expect("USER must identify an ACL principal");
    let disposition = if allow { "allow" } else { "deny" };
    let entry = format!("user:{user} {disposition} write");
    let output = Command::new("/bin/chmod")
        .args(["+a", &entry])
        .arg(path)
        .output()
        .expect("the physical APFS ACL receipt requires /bin/chmod");
    assert!(
        output.status.success(),
        "chmod +a failed: entry={entry:?}, status={:?}, stdout={}, stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let retained =
        File::open(path).expect("the ACL-bearing APFS fixture must remain openable for ACL proof");
    assert_eq!(
        extended_acl_presence(retained.as_fd())
            .expect("the descriptor-bound ACL presence probe must succeed"),
        ExtendedAclPresence::Present,
        "the installed {disposition} ACL must be observed through the retained descriptor"
    );
}

pub(super) fn run_chflags(flag: &str, path: &Path) {
    let output = Command::new("/usr/bin/chflags")
        .arg(flag)
        .arg(path)
        .output()
        .expect("the physical APFS object-flag receipt requires chflags");
    assert!(
        output.status.success(),
        "chflags failed: flag={flag}, status={:?}, stdout={}, stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

pub(super) fn fixture_root(label: &str) -> PathBuf {
    let slot = FIXTURE_SLOT_SERIAL.fetch_add(1, Ordering::Relaxed);
    assert!(
        slot < FIXTURE_SLOT_COUNT,
        "the retained APFS fixture-slot pool must cover every test root"
    );
    let slot_path = secure_fixture_base().join(format!("slot-{slot:03}"));
    fixture_root_under(&slot_path, label)
}

fn secure_fixture_base() -> &'static Path {
    FIXTURE_BASE
        .get_or_init(|| {
            let private_tmp = std::env::temp_dir()
                .canonicalize()
                .expect("the process-private macOS temporary route must canonicalize");
            assert!(
                private_tmp.is_absolute(),
                "the canonical macOS temporary route must be absolute"
            );
            let epoch_nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos());
            for attempt in 0_u64..64 {
                let candidate = private_tmp.join(format!(
                    "frankensearch-generation-root-tests-{}-{epoch_nanos}-{attempt}",
                    std::process::id()
                ));
                match DirBuilder::new().mode(0o700).create(&candidate) {
                    Ok(()) => {
                        fs::set_permissions(&candidate, fs::Permissions::from_mode(0o700))
                            .expect("persistent APFS fixture base mode should be settable");
                        set_empty_acl(&candidate);
                        for slot in 0..FIXTURE_SLOT_COUNT {
                            let slot_path = candidate.join(format!("slot-{slot:03}"));
                            DirBuilder::new()
                                .mode(0o700)
                                .create(&slot_path)
                                .expect("each retained APFS fixture slot should be creatable");
                            fs::set_permissions(&slot_path, fs::Permissions::from_mode(0o700))
                                .expect("each retained APFS fixture slot mode should be private");
                            set_empty_acl(&slot_path);
                        }
                        return candidate;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(error) => {
                        panic!("failed to create retained APFS fixture base: {error}");
                    }
                }
            }
            panic!("failed to allocate a retained process APFS fixture base");
        })
        .as_path()
}

fn fixture_root_under(base: &Path, label: &str) -> PathBuf {
    fs::create_dir_all(base).expect("persistent APFS fixture base should be creatable");
    let epoch_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    for attempt in 0_u64..64 {
        let serial = FIXTURE_SERIAL.fetch_add(1, Ordering::Relaxed);
        let candidate = base.join(format!(
            "{}-{}-{epoch_nanos}-{serial}-{attempt}",
            std::process::id(),
            label
        ));
        match DirBuilder::new().mode(0o700).create(&candidate) {
            Ok(()) => {
                fs::set_permissions(&candidate, fs::Permissions::from_mode(0o700))
                    .expect("persistent APFS fixture mode should be settable");
                set_empty_acl(&candidate);
                return candidate;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                panic!("failed to create persistent APFS test fixture: {error}");
            }
        }
    }
    panic!("failed to allocate a unique persistent APFS test fixture");
}

pub(super) fn private_dir(parent: &Path, name: &str) -> PathBuf {
    let path = parent.join(name);
    DirBuilder::new()
        .mode(0o700)
        .create(&path)
        .expect("private APFS directory should be creatable");
    fs::set_permissions(&path, fs::Permissions::from_mode(0o700))
        .expect("private APFS directory mode should be settable");
    set_empty_acl(&path);
    path
}

pub(super) fn observe_case_sensitive(root: &Path) -> bool {
    private_file(root, "case-probe", b"a");
    let upper = root.join("CASE-PROBE");
    match OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&upper)
    {
        Ok(mut file) => {
            file.write_all(b"b")
                .expect("case-sensitivity probe byte should be writable");
            file.sync_all()
                .expect("case-sensitivity probe byte should be durable");
            fs::set_permissions(&upper, fs::Permissions::from_mode(0o400))
                .expect("case-sensitivity probe should be sealed read-only");
            set_empty_acl(&upper);
            true
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => false,
        Err(error) => panic!("case-sensitivity probe failed unexpectedly: {error}"),
    }
}

pub(super) fn private_file(parent: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let path = parent.join(name);
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&path)
        .expect("private APFS file should be creatable");
    file.write_all(bytes)
        .expect("persistent APFS fixture bytes should be writable");
    file.sync_all()
        .expect("persistent APFS fixture bytes should be durable");
    fs::set_permissions(&path, fs::Permissions::from_mode(0o400))
        .expect("immutable APFS fixture should be sealed read-only");
    set_empty_acl(&path);
    path
}

pub(super) fn control_file(parent: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let path = parent.join(name);
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&path)
        .expect("private APFS control file should be creatable");
    file.write_all(bytes)
        .expect("persistent APFS control bytes should be writable");
    file.sync_all()
        .expect("persistent APFS control bytes should be durable");
    set_empty_acl(&path);
    path
}

pub(super) fn authority_image(fill: u8) -> Vec<u8> {
    vec![
        fill;
        usize::try_from(GENERATION_ROOT_AUTHORITY_FILE_BYTES)
            .expect("the frozen authority length fits usize")
    ]
}

pub(super) fn named_root_fixture(
    label: &str,
    lock_bytes: &[u8],
    authority_fill: u8,
) -> (PathBuf, PathBuf, PathBuf, GenerationRootAnchorLayout) {
    let root = fixture_root(label);
    let lock_path = control_file(&root, "LOCK", lock_bytes);
    let authority_path = control_file(&root, "AUTHORITY", &authority_image(authority_fill));
    let layout = GenerationRootAnchorLayout::new(
        u64::try_from(lock_bytes.len()).expect("the lock fixture length fits u64"),
    )
    .expect("the named lock fixture length is bounded");
    (root, lock_path, authority_path, layout)
}

pub(super) fn admit_named_root(
    root_path: &Path,
    lock_path: &Path,
    authority_path: &Path,
    layout: GenerationRootAnchorLayout,
) -> NamedGenerationRoot {
    NamedGenerationRoot::admit_preopened(
        root_path,
        layout,
        PreopenedGenerationRootAnchors::new(
            preopened_control(lock_path),
            preopened_control(authority_path),
        ),
    )
    .expect("the exact provider-labelled anchor tuple should qualify")
}

pub(super) fn rewrite_control_file(path: &Path, bytes: &[u8]) {
    let expected_len = u64::try_from(bytes.len()).expect("fixture length fits u64");
    assert_eq!(
        fs::metadata(path)
            .expect("control metadata should be readable")
            .len(),
        expected_len,
        "in-place control rewrites must retain the frozen physical length"
    );
    let mut file = OpenOptions::new()
        .write(true)
        .open(path)
        .expect("the mutable APFS control anchor should be writable");
    file.seek(SeekFrom::Start(0))
        .expect("the APFS control anchor should seek to its first byte");
    file.write_all(bytes)
        .expect("the complete in-place APFS control image should be writable");
    file.sync_all()
        .expect("the in-place APFS control rewrite should be durable");
}

pub(super) fn mapped_control_file(parent: &Path, name: &str) -> (PathBuf, crate::VectorIndex) {
    let path = parent.join(name);
    let mut writer = crate::VectorIndex::create(&path, "generation-root-test", 1)
        .expect("mapped APFS control fixture writer should be creatable");
    writer
        .write_record("control", &[1.0])
        .expect("mapped APFS control fixture record should be valid");
    writer
        .finish()
        .expect("mapped APFS control fixture should be durable");
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600))
        .expect("mapped APFS control fixture should use the control mode");
    set_empty_acl(&path);
    let mapping = crate::VectorIndex::open(&path)
        .expect("mapped APFS control fixture should retain a shared writable mapping");
    (path, mapping)
}

pub(super) fn mapped_immutable_file(parent: &Path, name: &str) -> (PathBuf, crate::VectorIndex) {
    let (path, mapping) = mapped_control_file(parent, name);
    fs::set_permissions(&path, fs::Permissions::from_mode(0o400))
        .expect("pre-held APFS mapping should seal as an immutable artifact");
    (path, mapping)
}

pub(super) fn mutate_mapping(mapping: &mut crate::VectorIndex) {
    let crate::VectorIndexData::Mutable(bytes) = &mut mapping.data else {
        panic!("legacy VectorIndex::open must retain a writable mapping");
    };
    let byte = bytes
        .last_mut()
        .expect("mapped APFS control fixture must contain at least one byte");
    *byte ^= 0x01;
}

pub(super) fn preopened_immutable(path: &Path) -> OwnedFd {
    File::open(path)
        .expect("trusted test provider should open the known regular artifact read-only")
        .into()
}

pub(super) fn preopened_control(path: &Path) -> OwnedFd {
    OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .expect("trusted test provider should open the known regular control read-write")
        .into()
}

pub(super) fn preopened_with_status_flag(path: &Path, flag: libc::c_int) -> OwnedFd {
    rustix::fs::open(
        path,
        rustix::fs::OFlags::from_bits_retain(flag.unsigned_abs()) | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .expect("trusted test provider should open the status-flag fixture")
}

pub(super) fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

pub(super) fn immutable_expectation(bytes: &[u8]) -> GenerationFileExpectation {
    GenerationFileExpectation::immutable(
        u64::try_from(bytes.len()).expect("fixture length should fit u64"),
        digest(bytes),
    )
    .expect("small fixture expectation should be accepted")
}

pub(super) fn confined(path: &str) -> ConfinedGenerationPath {
    ConfinedGenerationPath::parse(Path::new(path))
        .expect("static relative APFS fixture path should parse")
}

pub(super) fn injected(stage: GenerationRootStage) -> GenerationRootError {
    GenerationRootError::new(GenerationRootErrorKind::Io, stage)
}

mod durability_and_apfs;
mod guards_and_routes;
