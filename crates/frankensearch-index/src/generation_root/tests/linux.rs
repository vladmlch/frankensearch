use super::super::platform::{ProcBridgeProbePhase, TestBoundary, install_test_hook};
use super::super::{
    ConfinedGenerationPath, GENERATION_ROOT_AUTHORITY_FILE_BYTES, GenerationFileExpectation,
    GenerationRootAnchorLayout, GenerationRootError, GenerationRootErrorKind, GenerationRootStage,
    QualifiedGenerationDirectory as QualifiedGenerationRoot,
};
use sha2::{Digest, Sha256};
use std::ffi::OsStr;
use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::os::fd::AsFd;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt, symlink};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

static FIXTURE_SERIAL: AtomicU64 = AtomicU64::new(0);
static FIXTURE_SLOT_SERIAL: AtomicU64 = AtomicU64::new(0);
static FIXTURE_BASE: OnceLock<PathBuf> = OnceLock::new();
const FIXTURE_SLOT_COUNT: u64 = 512;

pub(super) fn fixture_root(label: &str) -> PathBuf {
    let slot = FIXTURE_SLOT_SERIAL.fetch_add(1, Ordering::Relaxed);
    assert!(
        slot < FIXTURE_SLOT_COUNT,
        "the retained fixture-slot pool must cover every test root"
    );
    let slot_path = secure_fixture_base().join(format!("slot-{slot:03}"));
    fixture_root_under(&slot_path, label)
}

fn secure_fixture_base() -> &'static Path {
    FIXTURE_BASE
        .get_or_init(|| {
            let home = std::env::var_os("HOME")
                .map(PathBuf::from)
                .expect("HOME must identify the process-private fixture ancestor");
            assert!(home.is_absolute(), "HOME must be an absolute route");

            // bd-54h2l LOW A: the per-run base still embeds pid + epoch
            // nanos (route qualification requires an ancestor chain this
            // user owns privately, so these fixtures cannot move to a
            // shared scratch root — see the note below), but it now nests
            // under ONE stable private parent instead of littering $HOME
            // with a fresh top-level entry per test process. Per-run
            // isolation is unchanged and no directory is deleted here;
            // the sprawl is simply confined to a single, purgeable place.
            //
            // Rooting these under `std::env::temp_dir()` was tried and
            // REVERTED: it fails route qualification with
            // `WrongOwner` at component_index 0, because a shared temp
            // root is not owned by this user. $HOME is load-bearing.
            let parent = home.join(".frankensearch-generation-root-tests");
            match DirBuilder::new().mode(0o700).create(&parent) {
                Ok(()) => {
                    fs::set_permissions(&parent, fs::Permissions::from_mode(0o700))
                        .expect("fixture parent mode should be settable");
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    // Adopting a directory left by an earlier run is
                    // fine, but only after proving it is still the thing
                    // we created: a real directory (never a symlink),
                    // owned by whoever owns HOME, and private.
                    let existing = fs::symlink_metadata(&parent)
                        .expect("the retained fixture parent should be stattable");
                    assert!(
                        existing.is_dir(),
                        "the retained fixture parent must be a directory"
                    );
                    let home_owner = fs::symlink_metadata(&home)
                        .expect("HOME should be stattable")
                        .uid();
                    assert_eq!(
                        existing.uid(),
                        home_owner,
                        "the retained fixture parent must be owned by the HOME owner"
                    );
                    assert_eq!(
                        existing.permissions().mode() & 0o777,
                        0o700,
                        "the retained fixture parent must stay private"
                    );
                }
                Err(error) => {
                    panic!("failed to create the retained fixture parent: {error}")
                }
            }
            let epoch_nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos());
            for attempt in 0_u64..64 {
                let candidate =
                    parent.join(format!("{}-{epoch_nanos}-{attempt}", std::process::id()));
                match DirBuilder::new().mode(0o700).create(&candidate) {
                    Ok(()) => {
                        fs::set_permissions(&candidate, fs::Permissions::from_mode(0o700))
                            .expect("fixture base mode should be settable");
                        for slot in 0..FIXTURE_SLOT_COUNT {
                            let slot_path = candidate.join(format!("slot-{slot:03}"));
                            DirBuilder::new()
                                .mode(0o700)
                                .create(&slot_path)
                                .expect("each retained fixture slot should be creatable");
                            fs::set_permissions(&slot_path, fs::Permissions::from_mode(0o700))
                                .expect("each retained fixture slot mode should be private");
                        }
                        return candidate;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(error) => panic!("failed to create retained fixture base: {error}"),
                }
            }
            panic!("failed to allocate a retained process fixture base");
        })
        .as_path()
}

fn fixture_root_under(base: &Path, label: &str) -> PathBuf {
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
                    .expect("fixture root mode should be settable");
                return candidate;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => panic!("failed to create persistent test fixture: {error}"),
        }
    }
    panic!("failed to allocate a unique persistent test fixture");
}

pub(super) fn run_btrfs_fixture_command(arguments: &[&OsStr]) {
    let output = Command::new("btrfs")
        .args(arguments)
        .output()
        .expect("the physical Btrfs receipt requires the btrfs utility");
    assert!(
        output.status.success(),
        "Btrfs fixture command failed: status={:?}, stdout={}, stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn run_bind_mount_fixture_command(program: &str, arguments: &[&OsStr]) {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .expect("the physical absolute-route receipt requires mount utilities");
    assert!(
        output.status.success(),
        "bind-mount fixture command failed: program={program}, status={:?}, stdout={}, stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

pub(super) struct BindMountFixture {
    source: PathBuf,
    target: PathBuf,
    mounted: bool,
}

impl BindMountFixture {
    pub(super) fn new(source: &Path, target: &Path) -> Self {
        run_bind_mount_fixture_command(
            "mount",
            &[OsStr::new("--bind"), source.as_os_str(), target.as_os_str()],
        );
        Self {
            source: source.to_path_buf(),
            target: target.to_path_buf(),
            mounted: true,
        }
    }

    pub(super) fn remount_same_source(&mut self) {
        if self.mounted {
            run_bind_mount_fixture_command(
                "umount",
                &[OsStr::new("--lazy"), self.target.as_os_str()],
            );
            self.mounted = false;
        }
        run_bind_mount_fixture_command(
            "mount",
            &[
                OsStr::new("--bind"),
                self.source.as_os_str(),
                self.target.as_os_str(),
            ],
        );
        self.mounted = true;
    }
}

impl Drop for BindMountFixture {
    fn drop(&mut self) {
        if self.mounted {
            let _ = Command::new("umount").arg(&self.target).status();
            self.mounted = false;
        }
    }
}

pub(super) fn run_chattr_fixture_command(flag: &str, path: &Path) {
    let output = Command::new("chattr")
        .arg(flag)
        .arg(path)
        .output()
        .expect("the physical attribute receipt requires the chattr utility");
    assert!(
        output.status.success(),
        "chattr fixture command failed: flag={flag}, status={:?}, stdout={}, stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

pub(super) struct ChattrReset {
    flag: &'static str,
    path: PathBuf,
    armed: bool,
}

impl ChattrReset {
    pub(super) fn new(flag: &'static str, path: &Path) -> Self {
        Self {
            flag,
            path: path.to_path_buf(),
            armed: true,
        }
    }

    pub(super) fn clear(&mut self) {
        if self.armed {
            run_chattr_fixture_command(self.flag, &self.path);
            self.armed = false;
        }
    }
}

impl Drop for ChattrReset {
    fn drop(&mut self) {
        if self.armed {
            let _ = Command::new("chattr")
                .arg(self.flag)
                .arg(&self.path)
                .status();
        }
    }
}

pub(super) fn private_dir(parent: &Path, name: &str) -> PathBuf {
    let path = parent.join(name);
    DirBuilder::new()
        .mode(0o700)
        .create(&path)
        .expect("private directory should be creatable");
    fs::set_permissions(&path, fs::Permissions::from_mode(0o700))
        .expect("private directory mode should be settable");
    path
}

pub(super) fn private_file(parent: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let path = parent.join(name);
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&path)
        .expect("private file should be creatable");
    file.write_all(bytes)
        .expect("fixture bytes should be writable");
    file.sync_all().expect("fixture bytes should be durable");
    fs::set_permissions(&path, fs::Permissions::from_mode(0o400))
        .expect("immutable fixture should be sealed read-only");
    path
}

pub(super) fn control_file(parent: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let path = parent.join(name);
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&path)
        .expect("control file should be creatable");
    file.write_all(bytes)
        .expect("control bytes should be writable");
    file.sync_all().expect("control bytes should be durable");
    path
}

pub(super) fn named_root_fixture(
    label: &str,
    lock_bytes: &[u8],
    authority_fill: u8,
) -> (PathBuf, PathBuf, GenerationRootAnchorLayout) {
    let root = fixture_root(label);
    control_file(&root, "LOCK", lock_bytes);
    let authority = authority_image(authority_fill);
    let authority_path = control_file(&root, "AUTHORITY", &authority);
    let layout = GenerationRootAnchorLayout::new(
        u64::try_from(lock_bytes.len()).expect("the lock fixture length fits u64"),
    )
    .expect("the named lock fixture length is bounded");
    (root, authority_path, layout)
}

pub(super) fn authority_image(fill: u8) -> Vec<u8> {
    vec![
        fill;
        usize::try_from(GENERATION_ROOT_AUTHORITY_FILE_BYTES)
            .expect("the frozen authority length fits usize")
    ]
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
        .expect("the mutable control anchor should be writable");
    file.seek(SeekFrom::Start(0))
        .expect("the control anchor should seek to its first byte");
    file.write_all(bytes)
        .expect("the complete in-place control image should be writable");
    file.sync_all()
        .expect("the in-place control rewrite should be durable");
}

pub(super) fn install_linux_acl_xattr(
    descriptor: &File,
    name: &str,
    owner_permissions: u16,
    named_permissions: u16,
    group_permissions: u16,
    mask_permissions: u16,
    other_permissions: u16,
) {
    const ACL_UNDEFINED_ID: u32 = u32::MAX;
    const ACL_USER_OBJ: u16 = 0x01;
    const ACL_USER: u16 = 0x02;
    const ACL_GROUP_OBJ: u16 = 0x04;
    const ACL_MASK: u16 = 0x10;
    const ACL_OTHER: u16 = 0x20;

    let mut bytes = Vec::with_capacity(44);
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    for (tag, permissions, id) in [
        (ACL_USER_OBJ, owner_permissions, ACL_UNDEFINED_ID),
        (ACL_USER, named_permissions, 424_242),
        (ACL_GROUP_OBJ, group_permissions, ACL_UNDEFINED_ID),
        (ACL_MASK, mask_permissions, ACL_UNDEFINED_ID),
        (ACL_OTHER, other_permissions, ACL_UNDEFINED_ID),
    ] {
        bytes.extend_from_slice(&tag.to_le_bytes());
        bytes.extend_from_slice(&permissions.to_le_bytes());
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    rustix::fs::fsetxattr(
        descriptor.as_fd(),
        name,
        &bytes,
        rustix::fs::XattrFlags::empty(),
    )
    .expect("raw Linux ACL fixture must install without an external utility");
}

pub(super) fn mapped_control_file(parent: &Path, name: &str) -> (PathBuf, crate::VectorIndex) {
    let path = parent.join(name);
    let mut writer = crate::VectorIndex::create(&path, "generation-root-test", 1)
        .expect("mapped control fixture writer should be creatable");
    writer
        .write_record("control", &[1.0])
        .expect("mapped control fixture record should be valid");
    writer
        .finish()
        .expect("mapped control fixture should be durable");
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600))
        .expect("mapped control fixture should use the control mode");
    let mapping = crate::VectorIndex::open(&path)
        .expect("mapped control fixture should retain a shared writable mapping");
    (path, mapping)
}

pub(super) fn mapped_immutable_file(parent: &Path, name: &str) -> (PathBuf, crate::VectorIndex) {
    let (path, mapping) = mapped_control_file(parent, name);
    fs::set_permissions(&path, fs::Permissions::from_mode(0o400))
        .expect("pre-held mapped fixture should seal as an immutable artifact");
    (path, mapping)
}

pub(super) fn mutate_mapping(mapping: &mut crate::VectorIndex) {
    let crate::VectorIndexData::Mutable(bytes) = &mut mapping.data else {
        panic!("legacy VectorIndex::open must retain a writable mapping");
    };
    let byte = bytes
        .last_mut()
        .expect("mapped control fixture must contain at least one byte");
    *byte ^= 0x01;
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
        .expect("static relative fixture path should parse")
}

pub(super) fn injected(stage: GenerationRootStage) -> GenerationRootError {
    GenerationRootError::new(GenerationRootErrorKind::Io, stage)
}

#[test]
#[ignore = "physical nested-PID receipt requires root, util-linux unshare, and exact isolated execution"]
fn physical_linux_foreign_procfs_pid_namespace_fails_before_mountinfo_open() {
    const CHILD_FLAG: &str = "FRANKENSEARCH_FOREIGN_PROCFS_PIDNS_CHILD";
    const ROOT_PATH: &str = "FRANKENSEARCH_FOREIGN_PROCFS_PIDNS_ROOT";
    const TEST_NAME: &str = "generation_root::tests::linux::physical_linux_foreign_procfs_pid_namespace_fails_before_mountinfo_open";

    assert!(
        rustix::process::geteuid().is_root(),
        "the foreign-procfs PID-namespace receipt requires root"
    );
    if std::env::var_os(CHILD_FLAG).is_none() {
        let root_path = fixture_root("foreign-procfs-pid-namespace");
        let executable =
            std::env::current_exe().expect("the current test ELF path must be available");
        let output = Command::new("unshare")
            .args(["--pid", "--fork"])
            .arg(executable)
            .args(["--exact", TEST_NAME, "--ignored", "--nocapture"])
            .env(CHILD_FLAG, "1")
            .env(ROOT_PATH, &root_path)
            .output()
            .expect("the physical receipt requires util-linux unshare");
        assert!(
            output.status.success(),
            "nested-PID child failed: status={:?}, stdout={}, stderr={}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    assert_eq!(
        std::process::id(),
        1,
        "the nested PID-namespace test process must be namespace PID 1"
    );
    assert_ne!(
        fs::read_link("/proc/self").expect("the inherited outer procfs self link must read"),
        PathBuf::from("1"),
        "the receipt must retain a procfs mounted for the outer PID namespace"
    );
    let root_path = PathBuf::from(
        std::env::var_os(ROOT_PATH).expect("the parent must pass the persistent root fixture"),
    );
    let boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_boundaries = Arc::clone(&boundaries);
    let _guard = install_test_hook(move |boundary| {
        hook_boundaries
            .lock()
            .expect("boundary log should lock")
            .push(boundary);
        Ok(())
    });
    let error = QualifiedGenerationRoot::admit(&root_path)
        .expect_err("foreign PID-namespace procfs must fail closed");
    assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert_eq!(error.stage(), GenerationRootStage::QualifyFilesystem);
    let observed = boundaries.lock().expect("boundary log should lock");
    assert!(
        observed.contains(&TestBoundary::BeforeProcCapabilityRootOpen),
        "the verified proc-root qualification must be attempted"
    );
    assert!(
        !observed.contains(&TestBoundary::BeforeMountInfoOpen),
        "foreign procfs must fail binding before the mountinfo file is opened"
    );
    drop(observed);
}

#[test]
#[ignore = "run the test ELF as root with FRANKENSEARCH_LINUX_PROC_OVERMOUNT_TEST_ROOT on native ext4/Btrfs"]
fn physical_linux_proc_fd_overmount_is_rejected_before_data_open() {
    const CHILD_FLAG: &str = "FRANKENSEARCH_PROC_OVERMOUNT_CHILD";
    const TEST_NAME: &str = "generation_root::tests::linux::physical_linux_proc_fd_overmount_is_rejected_before_data_open";

    assert!(
        rustix::process::geteuid().is_root(),
        "physical procfs overmount receipt must run as root"
    );
    let base = std::env::var_os("FRANKENSEARCH_LINUX_PROC_OVERMOUNT_TEST_ROOT")
        .map(PathBuf::from)
        .expect("FRANKENSEARCH_LINUX_PROC_OVERMOUNT_TEST_ROOT must identify native ext4/Btrfs");
    if std::env::var_os(CHILD_FLAG).is_none() {
        let executable =
            std::env::current_exe().expect("the current test ELF path must be available");
        let output = Command::new("unshare")
            .args(["--mount", "--fork", "--propagation", "private"])
            .arg(executable)
            .args(["--exact", TEST_NAME, "--ignored", "--nocapture"])
            .env(CHILD_FLAG, "1")
            .output()
            .expect("the physical receipt requires util-linux unshare");
        assert!(
            output.status.success(),
            "mount-namespace child failed: status={:?}, stdout={}, stderr={}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    let filesystem =
        rustix::fs::statfs(&base).expect("physical procfs fixture base must be statable");
    assert!(
        i128::from(filesystem.f_type) == i128::from(libc::EXT4_SUPER_MAGIC)
            || i128::from(filesystem.f_type) == i128::from(libc::BTRFS_SUPER_MAGIC),
        "physical procfs receipt requires ext4 or Btrfs"
    );
    let root_path = fixture_root_under(&base, "proc-fd-overmount");
    private_file(&root_path, "artifact", b"sealed");
    let fifo = root_path.join("proc-fd-overmount-fifo");
    rustix::fs::mkfifoat(
        rustix::fs::CWD,
        &fifo,
        rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
    )
    .expect("the hostile overmount fixture FIFO must be creatable");
    let decoy = private_dir(&root_path, "forged-proc-fd-decoy");
    for descriptor in 0_u32..=1024 {
        symlink(&fifo, decoy.join(descriptor.to_string()))
            .expect("each likely descriptor route must have a forged FIFO symlink");
    }
    let before_error = OpenOptions::new()
        .write(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(&fifo)
        .expect_err("the hostile FIFO must have no reader before the bridge attempt");
    assert_eq!(
        before_error.raw_os_error(),
        Some(libc::ENXIO),
        "the hostile FIFO fixture must prove its no-reader precondition"
    );
    let root = QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
    let process_id = std::process::id();
    let thread_id = rustix::thread::gettid();
    let proc_fd = PathBuf::from(format!("/proc/{process_id}/task/{thread_id}/fd"));
    let mount = Command::new("mount")
        .args(["--bind"])
        .arg(&decoy)
        .arg(&proc_fd)
        .output()
        .expect("the physical receipt requires mount");
    assert!(
        mount.status.success(),
        "proc-fd bind overmount failed: status={:?}, stdout={}, stderr={}",
        mount.status.code(),
        String::from_utf8_lossy(&mount.stdout),
        String::from_utf8_lossy(&mount.stderr)
    );

    let boundaries = Arc::new(Mutex::new(Vec::new()));
    let hook_boundaries = Arc::clone(&boundaries);
    let _guard = install_test_hook(move |boundary| {
        hook_boundaries
            .lock()
            .expect("boundary log should lock")
            .push(boundary);
        Ok(())
    });
    let error = root
        .admit_file(&confined("artifact"), immutable_expectation(b"sealed"))
        .expect_err("an overmounted proc-fd bridge must fail closed");
    assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
    assert_eq!(error.stage(), GenerationRootStage::OpenRegularFile);
    let observed = boundaries.lock().expect("boundary log should lock");
    assert!(
        observed.contains(&TestBoundary::BeforeProcBridgeProbe {
            index: 0,
            phase: ProcBridgeProbePhase::BeforeDataOpen,
        }),
        "the exact proc-fd bridge preflight must be attempted"
    );
    assert!(
        !observed.iter().any(|boundary| matches!(
            boundary,
            TestBoundary::AfterProcBridgeProbe { .. }
                | TestBoundary::AfterProcFdReopen { .. }
                | TestBoundary::BeforeRead { .. }
        )),
        "the overmount must fail before the data descriptor or content path"
    );
    drop(observed);
    let after_error = OpenOptions::new()
        .write(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(&fifo)
        .expect_err("the hostile FIFO must still have no reader after bridge rejection");
    assert_eq!(
        after_error.raw_os_error(),
        Some(libc::ENXIO),
        "bridge rejection must not establish a reader on any forged numeric fd route"
    );
}

mod admission_and_races;
mod boundaries_and_guards;
mod btrfs_and_faults;
mod locks_and_fork;
