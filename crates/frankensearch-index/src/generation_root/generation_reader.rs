//! Fresh-reader admission over the fixed LOCK/AUTHORITY anchors (bd-cdny8).
//!
//! A reader never reconstructs paths from a serialized "current" record. It
//! takes the shared anchor flock, resolves the exact slot pair, LOCK frames,
//! and external floor under one explicit security profile, opens the immutable
//! activation manifest the resolved head addresses (exact length and digest,
//! descriptor-relative), then rereads the anchors under a second shared flock
//! and admits only if the resolution is identical. The result is one owned,
//! nonserializable [`OpenedGenerationSnapshotV1`] behind an `Arc`: it retains
//! the manifest descriptor and bytes, so a later publication, rollback, or
//! root pathname replacement never changes what an admitted snapshot serves.
//!
//! The activation manifest is opened first, then every vector/lexical/ANN/
//! metadata component object it declares — content-addressed as
//! `<sha256 hex>.<role>` — at the exact length and digest of its receipt
//! (`GenerationClosureV1`), and finally the root inventory taken under the
//! second shared lock must be a clean closure (`ClosureViolationV1`).
//! Publication, reconciliation, and deletion are out of scope (see
//! `authority_publisher`).

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use super::authority_publisher::{
    ATTEMPT_FRAME_OFFSET, AUTHORITY_PUBLISHER_LOCK_BYTES_V1, AntiRollbackFloorProviderV1,
    ExpectedAuthorityPairV1, OWNER_FRAME_OFFSET, decode_pair, floor_head,
};
use super::{
    ConfinedGenerationPath, GENERATION_ROOT_AUTHORITY_FILE_NAME, GENERATION_ROOT_LOCK_FILE_NAME,
    GenerationFileExpectation, GenerationRootAnchorWitness, GenerationRootEntryKind,
    GenerationRootErrorKind, GenerationRootInventory, GenerationRootObjectWitness,
    GenerationRootReadGuard, GenerationRootResult, QualifiedGenerationFile,
    QualifiedGenerationRoot,
};
use frankensearch_core::generation::{
    ActivationManifest, AntiRollbackFloorRecordV1, AuthorityRefV1, AuthoritySlotV1,
    GENERATION_LOCK_FRAME_BYTES_V1, GenerationAuthorityErrorV1, GenerationComponentReceiptV1,
    GenerationComponentRole, GenerationLockFrameKindV1, GenerationLockFrameV1,
    GenerationRootSecurityProfileV1, resolve_authority_slots_with_profile_v1,
    verify_authority_manifest_reference,
};

/// Bounded number of resolve→open→reresolve rounds before a fresh open
/// reports a concurrent switch instead of spinning behind a publisher.
pub const SNAPSHOT_OPEN_MAX_ATTEMPTS_V1: usize = 3;

/// Suffix of an immutable activation-manifest object name.
pub const ACTIVATION_MANIFEST_SUFFIX_V1: &str = ".activation";

/// Confined relative name of the immutable activation manifest addressed
/// by `object_id`: 32 lowercase hex digits plus
/// [`ACTIVATION_MANIFEST_SUFFIX_V1`], directly under the generation root.
///
/// The name is a pure function of the random object id so a reader can
/// find the object without any directory scan or pathname adoption.
///
/// # Errors
///
/// Returns the platform's route error when relative names are unsupported.
pub fn activation_manifest_path_v1(
    object_id: [u8; 16],
) -> GenerationRootResult<ConfinedGenerationPath> {
    ConfinedGenerationPath::parse(Path::new(&activation_manifest_name_v1(object_id)))
}

/// Bare name of the activation manifest object for `object_id`.
#[must_use]
pub fn activation_manifest_name_v1(object_id: [u8; 16]) -> String {
    let mut name = lower_hex(&object_id);
    name.push_str(ACTIVATION_MANIFEST_SUFFIX_V1);
    name
}

/// Suffix (after the dot) of each content-addressed component object.
const COMPONENT_OBJECT_SUFFIXES_V1: [&str; 4] = ["vector", "lexical", "ann", "metadata"];

/// Suffix a write-ahead log carries beside the artifact it shadows; a WAL
/// inside a sealed generation root is never admissible.
pub const WRITE_AHEAD_LOG_SUFFIX_V1: &str = ".wal";

/// Bare name of one immutable component object.
///
/// The name is the 64 lowercase hex digits
/// of its exact SHA-256 plus `.<role>` (`vector`, `lexical`, `ann`,
/// `metadata`). Content addressing makes the manifest's receipt the only
/// locator a reader needs, and two generations that share identical
/// bytes for a role share one retained object.
#[must_use]
pub fn component_object_name_v1(role: GenerationComponentRole, sha256: [u8; 32]) -> String {
    let mut name = lower_hex(&sha256);
    name.push('.');
    name.push_str(role.as_str());
    name
}

/// Confined relative path of one component object.
///
/// # Errors
///
/// Returns the platform's route error when relative names are unsupported.
pub fn component_object_path_v1(
    role: GenerationComponentRole,
    sha256: [u8; 32],
) -> GenerationRootResult<ConfinedGenerationPath> {
    ConfinedGenerationPath::parse(Path::new(&component_object_name_v1(role, sha256)))
}

fn lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        hex.push(char::from(HEX[usize::from(byte >> 4)]));
        hex.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    hex
}

fn is_lower_hex(bytes: &[u8], len: usize) -> bool {
    bytes.len() == len
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
}

/// Whether `name` has the shape of a retained immutable object: an
/// activation manifest (`<32 hex>.activation`) or a component object
/// (`<64 hex>.<role>`). Shape only — identity is proven by admission.
#[must_use]
pub fn is_retained_object_name_v1(name: &[u8]) -> bool {
    if let Some(stem) = name.strip_suffix(ACTIVATION_MANIFEST_SUFFIX_V1.as_bytes()) {
        return is_lower_hex(stem, 32);
    }
    COMPONENT_OBJECT_SUFFIXES_V1.iter().any(|role| {
        name.strip_suffix(role.as_bytes())
            .and_then(|stem| stem.strip_suffix(b"."))
            .is_some_and(|stem| is_lower_hex(stem, 64))
    })
}

/// Why the retained root's contents are not a clean closure over
/// immutable objects. Bounded facts only — never a name or path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClosureViolationV1 {
    /// A write-ahead log sits beside a sealed object.
    WriteAheadLog,
    /// An entry that is neither an anchor nor a retained-object shape.
    UndeclaredEntry {
        /// Object type of the offending entry.
        kind: GenerationRootEntryKind,
    },
    /// Two or more names resolve to one inode, or an object carries a
    /// second link from outside the root.
    Alias,
    /// An object lives on a different device than the root.
    ForeignDevice,
}

/// The retained component objects of one admitted generation.
///
/// Each object is admitted descriptor-relatively at the exact length and
/// digest its manifest receipt names. Vector, lexical and metadata are
/// mandatory. ANN is present only when the selected manifest declares it; historical ANN
/// objects elsewhere in the root never supply an absent accelerator.
pub struct GenerationClosureV1 {
    vector: QualifiedGenerationFile,
    lexical: QualifiedGenerationFile,
    ann: Option<QualifiedGenerationFile>,
    metadata: QualifiedGenerationFile,
}

impl GenerationClosureV1 {
    /// Retained object for `role`; absent ANN is explicitly `None`.
    #[must_use]
    pub const fn file(&self, role: GenerationComponentRole) -> Option<&QualifiedGenerationFile> {
        match role {
            GenerationComponentRole::Vector => Some(&self.vector),
            GenerationComponentRole::Lexical => Some(&self.lexical),
            GenerationComponentRole::Ann => self.ann.as_ref(),
            GenerationComponentRole::Metadata => Some(&self.metadata),
        }
    }

    /// Exact bytes of the object for `role` (shared, immutable), or `None`
    /// when the selected manifest explicitly omits ANN.
    #[must_use]
    pub fn bytes(&self, role: GenerationComponentRole) -> Option<Arc<[u8]>> {
        self.file(role).map(QualifiedGenerationFile::bytes)
    }
}

impl fmt::Debug for GenerationClosureV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GenerationClosureV1")
            .field("vector", &self.vector.witness())
            .field("lexical", &self.lexical.witness())
            .field(
                "ann",
                &self.ann.as_ref().map(QualifiedGenerationFile::witness),
            )
            .field("metadata", &self.metadata.witness())
            .finish()
    }
}

/// Prove the root is a clean closure: every entry is an anchor, one of
/// this generation's declared objects, or a retained object of another
/// generation with the immutable-object shape, single-linked, on the
/// root's device; no WAL and no alias anywhere.
fn check_closure(
    inventory: &GenerationRootInventory,
    root_device: u64,
    declared: &[&str],
) -> Result<(), ClosureViolationV1> {
    for entry in inventory.entries() {
        // `as_encoded_bytes` is the portable spelling of the Unix
        // `OsStrExt::as_bytes` view: byte-identical on Unix, and defined
        // on every target, so this closure check compiles for Windows
        // even though the inventory that feeds it is Unix-only. Every use
        // below is a whole-string comparison or an ASCII suffix test, the
        // operations `as_encoded_bytes` explicitly permits.
        let name = entry.name().as_encoded_bytes();
        if name.ends_with(WRITE_AHEAD_LOG_SUFFIX_V1.as_bytes()) {
            return Err(ClosureViolationV1::WriteAheadLog);
        }
        if declared.iter().any(|declared| declared.as_bytes() == name) {
            continue;
        }
        if entry.kind() != GenerationRootEntryKind::RegularFile || !is_retained_object_name_v1(name)
        {
            return Err(ClosureViolationV1::UndeclaredEntry { kind: entry.kind() });
        }
        if entry.hard_links() != 1 {
            return Err(ClosureViolationV1::Alias);
        }
        if entry.device() != root_device {
            return Err(ClosureViolationV1::ForeignDevice);
        }
    }
    if !inventory.aliases().is_empty() {
        return Err(ClosureViolationV1::Alias);
    }
    Ok(())
}

/// Bounded health of an admitted snapshot relative to the external floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SnapshotHealthV1 {
    /// The external floor names exactly the served head.
    ExternallyAnchored {
        /// Floor compare-and-advance version that proved the head.
        cas_version: u64,
    },
    /// The external floor names the served head, but the local anchors
    /// already hold its immediate successor whose floor advance has not
    /// been observed (a publisher stopped between slot write and floor
    /// CAS). The old head is served read-only; mutation and eligibility
    /// decisions must stay frozen until the publisher reconciles.
    PendingLocalSuccessor {
        /// The locally committed but externally unproven successor.
        pending: AuthorityRefV1,
    },
    /// The selected profile does not anchor this root externally
    /// (`CooperativeLocal` or `ReadOnlyUnanchored`).
    Unanchored {
        /// Profile the caller selected explicitly.
        profile: GenerationRootSecurityProfileV1,
    },
}

/// Why a fresh open did not admit a snapshot. Never a silent fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotRefusalV1 {
    /// A publisher holds the exclusive anchor lock; retry later.
    LockContended,
    /// The LOCK anchor is not the owner+attempt layout this reader understands.
    LockLayout {
        /// Observed LOCK byte length.
        lock_byte_len: u64,
    },
    /// The anchors changed between the two shared-lock resolutions on
    /// every bounded attempt; the caller may retry after the publisher
    /// finishes.
    ConcurrentSwitch {
        /// Rounds attempted before giving up.
        attempts: usize,
    },
    /// A persisted publication attempt has no completing owner frame; the
    /// outcome must be reconciled by the publisher before any read.
    PendingReconciliation,
    /// Both slots are empty: the root has never published a head.
    NoAuthority,
    /// The external floor names a head that neither slot retains; the
    /// local anchors cannot serve the anchored generation.
    FloorHeadNotRetained,
    /// Slot, frame, floor, or profile resolution failed with a typed
    /// authority error (fork, gap, below floor, missing required floor…).
    Authority(GenerationAuthorityErrorV1),
    /// The manifest object opened, but it is not the exact object the
    /// resolved authority addresses.
    Manifest(GenerationAuthorityErrorV1),
    /// The manifest or a component object could not be admitted exactly
    /// (missing, wrong length or digest, wrong owner/mode/link identity…).
    Closure(GenerationRootErrorKind),
    /// Every declared object admitted, but the root also holds something
    /// a sealed generation root must never hold.
    ClosureViolation(ClosureViolationV1),
    /// The fresh head is older than the snapshot already installed in a
    /// cell for the same root identity (stale replica or replayed root).
    RegressedHead {
        /// Sequence of the installed snapshot's head.
        installed: u64,
        /// Sequence the fresh open observed.
        observed: u64,
    },
}

/// Outcome of one fresh open.
#[derive(Debug)]
pub enum SnapshotOpenOutcomeV1 {
    /// One owned, fully validated snapshot.
    Opened(Arc<OpenedGenerationSnapshotV1>),
    /// Typed refusal with zero retained state.
    Refused(SnapshotRefusalV1),
}

/// One admitted generation: exact authority, floor receipt, manifest
/// bytes, and the retained manifest descriptor.
///
/// Deliberately not serializable: a snapshot is a capability bound to
/// this process, not a record another process could reconstruct by path.
pub struct OpenedGenerationSnapshotV1 {
    root: GenerationRootObjectWitness,
    anchors: GenerationRootAnchorWitness,
    root_id: [u8; 16],
    profile: GenerationRootSecurityProfileV1,
    pair: ExpectedAuthorityPairV1,
    head: AuthoritySlotV1,
    floor: Option<AntiRollbackFloorRecordV1>,
    manifest_file: QualifiedGenerationFile,
    manifest: ActivationManifest,
    closure: GenerationClosureV1,
    health: SnapshotHealthV1,
}

impl OpenedGenerationSnapshotV1 {
    /// The retained component objects this generation declares.
    #[must_use]
    pub const fn closure(&self) -> &GenerationClosureV1 {
        &self.closure
    }

    /// Root object witness at admission.
    #[must_use]
    pub const fn root_witness(&self) -> GenerationRootObjectWitness {
        self.root
    }

    /// LOCK/AUTHORITY witnesses observed under the final shared lock.
    #[must_use]
    pub const fn anchor_witnesses(&self) -> GenerationRootAnchorWitness {
        self.anchors
    }

    /// Root identity every slot and floor was authenticated against.
    #[must_use]
    pub const fn root_id(&self) -> [u8; 16] {
        self.root_id
    }

    /// Profile the caller selected explicitly.
    #[must_use]
    pub const fn profile(&self) -> GenerationRootSecurityProfileV1 {
        self.profile
    }

    /// Exact slot pair resolved (and reread identically) at admission.
    #[must_use]
    pub const fn authority_pair(&self) -> ExpectedAuthorityPairV1 {
        self.pair
    }

    /// The served head: the slot whose manifest this snapshot retains.
    #[must_use]
    pub const fn head(&self) -> AuthoritySlotV1 {
        self.head
    }

    /// External floor record observed at admission, when a provider was given.
    #[must_use]
    pub const fn floor(&self) -> Option<AntiRollbackFloorRecordV1> {
        self.floor
    }

    /// Decoded, reference-verified activation manifest.
    #[must_use]
    pub const fn manifest(&self) -> &ActivationManifest {
        &self.manifest
    }

    /// Exact manifest object bytes (shared, immutable).
    #[must_use]
    pub fn manifest_bytes(&self) -> Arc<[u8]> {
        self.manifest_file.bytes()
    }

    /// Retained manifest object (descriptor + witness).
    #[must_use]
    pub const fn manifest_file(&self) -> &QualifiedGenerationFile {
        &self.manifest_file
    }

    /// Bounded floor health.
    #[must_use]
    pub const fn health(&self) -> SnapshotHealthV1 {
        self.health
    }
}

impl fmt::Debug for OpenedGenerationSnapshotV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenedGenerationSnapshotV1")
            .field("root", &self.root)
            .field("profile", &self.profile)
            .field("head_sequence", &self.head.authority.sequence)
            .field("head_slot", &self.head.slot_index)
            .field(
                "floor_cas_version",
                &self.floor.map(|record| record.cas_version),
            )
            .field("health", &self.health)
            .finish_non_exhaustive()
    }
}

/// Everything one shared-lock round resolved; two rounds must agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Resolution {
    pair: ExpectedAuthorityPairV1,
    head: AuthoritySlotV1,
    floor: Option<AntiRollbackFloorRecordV1>,
    health: SnapshotHealthV1,
}

fn decode_frame(
    lock_bytes: &[u8],
    offset: u64,
    root_id: [u8; 16],
    expected_kind: GenerationLockFrameKindV1,
) -> Result<Option<GenerationLockFrameV1>, GenerationAuthorityErrorV1> {
    let start =
        usize::try_from(offset).map_err(|_| GenerationAuthorityErrorV1::InvalidSlotLength)?;
    let bytes = lock_bytes
        .get(start..start + GENERATION_LOCK_FRAME_BYTES_V1)
        .ok_or(GenerationAuthorityErrorV1::InvalidSlotLength)?;
    if bytes.iter().all(|byte| *byte == 0) {
        return Ok(None);
    }
    let frame = GenerationLockFrameV1::from_authenticated_bytes(bytes, root_id)?;
    // ubs:ignore — lock kinds are public fixed-format state tags.
    if frame.kind != expected_kind {
        return Err(GenerationAuthorityErrorV1::InvalidField {
            field: "generation_lock.frame.kind",
        });
    }
    Ok(Some(frame))
}

fn resolve_under_guard(
    guard: &GenerationRootReadGuard<'_>,
    root_id: [u8; 16],
    profile: GenerationRootSecurityProfileV1,
    floor: Option<&dyn AntiRollbackFloorProviderV1>,
) -> Result<Resolution, SnapshotRefusalV1> {
    let lock_bytes = guard.lock_bytes();
    let lock_byte_len = u64::try_from(lock_bytes.len()).unwrap_or(u64::MAX);
    if lock_byte_len != AUTHORITY_PUBLISHER_LOCK_BYTES_V1 {
        return Err(SnapshotRefusalV1::LockLayout { lock_byte_len });
    }
    let owner = decode_frame(
        lock_bytes,
        OWNER_FRAME_OFFSET,
        root_id,
        GenerationLockFrameKindV1::Owner,
    )
    .map_err(SnapshotRefusalV1::Authority)?;
    let attempt = decode_frame(
        lock_bytes,
        ATTEMPT_FRAME_OFFSET,
        root_id,
        GenerationLockFrameKindV1::Attempt,
    )
    .map_err(SnapshotRefusalV1::Authority)?;
    // The publisher leaves its attempt frame in place and records
    // completion by writing an owner frame carrying the same attempt id.
    // An attempt without that owner frame is an outcome nobody has
    // reconciled: fail closed rather than pick an older pair.
    if let Some(attempt) = attempt
        && !owner.is_some_and(|owner| {
            owner.attempt_id == attempt.attempt_id && owner.fence >= attempt.fence
        })
    {
        return Err(SnapshotRefusalV1::PendingReconciliation);
    }
    let pair =
        decode_pair(guard.authority_bytes(), root_id).map_err(SnapshotRefusalV1::Authority)?;
    let (floor_record, floor_ref) =
        floor_head(floor, root_id).map_err(SnapshotRefusalV1::Authority)?;
    let resolved = resolve_authority_slots_with_profile_v1(
        pair.first,
        pair.second,
        owner,
        None,
        profile,
        floor_ref,
    )
    .map_err(SnapshotRefusalV1::Authority)?
    .ok_or(SnapshotRefusalV1::NoAuthority)?;

    let (head, health) = match (profile, floor_ref, floor_record) {
        (GenerationRootSecurityProfileV1::RequiredExternal, Some(floor), Some(record)) => {
            // ubs:ignore — authority references are public immutable identities.
            if resolved.authority == floor.authority {
                (
                    resolved,
                    SnapshotHealthV1::ExternallyAnchored {
                        cas_version: record.cas_version,
                    },
                )
            } else {
                // The profile resolver admitted `resolved` only as the
                // floor's immediate predecessor-linked successor. Serve the
                // externally proven head, which must still occupy the
                // other physical slot.
                let anchored = [pair.first, pair.second]
                    .into_iter()
                    .flatten()
                    // ubs:ignore — authority references are public immutable identities.
                    .find(|slot| slot.authority == floor.authority)
                    .ok_or(SnapshotRefusalV1::FloorHeadNotRetained)?;
                (
                    anchored,
                    SnapshotHealthV1::PendingLocalSuccessor {
                        pending: resolved.authority,
                    },
                )
            }
        }
        // The resolver already required a floor for RequiredExternal;
        // reaching here without one is impossible, but never downgrade.
        (GenerationRootSecurityProfileV1::RequiredExternal, _, _) => {
            return Err(SnapshotRefusalV1::Authority(
                GenerationAuthorityErrorV1::ExternalFloorRequired,
            ));
        }
        (profile, _, _) => (resolved, SnapshotHealthV1::Unanchored { profile }),
    };
    Ok(Resolution {
        pair,
        head,
        floor: floor_record,
        health,
    })
}

/// Whether a guard acquisition/release failure means "the anchors moved
/// under us — resolve again" rather than a hard fault.
const fn is_transient(kind: GenerationRootErrorKind) -> bool {
    matches!(
        kind,
        GenerationRootErrorKind::ObjectChanged | GenerationRootErrorKind::HardLinked
    )
}

enum Round {
    Resolved(
        Box<(
            Resolution,
            GenerationRootAnchorWitness,
            GenerationRootInventory,
        )>,
    ),
    Retry,
    Refused(SnapshotRefusalV1),
}

fn admit_component(
    root: &QualifiedGenerationRoot,
    role: GenerationComponentRole,
    receipt: GenerationComponentReceiptV1,
) -> GenerationRootResult<Result<QualifiedGenerationFile, SnapshotRefusalV1>> {
    let path = component_object_path_v1(role, receipt.sha256)?;
    let expectation = GenerationFileExpectation::immutable(receipt.byte_len, receipt.sha256)?;
    Ok(root
        .admit_file(&path, expectation)
        .map_err(|error| SnapshotRefusalV1::Closure(error.kind())))
}

impl QualifiedGenerationRoot {
    fn resolution_round(
        &self,
        root_id: [u8; 16],
        profile: GenerationRootSecurityProfileV1,
        floor: Option<&dyn AntiRollbackFloorProviderV1>,
    ) -> GenerationRootResult<Round> {
        let guard = match self.read_guard() {
            Ok(guard) => guard,
            Err(error) if error.kind() == GenerationRootErrorKind::LockContended => {
                return Ok(Round::Refused(SnapshotRefusalV1::LockContended));
            }
            Err(error) if is_transient(error.kind()) => return Ok(Round::Retry),
            Err(error) => return Err(error),
        };
        let resolution = match resolve_under_guard(&guard, root_id, profile, floor) {
            Ok(resolution) => resolution,
            Err(refusal) => {
                let _ = guard.release();
                return Ok(Round::Refused(refusal));
            }
        };
        let anchors = guard.witnesses();
        // The closure inventory is taken while the shared lock is held so
        // it describes the same instant as the resolution.
        let inventory = match guard.inventory() {
            Ok(inventory) => inventory,
            Err(error) if is_transient(error.kind()) => {
                let _ = guard.release();
                return Ok(Round::Retry);
            }
            Err(error) => {
                let _ = guard.release();
                return Err(error);
            }
        };
        match guard.release() {
            Ok(()) => Ok(Round::Resolved(Box::new((resolution, anchors, inventory)))),
            Err(error) if is_transient(error.kind()) => Ok(Round::Retry),
            Err(error) => Err(error),
        }
    }

    /// Open one owned snapshot of the current generation under `profile`.
    ///
    /// Two shared-lock resolutions bracket the descriptor-relative open
    /// of the activation manifest; the snapshot is admitted only if both
    /// resolutions are identical and the manifest is exactly the object
    /// the resolved authority addresses. Nothing is retained on refusal,
    /// and a refusal never invalidates a snapshot admitted earlier.
    ///
    /// # Errors
    ///
    /// Infrastructure failures (root revalidation, descriptor faults, a
    /// forked process) surface as [`super::GenerationRootError`]; every
    /// authority-level condition is a typed
    /// [`SnapshotOpenOutcomeV1::Refused`].
    pub fn open_generation_snapshot(
        &self,
        root_id: [u8; 16],
        profile: GenerationRootSecurityProfileV1,
        floor: Option<&dyn AntiRollbackFloorProviderV1>,
    ) -> GenerationRootResult<SnapshotOpenOutcomeV1> {
        if root_id == [0; 16] {
            return Ok(SnapshotOpenOutcomeV1::Refused(
                SnapshotRefusalV1::Authority(GenerationAuthorityErrorV1::InvalidField {
                    field: "generation_reader.root_id",
                }),
            ));
        }
        for _ in 0..SNAPSHOT_OPEN_MAX_ATTEMPTS_V1 {
            let (first, _, _) = match self.resolution_round(root_id, profile, floor)? {
                Round::Resolved(resolved) => *resolved,
                Round::Retry => continue,
                Round::Refused(refusal) => return Ok(SnapshotOpenOutcomeV1::Refused(refusal)),
            };

            // Open the immutable closure outside the lock: exact length
            // and digest from the authority, descriptor-relative route.
            let authority = first.head.authority;
            let path = activation_manifest_path_v1(authority.object_id)?;
            let expectation = GenerationFileExpectation::immutable(
                authority.manifest_len,
                authority.manifest_sha256,
            )?;
            let manifest_file = match self.admit_file(&path, expectation) {
                Ok(file) => file,
                Err(error) => {
                    return Ok(SnapshotOpenOutcomeV1::Refused(SnapshotRefusalV1::Closure(
                        error.kind(),
                    )));
                }
            };
            let manifest = match ActivationManifest::from_canonical_bytes(manifest_file.as_bytes())
                .and_then(|manifest| {
                    verify_authority_manifest_reference(&authority, &manifest).map(|()| manifest)
                }) {
                Ok(manifest) => manifest,
                Err(error) => {
                    return Ok(SnapshotOpenOutcomeV1::Refused(SnapshotRefusalV1::Manifest(
                        error,
                    )));
                }
            };

            // Every declared component object, exact length and digest
            // from the manifest receipts, descriptor-relative.
            let receipts = manifest.components();
            let mut vector = None;
            let mut lexical = None;
            let mut ann = None;
            let mut metadata = None;
            for (role, receipt) in receipts.iter() {
                match admit_component(self, role, receipt)? {
                    Ok(file) => match role {
                        GenerationComponentRole::Vector => vector = Some(file),
                        GenerationComponentRole::Lexical => lexical = Some(file),
                        GenerationComponentRole::Ann => ann = Some(file),
                        GenerationComponentRole::Metadata => metadata = Some(file),
                    },
                    Err(refusal) => return Ok(SnapshotOpenOutcomeV1::Refused(refusal)),
                }
            }
            // Off the native platforms `QualifiedGenerationFile` is
            // uninhabited (the fallback handle types are empty enums), so
            // rustc correctly reports every field after the first as
            // unreachable. The path is genuinely unreachable there —
            // `admit_component` cannot return a value — and this is the
            // shape that keeps the code compiling on such a target.
            #[cfg_attr(
                not(any(target_os = "linux", target_os = "macos")),
                allow(unreachable_code)
            )]
            let closure = GenerationClosureV1 {
                vector: vector.expect("mandatory vector component admitted"),
                lexical: lexical.expect("mandatory lexical component admitted"),
                ann,
                metadata: metadata.expect("mandatory metadata component admitted"),
            };

            // Reread under a fresh shared lock: identical or start over.
            let (second, anchors, inventory) =
                match self.resolution_round(root_id, profile, floor)? {
                    Round::Resolved(resolved) => *resolved,
                    Round::Retry => continue,
                    Round::Refused(refusal) => {
                        return Ok(SnapshotOpenOutcomeV1::Refused(refusal));
                    }
                };
            if second != first {
                continue;
            }

            // Closure over the retained root as it was under that lock.
            let manifest_name = activation_manifest_name_v1(authority.object_id);
            let component_names: Vec<_> = receipts
                .iter()
                .map(|(role, receipt)| component_object_name_v1(role, receipt.sha256))
                .collect();
            let declared: Vec<_> = [
                GENERATION_ROOT_LOCK_FILE_NAME,
                GENERATION_ROOT_AUTHORITY_FILE_NAME,
                manifest_name.as_str(),
            ]
            .into_iter()
            .chain(component_names.iter().map(String::as_str))
            .collect();
            if let Err(violation) = check_closure(&inventory, self.witness().device(), &declared) {
                return Ok(SnapshotOpenOutcomeV1::Refused(
                    SnapshotRefusalV1::ClosureViolation(violation),
                ));
            }

            return Ok(SnapshotOpenOutcomeV1::Opened(Arc::new(
                OpenedGenerationSnapshotV1 {
                    root: self.witness(),
                    anchors,
                    root_id,
                    profile,
                    pair: second.pair,
                    head: second.head,
                    floor: second.floor,
                    manifest_file,
                    manifest,
                    closure,
                    health: second.health,
                },
            )));
        }
        Ok(SnapshotOpenOutcomeV1::Refused(
            SnapshotRefusalV1::ConcurrentSwitch {
                attempts: SNAPSHOT_OPEN_MAX_ATTEMPTS_V1,
            },
        ))
    }
}

/// Lock-free owned cell the query hot path reads a prebuilt snapshot from.
///
/// The cell never builds anything itself: [`Self::refresh`] opens a fresh
/// snapshot through [`QualifiedGenerationRoot::open_generation_snapshot`]
/// and installs the already-built `Arc` only when the open succeeded and
/// the head did not regress. A refusal or infrastructure error leaves
/// whatever was installed exactly as it was, so readers keep serving the
/// last admitted generation. Old `Arc`s handed out by [`Self::load`] stay
/// valid and byte-exact after any later install.
pub struct GenerationSnapshotCellV1 {
    inner: arc_swap::ArcSwapOption<OpenedGenerationSnapshotV1>,
}

impl Default for GenerationSnapshotCellV1 {
    fn default() -> Self {
        Self::new()
    }
}

impl GenerationSnapshotCellV1 {
    /// An empty cell: nothing admitted yet.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner: arc_swap::ArcSwapOption::const_empty(),
        }
    }

    /// The currently installed snapshot, if any (one atomic load).
    #[must_use]
    pub fn load(&self) -> Option<Arc<OpenedGenerationSnapshotV1>> {
        self.inner.load_full()
    }

    /// Install an already-admitted snapshot; returns the previous one.
    pub fn install(
        &self,
        snapshot: Arc<OpenedGenerationSnapshotV1>,
    ) -> Option<Arc<OpenedGenerationSnapshotV1>> {
        self.inner.swap(Some(snapshot))
    }

    /// Open a fresh snapshot of `root` and install it on success.
    ///
    /// The new head must not be older than the installed one for the same
    /// root identity: a stale replica or replayed root is refused typed
    /// ([`SnapshotRefusalV1::RegressedHead`]) and the installed snapshot is
    /// kept. Equal sequences must name the same exact authority. Concurrent
    /// refreshes recheck the winning snapshot before installation; expensive
    /// admission runs outside this atomic update. On refusal or error nothing
    /// is installed.
    ///
    /// # Errors
    ///
    /// Same as [`QualifiedGenerationRoot::open_generation_snapshot`].
    pub fn refresh(
        &self,
        root: &QualifiedGenerationRoot,
        root_id: [u8; 16],
        profile: GenerationRootSecurityProfileV1,
        floor: Option<&dyn AntiRollbackFloorProviderV1>,
    ) -> GenerationRootResult<SnapshotOpenOutcomeV1> {
        let outcome = root.open_generation_snapshot(root_id, profile, floor)?;
        let SnapshotOpenOutcomeV1::Opened(fresh) = &outcome else {
            return Ok(outcome);
        };
        let mut installed = self.inner.load();
        loop {
            if let Some(current) = installed.as_ref()
                && current.root_id() == fresh.root_id()
            {
                let current_head = current.head().authority;
                let fresh_head = fresh.head().authority;
                if current_head.sequence > fresh_head.sequence {
                    return Ok(SnapshotOpenOutcomeV1::Refused(
                        SnapshotRefusalV1::RegressedHead {
                            installed: current_head.sequence,
                            observed: fresh_head.sequence,
                        },
                    ));
                }
                if current_head.sequence == fresh_head.sequence && current_head != fresh_head {
                    return Ok(SnapshotOpenOutcomeV1::Refused(
                        SnapshotRefusalV1::Authority(GenerationAuthorityErrorV1::EqualSequenceFork),
                    ));
                }
            }
            #[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
            super::platform::test_boundary(
                super::platform::TestBoundary::BeforeSnapshotCellCompareExchange,
            )?;
            let previous = self
                .inner
                .compare_and_swap(&*installed, Some(Arc::clone(fresh)));
            let exchanged = match (&*installed, &*previous) {
                (None, None) => true,
                (Some(expected), Some(previous)) => Arc::ptr_eq(expected, previous),
                _ => false,
            };
            if exchanged {
                return Ok(outcome);
            }
            // Another refresher won after our observation. Recheck its
            // exact head rather than overwriting it with a stale admission.
            installed = previous;
        }
    }
}

impl fmt::Debug for GenerationSnapshotCellV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GenerationSnapshotCellV1")
            .field(
                "installed_head_sequence",
                &self
                    .load()
                    .map(|snapshot| snapshot.head().authority.sequence),
            )
            .finish()
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests;
