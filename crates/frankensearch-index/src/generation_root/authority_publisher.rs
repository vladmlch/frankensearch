//! Fixed AUTHORITY and LOCK publisher over a qualified generation root
//! (bd-ycng6, first slice: publication, typed outcomes, reconciliation).
//!
//! The AUTHORITY anchor is one precreated, never-renamed, never-unlinked
//! 8192-byte inode holding two fixed 4096-byte authenticated slots; the LOCK
//! anchor holds an owner frame at `[0, 4096)` and an attempt frame at
//! `[4096, 8192)`. The kernel flock on LOCK is the sole live-owner authority;
//! every identity in the frames is a diagnostic and a mutation fence, never a
//! liveness proof or a timeout-stealable lease.
//!
//! Publication order under the exclusive flock: fence the exact expected slot
//! pair and the floor head, persist the attempt frame (`InFlight`), positionally
//! overwrite only the slot selected by `sequence & 1`, run the durability
//! barrier, exactly reread the slot, compare-and-advance the external floor,
//! persist the owner frame (`Completed`), unlock. Before the attempt frame is
//! written every failure is an ordinary [`PublicationOutcomeV1::NotCommitted`]
//! with zero authority mutation; from the first possibly issued slot byte or
//! floor request, an immediate failure is
//! [`PublicationOutcomeV1::CommitOutcomeUnknown`] carrying a bounded exact
//! [`AttemptPermitV1`] that [`AuthorityPublisherV1::reconcile`] resolves from
//! the exact local slot, floor, and frame state.
//!
//! This slice computes no GC, deletes nothing, and installs no other pointer.

use super::{
    GENERATION_ROOT_AUTHORITY_FILE_BYTES, GenerationRootError, GenerationRootErrorKind,
    GenerationRootResult, GenerationRootStage, QualifiedGenerationRoot, platform,
};
use frankensearch_core::generation::{
    AntiRollbackFloorRecordV1, AuthorityFloorV1, AuthorityRefV1, AuthoritySlotV1,
    GENERATION_AUTHORITY_SLOT_BYTES_V1, GENERATION_LOCK_FRAME_BYTES_V1, GenerationAuthorityErrorV1,
    GenerationLockFrameKindV1, GenerationLockFrameV1, GenerationRootSecurityProfileV1,
    InMemoryAntiRollbackFloorStoreV1, LocalAntiRollbackFloorStoreV1,
    resolve_authority_slots_with_profile_v1,
};

/// Exact LOCK anchor length this publisher requires: owner frame + attempt frame.
pub const AUTHORITY_PUBLISHER_LOCK_BYTES_V1: u64 = 2 * GENERATION_LOCK_FRAME_BYTES_V1 as u64;
pub(super) const OWNER_FRAME_OFFSET: u64 = 0;
pub(super) const ATTEMPT_FRAME_OFFSET: u64 = GENERATION_LOCK_FRAME_BYTES_V1 as u64;

/// External anti-rollback floor as the publisher consumes it.
///
/// Both reference stores in `frankensearch-core` implement it; consumer-owned
/// remote or OS-monotonic stores implement the same two calls.
pub trait AntiRollbackFloorProviderV1 {
    /// Current floor for `root_id`.
    ///
    /// # Errors
    ///
    /// Provider-specific typed failure; every error refuses publication.
    fn load(
        &self,
        root_id: [u8; 16],
    ) -> Result<Option<AntiRollbackFloorRecordV1>, GenerationAuthorityErrorV1>;

    /// Linearizable compare-and-advance.
    ///
    /// # Errors
    ///
    /// Provider-specific typed failure; once requested, any error leaves the
    /// publication outcome unknown until reconciled.
    fn compare_and_advance(
        &self,
        expected: Option<AntiRollbackFloorRecordV1>,
        next: AuthorityFloorV1,
        idempotency_key: [u8; 16],
    ) -> Result<AntiRollbackFloorRecordV1, GenerationAuthorityErrorV1>;
}

impl AntiRollbackFloorProviderV1 for InMemoryAntiRollbackFloorStoreV1 {
    fn load(
        &self,
        root_id: [u8; 16],
    ) -> Result<Option<AntiRollbackFloorRecordV1>, GenerationAuthorityErrorV1> {
        Self::load(self, root_id)
    }

    fn compare_and_advance(
        &self,
        expected: Option<AntiRollbackFloorRecordV1>,
        next: AuthorityFloorV1,
        idempotency_key: [u8; 16],
    ) -> Result<AntiRollbackFloorRecordV1, GenerationAuthorityErrorV1> {
        Self::compare_and_advance(self, expected, next, idempotency_key)
    }
}

impl AntiRollbackFloorProviderV1 for LocalAntiRollbackFloorStoreV1 {
    fn load(
        &self,
        root_id: [u8; 16],
    ) -> Result<Option<AntiRollbackFloorRecordV1>, GenerationAuthorityErrorV1> {
        Self::load(self, root_id)
    }

    fn compare_and_advance(
        &self,
        expected: Option<AntiRollbackFloorRecordV1>,
        next: AuthorityFloorV1,
        idempotency_key: [u8; 16],
    ) -> Result<AntiRollbackFloorRecordV1, GenerationAuthorityErrorV1> {
        Self::compare_and_advance(self, expected, next, idempotency_key)
    }
}

/// The exact slot pair a publication expects to find under the lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExpectedAuthorityPairV1 {
    /// Physical slot 0.
    pub first: Option<AuthoritySlotV1>,
    /// Physical slot 1.
    pub second: Option<AuthoritySlotV1>,
}

/// Why a publication was refused before any authority byte was issued.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotCommittedV1 {
    /// The selected profile forbids mutation.
    ReadOnlyProfile,
    /// Another live owner holds the exclusive flock.
    LockContended,
    /// The anchors mutated between this publisher's open and its lock
    /// (another publication landed); reload the pair and retry.
    AnchorsChanged,
    /// The LOCK anchor is not exactly two 4096-byte frames.
    LockLayout {
        /// Observed physical LOCK length.
        lock_byte_len: u64,
    },
    /// The slots under the lock are not the expected pair.
    ExpectedPairMismatch {
        /// What the lock holder actually observed (boxed: two full slots).
        observed: Box<ExpectedAuthorityPairV1>,
    },
    /// A non-empty slot failed authentication; publication needs repair first.
    UnreadableSlot(GenerationAuthorityErrorV1),
    /// The candidate does not extend the resolved head by exactly one sequence.
    InvalidCandidate(GenerationAuthorityErrorV1),
    /// The floor head could not be proven or already exceeds the pair.
    FloorRefused(GenerationAuthorityErrorV1),
}

/// Stage at which an attempt's outcome became unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptStageV1 {
    /// Persisting the `InFlight` attempt frame.
    AttemptFrame,
    /// Positionally writing the target slot.
    SlotWrite,
    /// Exact reread of the written slot after the barrier.
    SlotReread,
    /// External floor compare-and-advance.
    FloorAdvance,
    /// Persisting the `Completed` owner frame.
    OwnerFrame,
}

/// Bounded exact record of an attempt whose outcome is not yet known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttemptPermitV1 {
    /// Root the attempt targeted.
    pub root_id: [u8; 16],
    /// Writer that issued the attempt.
    pub writer_id: [u8; 16],
    /// Random per-attempt identity persisted in the attempt frame.
    pub attempt_id: [u8; 16],
    /// Physical slot selected by `candidate.sequence & 1`.
    pub target_slot: u8,
    /// Authority the attempt tried to install.
    pub candidate: AuthorityRefV1,
    /// Pair the attempt fenced against.
    pub expected: ExpectedAuthorityPairV1,
    /// Stage that failed.
    pub stage: AttemptStageV1,
    /// Idempotency key used for the floor request.
    pub idempotency_key: [u8; 16],
}

/// Result of one publication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublicationOutcomeV1 {
    /// Slot durable, reread exact, floor advanced (when provided), owner frame persisted.
    Committed {
        /// Installed slot.
        slot: AuthoritySlotV1,
        /// Floor receipt when a provider was supplied.
        floor: Option<AntiRollbackFloorRecordV1>,
    },
    /// Zero authority mutation occurred.
    NotCommitted(NotCommittedV1),
    /// A slot byte or floor request may have been issued; reconcile before any further mutation.
    CommitOutcomeUnknown(AttemptPermitV1),
}

/// Fresh-process reconciliation verdict for an [`AttemptPermitV1`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconcileOutcomeV1 {
    /// The candidate is the durable head and the floor (if any) agrees.
    Committed {
        /// Installed slot.
        slot: AuthoritySlotV1,
        /// Floor receipt when a provider was supplied.
        floor: Option<AntiRollbackFloorRecordV1>,
    },
    /// The target slot never received the candidate; prior authority stands.
    NotCommitted,
    /// Local state shows the candidate but the floor could not be proven or advanced.
    StillUnknown(AttemptPermitV1),
    /// The target slot holds a different authority at the candidate's sequence.
    Fork {
        /// What the slot holds instead.
        observed: AuthoritySlotV1,
    },
    /// The target slot is torn or foreign and must be repaired before any decision.
    RepairRequired(GenerationAuthorityErrorV1),
    /// The candidate committed and a newer authority has since been published.
    CommittedThenSuperseded {
        /// Current resolved head.
        head: AuthoritySlotV1,
    },
}

/// Writer-side handle bound to one qualified root and one security profile.
#[derive(Debug)]
pub struct AuthorityPublisherV1<'root> {
    root: &'root QualifiedGenerationRoot,
    root_id: [u8; 16],
    writer_id: [u8; 16],
    profile: GenerationRootSecurityProfileV1,
}

impl QualifiedGenerationRoot {
    /// Bind a publisher to this root.
    ///
    /// # Errors
    ///
    /// Rejects an all-zero `root_id` or `writer_id`.
    pub fn authority_publisher(
        &self,
        root_id: [u8; 16],
        writer_id: [u8; 16],
        profile: GenerationRootSecurityProfileV1,
    ) -> Result<AuthorityPublisherV1<'_>, GenerationAuthorityErrorV1> {
        if root_id == [0; 16] {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_publisher.root_id",
            });
        }
        if writer_id == [0; 16] {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_publisher.writer_id",
            });
        }
        Ok(AuthorityPublisherV1 {
            root: self,
            root_id,
            writer_id,
            profile,
        })
    }
}

fn slot_offset(slot_index: u8) -> u64 {
    u64::from(slot_index) * GENERATION_AUTHORITY_SLOT_BYTES_V1 as u64
}

fn slot_range(slot_index: u8) -> std::ops::Range<usize> {
    let start = usize::from(slot_index) * GENERATION_AUTHORITY_SLOT_BYTES_V1;
    start..start + GENERATION_AUTHORITY_SLOT_BYTES_V1
}

/// Decode one physical slot: all-zero bytes are the empty (genesis) slot.
pub(super) fn decode_slot(
    authority_bytes: &[u8],
    slot_index: u8,
    root_id: [u8; 16],
) -> Result<Option<AuthoritySlotV1>, GenerationAuthorityErrorV1> {
    let bytes = authority_bytes
        .get(slot_range(slot_index))
        .ok_or(GenerationAuthorityErrorV1::InvalidSlotLength)?;
    if bytes.iter().all(|byte| *byte == 0) {
        return Ok(None);
    }
    AuthoritySlotV1::from_authenticated_bytes(bytes, slot_index, root_id).map(Some)
}

pub(super) fn decode_pair(
    authority_bytes: &[u8],
    root_id: [u8; 16],
) -> Result<ExpectedAuthorityPairV1, GenerationAuthorityErrorV1> {
    if u64::try_from(authority_bytes.len()).ok() != Some(GENERATION_ROOT_AUTHORITY_FILE_BYTES) {
        return Err(GenerationAuthorityErrorV1::InvalidSlotLength);
    }
    Ok(ExpectedAuthorityPairV1 {
        first: decode_slot(authority_bytes, 0, root_id)?,
        second: decode_slot(authority_bytes, 1, root_id)?,
    })
}

pub(super) fn floor_head(
    provider: Option<&dyn AntiRollbackFloorProviderV1>,
    root_id: [u8; 16],
) -> Result<(Option<AntiRollbackFloorRecordV1>, Option<AuthorityFloorV1>), GenerationAuthorityErrorV1>
{
    let Some(provider) = provider else {
        return Ok((None, None));
    };
    let record = provider.load(root_id)?;
    let floor = record
        .map(|record| AuthorityFloorV1::new(root_id, record.authority))
        .transpose()?;
    Ok((record, floor))
}

/// Writer identity for owner/attempt frames: diagnostics and a mutation
/// fence, never a liveness proof or a timeout-stealable lease.
///
/// `writer_id` is the first 16 bytes of a SHA-256 over the host
/// fingerprint, boot identity, PID, process start ticks and a fresh random
/// nonce, so it is unique per process incarnation and cannot be replayed
/// by a PID reuse on the same host or by the same PID on a foreign host.
/// The components are retained in clear so a stale owner frame can be
/// explained (same host? same boot? which process?) without ever being
/// trusted as proof that the writer is alive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriterIdentityV1 {
    /// Derived 16-byte identity used in LOCK frames.
    pub writer_id: [u8; 16],
    /// SHA-256 of the host identity source (`/etc/machine-id` or hostname).
    pub host_fingerprint: [u8; 32],
    /// Kernel boot identity (`/proc/sys/kernel/random/boot_id`), if readable.
    pub boot_id: Option<[u8; 16]>,
    /// Process id at derivation time.
    pub pid: u32,
    /// Process start time in clock ticks since boot (`/proc/self/stat`
    /// field 22), if readable.
    pub process_start_ticks: Option<u64>,
    /// Fresh random nonce bound into the identity.
    pub nonce: [u8; 16],
}

impl WriterIdentityV1 {
    /// Derive the identity of the current process incarnation.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationRootError`] only when no random nonce can be
    /// obtained; every other component degrades to `None`/hostname.
    pub fn for_current_process() -> GenerationRootResult<Self> {
        use sha2::{Digest as _, Sha256};
        let nonce = random_identity()?;
        let host_source = std::fs::read("/etc/machine-id")
            .ok()
            .filter(|bytes| !bytes.iter().all(u8::is_ascii_whitespace))
            .or_else(|| std::fs::read("/proc/sys/kernel/hostname").ok())
            .unwrap_or_default();
        let host_fingerprint: [u8; 32] = Sha256::digest(&host_source).into();
        let boot_id = std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
            .ok()
            .and_then(|text| parse_uuid_bytes(text.trim()));
        let pid = std::process::id();
        let process_start_ticks =
            std::fs::read_to_string("/proc/self/stat")
                .ok()
                .and_then(|stat| {
                    // Field 2 (comm) may contain spaces; fields after the
                    // closing parenthesis are stable, so index from there:
                    // state is field 3, start time is field 22.
                    let tail = &stat[stat.rfind(')')? + 1..];
                    tail.split_whitespace().nth(19)?.parse::<u64>().ok()
                });
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch.generation-root.writer-identity.v1");
        hasher.update(host_fingerprint);
        hasher.update(boot_id.unwrap_or([0; 16]));
        hasher.update(pid.to_be_bytes());
        hasher.update(process_start_ticks.unwrap_or(0).to_be_bytes());
        hasher.update(nonce);
        let digest: [u8; 32] = hasher.finalize().into();
        let mut writer_id = [0_u8; 16];
        writer_id.copy_from_slice(&digest[..16]);
        if writer_id == [0; 16] {
            writer_id = nonce;
        }
        Ok(Self {
            writer_id,
            host_fingerprint,
            boot_id,
            pid,
            process_start_ticks,
            nonce,
        })
    }

    /// Whether `other` was derived on this host during this boot.
    #[must_use]
    pub fn same_host_and_boot(&self, other: &Self) -> bool {
        self.host_fingerprint == other.host_fingerprint && self.boot_id == other.boot_id
    }
}

fn parse_uuid_bytes(text: &str) -> Option<[u8; 16]> {
    let hex: String = text.chars().filter(|c| *c != '-').collect();
    if hex.len() != 32 {
        return None;
    }
    let mut bytes = [0_u8; 16];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).ok()?;
    }
    Some(bytes)
}

fn random_identity() -> GenerationRootResult<[u8; 16]> {
    use std::io::Read as _;
    let mut bytes = [0_u8; 16];
    let mut source = std::fs::File::open("/dev/urandom").map_err(|error| {
        GenerationRootError::new(
            GenerationRootErrorKind::Io,
            GenerationRootStage::AcquireLock,
        )
        .with_raw_os_error(error.raw_os_error().unwrap_or(0))
    })?;
    for _ in 0..8 {
        source.read_exact(&mut bytes).map_err(|error| {
            GenerationRootError::new(
                GenerationRootErrorKind::Io,
                GenerationRootStage::AcquireLock,
            )
            .with_raw_os_error(error.raw_os_error().unwrap_or(0))
        })?;
        if bytes != [0; 16] {
            return Ok(bytes);
        }
    }
    Err(GenerationRootError::new(
        GenerationRootErrorKind::Io,
        GenerationRootStage::AcquireLock,
    ))
}

fn candidate_extends(
    candidate: AuthorityRefV1,
    head: Option<AuthoritySlotV1>,
) -> Result<(), GenerationAuthorityErrorV1> {
    candidate.validate()?;
    match head {
        None => {
            if candidate.sequence != 1 || candidate.predecessor.is_some() {
                return Err(GenerationAuthorityErrorV1::BrokenPredecessorLink);
            }
        }
        Some(head) => {
            if candidate.sequence != head.authority.next_sequence()?
                || candidate.predecessor != Some(head.authority.fingerprint())
            {
                return Err(GenerationAuthorityErrorV1::BrokenPredecessorLink);
            }
        }
    }
    Ok(())
}

impl AuthorityPublisherV1<'_> {
    /// Root this publisher writes.
    #[must_use]
    pub const fn root_id(&self) -> [u8; 16] {
        self.root_id
    }

    fn lock_layout_matches(&self) -> Result<(), NotCommittedV1> {
        let lock_byte_len = self.root.layout.lock_byte_len();
        if lock_byte_len == AUTHORITY_PUBLISHER_LOCK_BYTES_V1 {
            Ok(())
        } else {
            Err(NotCommittedV1::LockLayout { lock_byte_len })
        }
    }

    /// Publish `candidate` as the new head if the anchors still hold `expected`.
    ///
    /// # Errors
    ///
    /// Infrastructure failures before any attempt byte was issued (root
    /// revalidation, descriptor faults, randomness) surface as
    /// [`GenerationRootError`]; every authority-level refusal is a typed
    /// [`PublicationOutcomeV1::NotCommitted`], and every failure after the
    /// attempt frame is a [`PublicationOutcomeV1::CommitOutcomeUnknown`].
    pub fn publish(
        &self,
        candidate: AuthorityRefV1,
        expected: ExpectedAuthorityPairV1,
        floor: Option<&dyn AntiRollbackFloorProviderV1>,
        idempotency_key: [u8; 16],
    ) -> GenerationRootResult<PublicationOutcomeV1> {
        if self.profile.require_mutation_authorized().is_err() {
            return Ok(PublicationOutcomeV1::NotCommitted(
                NotCommittedV1::ReadOnlyProfile,
            ));
        }
        if let Err(refusal) = self.lock_layout_matches() {
            return Ok(PublicationOutcomeV1::NotCommitted(refusal));
        }
        if idempotency_key == [0; 16] {
            return Ok(PublicationOutcomeV1::NotCommitted(
                NotCommittedV1::InvalidCandidate(GenerationAuthorityErrorV1::InvalidField {
                    field: "anti_rollback_floor.idempotency_key",
                }),
            ));
        }

        let mut guard = match self.root.try_exclusive_anchor_guard() {
            Ok(guard) => guard,
            Err(error) if error.kind() == GenerationRootErrorKind::LockContended => {
                return Ok(PublicationOutcomeV1::NotCommitted(
                    NotCommittedV1::LockContended,
                ));
            }
            // The lock path witnesses LOCK at open time and revalidates it
            // mutation-stably after the flock lands: a publication that
            // completed in that window is a fence, not a fault.
            Err(error)
                if matches!(
                    error.kind(),
                    GenerationRootErrorKind::ObjectChanged | GenerationRootErrorKind::HardLinked
                ) =>
            {
                return Ok(PublicationOutcomeV1::NotCommitted(
                    NotCommittedV1::AnchorsChanged,
                ));
            }
            Err(error) => return Err(error),
        };

        // Fence: exact expected pair, authenticated, under the lock.
        let observed = match decode_pair(&guard.authority_image.bytes, self.root_id) {
            Ok(observed) => observed,
            Err(error) => {
                return Ok(PublicationOutcomeV1::NotCommitted(
                    NotCommittedV1::UnreadableSlot(error),
                ));
            }
        };
        if observed != expected {
            return Ok(PublicationOutcomeV1::NotCommitted(
                NotCommittedV1::ExpectedPairMismatch {
                    observed: Box::new(observed),
                },
            ));
        }

        // Fence: floor head and profile resolution.
        let (floor_record, floor_ref) = match floor_head(floor, self.root_id) {
            Ok(head) => head,
            Err(error) => {
                return Ok(PublicationOutcomeV1::NotCommitted(
                    NotCommittedV1::FloorRefused(error),
                ));
            }
        };
        let head = match resolve_authority_slots_with_profile_v1(
            expected.first,
            expected.second,
            None,
            None,
            self.profile,
            floor_ref,
        ) {
            Ok(head) => head,
            Err(error) => {
                return Ok(PublicationOutcomeV1::NotCommitted(
                    NotCommittedV1::FloorRefused(error),
                ));
            }
        };
        if let Err(error) = candidate_extends(candidate, head) {
            return Ok(PublicationOutcomeV1::NotCommitted(
                NotCommittedV1::InvalidCandidate(error),
            ));
        }
        let target_slot = u8::try_from(candidate.sequence & 1).unwrap_or(1);
        let slot = match AuthoritySlotV1::new(target_slot, self.root_id, candidate) {
            Ok(slot) => slot,
            Err(error) => {
                return Ok(PublicationOutcomeV1::NotCommitted(
                    NotCommittedV1::InvalidCandidate(error),
                ));
            }
        };
        let slot_bytes = match slot.encode() {
            Ok(bytes) => bytes,
            Err(error) => {
                return Ok(PublicationOutcomeV1::NotCommitted(
                    NotCommittedV1::InvalidCandidate(error),
                ));
            }
        };

        let attempt_id = random_identity()?;
        let mut permit = AttemptPermitV1 {
            root_id: self.root_id,
            writer_id: self.writer_id,
            attempt_id,
            target_slot,
            candidate,
            expected,
            stage: AttemptStageV1::AttemptFrame,
            idempotency_key,
        };
        let fingerprint = candidate.fingerprint();
        let attempt_frame = match GenerationLockFrameV1::new(
            GenerationLockFrameKindV1::Attempt,
            self.root_id,
            self.writer_id,
            attempt_id,
            candidate.sequence,
            fingerprint,
        )
        .and_then(GenerationLockFrameV1::encode)
        {
            Ok(frame) => frame,
            Err(error) => {
                return Ok(PublicationOutcomeV1::NotCommitted(
                    NotCommittedV1::InvalidCandidate(error),
                ));
            }
        };

        // From here on, every failure is CommitOutcomeUnknown: the attempt
        // frame is the durable witness that a slot byte may follow.
        #[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
        if platform::crash_point(platform::TestBoundary::BeforePublishAttemptFrame).is_err() {
            return Ok(PublicationOutcomeV1::CommitOutcomeUnknown(permit));
        }
        if platform::write_lock_range(&mut guard.flock, ATTEMPT_FRAME_OFFSET, &attempt_frame)
            .is_err()
        {
            return Ok(PublicationOutcomeV1::CommitOutcomeUnknown(permit));
        }
        #[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
        if platform::crash_point(platform::TestBoundary::AfterPublishAttemptFrame).is_err() {
            return Ok(PublicationOutcomeV1::CommitOutcomeUnknown(permit));
        }

        permit.stage = AttemptStageV1::SlotWrite;
        #[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
        if platform::crash_point(platform::TestBoundary::BeforePublishSlotWrite).is_err() {
            return Ok(PublicationOutcomeV1::CommitOutcomeUnknown(permit));
        }
        if platform::write_control_range(
            &self.root.authority.inner,
            slot_offset(target_slot),
            &slot_bytes,
        )
        .is_err()
        {
            return Ok(PublicationOutcomeV1::CommitOutcomeUnknown(permit));
        }
        #[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
        if platform::crash_point(platform::TestBoundary::AfterPublishSlotWrite).is_err() {
            return Ok(PublicationOutcomeV1::CommitOutcomeUnknown(permit));
        }

        permit.stage = AttemptStageV1::SlotReread;
        #[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
        if platform::crash_point(platform::TestBoundary::BeforePublishSlotReread).is_err() {
            return Ok(PublicationOutcomeV1::CommitOutcomeUnknown(permit));
        }
        let reread_matches = platform::capture_anchor_image(&self.root.authority.inner)
            .ok()
            .and_then(|image| {
                image
                    .bytes
                    .get(slot_range(target_slot))
                    .map(|bytes| bytes == slot_bytes.as_slice())
            })
            .unwrap_or(false);
        if !reread_matches {
            return Ok(PublicationOutcomeV1::CommitOutcomeUnknown(permit));
        }

        permit.stage = AttemptStageV1::FloorAdvance;
        #[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
        if platform::crash_point(platform::TestBoundary::BeforePublishFloorAdvance).is_err() {
            return Ok(PublicationOutcomeV1::CommitOutcomeUnknown(permit));
        }
        let floor_receipt = match floor {
            None => None,
            Some(provider) => {
                let Ok(next) = AuthorityFloorV1::new(self.root_id, candidate) else {
                    return Ok(PublicationOutcomeV1::CommitOutcomeUnknown(permit));
                };
                match provider.compare_and_advance(floor_record, next, idempotency_key) {
                    Ok(receipt) => Some(receipt),
                    Err(_) => return Ok(PublicationOutcomeV1::CommitOutcomeUnknown(permit)),
                }
            }
        };
        #[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
        if platform::crash_point(platform::TestBoundary::AfterPublishFloorAdvance).is_err() {
            return Ok(PublicationOutcomeV1::CommitOutcomeUnknown(permit));
        }

        permit.stage = AttemptStageV1::OwnerFrame;
        #[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
        if platform::crash_point(platform::TestBoundary::BeforePublishOwnerFrame).is_err() {
            return Ok(PublicationOutcomeV1::CommitOutcomeUnknown(permit));
        }
        let Ok(owner_frame) = GenerationLockFrameV1::new(
            GenerationLockFrameKindV1::Owner,
            self.root_id,
            self.writer_id,
            attempt_id,
            candidate.sequence,
            fingerprint,
        )
        .and_then(GenerationLockFrameV1::encode) else {
            return Ok(PublicationOutcomeV1::CommitOutcomeUnknown(permit));
        };
        if platform::write_lock_range(&mut guard.flock, OWNER_FRAME_OFFSET, &owner_frame).is_err() {
            return Ok(PublicationOutcomeV1::CommitOutcomeUnknown(permit));
        }

        // The authority is durable and verified; a failed unlock is a
        // process-local condition the kernel resolves at descriptor close.
        let _ = guard.release();
        Ok(PublicationOutcomeV1::Committed {
            slot,
            floor: floor_receipt,
        })
    }

    /// Resolve an unknown attempt from the exact current slot, floor, and frame state.
    ///
    /// # Errors
    ///
    /// Infrastructure failures surface as [`GenerationRootError`]; every
    /// authority-level state is a typed [`ReconcileOutcomeV1`].
    pub fn reconcile(
        &self,
        permit: &AttemptPermitV1,
        floor: Option<&dyn AntiRollbackFloorProviderV1>,
    ) -> GenerationRootResult<ReconcileOutcomeV1> {
        if permit.root_id != self.root_id {
            return Ok(ReconcileOutcomeV1::RepairRequired(
                GenerationAuthorityErrorV1::RootMismatch,
            ));
        }
        let mut guard = match self.root.try_exclusive_anchor_guard() {
            Ok(guard) => guard,
            // Same fence as in `publish`: the permit stays valid; retry.
            Err(error)
                if matches!(
                    error.kind(),
                    GenerationRootErrorKind::LockContended
                        | GenerationRootErrorKind::ObjectChanged
                        | GenerationRootErrorKind::HardLinked
                ) =>
            {
                return Ok(ReconcileOutcomeV1::StillUnknown(*permit));
            }
            Err(error) => return Err(error),
        };
        let observed_target = match decode_slot(
            &guard.authority_image.bytes,
            permit.target_slot,
            self.root_id,
        ) {
            Ok(slot) => slot,
            Err(error) => return Ok(ReconcileOutcomeV1::RepairRequired(error)),
        };
        let expected_target = if permit.target_slot == 0 {
            permit.expected.first
        } else {
            permit.expected.second
        };

        let outcome = match observed_target {
            None => ReconcileOutcomeV1::NotCommitted,
            Some(slot) if Some(slot) == expected_target => ReconcileOutcomeV1::NotCommitted,
            Some(slot) if slot.authority == permit.candidate => {
                // Locally committed. Prove or advance the floor, then make
                // sure the owner frame records completion.
                let floor_receipt = match floor {
                    None => None,
                    Some(provider) => match provider.load(self.root_id) {
                        Err(_) => {
                            return Ok(ReconcileOutcomeV1::StillUnknown(*permit));
                        }
                        Ok(Some(record))
                            if record.authority.sequence > permit.candidate.sequence =>
                        {
                            // Floor is past us: only a newer local head explains it.
                            let pair = match decode_pair(&guard.authority_image.bytes, self.root_id)
                            {
                                Ok(pair) => pair,
                                Err(error) => {
                                    return Ok(ReconcileOutcomeV1::RepairRequired(error));
                                }
                            };
                            return Ok(
                                match resolve_authority_slots_with_profile_v1(
                                    pair.first,
                                    pair.second,
                                    None,
                                    None,
                                    GenerationRootSecurityProfileV1::CooperativeLocal,
                                    None,
                                ) {
                                    Ok(Some(head))
                                        if head.authority.sequence > permit.candidate.sequence =>
                                    {
                                        ReconcileOutcomeV1::CommittedThenSuperseded { head }
                                    }
                                    _ => ReconcileOutcomeV1::StillUnknown(*permit),
                                },
                            );
                        }
                        Ok(Some(record)) if record.authority == permit.candidate => Some(record),
                        Ok(current) => {
                            let next = match AuthorityFloorV1::new(self.root_id, permit.candidate) {
                                Ok(next) => next,
                                Err(error) => {
                                    return Ok(ReconcileOutcomeV1::RepairRequired(error));
                                }
                            };
                            match provider.compare_and_advance(
                                current,
                                next,
                                permit.idempotency_key,
                            ) {
                                Ok(receipt) => Some(receipt),
                                Err(_) => {
                                    return Ok(ReconcileOutcomeV1::StillUnknown(*permit));
                                }
                            }
                        }
                    },
                };
                let owner_frame = GenerationLockFrameV1::new(
                    GenerationLockFrameKindV1::Owner,
                    self.root_id,
                    self.writer_id,
                    permit.attempt_id,
                    permit.candidate.sequence,
                    permit.candidate.fingerprint(),
                )
                .and_then(GenerationLockFrameV1::encode);
                match owner_frame {
                    Ok(frame) => {
                        platform::write_lock_range(&mut guard.flock, OWNER_FRAME_OFFSET, &frame)?;
                    }
                    Err(error) => return Ok(ReconcileOutcomeV1::RepairRequired(error)),
                }
                ReconcileOutcomeV1::Committed {
                    slot,
                    floor: floor_receipt,
                }
            }
            Some(observed) => ReconcileOutcomeV1::Fork { observed },
        };
        let _ = guard.release();
        Ok(outcome)
    }

    /// Decode the owner frame currently persisted in LOCK, if any.
    ///
    /// # Errors
    ///
    /// Infrastructure failures surface as [`GenerationRootError`]; an
    /// unauthenticated frame is `Ok(None)`.
    pub fn owner_frame(&self) -> GenerationRootResult<Option<GenerationLockFrameV1>> {
        let guard = self.root.try_exclusive_anchor_guard()?;
        let image = platform::read_lock_image(&guard.flock)?;
        let _ = guard.release();
        let start = usize::try_from(OWNER_FRAME_OFFSET).unwrap_or(usize::MAX);
        let frame = image
            .get(start..start + GENERATION_LOCK_FRAME_BYTES_V1)
            .and_then(|bytes| {
                GenerationLockFrameV1::from_authenticated_bytes(bytes, self.root_id).ok()
            });
        Ok(frame)
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests;
