use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use super::codec::CanonicalEncoder;

// ---------------------------------------------------------------------------
// Immutable generation authority
// ---------------------------------------------------------------------------

/// Schema carried by immutable generation-authority references and slot frames.
pub const GENERATION_AUTHORITY_SCHEMA_V1: u16 = 1;
/// Schema for the deterministic anti-rollback floor reference provider.
pub const ANTI_ROLLBACK_FLOOR_SCHEMA_V1: u16 = 1;
/// Exact byte size of one physical `AUTHORITY` slot frame.
pub const GENERATION_AUTHORITY_SLOT_BYTES_V1: usize = 4_096;
/// Maximum canonical activation-manifest size accepted before decoding.
pub const GENERATION_ACTIVATION_MANIFEST_MAX_BYTES_V1: usize = 4_096;
/// Schema for activation manifests that explicitly declare optional ANN.
pub const GENERATION_ACTIVATION_MANIFEST_SCHEMA_V2: u16 = 2;
/// Maximum canonical schema-2 activation-manifest size accepted before decoding.
pub const GENERATION_ACTIVATION_MANIFEST_MAX_BYTES_V2: usize = 4_096;
/// Exact byte size of one physical `LOCK` owner or attempt frame.
pub const GENERATION_LOCK_FRAME_BYTES_V1: usize = 4_096;

const AUTHORITY_SLOT_DIGEST_BYTES: usize = 32;
pub(super) const AUTHORITY_SLOT_BODY_BYTES: usize =
    GENERATION_AUTHORITY_SLOT_BYTES_V1 - AUTHORITY_SLOT_DIGEST_BYTES;
const AUTHORITY_SLOT_MAGIC_V1: [u8; 8] = *b"FSAUTH01";
const AUTHORITY_SLOT_HEADER_BYTES: usize = 131;
const AUTHORITY_REF_MAGIC_V1: [u8; 9] = *b"FSAUTHREF";
pub(super) const AUTHORITY_REF_BYTES_V1: usize = 108;
const LOCK_FRAME_DIGEST_BYTES: usize = 32;
pub(super) const LOCK_FRAME_BODY_BYTES: usize =
    GENERATION_LOCK_FRAME_BYTES_V1 - LOCK_FRAME_DIGEST_BYTES;
const LOCK_FRAME_MAGIC_V1: [u8; 8] = *b"FSLOCK01";
const LOCK_FRAME_HEADER_BYTES: usize = 104;

/// A bounded reason why an authority reference, physical slot, or slot pair was
/// rejected. The error deliberately contains no paths, opaque object contents,
/// or unbounded caller data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GenerationAuthorityErrorV1 {
    /// A required fixed-format field had an unsupported or reserved value.
    #[error("generation authority has an invalid {field}")]
    InvalidField {
        /// Stable field identifier.
        field: &'static str,
    },
    /// A fixed-size frame was truncated or extended.
    #[error("generation authority slot has an invalid length")]
    InvalidSlotLength,
    /// The frame checksum did not authenticate its exact body bytes.
    #[error("generation authority slot checksum mismatch")]
    ChecksumMismatch,
    /// A frame was copied into the wrong physical slot.
    #[error("generation authority slot index mismatch")]
    SlotIndexMismatch,
    /// A frame was copied between authority roots.
    #[error("generation authority root mismatch")]
    RootMismatch,
    /// Canonical zero padding contained data.
    #[error("generation authority slot contains non-canonical padding")]
    NonCanonicalPadding,
    /// Both slots represented an equal-sequence fork.
    #[error("generation authority slots contain an equal-sequence fork")]
    EqualSequenceFork,
    /// A duplicate authority was not the permitted sequence-one genesis form.
    #[error("generation authority slots contain a non-genesis duplicate")]
    NonGenesisDuplicate,
    /// Both resolver inputs claimed the same physical authority slot.
    #[error("generation authority resolver received one physical slot twice")]
    DuplicatePhysicalSlot,
    /// A non-genesis authority was stored in the wrong physical slot.
    #[error("generation authority slot does not match its sequence parity")]
    SlotSequenceParity,
    /// The newer slot skipped a sequence or lacked an exact predecessor link.
    #[error("generation authority slots lack a consecutive predecessor link")]
    BrokenPredecessorLink,
    /// A lock frame belonged to a different authority root than the slots.
    #[error("generation authority lock root does not match the resolved authority")]
    LockRootMismatch,
    /// An owner lock did not attest the resolved authority head.
    #[error("generation authority lock does not attest the resolved authority")]
    LockAuthorityMismatch,
    /// A valid publication attempt remains unresolved and must be reconciled.
    #[error("generation authority publication attempt remains unresolved")]
    UnresolvedAttempt,
    /// The immutable authority counter cannot advance beyond `u64::MAX`.
    #[error("generation authority sequence is exhausted")]
    SequenceExhausted,
    /// A selected authority is missing or predates the required external floor.
    #[error("generation authority does not satisfy the required external floor")]
    AuthorityBelowFloor,
    /// A required-external profile was selected without an injected floor.
    #[error("generation authority required-external profile has no external floor")]
    ExternalFloorRequired,
    /// A mutation was attempted through an inspection-only profile.
    #[error("generation authority read-only profile forbids mutation")]
    ReadOnlyProfile,
    /// An anti-rollback floor compare-and-advance did not name the current
    /// exact record.
    #[error("generation authority anti-rollback floor compare-and-advance conflicted")]
    FloorCompareAndAdvanceConflict,
    /// An anti-rollback floor update did not advance its authority sequence.
    #[error("generation authority anti-rollback floor did not advance")]
    FloorSequenceRegression,
    /// An idempotency key was reused for a different floor operation.
    #[error("generation authority anti-rollback floor idempotency key conflicted")]
    FloorIdempotencyConflict,
    /// The deterministic reference provider's CAS version was exhausted.
    #[error("generation authority anti-rollback floor CAS version is exhausted")]
    FloorVersionExhausted,
    /// The deterministic reference provider cannot safely continue after a
    /// poisoned lock.
    #[error("generation authority anti-rollback floor store is unavailable")]
    FloorStoreUnavailable,
    /// The immutable activation manifest did not match its stored self-seal.
    #[error("generation activation manifest self-seal mismatch")]
    ManifestSelfSealMismatch,
    /// An externally addressed manifest did not match its authority reference.
    #[error("generation activation manifest does not match its authority reference")]
    ManifestReferenceMismatch,
}

/// Immutable pointer to one activation-manifest object.
///
/// This is intentionally distinct from [`super::ArtifactGenerationIdentityV1`]: the
/// latter identifies one artifact build, while this counter is the monotone
/// activation authority for a root. Object IDs are random opaque bytes, never a
/// filename or display name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AuthorityRefV1 {
    /// [`GENERATION_AUTHORITY_SCHEMA_V1`].
    pub schema_version: u16,
    /// Monotone activation sequence, beginning at one.
    pub sequence: u64,
    /// Opaque immutable activation-manifest object identifier.
    pub object_id: [u8; 16],
    /// Exact activation-manifest byte length.
    pub manifest_len: u64,
    /// SHA-256 of the exact activation-manifest bytes.
    pub manifest_sha256: [u8; 32],
    /// SHA-256 fingerprint of the immediately preceding authority reference.
    /// Genesis has no predecessor.
    pub predecessor: Option<[u8; 32]>,
}

impl AuthorityRefV1 {
    /// Construct and validate one immutable activation authority.
    ///
    /// # Errors
    ///
    /// Returns a bounded typed error when a reserved identity value is used or
    /// the sequence/predecessor relation is not canonical.
    pub fn new(
        sequence: u64,
        object_id: [u8; 16],
        manifest_len: u64,
        manifest_sha256: [u8; 32],
        predecessor: Option<[u8; 32]>,
    ) -> Result<Self, GenerationAuthorityErrorV1> {
        let reference = Self {
            schema_version: GENERATION_AUTHORITY_SCHEMA_V1,
            sequence,
            object_id,
            manifest_len,
            manifest_sha256,
            predecessor,
        };
        reference.validate()?;
        Ok(reference)
    }

    /// Validate the fixed, forward-safe authority-reference contract.
    ///
    /// # Errors
    ///
    /// Returns a bounded typed error for every invalid field.
    pub fn validate(&self) -> Result<(), GenerationAuthorityErrorV1> {
        if self.schema_version != GENERATION_AUTHORITY_SCHEMA_V1 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.schema_version",
            });
        }
        if self.sequence == 0 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.sequence",
            });
        }
        if self.object_id == [0; 16] {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.object_id",
            });
        }
        if self.manifest_len == 0 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.manifest_len",
            });
        }
        if self.manifest_sha256 == [0; 32] {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.manifest_sha256",
            });
        }
        if self.sequence == 1 && self.predecessor.is_some() {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.predecessor",
            });
        }
        if self.sequence > 1 && self.predecessor.is_none() {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.predecessor",
            });
        }
        if self.predecessor == Some([0; 32]) {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.predecessor",
            });
        }
        Ok(())
    }

    /// Return the next authority sequence without allowing counter rollover.
    ///
    /// # Errors
    ///
    /// Returns a typed exhaustion error for the terminal `u64::MAX` authority
    /// rather than permitting an ABA-like wrap to an earlier sequence.
    pub fn next_sequence(&self) -> Result<u64, GenerationAuthorityErrorV1> {
        self.validate()?;
        self.sequence
            .checked_add(1)
            .ok_or(GenerationAuthorityErrorV1::SequenceExhausted)
    }

    /// Canonical bytes used to link consecutive authority references.
    #[must_use]
    pub fn canonical_bytes(&self) -> [u8; AUTHORITY_REF_BYTES_V1] {
        let mut bytes = [0_u8; AUTHORITY_REF_BYTES_V1];
        bytes[..9].copy_from_slice(&AUTHORITY_REF_MAGIC_V1);
        bytes[9..11].copy_from_slice(&self.schema_version.to_be_bytes());
        bytes[11..19].copy_from_slice(&self.sequence.to_be_bytes());
        bytes[19..35].copy_from_slice(&self.object_id);
        bytes[35..43].copy_from_slice(&self.manifest_len.to_be_bytes());
        bytes[43..75].copy_from_slice(&self.manifest_sha256);
        bytes[75] = u8::from(self.predecessor.is_some());
        if let Some(predecessor) = self.predecessor {
            bytes[76..].copy_from_slice(&predecessor);
        }
        bytes
    }

    /// Decode the exact fixed canonical representation of one authority
    /// reference without allocating from caller-controlled lengths.
    ///
    /// # Errors
    ///
    /// Returns a typed error for malformed, future, or non-canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, GenerationAuthorityErrorV1> {
        if bytes.len() != AUTHORITY_REF_BYTES_V1 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.canonical_bytes",
            });
        }
        if !bytes[..9].eq(AUTHORITY_REF_MAGIC_V1.as_slice()) {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.magic",
            });
        }
        let predecessor = match bytes[75] {
            0 => {
                if !bytes[76..].iter().all(|byte| byte.eq(&0)) {
                    return Err(GenerationAuthorityErrorV1::NonCanonicalPadding);
                }
                None
            }
            1 => Some(bytes[76..].try_into().map_err(|_| {
                GenerationAuthorityErrorV1::InvalidField {
                    field: "authority_ref.predecessor",
                }
            })?),
            _ => {
                return Err(GenerationAuthorityErrorV1::InvalidField {
                    field: "authority_ref.predecessor_present",
                });
            }
        };
        let reference = Self {
            schema_version: u16::from_be_bytes([bytes[9], bytes[10]]),
            sequence: u64::from_be_bytes(bytes[11..19].try_into().map_err(|_| {
                GenerationAuthorityErrorV1::InvalidField {
                    field: "authority_ref.sequence",
                }
            })?),
            object_id: bytes[19..35].try_into().map_err(|_| {
                GenerationAuthorityErrorV1::InvalidField {
                    field: "authority_ref.object_id",
                }
            })?,
            manifest_len: u64::from_be_bytes(bytes[35..43].try_into().map_err(|_| {
                GenerationAuthorityErrorV1::InvalidField {
                    field: "authority_ref.manifest_len",
                }
            })?),
            manifest_sha256: bytes[43..75].try_into().map_err(|_| {
                GenerationAuthorityErrorV1::InvalidField {
                    field: "authority_ref.manifest_sha256",
                }
            })?,
            predecessor,
        };
        reference.validate()?;
        if !reference.canonical_bytes().eq(bytes) {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.canonical_bytes",
            });
        }
        Ok(reference)
    }

    /// SHA-256 fingerprint of this exact canonical reference.
    #[must_use]
    pub fn fingerprint(&self) -> [u8; 32] {
        Sha256::digest(self.canonical_bytes()).into()
    }
}

pub(super) fn predecessor_matches_authority(
    predecessor: Option<[u8; 32]>,
    authority: AuthorityRefV1,
) -> bool {
    predecessor.is_some_and(|predecessor| {
        constant_time_fingerprint_eq(&predecessor, &authority.fingerprint())
    })
}

pub(super) fn constant_time_fingerprint_eq(left: &[u8; 32], right: &[u8; 32]) -> bool {
    let mut difference = 0_u8;
    for index in 0..left.len() {
        difference |= left[index] ^ right[index];
    }
    difference == 0
}

/// A decoded, authenticated physical authority slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuthoritySlotV1 {
    /// Physical slot selected by the publisher (`0` or `1`).
    pub slot_index: u8,
    /// Root identity bound into the frame.
    pub root_id: [u8; 16],
    /// Immutable authority reference carried by the frame.
    pub authority: AuthorityRefV1,
}

/// An externally retained immutable authority that bounds acceptable recovery.
///
/// It is pure evidence, not a filesystem pointer: consumers may require an
/// exact floor or its immediately predecessor-linked successor without
/// silently accepting a stale or unprovable authority head.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AuthorityFloorV1 {
    /// Root identity that prevents a floor from being replayed across roots.
    pub root_id: [u8; 16],
    /// Exact externally retained authority reference.
    pub authority: AuthorityRefV1,
}

impl AuthorityFloorV1 {
    /// Construct and validate one bounded external authority floor.
    ///
    /// # Errors
    ///
    /// Returns a typed error for reserved root or authority values.
    pub fn new(
        root_id: [u8; 16],
        authority: AuthorityRefV1,
    ) -> Result<Self, GenerationAuthorityErrorV1> {
        let floor = Self { root_id, authority };
        floor.validate()?;
        Ok(floor)
    }

    pub(super) fn validate(&self) -> Result<(), GenerationAuthorityErrorV1> {
        if self.root_id == [0; 16] {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_floor.root_id",
            });
        }
        self.authority.validate()
    }
}

/// Explicit security posture for opening one generation root.
///
/// There is deliberately no `Default` implementation: a caller must choose a
/// posture instead of silently falling back from an externally anchored root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenerationRootSecurityProfileV1 {
    /// Require a consumer-owned, externally retained authority floor before a
    /// root can be selected for use.
    RequiredExternal,
    /// Permit owner-controlled local recovery while reporting that it cannot
    /// detect a hostile whole-store rollback.
    CooperativeLocal,
    /// Permit inspection only; publication, repair, rollback, and garbage
    /// collection must be rejected by the caller before mutation.
    ReadOnlyUnanchored,
}

impl GenerationRootSecurityProfileV1 {
    /// Whether this profile authorizes a caller to mutate a generation root.
    #[must_use]
    pub const fn permits_mutation(self) -> bool {
        !matches!(self, Self::ReadOnlyUnanchored)
    }

    /// Reject a mutation through an inspection-only profile.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationAuthorityErrorV1::ReadOnlyProfile`] when this is
    /// [`Self::ReadOnlyUnanchored`].
    pub const fn require_mutation_authorized(self) -> Result<(), GenerationAuthorityErrorV1> {
        if self.permits_mutation() {
            Ok(())
        } else {
            Err(GenerationAuthorityErrorV1::ReadOnlyProfile)
        }
    }
}

/// Versioned, digest-bound result of one successful anti-rollback floor CAS.
///
/// The record is intentionally independent of any generation-root path: its
/// caller owns storage and replication of this receipt outside the replaceable
/// root it anchors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AntiRollbackFloorRecordV1 {
    /// [`ANTI_ROLLBACK_FLOOR_SCHEMA_V1`].
    pub schema_version: u16,
    /// Generation root anchored by this record.
    pub root_id: [u8; 16],
    /// Exact externally retained authority floor.
    pub authority: AuthorityRefV1,
    /// Monotone successful compare-and-advance version for this root.
    pub cas_version: u64,
    /// SHA-256 over the canonical record fields above.
    pub record_sha256: [u8; 32],
}

impl AntiRollbackFloorRecordV1 {
    pub(super) fn new(
        floor: AuthorityFloorV1,
        cas_version: u64,
    ) -> Result<Self, GenerationAuthorityErrorV1> {
        floor.validate()?;
        if cas_version == 0 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "anti_rollback_floor.cas_version",
            });
        }
        let mut record = Self {
            schema_version: ANTI_ROLLBACK_FLOOR_SCHEMA_V1,
            root_id: floor.root_id,
            authority: floor.authority,
            cas_version,
            record_sha256: [0; 32],
        };
        record.record_sha256 = record.computed_record_sha256();
        Ok(record)
    }

    pub(super) fn validate(&self) -> Result<(), GenerationAuthorityErrorV1> {
        if self.schema_version != ANTI_ROLLBACK_FLOOR_SCHEMA_V1 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "anti_rollback_floor.schema_version",
            });
        }
        AuthorityFloorV1::new(self.root_id, self.authority)?;
        if self.cas_version == 0 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "anti_rollback_floor.cas_version",
            });
        }
        if self.record_sha256 != self.computed_record_sha256() {
            return Err(GenerationAuthorityErrorV1::ChecksumMismatch);
        }
        Ok(())
    }

    pub(super) fn computed_record_sha256(&self) -> [u8; 32] {
        let mut encoder = CanonicalEncoder::new(b"frankensearch.anti-rollback-floor-record.v1");
        encoder.u16(self.schema_version);
        encoder.bytes(&self.root_id);
        encoder.bytes(&self.authority.canonical_bytes());
        encoder.u64(self.cas_version);
        Sha256::digest(encoder.finish()).into()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AntiRollbackFloorRequestV1 {
    root_id: [u8; 16],
    expected_record_sha256: Option<[u8; 32]>,
    next_authority: AuthorityRefV1,
    result: AntiRollbackFloorRecordV1,
}

#[derive(Default)]
pub(super) struct AntiRollbackFloorStoreStateV1 {
    pub(super) records: BTreeMap<[u8; 16], AntiRollbackFloorRecordV1>,
    requests: BTreeMap<[u8; 16], AntiRollbackFloorRequestV1>,
}

/// Deterministic, linearizable in-memory reference provider for anti-rollback
/// floor conformance tests and consumer integration tests.
///
/// This provider is deliberately not a crash-durable authority. Production
/// callers must inject a consumer-owned provider outside the generation root;
/// this implementation makes the exact load/CAS/idempotency contract testable
/// without selecting a storage backend.
#[derive(Default)]
pub struct InMemoryAntiRollbackFloorStoreV1 {
    pub(super) state: std::sync::Mutex<AntiRollbackFloorStoreStateV1>,
}

impl InMemoryAntiRollbackFloorStoreV1 {
    /// Construct an empty deterministic reference provider.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Load the exact current floor record for one root.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationAuthorityErrorV1::FloorStoreUnavailable`] when a
    /// prior panic poisoned the reference-provider lock.
    pub fn load(
        &self,
        root_id: [u8; 16],
    ) -> Result<Option<AntiRollbackFloorRecordV1>, GenerationAuthorityErrorV1> {
        if root_id == [0; 16] {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "anti_rollback_floor.root_id",
            });
        }
        let state = self
            .state
            .lock()
            .map_err(|_| GenerationAuthorityErrorV1::FloorStoreUnavailable)?;
        Ok(state.records.get(&root_id).copied())
    }

    /// Atomically compare and advance one root's anti-rollback floor.
    ///
    /// `expected` must be the exact record returned by a prior [`Self::load`]
    /// call (or `None` for a previously unanchored root). Repeating the same
    /// operation with the same non-zero `idempotency_key` returns the original
    /// successful receipt; reusing that key for another operation fails closed.
    ///
    /// # Errors
    ///
    /// Returns a typed conflict, regression, idempotency, exhaustion, or store
    /// availability error without changing the recorded floor.
    pub fn compare_and_advance(
        &self,
        expected: Option<AntiRollbackFloorRecordV1>,
        next: AuthorityFloorV1,
        idempotency_key: [u8; 16],
    ) -> Result<AntiRollbackFloorRecordV1, GenerationAuthorityErrorV1> {
        next.validate()?;
        if idempotency_key == [0; 16] {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "anti_rollback_floor.idempotency_key",
            });
        }
        if let Some(expected) = expected {
            expected.validate()?;
            if expected.root_id != next.root_id {
                return Err(GenerationAuthorityErrorV1::RootMismatch);
            }
        }

        let mut state = self
            .state
            .lock()
            .map_err(|_| GenerationAuthorityErrorV1::FloorStoreUnavailable)?;
        let expected_record_sha256 = expected.map(|record| record.record_sha256);
        if let Some(previous) = state.requests.get(&idempotency_key) {
            if previous.root_id == next.root_id
                && previous.expected_record_sha256 == expected_record_sha256
                && previous.next_authority == next.authority
            {
                return Ok(previous.result);
            }
            return Err(GenerationAuthorityErrorV1::FloorIdempotencyConflict);
        }

        let current = state.records.get(&next.root_id).copied();
        if current != expected {
            return Err(GenerationAuthorityErrorV1::FloorCompareAndAdvanceConflict);
        }
        match current {
            Some(record) => {
                let expected_sequence = record.authority.next_sequence()?;
                if next.authority.sequence <= record.authority.sequence {
                    return Err(GenerationAuthorityErrorV1::FloorSequenceRegression);
                }
                if next.authority.sequence != expected_sequence
                    || !predecessor_matches_authority(next.authority.predecessor, record.authority)
                {
                    return Err(GenerationAuthorityErrorV1::BrokenPredecessorLink);
                }
            }
            None if next.authority.sequence != 1 || next.authority.predecessor.is_some() => {
                return Err(GenerationAuthorityErrorV1::BrokenPredecessorLink);
            }
            None => {}
        }
        let cas_version = current
            .map(|record| record.cas_version)
            .unwrap_or(0)
            .checked_add(1)
            .ok_or(GenerationAuthorityErrorV1::FloorVersionExhausted)?;
        let result = AntiRollbackFloorRecordV1::new(next, cas_version)?;
        state.records.insert(next.root_id, result);
        state.requests.insert(
            idempotency_key,
            AntiRollbackFloorRequestV1 {
                root_id: next.root_id,
                expected_record_sha256,
                next_authority: next.authority,
                result,
            },
        );
        drop(state);
        Ok(result)
    }
}

/// Kind of fixed `LOCK` frame. Owner and attempt remain distinct on disk so an
/// interrupted attempt cannot be silently mistaken for authority ownership.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenerationLockFrameKindV1 {
    /// Current writer ownership record.
    Owner,
    /// In-progress publication attempt record.
    Attempt,
}

impl GenerationLockFrameKindV1 {
    const fn tag(self) -> u8 {
        match self {
            Self::Owner => 1,
            Self::Attempt => 2,
        }
    }

    fn from_tag(tag: u8) -> Result<Self, GenerationAuthorityErrorV1> {
        match tag {
            1 => Ok(Self::Owner),
            2 => Ok(Self::Attempt),
            _ => Err(GenerationAuthorityErrorV1::InvalidField {
                field: "generation_lock.kind",
            }),
        }
    }
}

/// Bounded owner/attempt evidence for the fixed generation-root `LOCK` path.
///
/// It is an immutable byte contract only: taking a filesystem lock or deciding
/// whether an attempt may publish belongs to the publisher lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenerationLockFrameV1 {
    /// Owner or attempt frame classification.
    pub kind: GenerationLockFrameKindV1,
    /// Root ID that prevents cross-root frame replay.
    pub root_id: [u8; 16],
    /// Opaque writer identity.
    pub writer_id: [u8; 16],
    /// Opaque immutable publication attempt identity.
    pub attempt_id: [u8; 16],
    /// Monotone writer fence, beginning at one.
    pub fence: u64,
    /// Fingerprint of the authority reference the lock state observed.
    pub authority_fingerprint: [u8; 32],
}

impl GenerationLockFrameV1 {
    /// Construct and validate a bounded lock owner/attempt frame.
    ///
    /// # Errors
    ///
    /// Returns a typed error when any identity or fence is reserved.
    pub fn new(
        kind: GenerationLockFrameKindV1,
        root_id: [u8; 16],
        writer_id: [u8; 16],
        attempt_id: [u8; 16],
        fence: u64,
        authority_fingerprint: [u8; 32],
    ) -> Result<Self, GenerationAuthorityErrorV1> {
        let frame = Self {
            kind,
            root_id,
            writer_id,
            attempt_id,
            fence,
            authority_fingerprint,
        };
        frame.validate()?;
        Ok(frame)
    }

    /// Validate the fixed lock-frame facts.
    ///
    /// # Errors
    ///
    /// Returns a typed error when any fact is reserved or cannot identify a
    /// specific publication attempt.
    pub fn validate(&self) -> Result<(), GenerationAuthorityErrorV1> {
        if self.root_id == [0; 16]
            || self.writer_id == [0; 16]
            || self.attempt_id == [0; 16]
            || self.fence == 0
            || self.authority_fingerprint == [0; 32]
        {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "generation_lock.frame",
            });
        }
        Ok(())
    }

    /// Encode this owner/attempt frame into its fixed canonical form.
    ///
    /// # Errors
    ///
    /// Returns a typed error instead of serializing a reserved frame.
    pub fn encode(
        self,
    ) -> Result<[u8; GENERATION_LOCK_FRAME_BYTES_V1], GenerationAuthorityErrorV1> {
        self.validate()?;
        let mut bytes = [0_u8; GENERATION_LOCK_FRAME_BYTES_V1];
        bytes[..8].copy_from_slice(&LOCK_FRAME_MAGIC_V1);
        bytes[8..10].copy_from_slice(&GENERATION_AUTHORITY_SCHEMA_V1.to_be_bytes());
        bytes[10] = self.kind.tag();
        bytes[16..32].copy_from_slice(&self.root_id);
        bytes[32..48].copy_from_slice(&self.writer_id);
        bytes[48..64].copy_from_slice(&self.attempt_id);
        bytes[64..72].copy_from_slice(&self.fence.to_be_bytes());
        bytes[72..104].copy_from_slice(&self.authority_fingerprint);
        let digest = Sha256::digest(&bytes[..LOCK_FRAME_BODY_BYTES]);
        bytes[LOCK_FRAME_BODY_BYTES..].copy_from_slice(&digest);
        Ok(bytes)
    }

    /// Parse and authenticate a fixed `LOCK` frame for the expected root.
    ///
    /// # Errors
    ///
    /// Returns a typed error for malformed, copied, tampered, or noncanonical
    /// owner/attempt frames.
    pub fn from_authenticated_bytes(
        bytes: &[u8],
        expected_root_id: [u8; 16],
    ) -> Result<Self, GenerationAuthorityErrorV1> {
        if bytes.len() != GENERATION_LOCK_FRAME_BYTES_V1 {
            return Err(GenerationAuthorityErrorV1::InvalidSlotLength);
        }
        if !bytes[..8].eq(LOCK_FRAME_MAGIC_V1.as_slice())
            || !bytes[8..10].eq(GENERATION_AUTHORITY_SCHEMA_V1.to_be_bytes().as_slice())
            || !bytes[11..16].iter().all(|byte| byte.eq(&0))
        {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "generation_lock.header",
            });
        }
        if !bytes[16..32].eq(expected_root_id.as_slice()) {
            return Err(GenerationAuthorityErrorV1::RootMismatch);
        }
        let expected_digest = Sha256::digest(&bytes[..LOCK_FRAME_BODY_BYTES]);
        if !bytes[LOCK_FRAME_BODY_BYTES..].eq(&expected_digest[..]) {
            return Err(GenerationAuthorityErrorV1::ChecksumMismatch);
        }
        if bytes[LOCK_FRAME_HEADER_BYTES..LOCK_FRAME_BODY_BYTES]
            .iter()
            .any(|byte| !byte.eq(&0))
        {
            return Err(GenerationAuthorityErrorV1::NonCanonicalPadding);
        }
        let mut writer_id = [0_u8; 16];
        writer_id.copy_from_slice(&bytes[32..48]);
        let mut attempt_id = [0_u8; 16];
        attempt_id.copy_from_slice(&bytes[48..64]);
        let mut authority_fingerprint = [0_u8; 32];
        authority_fingerprint.copy_from_slice(&bytes[72..104]);
        Self::new(
            GenerationLockFrameKindV1::from_tag(bytes[10])?,
            expected_root_id,
            writer_id,
            attempt_id,
            u64::from_be_bytes(bytes[64..72].try_into().map_err(|_| {
                GenerationAuthorityErrorV1::InvalidField {
                    field: "generation_lock.fence",
                }
            })?),
            authority_fingerprint,
        )
    }
}

impl AuthoritySlotV1 {
    /// Construct one physical authority-slot frame.
    ///
    /// # Errors
    ///
    /// Returns a bounded typed error when the physical slot/root or its
    /// authority reference is invalid.
    pub fn new(
        slot_index: u8,
        root_id: [u8; 16],
        authority: AuthorityRefV1,
    ) -> Result<Self, GenerationAuthorityErrorV1> {
        let slot = Self {
            slot_index,
            root_id,
            authority,
        };
        slot.validate()?;
        Ok(slot)
    }

    /// Validate this in-memory physical slot before encoding or resolving it.
    ///
    /// # Errors
    ///
    /// Returns a bounded typed error for invalid physical binding data.
    pub fn validate(&self) -> Result<(), GenerationAuthorityErrorV1> {
        if self.slot_index > 1 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_slot.slot_index",
            });
        }
        if self.root_id == [0; 16] {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_slot.root_id",
            });
        }
        self.authority.validate()
    }

    /// Encode this slot into its fixed 4096-byte canonical representation.
    ///
    /// # Errors
    ///
    /// Returns a bounded typed error instead of serializing an invalid slot.
    pub fn encode(
        self,
    ) -> Result<[u8; GENERATION_AUTHORITY_SLOT_BYTES_V1], GenerationAuthorityErrorV1> {
        self.validate()?;
        let mut bytes = [0_u8; GENERATION_AUTHORITY_SLOT_BYTES_V1];
        bytes[..8].copy_from_slice(&AUTHORITY_SLOT_MAGIC_V1);
        bytes[8..10].copy_from_slice(&GENERATION_AUTHORITY_SCHEMA_V1.to_be_bytes());
        bytes[10] = self.slot_index;
        bytes[16..32].copy_from_slice(&self.root_id);
        bytes[32..34].copy_from_slice(&self.authority.schema_version.to_be_bytes());
        bytes[34..42].copy_from_slice(&self.authority.sequence.to_be_bytes());
        bytes[42..58].copy_from_slice(&self.authority.object_id);
        bytes[58..66].copy_from_slice(&self.authority.manifest_len.to_be_bytes());
        bytes[66..98].copy_from_slice(&self.authority.manifest_sha256);
        bytes[98] = u8::from(self.authority.predecessor.is_some());
        if let Some(predecessor) = self.authority.predecessor {
            bytes[99..131].copy_from_slice(&predecessor);
        }
        let digest = Sha256::digest(&bytes[..AUTHORITY_SLOT_BODY_BYTES]);
        bytes[AUTHORITY_SLOT_BODY_BYTES..].copy_from_slice(&digest);
        Ok(bytes)
    }

    /// Decode and authenticate one physical slot for the expected root and
    /// physical index.
    ///
    /// # Errors
    ///
    /// Returns a typed error for checksum, root, index, canonical-padding, or
    /// authority-reference violations.
    pub fn from_authenticated_bytes(
        bytes: &[u8],
        expected_slot_index: u8,
        expected_root_id: [u8; 16],
    ) -> Result<Self, GenerationAuthorityErrorV1> {
        if bytes.len() != GENERATION_AUTHORITY_SLOT_BYTES_V1 {
            return Err(GenerationAuthorityErrorV1::InvalidSlotLength);
        }
        if expected_slot_index > 1 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_slot.expected_slot_index",
            });
        }
        if !bytes[..8].eq(AUTHORITY_SLOT_MAGIC_V1.as_slice())
            || !bytes[8..10].eq(GENERATION_AUTHORITY_SCHEMA_V1.to_be_bytes().as_slice())
            || !bytes[11..16].iter().all(|byte| byte.eq(&0))
        {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_slot.header",
            });
        }
        if !bytes[10].eq(&expected_slot_index) {
            return Err(GenerationAuthorityErrorV1::SlotIndexMismatch);
        }
        if !bytes[16..32].eq(expected_root_id.as_slice()) {
            return Err(GenerationAuthorityErrorV1::RootMismatch);
        }
        let expected_digest = Sha256::digest(&bytes[..AUTHORITY_SLOT_BODY_BYTES]);
        if !bytes[AUTHORITY_SLOT_BODY_BYTES..].eq(&expected_digest[..]) {
            return Err(GenerationAuthorityErrorV1::ChecksumMismatch);
        }
        if bytes[AUTHORITY_SLOT_HEADER_BYTES..AUTHORITY_SLOT_BODY_BYTES]
            .iter()
            .any(|byte| !byte.eq(&0))
        {
            return Err(GenerationAuthorityErrorV1::NonCanonicalPadding);
        }

        let mut object_id = [0_u8; 16];
        object_id.copy_from_slice(&bytes[42..58]);
        let mut manifest_sha256 = [0_u8; 32];
        manifest_sha256.copy_from_slice(&bytes[66..98]);
        let predecessor = match bytes[98] {
            0 => {
                if !bytes[99..131].iter().all(|byte| byte.eq(&0)) {
                    return Err(GenerationAuthorityErrorV1::NonCanonicalPadding);
                }
                None
            }
            1 => {
                let mut predecessor = [0_u8; 32];
                predecessor.copy_from_slice(&bytes[99..131]);
                Some(predecessor)
            }
            _ => {
                return Err(GenerationAuthorityErrorV1::InvalidField {
                    field: "authority_slot.predecessor_present",
                });
            }
        };
        let authority = AuthorityRefV1 {
            schema_version: u16::from_be_bytes([bytes[32], bytes[33]]),
            sequence: u64::from_be_bytes(bytes[34..42].try_into().map_err(|_| {
                GenerationAuthorityErrorV1::InvalidField {
                    field: "authority_slot.sequence",
                }
            })?),
            object_id,
            manifest_len: u64::from_be_bytes(bytes[58..66].try_into().map_err(|_| {
                GenerationAuthorityErrorV1::InvalidField {
                    field: "authority_slot.manifest_len",
                }
            })?),
            manifest_sha256,
            predecessor,
        };
        authority.validate()?;
        Self::new(expected_slot_index, expected_root_id, authority)
    }
}

/// Resolve the two physical authority slots without performing I/O.
///
/// A single surviving valid slot remains usable. Two different slots must be
/// exactly consecutive and predecessor-linked; an equal reference is legal
/// only for the duplicated sequence-one genesis record.
///
/// # Errors
///
/// Returns a typed error instead of selecting a structurally credible fork or
/// skipping a missing authority history edge.
pub fn resolve_authority_slots_v1(
    first: Option<AuthoritySlotV1>,
    second: Option<AuthoritySlotV1>,
) -> Result<Option<AuthoritySlotV1>, GenerationAuthorityErrorV1> {
    if let Some(slot) = first {
        slot.validate()?;
    }
    if let Some(slot) = second {
        slot.validate()?;
    }
    match (first, second) {
        (None, None) => Ok(None),
        (Some(slot), None) | (None, Some(slot)) => {
            if !slot_matches_authority_sequence(slot) {
                return Err(GenerationAuthorityErrorV1::SlotSequenceParity);
            }
            Ok(Some(slot))
        }
        (Some(first), Some(second)) => {
            // ubs:ignore — slot indices and root IDs are public structural identities.
            if first.root_id != second.root_id {
                return Err(GenerationAuthorityErrorV1::InvalidField {
                    field: "authority_slot.pair",
                });
            }
            // ubs:ignore — physical slot indices are public structural identities.
            if first.slot_index == second.slot_index {
                return Err(GenerationAuthorityErrorV1::DuplicatePhysicalSlot);
            }
            // ubs:ignore — authority sequences are public monotone counters.
            if first.authority.sequence == second.authority.sequence {
                // ubs:ignore — authority references carry only public immutable identities.
                if first.authority != second.authority {
                    return Err(GenerationAuthorityErrorV1::EqualSequenceFork);
                }
                if first.authority.sequence != 1 {
                    return Err(GenerationAuthorityErrorV1::NonGenesisDuplicate);
                }
                return Ok(Some(first));
            }

            let (older, newer) = if first.authority.sequence < second.authority.sequence {
                (first, second)
            } else {
                (second, first)
            };
            if !slot_matches_authority_sequence(older) || !slot_matches_authority_sequence(newer) {
                return Err(GenerationAuthorityErrorV1::SlotSequenceParity);
            }
            // ubs:ignore — sequence/predecessor fingerprints are public history identities.
            if newer.authority.sequence != older.authority.sequence.saturating_add(1)
                // ubs:ignore — predecessor fingerprints are public immutable history identities.
                || newer.authority.predecessor != Some(older.authority.fingerprint())
            {
                return Err(GenerationAuthorityErrorV1::BrokenPredecessorLink);
            }
            Ok(Some(newer))
        }
    }
}

/// Decode and resolve the two physical authority frames without filesystem
/// access.
///
/// A present but malformed frame is never converted to `None`: callers must
/// reconcile that corruption explicitly rather than treating it as one-slot
/// survival. Only an absent physical frame may enter the survival path.
///
/// # Errors
///
/// Returns the exact bounded slot decode error for malformed, copied, or
/// cross-root frames before attempting pair resolution.
pub fn resolve_authority_slot_frames_v1(
    first: Option<&[u8]>,
    second: Option<&[u8]>,
    expected_root_id: [u8; 16],
) -> Result<Option<AuthoritySlotV1>, GenerationAuthorityErrorV1> {
    let first = first
        .map(|bytes| AuthoritySlotV1::from_authenticated_bytes(bytes, 0, expected_root_id))
        .transpose()?;
    let second = second
        .map(|bytes| AuthoritySlotV1::from_authenticated_bytes(bytes, 1, expected_root_id))
        .transpose()?;
    resolve_authority_slots_v1(first, second)
}

/// Resolve two authority slots against one exact externally retained floor.
///
/// The selected authority must be the exact floor or its immediate
/// predecessor-linked successor. A later but unprovable head is not selected;
/// callers must supply its intervening authority evidence explicitly.
///
/// # Errors
///
/// Returns a typed error for missing/stale authorities, cross-root floors,
/// equal-sequence divergence, gaps, or sequence exhaustion.
pub fn resolve_authority_slots_at_floor_v1(
    first: Option<AuthoritySlotV1>,
    second: Option<AuthoritySlotV1>,
    floor: AuthorityFloorV1,
) -> Result<AuthoritySlotV1, GenerationAuthorityErrorV1> {
    let resolved = resolve_authority_slots_v1(first, second)?
        .ok_or(GenerationAuthorityErrorV1::AuthorityBelowFloor)?;
    resolve_selected_authority_at_floor_v1(resolved, floor)
}

fn resolve_selected_authority_at_floor_v1(
    resolved: AuthoritySlotV1,
    floor: AuthorityFloorV1,
) -> Result<AuthoritySlotV1, GenerationAuthorityErrorV1> {
    floor.validate()?;
    // ubs:ignore — root IDs are public structural identities.
    if resolved.root_id != floor.root_id {
        return Err(GenerationAuthorityErrorV1::RootMismatch);
    }
    if resolved.authority.sequence < floor.authority.sequence {
        return Err(GenerationAuthorityErrorV1::AuthorityBelowFloor);
    }
    // ubs:ignore — authority sequences are public monotone counters.
    if resolved.authority.sequence == floor.authority.sequence {
        // ubs:ignore — authority references are public immutable identities.
        if resolved.authority != floor.authority {
            return Err(GenerationAuthorityErrorV1::EqualSequenceFork);
        }
        return Ok(resolved);
    }
    let successor = floor
        .authority
        .next_sequence()
        .map_err(|_| GenerationAuthorityErrorV1::SequenceExhausted)?;
    // ubs:ignore — authority sequences are public monotone counters.
    if resolved.authority.sequence != successor {
        return Err(GenerationAuthorityErrorV1::BrokenPredecessorLink);
    }
    // ubs:ignore — predecessor fingerprints are public immutable history identities.
    if resolved.authority.predecessor != Some(floor.authority.fingerprint()) {
        return Err(GenerationAuthorityErrorV1::BrokenPredecessorLink);
    }
    Ok(resolved)
}

/// Resolve authority slots while refusing to silently step around a valid
/// owner/attempt `LOCK` state.
///
/// A persisted attempt is evidence of an outcome the caller has not yet
/// reconciled, so it fails closed even if an older authority pair resolves.
/// The helper performs no filesystem I/O and does not mutate either frame.
///
/// # Errors
///
/// Returns a bounded typed error when frame kind, root binding, owner binding,
/// or a pending attempt prevents a safe selection.
pub fn resolve_authority_slots_with_locks_v1(
    first: Option<AuthoritySlotV1>,
    second: Option<AuthoritySlotV1>,
    owner: Option<GenerationLockFrameV1>,
    attempt: Option<GenerationLockFrameV1>,
) -> Result<Option<AuthoritySlotV1>, GenerationAuthorityErrorV1> {
    let resolved = resolve_authority_slots_v1(first, second)?;
    if let Some(owner) = owner {
        owner.validate()?;
        // ubs:ignore — lock kinds are public fixed-format state tags.
        if owner.kind != GenerationLockFrameKindV1::Owner {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "generation_lock.owner.kind",
            });
        }
    }
    if let Some(attempt) = attempt {
        attempt.validate()?;
        // ubs:ignore — lock kinds are public fixed-format state tags.
        if attempt.kind != GenerationLockFrameKindV1::Attempt {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "generation_lock.attempt.kind",
            });
        }
    }
    if let (Some(owner), Some(attempt)) = (owner, attempt) {
        // ubs:ignore — root IDs are public structural identities.
        if owner.root_id != attempt.root_id {
            return Err(GenerationAuthorityErrorV1::LockRootMismatch);
        }
    }
    if let Some(slot) = resolved {
        // ubs:ignore — root IDs are public structural identities.
        if owner.is_some_and(|frame| frame.root_id != slot.root_id) {
            return Err(GenerationAuthorityErrorV1::LockRootMismatch);
        }
        // ubs:ignore — root IDs are public structural identities.
        if attempt.is_some_and(|frame| frame.root_id != slot.root_id) {
            return Err(GenerationAuthorityErrorV1::LockRootMismatch);
        }
    }
    if attempt.is_some() {
        return Err(GenerationAuthorityErrorV1::UnresolvedAttempt);
    }
    if let Some(owner) = owner {
        let resolved = resolved.ok_or(GenerationAuthorityErrorV1::LockAuthorityMismatch)?;
        // ubs:ignore — authority fingerprints are public immutable-integrity identities.
        if owner.authority_fingerprint != resolved.authority.fingerprint() {
            return Err(GenerationAuthorityErrorV1::LockAuthorityMismatch);
        }
    }
    Ok(resolved)
}

/// Resolve authority slots under one explicit generation-root security profile.
///
/// The lock-aware resolver always runs first, so an unresolved publication
/// attempt cannot be bypassed by selecting a weaker profile. Required external
/// roots additionally require an injected retained floor; local and inspection
/// profiles intentionally remain unanchored and are distinguishable to callers.
///
/// # Errors
///
/// Returns a typed error when lock reconciliation fails, an external floor is
/// absent or not satisfied, or the underlying slots are malformed.
pub fn resolve_authority_slots_with_profile_v1(
    first: Option<AuthoritySlotV1>,
    second: Option<AuthoritySlotV1>,
    owner: Option<GenerationLockFrameV1>,
    attempt: Option<GenerationLockFrameV1>,
    profile: GenerationRootSecurityProfileV1,
    external_floor: Option<AuthorityFloorV1>,
) -> Result<Option<AuthoritySlotV1>, GenerationAuthorityErrorV1> {
    let resolved = resolve_authority_slots_with_locks_v1(first, second, owner, attempt)?;
    match profile {
        GenerationRootSecurityProfileV1::RequiredExternal => {
            let floor = external_floor.ok_or(GenerationAuthorityErrorV1::ExternalFloorRequired)?;
            let resolved = resolved.ok_or(GenerationAuthorityErrorV1::AuthorityBelowFloor)?;
            resolve_selected_authority_at_floor_v1(resolved, floor).map(Some)
        }
        GenerationRootSecurityProfileV1::CooperativeLocal
        | GenerationRootSecurityProfileV1::ReadOnlyUnanchored => Ok(resolved),
    }
}

fn slot_matches_authority_sequence(slot: AuthoritySlotV1) -> bool {
    if slot.authority.sequence == 1 {
        return true;
    }
    let expected_slot = (slot.authority.sequence & 1) as u8;
    slot.slot_index == expected_slot
}
