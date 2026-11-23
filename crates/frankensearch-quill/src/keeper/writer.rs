use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use asupersync::Cx;
use asupersync::runtime::spawn_blocking;
use asupersync::sync::LockError;

use crate::schema::SchemaDescriptor;
use crate::segment::{EncodedSegment, PendingSegmentFile};

use super::admission::{WriterAdmissionInner, acquire_writer_admission};
use super::compaction::{CompactionArtifact, ConcatMergeArtifact, build_compaction};
use super::concat_merge::{
    build_concat_merge, open_published_snapshot_blocking, open_snapshot_blocking,
    reconcile_encoded_segment, reconcile_published_segment, writer_mutation_guard,
};
#[cfg(feature = "bench-internals")]
use super::constants::BenchmarkQuillDirectorySyncState;
use super::errors::{
    CompactionError, CompactionPolicy, CompactionReport, ConcatMergeError, KeeperError,
    PublishIntent,
};
use super::gc::{
    GarbageCollectionOptions, GarbageCollectionReport, collect_writer_garbage_under_lock,
};
#[cfg(unix)]
use super::gc::{collect_writer_garbage_with_snapshot, witness_orphaned_segments_with_snapshot};
use super::manifest::Manifest;
use super::manifest_load::validate_proposed_manifest_segments;
use super::manifest_validation::{
    ErrorClass, manifest_encoded_len, validate_manifest_shape, validate_manifest_successor,
};
use super::publish::{
    GenerationClaimGuard, ManifestPublisher, normalize_publish_directory, publish_pending_segment,
};
use super::recovery::recover_writer_directory;
#[cfg(not(windows))]
use super::slot_io::sync_directory;
use super::snapshot::{KeeperSnapshot, manifest_matches_proposal};

#[derive(Clone)]
pub(in crate::keeper) enum WriterProtection {
    Disabled,
}

/// Retained evidence that one MANIFEST publication may have become durable
/// without its owning future surviving to install the successor snapshot.
///
/// A durable publication finishes inside a blocking closure that outlives a
/// dropped future, so `rename`+`fsync` can install generation N+1 while the
/// caller's future never resumes to assign the refreshed snapshot. The marker
/// is written synchronously in the same poll that reaches the publication, so
/// a drop, panic, or cancellation anywhere below it cannot erase the fact that
/// disk may now be ahead of the retained view.
#[derive(Debug, Clone, Copy)]
pub(in crate::keeper) struct PendingPublication {
    /// Generation the retained snapshot held when the attempt began.
    from_generation: u64,
    /// Generation the attempt proposed to install.
    proposed_generation: u64,
}

/// The publication phase carried alongside the last fully published
/// generation in [`PublicationReadState`].
///
/// The phase is deliberately part of the same atomic word as the generation:
/// an authority reader must never combine a fresh phase with a stale
/// generation (or vice versa).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PublicationAuthorityPhase {
    /// The local published `Arc` and durable MANIFEST name this generation.
    Stable = 0,
    /// A blocking publisher is preparing a successor; no slot has moved yet.
    Preparing = 1,
    /// The durable MANIFEST may be ahead of the local published `Arc`.
    DurableAhead = 2,
    /// A blocking publication unwound around a slot transition; reopen first.
    Indeterminate = 3,
}

impl PublicationAuthorityPhase {
    pub(in crate::keeper) const fn from_bits(bits: u64) -> Self {
        match bits {
            0 => Self::Stable,
            1 => Self::Preparing,
            2 => Self::DurableAhead,
            3 => Self::Indeterminate,
            _ => unreachable!(),
        }
    }
}

/// One atomically-readable authority generation and phase.
///
/// `generation` is the generation of the currently published process-local
/// snapshot. In `DurableAhead` the next generation is externally visible; in
/// `Indeterminate` its extent must be established by reconciliation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublicationAuthorityState {
    packed: u64,
}

impl PublicationAuthorityState {
    /// Generation currently represented by this word.
    #[must_use]
    pub const fn generation(self) -> u64 {
        self.packed >> 2
    }

    /// Publication phase carried with [`Self::generation`].
    #[must_use]
    pub const fn phase(self) -> PublicationAuthorityPhase {
        PublicationAuthorityPhase::from_bits(self.packed & 0b11)
    }

    /// Whether a process-local snapshot at this generation remains readable.
    #[must_use]
    pub const fn is_readable(self) -> bool {
        matches!(
            self.phase(),
            PublicationAuthorityPhase::Stable | PublicationAuthorityPhase::Preparing
        )
    }
}

/// Shared publication authority for one `KeeperWriter` and its lock-free
/// `QuillReader` facade.
///
/// This owns exactly one atomic word: `(generation << 2) | phase`. The word is
/// installed by the blocking MANIFEST publisher before observer callbacks can
/// expose a rename, and readers load it with `Acquire` around one pinned
/// process-local snapshot `Arc`.
#[derive(Debug, Clone)]
pub struct PublicationReadState {
    authority: Arc<AtomicU64>,
}

impl PublicationReadState {
    const PHASE_BITS: u32 = 2;
    pub const MAX_GENERATION: u64 = u64::MAX >> Self::PHASE_BITS;

    /// Construct the stable initial authority word for an opened snapshot.
    pub fn new(generation: u64) -> Result<Self, KeeperError> {
        Ok(Self {
            authority: Arc::new(AtomicU64::new(Self::pack(
                generation,
                PublicationAuthorityPhase::Stable,
            )?)),
        })
    }

    pub(in crate::keeper) fn pack(
        generation: u64,
        phase: PublicationAuthorityPhase,
    ) -> Result<u64, KeeperError> {
        if generation > Self::MAX_GENERATION {
            return Err(KeeperError::InvalidTransition {
                detail: format!(
                    "generation {generation} cannot be represented in the packed publication authority word"
                ),
            });
        }
        Ok((generation << Self::PHASE_BITS) | u64::from(phase as u8))
    }

    /// Reject a MANIFEST generation that cannot fit in the reader-visible
    /// packed authority word before a process-local or owned-buffer mutation.
    pub fn validate_generation(generation: u64) -> Result<(), KeeperError> {
        let _ = Self::pack(generation, PublicationAuthorityPhase::Stable)?;
        Ok(())
    }

    /// Load one indivisible generation/phase observation for a lock-free read.
    #[must_use]
    pub fn load(&self) -> PublicationAuthorityState {
        PublicationAuthorityState {
            packed: self.authority.load(AtomicOrdering::Acquire),
        }
    }

    /// Mark the exact process-local successor readable only after its `Arc` is
    /// installed. This is also the only normal path out of `DurableAhead`.
    pub fn stabilize(&self, generation: u64) -> Result<(), KeeperError> {
        let stable = Self::pack(generation, PublicationAuthorityPhase::Stable)?;
        loop {
            let observed = self.load();
            if generation < observed.generation() {
                return Ok(());
            }
            if generation == observed.generation() {
                if observed.phase() == PublicationAuthorityPhase::Stable {
                    return Ok(());
                }
                return Err(KeeperError::InvalidTransition {
                    detail: format!(
                        "generic stabilization cannot clear same-generation {:?} publication authority at generation {generation}",
                        observed.phase()
                    ),
                });
            }
            if observed.phase() == PublicationAuthorityPhase::Preparing {
                return Err(KeeperError::InvalidTransition {
                    detail: format!(
                        "cannot stabilize generation {generation} while generation {} has an active publication",
                        observed.generation()
                    ),
                });
            }
            if self
                .authority
                .compare_exchange(
                    observed.packed,
                    stable,
                    AtomicOrdering::AcqRel,
                    AtomicOrdering::Acquire,
                )
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    /// Clear same-generation indeterminacy only after Keeper reopened the
    /// durable MANIFEST and proved that the retained process-local generation
    /// is still authoritative.
    pub fn stabilize_reconciled_same_generation(&self, generation: u64) -> Result<(), KeeperError> {
        let stable = Self::pack(generation, PublicationAuthorityPhase::Stable)?;
        let indeterminate = Self::pack(generation, PublicationAuthorityPhase::Indeterminate)?;
        match self.authority.compare_exchange(
            indeterminate,
            stable,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
        ) {
            Ok(_) => Ok(()),
            Err(observed) if observed == stable => Ok(()),
            Err(observed) => {
                let observed = PublicationAuthorityState { packed: observed };
                Err(KeeperError::InvalidTransition {
                    detail: format!(
                        "same-generation reconciliation for {generation} cannot clear {:?} authority at generation {}",
                        observed.phase(),
                        observed.generation()
                    ),
                })
            }
        }
    }
}

/// Blocking-closure ownership of one MANIFEST publication transition.
///
/// Its `Drop` implementation is intentionally the only unwind classifier: a
/// still-preparing closure becomes indeterminate. Once a slot has moved the
/// word has already left `Preparing`, so the compare-exchange cannot downgrade
/// a post-rename state back to a readable one.
pub(in crate::keeper) struct PublicationAuthorityTransition {
    pub(in crate::keeper) state: PublicationReadState,
    stable: u64,
    preparing: u64,
    indeterminate: u64,
    durable_ahead: u64,
}

impl PublicationAuthorityTransition {
    pub(in crate::keeper) fn begin(
        state: PublicationReadState,
        proposed_generation: u64,
    ) -> Result<Self, KeeperError> {
        let before = state.load();
        if before.phase() != PublicationAuthorityPhase::Stable {
            return Err(KeeperError::InvalidTransition {
                detail: format!(
                    "cannot begin MANIFEST publication from {:?} generation {}",
                    before.phase(),
                    before.generation()
                ),
            });
        }
        let expected_proposed =
            before
                .generation()
                .checked_add(1)
                .ok_or_else(|| KeeperError::GenerationExhausted {
                    current: before.generation(),
                })?;
        if proposed_generation != expected_proposed {
            return Err(KeeperError::InvalidTransition {
                detail: format!(
                    "packed publication authority at generation {} cannot publish generation {proposed_generation}",
                    before.generation()
                ),
            });
        }
        let _ = PublicationReadState::pack(proposed_generation, PublicationAuthorityPhase::Stable)?;
        let stable =
            PublicationReadState::pack(before.generation(), PublicationAuthorityPhase::Stable)?;
        let preparing =
            PublicationReadState::pack(before.generation(), PublicationAuthorityPhase::Preparing)?;
        let indeterminate = PublicationReadState::pack(
            before.generation(),
            PublicationAuthorityPhase::Indeterminate,
        )?;
        let durable_ahead = PublicationReadState::pack(
            before.generation(),
            PublicationAuthorityPhase::DurableAhead,
        )?;
        state
            .authority
            .compare_exchange(
                stable,
                preparing,
                AtomicOrdering::AcqRel,
                AtomicOrdering::Acquire,
            )
            .map_err(|observed| KeeperError::InvalidTransition {
                detail: format!(
                    "publication authority changed while beginning generation {proposed_generation}: observed packed word {observed:#018x}"
                ),
            })?;
        Ok(Self {
            state,
            stable,
            preparing,
            indeterminate,
            durable_ahead,
        })
    }

    /// A known failure before either MANIFEST slot has moved is retry-safe.
    pub(in crate::keeper) fn restore_known_pre_rename_failure(&self) {
        let _ = self.state.authority.compare_exchange(
            self.preparing,
            self.stable,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
        );
    }

    /// A successful claim is the conservative boundary after which the
    /// blocking closure may reach either slot rename. Readers fail closed
    /// before the first slot move until reconciliation settles the attempt.
    pub(in crate::keeper) fn mark_indeterminate(&self) {
        let _ = self.state.authority.compare_exchange(
            self.preparing,
            self.indeterminate,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
        );
    }

    /// `MANIFEST` now names the successor. Store before any checkpoint
    /// observer runs so a parked callback is already inside the fail-closed
    /// window.
    pub(in crate::keeper) fn mark_durable_ahead(&self) {
        self.state
            .authority
            .store(self.durable_ahead, AtomicOrdering::Release);
    }
}

impl Drop for PublicationAuthorityTransition {
    fn drop(&mut self) {
        let _ = self.state.authority.compare_exchange(
            self.preparing,
            self.indeterminate,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
        );
    }
}

/// Sole cross-process mutation capability for one Quill index directory.
///
/// This type is intentionally not `Clone`. Its internal admission token may be
/// retained by an in-flight blocking publication so cancellation cannot release
/// the OS lock while filesystem mutation is still running.
///
/// For the same reason a successful publication can outlive the future that
/// awaited it. This writer therefore retains a pending-publication marker
/// across such a drop, and every later authority read reconciles the durable
/// MANIFEST before [`Self::snapshot`] is used as publication authority.
///
/// Every process that mutates the index directory must honor Quill's `LOCK`.
/// Crash consistency does not cover an out-of-band process replacing admitted
/// directory entries while a writer operation is in flight.
pub struct KeeperWriter {
    pub(in crate::keeper) admission: Arc<WriterAdmissionInner>,
    pub(in crate::keeper) snapshot: KeeperSnapshot,
    pub(in crate::keeper) garbage_options: GarbageCollectionOptions,
    pub(in crate::keeper) protection: WriterProtection,
    /// Benchmark-only observer for the exact post-MANIFEST-sync checkpoint.
    #[cfg(feature = "bench-internals")]
    pub(in crate::keeper) benchmark_qg4_directory_sync_state:
        Option<Arc<BenchmarkQuillDirectorySyncState>>,
    /// Bound only by the process-local Quill facade. Standalone Keeper users
    /// have no `ArcSwap` reader to coordinate, so they do not install this
    /// state word.
    pub(in crate::keeper) publication_read_state: Option<PublicationReadState>,
    /// Set synchronously before a publication is awaited; cleared only once the
    /// retained snapshot has been proven against the durable generation that
    /// attempt may have installed.
    pub(in crate::keeper) pending_publication: Option<PendingPublication>,
}

impl KeeperWriter {
    /// Open an existing index for mutation with the default GC grace period.
    ///
    /// # Errors
    ///
    /// Returns a typed admission, recovery, schema, segment, or I/O failure.
    pub async fn open(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
    ) -> Result<Self, KeeperError> {
        Self::open_inner(
            cx,
            directory.into(),
            schema,
            false,
            GarbageCollectionOptions::default(),
            WriterProtection::Disabled,
        )
        .await
    }

    /// Create a genesis index or open an existing one for mutation.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::open`] plus directory creation or
    /// genesis-publication failures.
    pub async fn create(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
    ) -> Result<Self, KeeperError> {
        Self::open_inner(
            cx,
            directory.into(),
            schema,
            true,
            GarbageCollectionOptions::default(),
            WriterProtection::Disabled,
        )
        .await
    }

    async fn open_inner(
        cx: &Cx,
        directory: PathBuf,
        schema: SchemaDescriptor,
        create: bool,
        garbage_options: GarbageCollectionOptions,
        protection: WriterProtection,
    ) -> Result<Self, KeeperError> {
        if cx.is_cancel_requested() {
            return Err(KeeperError::WriterAdmissionCancelled);
        }
        let mut directory = normalize_publish_directory(directory);
        if create {
            let create_path = directory.clone();
            spawn_blocking(move || {
                std::fs::create_dir_all(&create_path).map_err(|source| KeeperError::Io {
                    operation: "create index directory",
                    path: create_path,
                    source,
                })
            })
            .await?;
            directory = normalize_publish_directory(directory);
        }
        let admission_directory = directory.clone();
        let admission =
            spawn_blocking(move || acquire_writer_admission(&admission_directory)).await?;
        if cx.is_cancel_requested() {
            return Err(KeeperError::WriterAdmissionCancelled);
        }

        let recovery_admission = Arc::clone(&admission);
        let recovery_protection = protection.clone();
        spawn_blocking(move || {
            recover_writer_directory(&recovery_admission, schema, &recovery_protection)
        })
        .await?;

        let snapshot = match open_snapshot_blocking(directory.clone(), schema).await {
            Ok(snapshot) => snapshot,
            Err(KeeperError::IndexNotFound { .. }) if create => {
                let schema_id = schema
                    .schema_id()
                    .map_err(|source| KeeperError::InvalidSchema { source })?;
                let genesis = Manifest::empty(1, schema_id, 0);
                let claim_admission = Arc::clone(&admission);
                let publisher = ManifestPublisher::new(&directory);
                match &protection {
                    WriterProtection::Disabled => {
                        publisher
                            .publish_with_generation_claim(cx, &genesis, move |_, generation| {
                                GenerationClaimGuard::acquire(claim_admission, generation)
                            })
                            .await?;
                    }
                }
                open_snapshot_blocking(directory.clone(), schema).await?
            }
            Err(error) => return Err(error),
        };

        // The snapshot above was opened after recovery, under this admission,
        // before this writer published anything, so it is the writer's view:
        // the witness and the sweep below are judged against it rather than
        // reopening (and re-verifying every segment) twice more, neither
        // removes anything it references, and both refuse to run unless the
        // selected MANIFEST still equals it.
        #[cfg(unix)]
        {
            let gc_admission = Arc::clone(&admission);
            let gc_directory = directory.clone();
            let gc_snapshot = snapshot.clone();
            spawn_blocking(move || {
                gc_admission.ensure_directory_identity()?;
                // Witness before sweeping, and only here: this writer has
                // published nothing of its own yet, so nothing on disk is a
                // staged-but-unpublished segment that a sighting would misread
                // as an orphan. The sweep itself never mints provenance.
                witness_orphaned_segments_with_snapshot(&gc_directory, &gc_snapshot)?;
                gc_admission.ensure_directory_identity()?;
                collect_writer_garbage_with_snapshot(&gc_directory, &gc_snapshot, garbage_options)
            })
            .await?;
        }
        #[cfg(windows)]
        tracing::debug!(
            path = %directory.display(),
            "Windows writer admission is active; descriptor-relative garbage collection remains unavailable"
        );
        Ok(Self {
            admission,
            snapshot,
            garbage_options,
            protection,
            #[cfg(feature = "bench-internals")]
            benchmark_qg4_directory_sync_state: None,
            publication_read_state: None,
            pending_publication: None,
        })
    }

    /// Return this writer's current authoritative immutable reader view.
    ///
    /// A dropped [`Self::publish`] future can leave a durable successor on disk
    /// before this writer refreshes its retained view. In that state this method
    /// fails closed; call [`Self::reconcile_publication`] and retry instead of
    /// treating the retained generation as authority.
    ///
    /// # Errors
    ///
    /// Returns [`KeeperError::PublicationReconciliationRequired`] when a prior
    /// publication must be reconciled before this view can be used as authority.
    pub fn snapshot(&self) -> Result<&KeeperSnapshot, KeeperError> {
        let Some(pending) = self.pending_publication else {
            return Ok(&self.snapshot);
        };
        Err(KeeperError::PublicationReconciliationRequired {
            retained_generation: self.snapshot.loaded_manifest().manifest.generation,
            proposed_generation: pending.proposed_generation,
        })
    }

    /// Retained writer view for crate-internal bookkeeping only.
    ///
    /// This is deliberately not a public authority reader. In particular, it
    /// can lag disk while [`Self::publication_awaits_reconciliation`] is true.
    #[must_use]
    pub(crate) const fn retained_snapshot_for_bookkeeping(&self) -> &KeeperSnapshot {
        &self.snapshot
    }

    /// Bind the process-local reader's packed publication authority word.
    ///
    /// This is crate-private because only `QuillWriterState::from_backend`
    /// constructs the matching immutable reader `Arc`.
    pub(crate) fn bind_publication_read_state(&mut self, state: PublicationReadState) {
        self.publication_read_state = Some(state);
    }

    #[cfg(feature = "bench-internals")]
    pub(crate) fn bind_benchmark_qg4_directory_sync_state(
        &mut self,
        state: Arc<BenchmarkQuillDirectorySyncState>,
    ) {
        self.benchmark_qg4_directory_sync_state = Some(state);
    }

    fn publication_read_state_is_stable_at(&self, generation: u64) -> bool {
        self.publication_read_state.as_ref().is_some_and(|state| {
            let observed = state.load();
            observed.phase() == PublicationAuthorityPhase::Stable
                && observed.generation() == generation
        })
    }

    /// Whether a publication may have become durable without this writer
    /// installing its successor snapshot.
    ///
    /// True only between an abandoned [`Self::publish`] future and the next
    /// authority read that proves the retained snapshot against disk.
    #[must_use]
    pub const fn publication_awaits_reconciliation(&self) -> bool {
        self.pending_publication.is_some()
    }

    /// Prove the retained snapshot against the durable MANIFEST.
    ///
    /// This is a no-op when no publication was abandoned. Otherwise durable
    /// authority is re-read under the publication guard — so an abandoned
    /// attempt whose blocking choreography is still running cannot be observed
    /// mid-rename — and adopted. Authority never moves backwards.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, admission-identity, snapshot-open, or
    /// rollback failures. The marker survives every one of them, so a failed
    /// reconciliation cannot be mistaken for an installed successor.
    pub async fn reconcile_publication(&mut self, cx: &Cx) -> Result<&KeeperSnapshot, KeeperError> {
        self.reconcile_pending_publication(cx).await?;
        Ok(self.retained_snapshot_for_bookkeeping())
    }

    /// Adopt the durable generation an abandoned publication may have
    /// installed, and report whether a marker was reconciled.
    ///
    /// The publication guard is taken first: a dropped future leaves its
    /// blocking closure holding that guard until the choreography ends, so
    /// acquiring it is what makes the following read an authority read rather
    /// than a sighting of a half-renamed directory.
    async fn reconcile_pending_publication(&mut self, cx: &Cx) -> Result<bool, KeeperError> {
        let Some(pending) = self.pending_publication else {
            return Ok(false);
        };
        let guard = writer_mutation_guard(cx, &self.admission.directory).await?;
        let admission = Arc::clone(&self.admission);
        let protection = self.protection.clone();
        let schema = self.snapshot.schema();
        let installed = spawn_blocking(move || {
            let _guard = guard;
            admission.ensure_directory_identity()?;
            let _ = protection;
            let installed = KeeperSnapshot::open(&admission.directory, schema)?;
            admission.ensure_directory_identity()?;
            Ok::<KeeperSnapshot, KeeperError>(installed)
        })
        .await?;
        let retained_generation = self.snapshot.loaded_manifest().manifest.generation;
        let installed_generation = installed.loaded_manifest().manifest.generation;
        if installed_generation < retained_generation {
            return Err(KeeperError::InvalidTransition {
                detail: format!(
                    "durable generation {installed_generation} is behind the retained writer \
                     snapshot at {retained_generation} after an abandoned publication of \
                     generation {} from generation {}",
                    pending.proposed_generation, pending.from_generation
                ),
            });
        }
        self.snapshot = installed;
        self.pending_publication = None;
        Ok(true)
    }

    /// Publish exactly the next MANIFEST generation through an `O_EXCL` claim.
    ///
    /// The proposal's tombstones and segment removals are taken as the
    /// caller's explicit deletion intent ([`PublishIntent::ExplicitDeletion`]).
    /// A proposal derived from an additive or replacing mutation should go
    /// through [`Self::publish_with_intent`] with
    /// [`PublishIntent::PreserveLiveDocuments`] so a proposal that lost its
    /// replacement rows is refused instead of published (gh#45).
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, claim, transition, durability, or I/O
    /// failures. An ambiguous prior attempt is reconciled to the exact
    /// installed proposal; a differing installed generation remains an error.
    pub async fn publish(
        &mut self,
        cx: &Cx,
        manifest: &Manifest,
    ) -> Result<&KeeperSnapshot, KeeperError> {
        self.publish_with_intent(cx, manifest, PublishIntent::ExplicitDeletion)
            .await
    }

    /// [`Self::publish`] with an explicit [`PublishIntent`].
    ///
    /// # Errors
    ///
    /// As [`Self::publish`], plus [`KeeperError::LiveDocumentFloor`] when
    /// `intent` is [`PublishIntent::PreserveLiveDocuments`] and the proposal
    /// would serve fewer live documents than the retained generation. The
    /// refusal happens before the generation claim, the temp MANIFEST, or the
    /// pending-publication marker exist; the outgoing generation stays
    /// authoritative and the writer remains usable.
    pub async fn publish_with_intent(
        &mut self,
        cx: &Cx,
        manifest: &Manifest,
        intent: PublishIntent,
    ) -> Result<&KeeperSnapshot, KeeperError> {
        if cx.is_cancel_requested() {
            return Err(KeeperError::PublishLock {
                source: LockError::Cancelled,
            });
        }
        self.admission.ensure_directory_identity()?;
        validate_manifest_shape(manifest, ErrorClass::Invalid)
            .map_err(|source| KeeperError::InvalidManifest { source })?;
        manifest_encoded_len(manifest).map_err(|source| KeeperError::InvalidManifest { source })?;
        manifest
            .validate()
            .map_err(|source| KeeperError::InvalidManifest { source })?;
        // An earlier attempt may have become durable after its future was
        // dropped. Prove the retained snapshot against disk before this
        // proposal is validated against it, otherwise a stale successor check
        // would admit a proposal for a generation that already exists. When the
        // abandoned attempt is exactly what won, adopt it instead of
        // republishing the same generation.
        if self.reconcile_pending_publication(cx).await?
            && manifest_matches_proposal(&self.snapshot.loaded_manifest().manifest, manifest)
        {
            return Ok(self.retained_snapshot_for_bookkeeping());
        }
        validate_manifest_successor(
            &self.snapshot.loaded_manifest().manifest,
            manifest,
            intent.live_document_floor(),
        )?;
        let directory = self.admission.directory.clone();
        let preflight_directory = directory.clone();
        let preflight_manifest = manifest.clone();
        let preflight_protection = self.protection.clone();
        let schema = self.snapshot.schema();
        let verified = spawn_blocking(move || {
            validate_proposed_manifest_segments(
                &preflight_directory,
                &preflight_manifest,
                schema,
                &preflight_protection,
            )
        })
        .await?;
        if cx.is_cancel_requested() {
            return Err(KeeperError::PublishLock {
                source: LockError::Cancelled,
            });
        }
        let claim_admission = Arc::clone(&self.admission);
        let publisher = ManifestPublisher::with_publication_read_state(
            &directory,
            self.publication_read_state.clone(),
        );
        #[cfg(feature = "bench-internals")]
        let publisher = publisher.with_benchmark_qg4_directory_sync_state(
            self.benchmark_qg4_directory_sync_state.clone(),
        );
        // Written in the same poll that reaches the publication and never
        // behind an await: dropping, unwinding, or cancelling out of anything
        // below cannot erase it, so the durable generation this attempt may
        // install stays reconcilable through the next authority read.
        let from_generation = self.snapshot.loaded_manifest().manifest.generation;
        self.pending_publication = Some(PendingPublication {
            from_generation,
            proposed_generation: manifest.generation,
        });
        let publish_result = match &self.protection {
            WriterProtection::Disabled => {
                publisher
                    .publish_with_generation_claim(cx, manifest, move |_, generation| {
                        GenerationClaimGuard::acquire(claim_admission, generation)
                    })
                    .await
            }
        };
        if let Err(error) = publish_result {
            // A closure-owned pre-slot failure restored the packed word to the
            // exact predecessor. No MANIFEST entry moved, so preserving the
            // pessimistic marker would needlessly turn valid readers into a
            // false reconciliation requirement. Retry state remains owned by
            // the caller because this returns the original publication error.
            if self.publication_read_state_is_stable_at(from_generation) {
                self.pending_publication = None;
                return Err(error);
            }
            if self.reconcile_manifest_proposal(cx, manifest).await? {
                self.pending_publication = None;
                return Ok(self.retained_snapshot_for_bookkeeping());
            }
            // The proposal did not win, but recovery may still have advanced
            // durable authority past the retained snapshot. Keep the marker so
            // the next authority read reconciles rather than assuming this
            // failure left disk where the retained view says it is.
            return Err(error);
        }
        self.admission.ensure_directory_identity()?;
        self.snapshot =
            open_published_snapshot_blocking(directory, self.snapshot.schema(), verified).await?;
        self.pending_publication = None;
        Ok(self.retained_snapshot_for_bookkeeping())
    }

    async fn reconcile_manifest_proposal(
        &mut self,
        cx: &Cx,
        proposed: &Manifest,
    ) -> Result<bool, KeeperError> {
        let guard = writer_mutation_guard(cx, &self.admission.directory).await?;
        let admission = Arc::clone(&self.admission);
        let protection = self.protection.clone();
        let schema = self.snapshot.schema();
        let directory = admission.directory.clone();
        spawn_blocking(move || {
            let _guard = guard;
            admission.ensure_directory_identity()?;
            recover_writer_directory(&admission, schema, &protection)?;
            #[cfg(not(windows))]
            sync_directory(&admission.directory)?;
            admission.ensure_directory_identity()
        })
        .await?;
        let recovered = open_snapshot_blocking(directory, schema).await?;
        if !manifest_matches_proposal(&recovered.loaded_manifest().manifest, proposed) {
            return Ok(false);
        }
        self.snapshot = recovered;
        Ok(true)
    }

    /// Atomically install one synced immutable segment while holding `LOCK`.
    ///
    /// # Errors
    ///
    /// Returns a typed path, collision, sidecar, or fsync failure.
    pub async fn publish_segment(
        &mut self,
        cx: &Cx,
        pending: PendingSegmentFile,
    ) -> Result<PathBuf, KeeperError> {
        let guard = writer_mutation_guard(cx, &self.admission.directory).await?;
        let admission = Arc::clone(&self.admission);
        let protection = self.protection.clone();
        spawn_blocking(move || {
            let _guard = guard;
            admission.ensure_directory_identity()?;
            let parent = pending.path().parent().ok_or_else(|| KeeperError::Io {
                operation: "resolve pending segment directory",
                path: pending.path().to_path_buf(),
                source: io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "pending segment path has no parent directory",
                ),
            })?;
            let parent = std::fs::canonicalize(parent).map_err(|source| KeeperError::Io {
                operation: "canonicalize pending segment directory",
                path: parent.to_path_buf(),
                source,
            })?;
            if parent != admission.directory {
                return Err(KeeperError::Io {
                    operation: "verify pending segment directory",
                    path: pending.path().to_path_buf(),
                    source: io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "pending segment belongs to a different index directory",
                    ),
                });
            }
            if let Some(published) = reconcile_published_segment(&admission, &protection, &pending)?
            {
                admission.ensure_directory_identity()?;
                return Ok(published);
            }
            let published = match &protection {
                WriterProtection::Disabled => publish_pending_segment(pending),
            }?;
            admission.ensure_directory_identity()?;
            Ok(published)
        })
        .await
    }

    /// Reconcile or install retained canonical bytes under one mutation guard.
    ///
    /// Unlike a caller-created [`PendingSegmentFile`], this operation acquires
    /// the process-wide writer guard before inspecting either the canonical
    /// destination or retry temp. A dropped future may leave its blocking
    /// closure running, but a retry cannot race that closure or manufacture a
    /// redundant temp after the exact canonical file already won.
    ///
    /// # Errors
    ///
    /// Returns a typed lock, byte-conflict, segment-write, sidecar, or fsync
    /// failure. Existing differing artifacts are preserved.
    pub(crate) async fn publish_encoded_segment_retryable(
        &self,
        cx: &Cx,
        encoded: Arc<EncodedSegment>,
    ) -> Result<PathBuf, KeeperError> {
        let guard = writer_mutation_guard(cx, &self.admission.directory).await?;
        let admission = Arc::clone(&self.admission);
        let protection = self.protection.clone();
        spawn_blocking(move || {
            let _guard = guard;
            admission.ensure_directory_identity()?;
            if let Some(published) = reconcile_encoded_segment(&admission, &protection, &encoded)? {
                admission.ensure_directory_identity()?;
                return Ok(published);
            }
            let pending = encoded
                .write_temp_retryable(&admission.directory)
                .map_err(|source| KeeperError::SegmentInstall { source })?;
            let published = match &protection {
                WriterProtection::Disabled => publish_pending_segment(pending),
            }?;
            admission.ensure_directory_identity()?;
            Ok(published)
        })
        .await
    }

    /// Build, install, and atomically publish one Q1-preserving concat merge.
    ///
    /// Source ids are interpreted in caller order and must name exactly one
    /// uninterrupted slice of the current manifest. Construction runs on the
    /// asupersync blocking lane. The fully synced immutable output remains
    /// unreachable until the exact successor MANIFEST is published; a
    /// cancellation between those two steps can therefore leave only a safe
    /// garbage-collectable orphan.
    ///
    /// # Errors
    ///
    /// Returns typed merge validation/codec failures plus the ordinary segment
    /// installation and reconciled MANIFEST-publication failures.
    pub async fn concat_merge(
        &mut self,
        cx: &Cx,
        source_segment_ids: &[u64],
        output_segment_id: u64,
        created_unix_s: i64,
    ) -> Result<&KeeperSnapshot, KeeperError> {
        if cx.is_cancel_requested() {
            return Err(ConcatMergeError::Cancelled.into());
        }
        // Plan against proven authority: an abandoned publication may have
        // advanced disk past the retained snapshot these source ids name.
        self.reconcile_pending_publication(cx).await?;
        let mut source_ids = Vec::new();
        source_ids
            .try_reserve_exact(source_segment_ids.len())
            .map_err(|_| ConcatMergeError::Allocation {
                resource: "source ids",
                count: source_segment_ids.len(),
            })?;
        source_ids.extend_from_slice(source_segment_ids);
        let snapshot = self.snapshot.clone();
        let artifact = spawn_blocking(move || {
            build_concat_merge(&snapshot, &source_ids, output_segment_id, created_unix_s)
        })
        .await?;
        if cx.is_cancel_requested() {
            return Err(ConcatMergeError::Cancelled.into());
        }

        let ConcatMergeArtifact { encoded, manifest } = artifact;
        let guard = writer_mutation_guard(cx, &self.admission.directory).await?;
        let admission = Arc::clone(&self.admission);
        let pending = spawn_blocking(move || {
            let _guard = guard;
            admission.ensure_directory_identity()?;
            encoded
                .write_temp_retryable(&admission.directory)
                .map_err(|source| KeeperError::from(ConcatMergeError::Segment { source }))
        })
        .await?;
        self.publish_segment(cx, pending).await?;
        // A merge rewrites rows; it never removes a live one, so it is held to
        // the live-document floor like any other additive publication.
        self.publish_with_intent(cx, &manifest, PublishIntent::PreserveLiveDocuments)
            .await
    }

    /// Rewrite every segment above the selected tombstone-density threshold.
    ///
    /// Replacement files are completely built, validated, and synced before
    /// the successor MANIFEST can reference them. Cancellation or a crash in
    /// that window therefore leaves the old generation authoritative and only
    /// unreferenced, grace-period-GC-eligible output files behind.
    ///
    /// # Errors
    ///
    /// Returns typed policy/rewrite failures plus ordinary segment install and
    /// MANIFEST publication failures.
    pub async fn compact(
        &mut self,
        cx: &Cx,
        policy: CompactionPolicy,
        created_unix_s: i64,
    ) -> Result<CompactionReport, KeeperError> {
        if cx.is_cancel_requested() {
            return Err(CompactionError::Cancelled.into());
        }
        // Rewrite against proven authority: a compaction planned from a
        // snapshot that an abandoned publication already superseded would
        // propose a generation that exists.
        self.reconcile_pending_publication(cx).await?;
        let snapshot = self.snapshot.clone();
        let artifact =
            spawn_blocking(move || build_compaction(&snapshot, policy, created_unix_s)).await?;
        if !artifact.report.changed() {
            return Ok(artifact.report);
        }
        if cx.is_cancel_requested() {
            return Err(CompactionError::Cancelled.into());
        }

        let CompactionArtifact {
            encoded,
            manifest,
            report,
        } = artifact;
        for segment in encoded {
            self.publish_encoded_segment_retryable(cx, Arc::new(segment))
                .await?;
        }
        if cx.is_cancel_requested() {
            return Err(CompactionError::Cancelled.into());
        }
        // Compaction folds tombstones into the rewritten segments; every live
        // row survives, so the live-document floor holds here as well.
        self.publish_with_intent(cx, &manifest, PublishIntent::PreserveLiveDocuments)
            .await?;
        Ok(report)
    }

    /// Run one grace-window garbage sweep under the held writer admission.
    ///
    /// # Errors
    ///
    /// Returns before removal on any recovery, identity, metadata, or path
    /// validation failure.
    pub async fn collect_garbage(
        &mut self,
        cx: &Cx,
    ) -> Result<GarbageCollectionReport, KeeperError> {
        let options = self.garbage_options;
        self.collect_garbage_with(cx, options).await
    }

    /// Run one garbage sweep with an explicit grace window under the held
    /// writer admission.
    ///
    /// The window protects a reader that resolved an older `MANIFEST` and has
    /// not opened its segments yet. Shorten it only for a directory no reader
    /// can have resolved, such as a private build directory that has never
    /// been published. Files either `MANIFEST` slot references are kept at any
    /// window.
    ///
    /// # Errors
    ///
    /// Returns before removal on any recovery, identity, metadata, or path
    /// validation failure.
    pub async fn collect_garbage_with(
        &mut self,
        cx: &Cx,
        options: GarbageCollectionOptions,
    ) -> Result<GarbageCollectionReport, KeeperError> {
        let guard = writer_mutation_guard(cx, &self.admission.directory).await?;
        let admission = Arc::clone(&self.admission);
        let schema = self.snapshot.schema();
        // With no publication awaiting reconciliation, the retained snapshot
        // is this admission's authority; the sweep refuses if it is not.
        #[cfg(unix)]
        let retained = self
            .pending_publication
            .is_none()
            .then(|| self.snapshot.clone());
        spawn_blocking(move || {
            let _guard = guard;
            admission.ensure_directory_identity()?;
            #[cfg(unix)]
            if let Some(snapshot) = retained {
                return collect_writer_garbage_with_snapshot(
                    &admission.directory,
                    &snapshot,
                    options,
                );
            }
            collect_writer_garbage_under_lock(&admission.directory, schema, options)
        })
        .await
    }
}
