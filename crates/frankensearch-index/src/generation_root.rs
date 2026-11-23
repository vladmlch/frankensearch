//! Descriptor-owned admission for composite search-index generation roots.
//!
//! This module is deliberately narrower than the generation publisher. It
//! qualifies an existing private root, admits existing immutable regular
//! files into retained descriptors plus exact owned bytes, exposes fail-closed
//! durability barriers, and provides non-blocking kernel `flock` guards.
//! Publication, renames, unlinks, truncation, garbage collection, recovery,
//! and consumer activation do not belong here.
//!
//! # Supported targets and process binding
//!
//! Linux admits path-opened roots and final files only on locally qualified,
//! writable ext4 or Btrfs. Apple Silicon admits path-opened local writable APFS
//! roots, but every final artifact/control/lock file must arrive as an owned
//! already-open descriptor from the trusted provider documented by the
//! preopened APIs. Every other target returns
//! [`GenerationRootErrorKind::UnsupportedPlatform`] before route parsing or
//! filesystem I/O. Root, file, control, and lock capabilities are bound to the
//! process that created them; using any operation that revalidates or acts on
//! those capabilities after `fork` returns
//! [`GenerationRootErrorKind::ForkedProcess`].
//!
//! # Synchronous execution and resource boundary
//!
//! Every operation in this module is synchronous. Each exact-name component
//! proof accepts up to [`GENERATION_ROOT_MAX_DIRECTORY_ENTRIES`] names and
//! [`GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES`] aggregate name bytes, and may
//! inspect one overflow entry/name to reject it; one route can require up to
//! [`GENERATION_ROOT_MAX_COMPONENTS`] such proofs.
//! A route walk can invoke more than one exact-name proof per component, and
//! one public operation can walk the same route several times. The component
//! ceiling bounds depth, not the aggregate number of directory scans.
//! Linux mount qualification stores at most
//! [`GENERATION_ROOT_MAX_MOUNTINFO_BYTES`] from the calling thread's
//! task-scoped mount table and may read one overflow sentinel byte.
//! Immutable artifacts can read/hash up to [`GENERATION_ROOT_MAX_FILE_BYTES`]
//! bytes; repeatedly checked control anchors have the smaller
//! [`GENERATION_ROOT_MAX_CONTROL_FILE_BYTES`] ceiling. Explicit immutable
//! durability performs two full-file reads, while control admission, locking,
//! and locked durability can perform multiple exact hash passes around route,
//! lock, and kernel-barrier checkpoints. These functions are neither async nor
//! cancellation-aware. Async consumers must run them in a bounded blocking
//! pool, propagate cancellation only at operation boundaries, and admit only
//! trusted local generation roots whose total work is independently budgeted.
//! The per-proof ceilings below bound individual work and memory; they are not
//! aggregate admission budgets or latency promises.
//!
//! # Linux retained-capability reopen
//!
//! Linux has one intentional, tightly bounded magic-link exception. After the
//! original route has produced and fully qualified a retained `O_PATH`
//! regular-file capability, the module opens a separately verified procfs root,
//! proves that its `self` link names the calling process and its `thread-self`
//! link names the calling thread, and preflights the exact final
//! `/proc/thread-self/fd/<held-O_PATH-fd>` magic link as a symlink on the same
//! procfs mount before and after deriving the data descriptor. Resolving
//! `thread-self` and the numeric fd entry in the data-open syscall binds the
//! reopen to the accessing thread's descriptor table even after
//! `unshare(CLONE_FILES)`; process-identity checks still reject child-side use
//! after `fork()`.
//! This bridge never re-resolves the original generation route. Descriptor-
//! bound directory traversal performs each final-file route revalidation with
//! an `O_PATH`-only final probe, and the retained probe, derived data
//! descriptor, final procfs probe, and final route probe must retain their
//! respective identities. As with every `/proc`-based capability protocol, the
//! calling mount namespace must be trusted against concurrent `CAP_SYS_ADMIN`
//! remounts; a process able to rewrite that namespace is outside this
//! substrate's authority boundary.

#![forbid(unsafe_code)]

mod handles;
mod platform;
mod root;
mod types;
mod witness;

pub mod authority_publisher;
pub mod generation_reader;

#[cfg(test)]
mod tests;

pub use handles::QualifiedGenerationFile;
pub(crate) use root::GenerationRootExclusiveAnchorGuard;
#[cfg(target_os = "macos")]
pub use root::PreopenedGenerationRootAnchors;
pub use root::{GenerationRootReadGuard, QualifiedGenerationRoot};
pub use types::{
    GENERATION_ROOT_AUTHORITY_FILE_BYTES, GENERATION_ROOT_AUTHORITY_FILE_NAME,
    GENERATION_ROOT_CONTROL_FILE_MODE, GENERATION_ROOT_DIRECTORY_MODE,
    GENERATION_ROOT_IMMUTABLE_FILE_MODE, GENERATION_ROOT_LOCK_FILE_NAME,
    GENERATION_ROOT_MAX_COMPONENT_BYTES, GENERATION_ROOT_MAX_COMPONENTS,
    GENERATION_ROOT_MAX_CONTROL_FILE_BYTES, GENERATION_ROOT_MAX_DIRECTORY_ENTRIES,
    GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES, GENERATION_ROOT_MAX_FILE_BYTES,
    GENERATION_ROOT_MAX_MOUNTINFO_BYTES, GENERATION_ROOT_MAX_ROUTE_BYTES, GenerationRootError,
    GenerationRootErrorKind, GenerationRootResult, GenerationRootStage,
};
pub use witness::{
    ConfinedGenerationPath, GenerationFileExpectation, GenerationFileRole,
    GenerationRootAnchorLayout, GenerationRootAnchorWitness, GenerationRootEntry,
    GenerationRootEntryKind, GenerationRootInventory, GenerationRootObjectWitness,
    QualifiedFilesystem,
};

use handles::GenerationRootLockMode;
#[cfg(test)]
use root::QualifiedGenerationDirectory;

impl QualifiedGenerationRoot {
    // This deliberately lands one bead before the writer-lease consumer.
    #[allow(dead_code)]
    #[cfg_attr(target_os = "macos", allow(clippy::unused_self))]
    pub(crate) fn try_exclusive_anchor_guard(
        &self,
    ) -> GenerationRootResult<GenerationRootExclusiveAnchorGuard<'_>> {
        self.try_exclusive_anchor_guard_impl()
    }

    #[cfg(target_os = "macos")]
    #[allow(dead_code)]
    pub(crate) fn try_exclusive_anchor_guard_preopened(
        &self,
        fresh_lock_descriptor: std::os::fd::OwnedFd,
    ) -> GenerationRootResult<GenerationRootExclusiveAnchorGuard<'_>> {
        self.try_exclusive_anchor_guard_preopened_impl(fresh_lock_descriptor)
    }
}
