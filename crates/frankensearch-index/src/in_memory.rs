//! In-memory vector index for zero-latency search.
//!
//! Unlike the file-backed [`crate::VectorIndex`] (memory-mapped FSVI), this
//! module stores all vectors in heap-allocated memory, guaranteeing no page
//! faults on access. Vectors are stored as f16 for 50% memory savings.
//!
//! # Usage
//!
//! ```rust,ignore
//! use frankensearch_index::in_memory::InMemoryVectorIndex;
//!
//! // From pre-computed f32 vectors
//! let index = InMemoryVectorIndex::from_vectors(
//!     doc_ids,
//!     vectors,
//!     256,
//! ).unwrap();
//!
//! let hits = index.search_top_k(&query, 10, None).unwrap();
//! ```

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::OnceLock;
#[cfg(target_os = "linux")]
use std::sync::atomic::AtomicU64;

use ahash::AHashMap;
use frankensearch_core::SearchResult;
use frankensearch_core::filter::BuildIdentityHasherU64;
use half::f16;

mod constructors;
mod residual_io;
mod residual_sidecar;
mod search_ops;
#[cfg(test)]
mod tests;
mod two_tier_index;

pub use self::two_tier_index::InMemoryTwoTierIndex;

use self::residual_io::residual_sidecar_error;
use self::residual_sidecar::{ExactResidualSidecar, ResidualSourceBinding};

#[cfg(any(test, target_os = "linux"))]
const EXACT_RESIDUAL_SIDECAR_MAGIC: [u8; 8] = *b"FSRSIDX1";
#[cfg(any(test, target_os = "linux"))]
const EXACT_RESIDUAL_SIDECAR_VERSION: u32 = 2;
const EXACT_RESIDUAL_BLOCK: usize = 32;
const EXACT_RESIDUAL_LANES: usize = 8;
const EXACT_RESIDUAL_SIDECAR_HEADER_BYTES: usize = 8 + 4 + 4 * 32 + 8 + 8 + 4 + 4;
const EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES: usize = 32;
/// Sidecars are optional, so opening one must never permit unbounded allocation
/// or turn an untrusted artifact into a process-wide memory-pressure event.
const EXACT_RESIDUAL_SIDECAR_MAX_BYTES: usize = 512 * 1024 * 1024;
/// This is deliberately far above supported embedding dimensions while keeping
/// the permutation and its admission bitmap bounded independently of file size.
const EXACT_RESIDUAL_SIDECAR_MAX_DIMENSION: usize = 65_536;
/// A residual scan that finds no lane to prove irrelevant during this probe
/// budget stops paying transform-bound overhead and completes on the incumbent
/// flat scanner. This is a semantic-preserving adaptive fallback, not a speed
/// claim: every result still uses the exact f16 scorer.
const EXACT_RESIDUAL_ADAPTIVE_PROBE_GROUPS: usize = 32;
/// Bound generation-name collision retries while publishing an optional cache
/// artifact. Discovery has separate fixed entry and comparison budgets; if
/// either is exhausted, the optional cache simply remains unavailable.
#[cfg(target_os = "linux")]
const EXACT_RESIDUAL_CACHE_ATTEMPTS: usize = 64;
/// Bound every optional-cache directory walk, including unrelated entries.
/// Reaching this fixed limit rejects the cache as unavailable for this open;
/// publication is skipped rather than making an unbounded directory a source
/// of source-derived allocation or I/O.
#[cfg(target_os = "linux")]
const EXACT_RESIDUAL_CACHE_DIRECTORY_ENTRY_LIMIT: usize = 256;
/// Bound the number of descriptor-stream comparisons after a candidate passes
/// the fixed-header admission. This remains above the prior sixty-four-entry
/// regression while preventing a cache full of tiny corrupt siblings from
/// turning a product open into unbounded repeated work.
#[cfg(target_os = "linux")]
const EXACT_RESIDUAL_CACHE_COMPARISON_CANDIDATE_LIMIT: usize = 128;
/// Bound the total bytes that descriptor-stream comparison can read during one
/// product open. The cap admits one maximum-sized sidecar, but never 1024 of
/// them. Exhaustion is an optional-cache miss and skips publication.
#[cfg(target_os = "linux")]
const EXACT_RESIDUAL_CACHE_COMPARISON_BYTE_BUDGET: usize = EXACT_RESIDUAL_SIDECAR_MAX_BYTES;

#[cfg(target_os = "linux")]
static EXACT_RESIDUAL_CACHE_NONCE: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
std::thread_local! {
    /// Deterministic seam for the cache-only query-transform allocation path.
    /// It is thread-local so concurrent tests cannot make another search take
    /// an unintended fallback.
    static FAIL_NEXT_RESIDUAL_QUERY_TRANSFORM_ALLOCATION: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// Independent seam for allocation of the query suffix envelopes.
    static FAIL_NEXT_RESIDUAL_QUERY_SUFFIX_ALLOCATION: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

#[cfg(all(test, target_os = "linux"))]
std::thread_local! {
    /// Fail the cache publication-capability acquisition before any derived
    /// sidecar allocation. This witnesses the optional-cache fail-open rule.
    static FAIL_NEXT_EXACT_RESIDUAL_PUBLICATION_ACQUISITION: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// Counts real source-derived sidecar builds after an individual test has
    /// reset it. It is intentionally thread-local with the failure seam.
    static EXACT_RESIDUAL_SIDECAR_BUILD_COUNT: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn fail_next_residual_query_transform_allocation() {
    FAIL_NEXT_RESIDUAL_QUERY_TRANSFORM_ALLOCATION.with(|failure| failure.set(true));
}

#[cfg(test)]
fn take_residual_query_transform_allocation_failure() -> bool {
    FAIL_NEXT_RESIDUAL_QUERY_TRANSFORM_ALLOCATION.with(|failure| failure.replace(false))
}

#[cfg(test)]
fn residual_query_transform_allocation_failure_is_pending() -> bool {
    FAIL_NEXT_RESIDUAL_QUERY_TRANSFORM_ALLOCATION.with(std::cell::Cell::get)
}

#[cfg(test)]
fn fail_next_residual_query_suffix_allocation() {
    FAIL_NEXT_RESIDUAL_QUERY_SUFFIX_ALLOCATION.with(|failure| failure.set(true));
}

#[cfg(test)]
fn take_residual_query_suffix_allocation_failure() -> bool {
    FAIL_NEXT_RESIDUAL_QUERY_SUFFIX_ALLOCATION.with(|failure| failure.replace(false))
}

#[cfg(test)]
fn residual_query_suffix_allocation_failure_is_pending() -> bool {
    FAIL_NEXT_RESIDUAL_QUERY_SUFFIX_ALLOCATION.with(std::cell::Cell::get)
}

#[cfg(all(test, target_os = "linux"))]
fn fail_next_exact_residual_publication_acquisition() {
    FAIL_NEXT_EXACT_RESIDUAL_PUBLICATION_ACQUISITION.with(|failure| failure.set(true));
}

#[cfg(all(test, target_os = "linux"))]
fn take_exact_residual_publication_acquisition_failure() -> bool {
    FAIL_NEXT_EXACT_RESIDUAL_PUBLICATION_ACQUISITION.with(|failure| failure.replace(false))
}

#[cfg(all(test, target_os = "linux"))]
fn exact_residual_publication_acquisition_failure_is_pending() -> bool {
    FAIL_NEXT_EXACT_RESIDUAL_PUBLICATION_ACQUISITION.with(std::cell::Cell::get)
}

#[cfg(all(test, target_os = "linux"))]
fn reset_exact_residual_sidecar_build_count() {
    EXACT_RESIDUAL_SIDECAR_BUILD_COUNT.with(|count| count.set(0));
}

#[cfg(all(test, target_os = "linux"))]
fn exact_residual_sidecar_build_count() -> usize {
    EXACT_RESIDUAL_SIDECAR_BUILD_COUNT.with(std::cell::Cell::get)
}

/// Fully-resident in-memory vector index with f16 quantization.
///
/// All vectors are stored in a contiguous `Vec<f16>` in row-major order,
/// eliminating memory-map page faults for deterministic sub-millisecond search.
#[derive(Debug, Clone)]
pub struct InMemoryVectorIndex {
    /// Document IDs, indexed by position.
    doc_ids: Vec<String>,
    /// Flat f16 vector slab: `doc_ids.len() * dimension` elements.
    vectors: Vec<f16>,
    /// Lazily-built flat int8 vector slab (same row-major layout) for the int8 ADC
    /// pass-1 of [`InMemoryVectorIndex::search_top_k_int8_two_pass`]. Quantized with
    /// a single corpus-wide max-abs scale, which preserves the dot-product ranking
    /// (the scale is a per-query constant). Built on first two-pass use so exact-only
    /// callers pay neither the quantization work nor its `N·d`-byte footprint.
    vectors_i8: OnceLock<Vec<i8>>,
    /// Lazily-built packed signed-4-bit quantization (2 dims/byte, `dim.div_ceil(2)`
    /// bytes/vector — half the int8 slab) for the optional 4-bit two-pass scan
    /// (`search_top_k_4bit_two_pass`). Built on first 4-bit-two-pass use.
    vectors_nibbles: OnceLock<Vec<u8>>,
    /// Lazily-built FNV-1a hashes of `doc_ids` (same `frankensearch_core::filter`
    /// hash that `BitsetFilter` uses). Lets the filtered scan call
    /// `SearchFilter::matches_doc_id_hash` with a precomputed hash instead of
    /// re-hashing each `doc_id` string per vector (the FSVI `search.rs` scan already
    /// does this). Built on first *filtered* search, so unfiltered callers pay
    /// neither the hashing nor the `8·N`-byte footprint.
    doc_id_hashes: OnceLock<Vec<u64>>,
    /// Lazily-built `doc_id → position` map for O(1) lookup, replacing the O(N)
    /// linear `doc_ids.iter().position(...)` scan in the per-hit quality-rerank
    /// path (`quality_scores_for_hits`), which was O(hits·N). Built on first
    /// doc-id lookup, so search-only callers pay nothing.
    doc_id_index: OnceLock<AHashMap<String, usize>>,
    /// Lazily-built `doc_id_hash → position` map (identity-hashed, same FNV-1a key
    /// space as `BitsetFilter`). Lets a *selective* filtered search gather the
    /// allow-set's positions directly — `O(|allow-set|)` exact dots instead of one
    /// filter probe per corpus document (`scan_gather` vs `scan_range`). Stored as
    /// `Option`: `None` means two `doc_ids` collide to the same hash, so the map is
    /// not a bijection and the gather fast-path is disabled (the per-document scan
    /// stays correct). Built on first selective-filter search; other callers pay
    /// neither the build nor its footprint.
    hash_to_pos: OnceLock<Option<HashMap<u64, usize, BuildIdentityHasherU64>>>,
    /// Lowercase hex SHA-256 fingerprint of the mathematical embedding space
    /// this index's vectors were produced in, when known (bd-9xuj T2-C3).
    ///
    /// `Some` only when the source declared it: an FSVI v2 file's validated
    /// identity header ([`Self::from_fsvi`]) or an explicit caller-supplied
    /// space ([`Self::from_vectors_with_identity`]). `None` is the typed
    /// legacy-unidentified state (v1 files, [`Self::from_vectors`]) — it is
    /// never fabricated from a default, and downstream seams route it as
    /// legacy rather than failing closed.
    space_fingerprint_hex: Option<String>,
    /// Embedder id recorded by the source that produced this index's vectors
    /// (bd-9xuj T2-C2): the FSVI header's `embedder_id` on the file-backed
    /// load paths, or the declared space's `logical_model_id` on
    /// [`Self::from_vectors_with_identity`]. Diagnostics only — compatibility
    /// decisions join on [`Self::space_fingerprint_hex`], never on this
    /// string. `None` means the source carried none ([`Self::from_vectors`]);
    /// absence stays typed, never defaulted.
    embedder_id: Option<String>,
    /// Embedder revision recorded by the source, captured under the same
    /// rules as [`Self::embedder_id`] (bd-9xuj T2-C2). Kept verbatim: a v1
    /// header's empty revision is preserved as `Some("")`, distinct from the
    /// `None` of a source with no header at all.
    embedder_revision: Option<String>,
    /// Whether [`Self::space_fingerprint_hex`] was derived from a validated
    /// FSVI v2 identity HEADER — i.e. read out of the artifact's own bytes
    /// through exact admission — rather than declared by a caller
    /// (bd-9xuj T2-C4-write, admission guards 2+8).
    ///
    /// `true` only on the [`Self::from_admitted_v2`] load path (the sole way
    /// a v2 header reaches this type: [`crate::VectorIndex::open`] is strictly v1).
    /// A caller-supplied space on [`Self::from_vectors_with_identity`] is a
    /// DECLARED identity: retained for joins and diagnostics, but it is a
    /// claim by the constructing process, not an attestation persisted in an
    /// artifact header, so it stays `false`. Attested-only seams (the refresh
    /// identity-bound merge) admit against attested identity exclusively.
    space_identity_attested: bool,
    /// Vector dimensionality.
    dimension: usize,
    /// Present only for an index loaded from an exactly admitted FSVI v2 image.
    /// A residual sidecar requires this binding; legacy FSVI and caller-built
    /// indexes deliberately remain on the established flat route.
    residual_source_binding: Option<ResidualSourceBinding>,
    /// Lazily attached after the complete sidecar header, source witnesses,
    /// dimensions, permutation, and every numeric payload have been admitted.
    /// An absent or rejected sidecar is intentionally indistinguishable from no
    /// optimization request to the public search path.
    exact_residual_sidecar: OnceLock<ExactResidualSidecar>,
}

/// Selectivity threshold for the gather fast-path: take it only when the filter's
/// allow-set is smaller than `corpus / GATHER_SELECTIVITY_DIVISOR`. Below this the
/// gather (exact f16 dots over the allow-set, parallel above `PARALLEL_CHUNK_SIZE`)
/// beats the parallel per-document scan; above it the scan wins because the gather's
/// serial setup (allow-set collect + position sort) grows with the allow-set. The
/// `filtered_gather` selectivity-sweep bench (N=50k, dim 384) measured, with the
/// parallel gather: 14× at 0.5%, 8.6× at 1%, 2.1× at 5%, 1.3× at 10%, then a loss by
/// 25% (crossover ~13%). Gate at **N/10 (10%)** — inside the winning region
/// (≥1.3× at the boundary) with margin for a machine whose core count shifts the
/// crossover. (Tiny allow-sets stay serial and still hit 6.9–50× — see the ledger.)
const GATHER_SELECTIVITY_DIVISOR: usize = 10;

// ─── Heap helpers (mirrors search.rs internals) ─────────────────────────────

#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    index: usize,
    score: f32,
}

impl HeapEntry {
    const fn new(index: usize, score: f32) -> Self {
        Self { index, score }
    }
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: "largest" == worst score, so peek() returns cutoff.
        match score_key(self.score).total_cmp(&score_key(other.score)) {
            Ordering::Less => Ordering::Greater,
            Ordering::Greater => Ordering::Less,
            Ordering::Equal => self.index.cmp(&other.index),
        }
    }
}

const fn score_key(score: f32) -> f32 {
    if score.is_nan() {
        f32::NEG_INFINITY
    } else {
        score
    }
}

/// Winners-count threshold above which the `limit_all` final sort uses a parallel
/// `par_sort_unstable_by` (mirrors `search::PAR_SORT_THRESHOLD`). Below it, rayon's
/// spawn/merge overhead is not amortized for the cheap `compare_best_first`.
const PAR_SORT_THRESHOLD: usize = 16_384;

fn compare_best_first(left: &HeapEntry, right: &HeapEntry) -> Ordering {
    match score_key(right.score).total_cmp(&score_key(left.score)) {
        Ordering::Equal => left.index.cmp(&right.index),
        other => other,
    }
}

fn insert_candidate(heap: &mut BinaryHeap<HeapEntry>, candidate: HeapEntry, limit: usize) {
    if limit == 0 {
        return;
    }
    if heap.len() < limit {
        heap.push(candidate);
        return;
    }
    if let Some(&worst) = heap.peek()
        && match score_key(candidate.score).total_cmp(&score_key(worst.score)) {
            Ordering::Greater => true,
            Ordering::Less => false,
            Ordering::Equal => candidate.index < worst.index,
        }
    {
        let _ = heap.pop();
        heap.push(candidate);
    }
}

fn bounded_heap(capacity: usize, field: &str) -> SearchResult<BinaryHeap<HeapEntry>> {
    let mut heap = BinaryHeap::new();
    heap.try_reserve_exact(capacity)
        .map_err(|_| residual_sidecar_error(field, "heap allocation failed"))?;
    Ok(heap)
}

fn merge_partial_heaps(
    partial_heaps: Vec<BinaryHeap<HeapEntry>>,
    limit: usize,
) -> SearchResult<BinaryHeap<HeapEntry>> {
    let mut total_elements = 0_usize;
    for heap in &partial_heaps {
        total_elements = total_elements.saturating_add(heap.len());
    }
    let capacity = limit.min(total_elements);
    let mut merged = bounded_heap(capacity, "merge_heap")?;
    for partial in partial_heaps {
        for entry in partial {
            insert_candidate(&mut merged, entry, limit);
        }
    }
    Ok(merged)
}
