//! Brute-force top-k vector search over an opened [`crate::VectorIndex`].

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::OnceLock;

use frankensearch_core::VectorHit;
use frankensearch_core::config::ZeroSignalReason;

mod quantized_two_pass;
mod scans;
#[cfg(test)]
mod tests;
mod top_k;

/// Record-count threshold where search switches from sequential to Rayon.
pub const PARALLEL_THRESHOLD: usize = 10_000;
/// Chunk size per Rayon task in the parallel scan path.
pub const PARALLEL_CHUNK_SIZE: usize = 1_024;
const INT8_PARALLEL_CHUNK_SIZE: usize = PARALLEL_CHUNK_SIZE * 4;
/// Selectivity threshold for the file-backed gather fast-path. A hash-addressable
/// filter must be smaller than `record_count / GATHER_SELECTIVITY_DIVISOR` before
/// we invert the loop and binary-search/gather the allowed hash ranges. The FSVI
/// crossover is lower than the in-memory gather because each allowed hash pays
/// `log2(N)` record-table probes; the short `filtered_gather` sweep keeps clear of
/// the measured 5% regression.
const GATHER_SELECTIVITY_DIVISOR: usize = 50;

/// Configurable parameters for vector search parallelism.
///
/// Controls when and how the brute-force scan switches from sequential
/// to Rayon-parallel execution. Use [`SearchParams::default()`] for the
/// standard settings (threshold = 10,000, chunk size = 1,024, parallel
/// enabled via `FRANKENSEARCH_PARALLEL_SEARCH` env var).
#[derive(Debug, Clone, Copy)]
pub struct SearchParams {
    /// Minimum record count to trigger parallel scanning.
    /// Below this threshold, search runs sequentially.
    pub parallel_threshold: usize,
    /// Number of records processed per Rayon chunk in parallel mode.
    pub parallel_chunk_size: usize,
    /// Whether parallel scanning is allowed at all. When `false`, search
    /// always runs sequentially regardless of record count.
    pub parallel_enabled: bool,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            parallel_threshold: PARALLEL_THRESHOLD,
            parallel_chunk_size: PARALLEL_CHUNK_SIZE,
            parallel_enabled: parallel_search_enabled(),
        }
    }
}

/// A top-k result plus a typed classification when it is empty.
///
/// The invariant `zero_signal.is_some() == hits.is_empty()` lets callers
/// distinguish a legitimately empty answer (benign request/state outcome)
/// from an unusable semantic lane (availability failure) without inferring
/// anything from bare emptiness.
#[derive(Debug, Clone)]
pub struct ClassifiedHits {
    /// Ranked hits, best first. May be empty.
    pub hits: Vec<VectorHit>,
    /// `Some(reason)` if and only if `hits` is empty.
    pub zero_signal: Option<ZeroSignalReason>,
}

impl ClassifiedHits {
    /// An empty result with its typed reason.
    #[must_use]
    pub const fn empty(reason: ZeroSignalReason) -> Self {
        Self {
            hits: Vec::new(),
            zero_signal: Some(reason),
        }
    }
}

static PARALLEL_SEARCH_ENABLED_CACHE: OnceLock<bool> = OnceLock::new();

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
        // BinaryHeap keeps the largest element at the top.
        // We define "largest" == "worst" so peek() returns the current cutoff.
        match score_key(self.score).total_cmp(&score_key(other.score)) {
            Ordering::Less => Ordering::Greater,
            Ordering::Greater => Ordering::Less,
            Ordering::Equal => self.index.cmp(&other.index),
        }
    }
}

/// Largest dimension whose entire `[-127 * 127 * dim, 127 * 127 * dim]` integer
/// dot range is exactly representable as f32. The common 384-dimensional path can
/// therefore rank raw i32 values while preserving the shipped i32-to-f32 order.
const MAX_EXACT_I8_DOT_DIM: usize = 1_040;

/// Packed pass-1 ordering key for the int8 two-pass scan. Larger means worse, so
/// `BinaryHeap::peek` remains the cutoff. The high word reverses score order
/// (higher scores become smaller keys); the low word preserves the full `usize`
/// index as the deterministic tiebreak (lower indices are better).
type Int8HeapKey = u128;

#[allow(clippy::inline_always)]
#[inline(always)]
const fn int8_heap_key(index: usize, score: i32) -> Int8HeapKey {
    let ascending_score = score.cast_unsigned() ^ 0x8000_0000;
    let descending_score = !ascending_score;
    ((descending_score as u128) << usize::BITS) | index as u128
}

#[allow(clippy::inline_always)]
#[inline(always)]
fn int8_heap_key_from_f32(index: usize, score: i32) -> Int8HeapKey {
    let score = score as f32;
    let bits = score.to_bits();
    let sign_mask = (bits >> 31).wrapping_neg() | 0x8000_0000;
    let ascending_score = bits ^ sign_mask;
    let descending_score = !ascending_score;
    (u128::from(descending_score) << usize::BITS) | index as u128
}

const fn int8_heap_index(key: Int8HeapKey) -> usize {
    key as usize
}

#[inline]
fn retain_int8_candidate(
    heap: &mut BinaryHeap<Int8HeapKey>,
    cutoff: &mut Int8HeapKey,
    candidate: Int8HeapKey,
    limit: usize,
) {
    if heap.len() < limit {
        heap.push(candidate);
        if heap.len() == limit {
            *cutoff = heap.peek().copied().unwrap_or(Int8HeapKey::MAX);
        }
    } else if candidate < *cutoff {
        let _ = heap.pop();
        heap.push(candidate);
        *cutoff = heap.peek().copied().unwrap_or(Int8HeapKey::MAX);
    }
}

/// Quantize an f32 query to int8 using its own max-abs scale (a per-query constant
/// that does not change the dot-product ranking).
#[allow(clippy::cast_possible_truncation)] // round()+clamp() bounds the f32->i8 cast
fn quantize_i8_query(query: &[f32]) -> Vec<i8> {
    let max_abs = query.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    if max_abs <= 0.0 {
        return vec![0; query.len()];
    }
    let scale = 127.0 / max_abs;
    query
        .iter()
        .map(|&x| (x * scale).round().clamp(-127.0, 127.0) as i8)
        .collect()
}

/// Quantize one component to a signed 4-bit nibble (`[-7, 7]`, 4-bit two's
/// complement in the low 4 bits) given a scale.
#[inline]
#[allow(clippy::cast_possible_truncation)] // round()+clamp() bounds the cast
fn nibble_of(value: f32, scale: f32) -> u8 {
    let q = (value * scale).round().clamp(-7.0, 7.0) as i8;
    q.cast_unsigned() & 0x0F
}

/// Pack an f32 query into signed 4-bit nibbles, 2 dims/byte (low = even dim, high =
/// odd dim), using the query's own max-abs scale (a per-query constant that does not
/// change the dot-product ranking). Matches `pack_4bit_f16_bytes`.
fn pack_4bit_query(query: &[f32]) -> Vec<u8> {
    let max_abs = query.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    let scale = if max_abs > 1e-9 { 7.0 / max_abs } else { 0.0 };
    let mut packed = vec![0_u8; query.len().div_ceil(2)];
    for (d, &x) in query.iter().enumerate() {
        let nib = nibble_of(x, scale);
        if d % 2 == 0 {
            packed[d / 2] |= nib;
        } else {
            packed[d / 2] |= nib << 4;
        }
    }
    packed
}

pub(crate) const fn score_key(score: f32) -> f32 {
    if score.is_nan() {
        f32::NEG_INFINITY
    } else {
        score
    }
}

/// Winners-count threshold above which the `limit_all` final sort uses a parallel
/// `par_sort_unstable_by` instead of the serial sort. Below it, rayon's spawn/merge
/// overhead is not amortized (the per-element comparison is cheap); at 50k winners
/// the parallel sort is ~2.81× faster (`winners_sort` bench). Bit-identical output.
const PAR_SORT_THRESHOLD: usize = 16_384;

// Strict total order: `score_key.total_cmp` then a unique-`index` tiebreak. Because
// no two distinct entries compare Equal, the `winners` sorts that use this run as
// `sort_unstable_by` (pdqsort, no scratch alloc) with output identical to a stable
// sort — a ~1.16× (top-k) to ~1.47× (limit_all, 50k winners) win on the final order.
fn compare_best_first(left: &HeapEntry, right: &HeapEntry) -> Ordering {
    match score_key(right.score).total_cmp(&score_key(left.score)) {
        Ordering::Equal => left.index.cmp(&right.index),
        other => other,
    }
}

fn candidate_is_better(left: HeapEntry, right: HeapEntry) -> bool {
    match score_key(left.score).total_cmp(&score_key(right.score)) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => left.index < right.index,
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
        && candidate_is_better(candidate, worst)
    {
        let _ = heap.pop();
        heap.push(candidate);
    }
}

fn merge_partial_heaps(
    partial_heaps: Vec<BinaryHeap<HeapEntry>>,
    limit: usize,
) -> BinaryHeap<HeapEntry> {
    let mut total_elements = 0_usize;
    for heap in &partial_heaps {
        total_elements = total_elements.saturating_add(heap.len());
    }
    let capacity = limit.min(total_elements).saturating_add(1);
    let mut merged = BinaryHeap::with_capacity(capacity);
    for heap in partial_heaps {
        for entry in heap {
            insert_candidate(&mut merged, entry, limit);
        }
    }
    merged
}

fn merge_int8_partial_heaps(
    partial_heaps: Vec<BinaryHeap<Int8HeapKey>>,
    limit: usize,
) -> BinaryHeap<Int8HeapKey> {
    let total_elements = partial_heaps
        .iter()
        .map(BinaryHeap::len)
        .fold(0_usize, usize::saturating_add);
    let mut merged = BinaryHeap::with_capacity(limit.min(total_elements).saturating_add(1));
    for heap in partial_heaps {
        for candidate in heap {
            if merged.len() < limit {
                merged.push(candidate);
            } else if merged.peek().is_some_and(|&worst| candidate < worst) {
                let _ = merged.pop();
                merged.push(candidate);
            }
        }
    }
    merged
}

fn parallel_search_enabled() -> bool {
    *PARALLEL_SEARCH_ENABLED_CACHE.get_or_init(|| {
        let value = std::env::var("FRANKENSEARCH_PARALLEL_SEARCH").ok();
        parse_parallel_search_env(value.as_deref())
    })
}

fn parse_parallel_search_env(value: Option<&str>) -> bool {
    value.is_none_or(|raw| {
        let normalized = raw.trim();
        !normalized.eq_ignore_ascii_case("0")
            && !normalized.eq_ignore_ascii_case("false")
            && !normalized.eq_ignore_ascii_case("no")
            && !normalized.eq_ignore_ascii_case("off")
    })
}
