use std::collections::HashMap;
use std::hash::{BuildHasher, BuildHasherDefault, Hasher};

use super::{DEFAULT_ARENA_CHUNK_BYTES, FIELD_PREFIX_BYTES};

#[cfg(test)]
std::thread_local! {
    static SORTED_IDS_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn reset_sorted_ids_calls() {
    SORTED_IDS_CALLS.with(|calls| calls.set(0));
}

#[cfg(test)]
pub(super) fn sorted_ids_calls() -> usize {
    SORTED_IDS_CALLS.with(std::cell::Cell::get)
}

/// A span into a [`ByteArena`]: which chunk, where, how long.
///
/// 12 bytes, `Copy`; the interner keeps one per term.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaSpan {
    chunk: u32,
    offset: u32,
    len: u32,
}

/// Chunked bump allocator for raw bytes.
///
/// Not a general allocator: append-only between resets, no per-item free.
/// [`reset`](Self::reset) retains every standard chunk at full capacity so a
/// steady-state flush cycle performs zero global allocations.
#[derive(Debug)]
pub struct ByteArena {
    chunks: Vec<Vec<u8>>,
    chunk_size: usize,
    /// Index of the chunk currently accepting writes.
    active: usize,
    /// Running sum of every chunk's capacity, so [`Self::bytes_reserved`] is
    /// O(1) instead of re-summing every chunk per call. This is the sibling of
    /// `TermInterner::running_bytes_used` (bd-w4j5) on the *reserved* path,
    /// which that fix left as an O(chunks) scan. Exact rather than
    /// approximate: `push` only ever grows capacity by appending a whole new
    /// chunk (the `needs_new` guard means `extend_from_slice` always fits an
    /// existing chunk).
    ///
    /// Every site that changes chunk capacity must maintain this, and there
    /// are exactly four: the three `chunks.push` calls in [`Self::push`]
    /// (oversized chunk, lazy first chunk, next standard chunk) and the
    /// `retain` in [`Self::reset`], which drops chunks and therefore
    /// recomputes. The `debug_assert_eq!` in [`Self::bytes_reserved`] is the
    /// backstop that catches a missed site.
    running_bytes_reserved: usize,
}

/// Hand-written so the running counter is recomputed rather than copied.
///
/// `Vec::clone` allocates exactly `len`, so a cloned chunk's capacity is not
/// the source chunk's — a derived `Clone` would carry the source's
/// `running_bytes_reserved` onto containers that reserve a different amount
/// and `bytes_reserved` would report the wrong number for the rest of the
/// clone's life. Cloning is off the hot path (it is not per document), so the
/// O(chunks) recompute here costs nothing that matters.
impl Clone for ByteArena {
    fn clone(&self) -> Self {
        let chunks = self.chunks.clone();
        let running_bytes_reserved = chunks.iter().map(Vec::capacity).sum();
        Self {
            chunks,
            chunk_size: self.chunk_size,
            active: self.active,
            running_bytes_reserved,
        }
    }
}

impl ByteArena {
    /// Create an arena with the given chunk size (min 4 KiB to bound the
    /// chunk-vector length on adversarial configs).
    #[must_use]
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        let chunk_size = chunk_size.max(4096);
        // Chunks are allocated lazily on first `push`, so nothing is reserved
        // yet and the running counter starts at zero.
        Self {
            chunks: Vec::new(),
            chunk_size,
            active: 0,
            running_bytes_reserved: 0,
        }
    }

    /// Copy `bytes` into the arena, returning its span.
    ///
    /// Oversized inputs (longer than the chunk size) get a dedicated
    /// exactly-sized chunk; the previously active chunk stays active so its
    /// remaining capacity is not wasted.
    ///
    /// # Panics
    /// Panics if chunk count, offset, or span length overflow `u32` — at the
    /// 64 MiB shard budget these are unreachable by orders of magnitude.
    pub fn push(&mut self, bytes: &[u8]) -> ArenaSpan {
        if bytes.len() > self.chunk_size {
            let mut chunk = Vec::with_capacity(bytes.len());
            chunk.extend_from_slice(bytes);
            self.running_bytes_reserved =
                self.running_bytes_reserved.saturating_add(chunk.capacity());
            self.chunks.push(chunk);
            let idx = self.chunks.len() - 1;
            return ArenaSpan {
                chunk: u32::try_from(idx).expect("arena chunk count exceeds u32"),
                offset: 0,
                len: u32::try_from(bytes.len()).expect("arena span exceeds u32"),
            };
        }
        if self.chunks.is_empty() {
            // Lazy first chunk: this is a capacity change like any other, so it
            // must be folded into the running counter or `bytes_reserved` would
            // report 0 for a non-empty arena.
            let chunk = Vec::with_capacity(self.chunk_size);
            self.running_bytes_reserved =
                self.running_bytes_reserved.saturating_add(chunk.capacity());
            self.chunks.push(chunk);
            self.active = 0;
        }
        let needs_new = {
            let chunk = &self.chunks[self.active];
            chunk.len() + bytes.len() > chunk.capacity()
        };
        if needs_new {
            // Find or create the next reusable standard chunk. Dedicated
            // oversized chunks (capacity != chunk_size) are skipped.
            let mut next = self.active + 1;
            while next < self.chunks.len() && self.chunks[next].capacity() != self.chunk_size {
                next += 1;
            }
            if next == self.chunks.len() {
                let chunk = Vec::with_capacity(self.chunk_size);
                self.running_bytes_reserved =
                    self.running_bytes_reserved.saturating_add(chunk.capacity());
                self.chunks.push(chunk);
            }
            self.active = next;
        }
        let chunk_idx = self.active;
        let chunk = &mut self.chunks[chunk_idx];
        let offset = chunk.len();
        chunk.extend_from_slice(bytes);
        ArenaSpan {
            chunk: u32::try_from(chunk_idx).expect("arena chunk count exceeds u32"),
            offset: u32::try_from(offset).expect("arena offset exceeds u32"),
            len: u32::try_from(bytes.len()).expect("arena span exceeds u32"),
        }
    }

    /// Resolve a span to its bytes.
    ///
    /// # Panics
    /// Panics if the span does not belong to this arena (spans are only ever
    /// produced by [`push`](Self::push) on the same arena; crossing arenas is
    /// a programming error, not a recoverable state).
    #[must_use]
    pub fn resolve(&self, span: ArenaSpan) -> &[u8] {
        let chunk = &self.chunks[span.chunk as usize];
        &chunk[span.offset as usize..span.offset as usize + span.len as usize]
    }

    /// Bytes currently stored (sum of chunk lengths).
    #[must_use]
    pub fn bytes_used(&self) -> usize {
        self.chunks.iter().map(Vec::len).sum()
    }

    /// Bytes currently reserved (sum of chunk capacities) — the RSS-relevant
    /// figure reported in flush-cycle tracing.
    #[must_use]
    pub fn bytes_reserved(&self) -> usize {
        debug_assert_eq!(
            self.running_bytes_reserved,
            self.chunks.iter().map(Vec::capacity).sum::<usize>(),
            "arena running reserved drifted from the chunk capacity sum"
        );
        self.running_bytes_reserved
    }

    /// Number of chunks (soak tests assert this stabilizes across cycles).
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Clear all content, retaining standard-chunk capacity. Dedicated
    /// oversized chunks are dropped (they are corpus outliers by definition;
    /// retaining them would ratchet RSS on pathological inputs).
    pub fn reset(&mut self) {
        self.chunks.retain(|c| c.capacity() == self.chunk_size);
        // `retain` drops the dedicated oversized chunks, so recompute rather
        // than track the deletions. Reset is once per flush cycle, not per
        // document, so an O(chunks) pass here is off the hot path — and
        // skipping it would leave `running_bytes_reserved` counting capacity
        // that was just freed, permanently over-reporting reserved bytes.
        self.running_bytes_reserved = self.chunks.iter().map(Vec::capacity).sum();
        for chunk in &mut self.chunks {
            chunk.clear();
        }
        self.active = 0;
    }
}

impl Default for ByteArena {
    fn default() -> Self {
        Self::with_chunk_size(DEFAULT_ARENA_CHUNK_BYTES)
    }
}

/// One interned term as seen from the `buckets` map: the dense id plus an
/// inline copy of the term's arena span.
///
/// The span is deliberately duplicated from `spans[id]` (P8 locality lever,
/// bd-e8h-w2-interner-arena-x9s38): the per-token hot path (`find_in_bucket`
/// → `matches`) verifies key bytes straight from the map entry, removing the
/// dependent `spans[id]` load — a random-access cache miss per repeated
/// token sitting on the longest load chain (map entry → span → arena bytes).
/// `spans` remains the id → span source of truth for the resolve APIs
/// (`composite_key`, `field_and_term`, `sorted_ids`).
///
/// INVARIANT (span-mirror): `entry.span == spans[entry.id as usize]` for
/// every entry reachable from `buckets`. Both are written exactly once,
/// together, in `intern_accounted`; spans are never mutated afterwards (the
/// arena is append-only between resets, and `reset` clears both containers).
#[derive(Debug, Clone, Copy)]
pub(super) struct BucketEntry {
    pub(super) id: u32,
    pub(super) span: ArenaSpan,
}

/// Collision bucket: hash → interned entry/entries. The `Many` arm is
/// exercised only on 64-bit hash collisions (or by tests injecting a
/// degenerate hasher).
#[derive(Debug, Clone)]
pub(super) enum Bucket {
    One(BucketEntry),
    Many(Vec<BucketEntry>),
}

pub const TERM_BUCKET_BYTES_ESTIMATE: usize = 8 + std::mem::size_of::<Bucket>() + 8;

// Flush-accounting contract (byte-identity obligation): the estimate feeds
// `running_bytes_used` — the shard flush trigger — and delta.rs's seal
// check, so it must stay numerically identical across `Bucket` layout
// changes or segment boundaries (and therefore on-disk bytes) would drift.
// The niche layout keeps `Bucket` at 24 bytes with `BucketEntry` inline
// (16-byte payload beside the `Vec` niche), the same size as the pre-P8
// `One(u32) | Many(Vec<u32>)` shape. If this assert fires, do NOT relax it:
// pin the constant to 40 and re-derive flush parity evidence.
const _: () = assert!(TERM_BUCKET_BYTES_ESTIMATE == 40);

/// Identity [`Hasher`] for the interner's `buckets` map.
///
/// INVARIANT (key-is-already-a-hash): every key of `buckets` is a finalized
/// 64-bit hash — the output of [`TermInterner::hash_parts`] (default: ahash
/// over the composite `(field_ord, term)` key). Its only producers are
/// `find`/`find_in_bucket`/`intern_accounted`, each of which computes the
/// key via `hash_parts` immediately before the map operation. Re-hashing
/// that u64 through the std map's `SipHash` was pure waste (~4.3% of the QG-2
/// quill ingest arm: `hash_one::<&u64>` + sip `Hasher::write`; P1 profile
/// card, bd-e8h-w2-u64-hasher-swap-vcfft).
///
/// Do NOT reuse this type for maps whose keys are not already uniformly
/// distributed hash outputs (sequential ids, pointers, lengths): identity
/// hashing such keys would degrade hashbrown's probe behavior. Under the
/// test-injected constant hasher every term collapses into one
/// `Bucket::Many` entry, so the degenerate path stays exercised. The map
/// hasher cannot affect on-disk bytes: TERMDICT order comes from
/// [`TermInterner::sorted_ids`] (a byte sort), and the only whole-map reads
/// (`bytes_reserved`, `clear`) are order-independent.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct PrehashedKeyHasher(u64);

impl Hasher for PrehashedKeyHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write_u64(&mut self, key: u64) {
        self.0 = key;
    }

    fn write(&mut self, bytes: &[u8]) {
        // Only u64 keys (which use `write_u64`) are expected; fold any other
        // input so the hasher stays total rather than silently truncating.
        for &byte in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(byte);
        }
    }
}

/// Per-shard composite-key term interner.
///
/// Keys are `(field_ord, term bytes)`, stored once in the arena as the
/// on-disk composite form (BE field prefix + term bytes). Ids are dense u32s
/// in first-insertion order.
///
/// The hasher is generic (default [`ahash::RandomState`] — in-memory only;
/// durable hashing elsewhere in Quill is xxh3 by contract, FSLX §2). Tests
/// inject a constant hasher to force every key through the `Many`
/// verification path.
#[derive(Debug)]
pub struct TermInterner<S: BuildHasher = ahash::RandomState> {
    pub(super) arena: ByteArena,
    pub(super) spans: Vec<ArenaSpan>,
    /// hash → term id(s), keyed by the ALREADY-FINALIZED `hash_parts` output,
    /// so the map hasher is the no-op [`PrehashedKeyHasher`] (see its
    /// key-is-already-a-hash invariant note).
    pub(super) buckets: HashMap<u64, Bucket, BuildHasherDefault<PrehashedKeyHasher>>,
    hasher: S,
    /// Scratch buffer for composite-key assembly (reused, never shrunk).
    pub(super) key_scratch: Vec<u8>,
    /// Running sum of every `intern_accounted` increment, so [`Self::bytes_used`]
    /// is O(1) instead of re-summing the arena chunks and every bucket per call
    /// (profile-directed, bd-w4j5: `should_flush` re-called `bytes_used` once per
    /// ingested document). Exactly equals the former recompute because each
    /// increment already accounts for the arena key, span, and bucket bytes.
    running_bytes_used: usize,
    /// Retained capacity of `Bucket::Many` collision-entry vectors.
    ///
    /// Maintaining this total at the two mutation sites keeps
    /// [`Self::bytes_reserved`] exact without scanning every distinct term
    /// after every ingested document.
    running_collision_bytes_reserved: usize,
}

impl<S: BuildHasher + Clone> Clone for TermInterner<S> {
    fn clone(&self) -> Self {
        let buckets = self.buckets.clone();
        // `Vec::clone` is allowed to choose a different capacity, so copy the
        // logical cache from the cloned collision vectors rather than from the
        // source interner.
        let running_collision_bytes_reserved =
            buckets.values().fold(0_usize, |reserved, bucket| {
                let bucket_reserved = match bucket {
                    Bucket::One(_) => 0,
                    Bucket::Many(entries) => entries
                        .capacity()
                        .saturating_mul(std::mem::size_of::<BucketEntry>()),
                };
                reserved.saturating_add(bucket_reserved)
            });
        Self {
            arena: self.arena.clone(),
            spans: self.spans.clone(),
            buckets,
            hasher: self.hasher.clone(),
            key_scratch: self.key_scratch.clone(),
            running_bytes_used: self.running_bytes_used,
            running_collision_bytes_reserved,
        }
    }
}

impl TermInterner<ahash::RandomState> {
    #[must_use]
    pub fn new() -> Self {
        Self::with_hasher(ahash::RandomState::new())
    }

    #[must_use]
    pub(crate) fn with_arena_chunk_size(arena_chunk_size: usize) -> Self {
        Self::with_hasher_and_arena_chunk_size(ahash::RandomState::new(), arena_chunk_size)
    }
}

impl Default for TermInterner<ahash::RandomState> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: BuildHasher> TermInterner<S> {
    #[must_use]
    pub fn with_hasher(hasher: S) -> Self {
        Self::with_hasher_and_arena_chunk_size(hasher, DEFAULT_ARENA_CHUNK_BYTES)
    }

    fn with_hasher_and_arena_chunk_size(hasher: S, arena_chunk_size: usize) -> Self {
        Self {
            arena: ByteArena::with_chunk_size(arena_chunk_size),
            spans: Vec::new(),
            buckets: HashMap::default(),
            hasher,
            key_scratch: Vec::with_capacity(64),
            running_bytes_used: 0,
            running_collision_bytes_reserved: 0,
        }
    }

    fn hash_parts(&self, field_ord: u16, term: &[u8]) -> u64 {
        let mut h = self.hasher.build_hasher();
        h.write(&field_ord.to_be_bytes());
        h.write(term);
        h.finish()
    }

    /// Verify a candidate's key bytes against `(field_ord, term)`.
    ///
    /// Takes the span (from the bucket entry's inline copy) rather than an
    /// id so the hit path never touches `spans` (see [`BucketEntry`]).
    fn matches(&self, span: ArenaSpan, field_ord: u16, term: &[u8]) -> bool {
        let key = self.arena.resolve(span);
        key.len() == FIELD_PREFIX_BYTES + term.len()
            && key[..FIELD_PREFIX_BYTES] == field_ord.to_be_bytes()
            && key[FIELD_PREFIX_BYTES..] == *term
    }

    fn find_in_bucket(&self, hash: u64, field_ord: u16, term: &[u8]) -> Option<u32> {
        match self.buckets.get(&hash)? {
            Bucket::One(entry) => self
                .matches(entry.span, field_ord, term)
                .then_some(entry.id),
            Bucket::Many(entries) => entries
                .iter()
                .find(|entry| self.matches(entry.span, field_ord, term))
                .map(|entry| entry.id),
        }
    }

    /// Find an existing composite term without mutating the interner.
    pub(crate) fn find(&self, field_ord: u16, term: &[u8]) -> Option<u32> {
        let hash = self.hash_parts(field_ord, term);
        self.find_in_bucket(hash, field_ord, term)
    }

    /// Intern `(field_ord, term)`, returning the dense local id.
    ///
    /// Hot path: existing terms cost one hash + one arena compare and perform
    /// zero allocations (the prefix and term bytes are hashed directly;
    /// composite-key scratch is populated only for a new term).
    ///
    /// # Panics
    /// Panics if the number of distinct terms exceeds `u32` — unreachable
    /// under the shard budget.
    pub fn intern(&mut self, field_ord: u16, term: &[u8]) -> u32 {
        self.intern_accounted(field_ord, term).0
    }

    /// Intern a term and return its exact increment to [`Self::bytes_used`].
    ///
    /// The delta segment uses the increment to keep its seal check O(1). The
    /// normal Scribe path needs only the dense id and uses [`Self::intern`].
    pub(crate) fn intern_accounted(&mut self, field_ord: u16, term: &[u8]) -> (u32, usize) {
        let hash = self.hash_parts(field_ord, term);
        if let Some(id) = self.find_in_bucket(hash, field_ord, term) {
            return (id, 0);
        }

        self.key_scratch.clear();
        self.key_scratch.extend_from_slice(&field_ord.to_be_bytes());
        self.key_scratch.extend_from_slice(term);

        // New term: copy the composite key into the arena, assign the next id.
        let span = self.arena.push(&self.key_scratch);
        let id = u32::try_from(self.spans.len()).expect("term id space exceeds u32");
        self.spans.push(span);
        // Written together with `spans` — the span-mirror invariant
        // (`entry.span == spans[entry.id]`, see [`BucketEntry`]) holds here
        // by construction and is never mutated afterwards.
        let entry = BucketEntry { id, span };
        // `bucket_bytes` (first tuple element) feeds `running_bytes_used`,
        // the shard flush trigger: its values are PINNED to the pre-P8
        // id-only increments (u32-based) as a byte-identity contract — the
        // real entry is larger, but changing these numbers would move flush
        // boundaries and thus on-disk segment bytes. The `reserved` deltas
        // (second element) are RSS diagnostics only and use the real
        // `BucketEntry` size.
        let (bucket_bytes, collision_reserved_delta) = match self.buckets.entry(hash) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(Bucket::One(entry));
                (TERM_BUCKET_BYTES_ESTIMATE, 0)
            }
            std::collections::hash_map::Entry::Occupied(mut o) => match o.get_mut() {
                Bucket::One(existing) => {
                    let existing = *existing;
                    let entries = vec![existing, entry];
                    let reserved = entries
                        .capacity()
                        .saturating_mul(std::mem::size_of::<BucketEntry>());
                    *o.get_mut() = Bucket::Many(entries);
                    (2 * std::mem::size_of::<u32>(), reserved)
                }
                Bucket::Many(entries) => {
                    let capacity_before = entries.capacity();
                    entries.push(entry);
                    let reserved_delta = entries
                        .capacity()
                        .saturating_sub(capacity_before)
                        .saturating_mul(std::mem::size_of::<BucketEntry>());
                    (std::mem::size_of::<u32>(), reserved_delta)
                }
            },
        };
        self.running_collision_bytes_reserved = self
            .running_collision_bytes_reserved
            .saturating_add(collision_reserved_delta);
        let added_bytes = FIELD_PREFIX_BYTES
            .saturating_add(term.len())
            .saturating_add(std::mem::size_of::<ArenaSpan>())
            .saturating_add(bucket_bytes);
        self.running_bytes_used = self.running_bytes_used.saturating_add(added_bytes);
        (id, added_bytes)
    }

    /// Number of distinct interned terms.
    #[must_use]
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Resolve an id to its composite key (BE field prefix + term bytes).
    ///
    /// # Panics
    /// Panics on an id not produced by this interner instance.
    #[must_use]
    pub fn composite_key(&self, id: u32) -> &[u8] {
        self.arena.resolve(self.spans[id as usize])
    }

    /// Resolve an id to `(field_ord, term bytes)`.
    ///
    /// # Panics
    /// Panics on an id not produced by this interner instance.
    #[must_use]
    pub fn field_and_term(&self, id: u32) -> (u16, &[u8]) {
        let key = self.composite_key(id);
        let field = u16::from_be_bytes([key[0], key[1]]);
        (field, &key[FIELD_PREFIX_BYTES..])
    }

    /// Ids sorted by composite key bytes — exactly the on-disk TERMDICT order
    /// (FSLX §5.1: the BE field prefix makes byte order equal (field, term)
    /// order). Used by flushes and immutable snapshot caches to materialize
    /// canonical term order. Large buckets use an in-place MSD radix
    /// partition so shared prefixes are inspected once per byte instead of by
    /// every comparison; small buckets finish with the comparison sorter to
    /// avoid clearing a 257-way histogram for a handful of ids.
    ///
    /// # Panics
    /// Panics if the id space exceeds `u32` (unreachable; see [`Self::intern`]).
    #[must_use]
    pub fn sorted_ids(&self) -> Vec<u32> {
        #[cfg(test)]
        SORTED_IDS_CALLS.with(|calls| calls.set(calls.get().saturating_add(1)));
        let mut ids: Vec<u32> =
            (0..u32::try_from(self.spans.len()).expect("term id space exceeds u32")).collect();
        self.sort_ids_msd(&mut ids);
        ids
    }

    fn sort_ids_msd(&self, ids: &mut [u32]) {
        const RADIX: usize = 257;
        const SMALL_BUCKET: usize = 24;

        if ids.len() < 2 {
            return;
        }

        let mut pending = Vec::with_capacity(64);
        pending.push((0_usize, ids.len(), 0_usize));
        let mut counts = [0_usize; RADIX];
        let mut starts = [0_usize; RADIX];
        let mut next = [0_usize; RADIX];

        while let Some((lo, hi, depth)) = pending.pop() {
            if hi - lo <= SMALL_BUCKET {
                ids[lo..hi]
                    .sort_unstable_by(|a, b| self.composite_key(*a).cmp(self.composite_key(*b)));
                continue;
            }

            counts.fill(0);
            for &id in &ids[lo..hi] {
                let digit = self
                    .composite_key(id)
                    .get(depth)
                    .map_or(0, |byte| usize::from(*byte) + 1);
                counts[digit] += 1;
            }

            let mut cursor = lo;
            for bucket in 0..RADIX {
                starts[bucket] = cursor;
                next[bucket] = cursor;
                cursor += counts[bucket];
            }
            debug_assert_eq!(cursor, hi);

            // American-flag partitioning: every swap fills one destination
            // slot, so the pass is linear and needs no second id array.
            for bucket in 0..RADIX {
                let end = starts[bucket] + counts[bucket];
                while next[bucket] < end {
                    let source = next[bucket];
                    let id = ids[source];
                    let destination_bucket = self
                        .composite_key(id)
                        .get(depth)
                        .map_or(0, |byte| usize::from(*byte) + 1);
                    if destination_bucket == bucket {
                        next[bucket] += 1;
                    } else {
                        let destination = next[destination_bucket];
                        debug_assert!(
                            destination < starts[destination_bucket] + counts[destination_bucket]
                        );
                        ids.swap(source, destination);
                        next[destination_bucket] += 1;
                    }
                }
            }

            // Bucket zero is the end-of-key sentinel and is already final.
            // Push in reverse so low-byte buckets are processed first, keeping
            // the traversal deterministic even though the partition is not.
            for bucket in (1..RADIX).rev() {
                if counts[bucket] > 1 {
                    pending.push((starts[bucket], starts[bucket] + counts[bucket], depth + 1));
                }
            }
        }
    }

    /// Approximate live bytes held, for the shard flush trigger.
    ///
    /// This deliberately uses lengths rather than capacities: reset retains
    /// allocations for reuse, and counting those retained capacities as live
    /// would make every post-flush document immediately flush again. Use
    /// [`Self::bytes_reserved`] for RSS diagnostics.
    #[must_use]
    pub fn bytes_used(&self) -> usize {
        self.running_bytes_used
    }

    /// Complete retained interner allocation for RSS/reuse diagnostics.
    #[must_use]
    pub fn bytes_reserved(&self) -> usize {
        self.arena
            .bytes_reserved()
            .saturating_add(
                self.spans
                    .capacity()
                    .saturating_mul(std::mem::size_of::<ArenaSpan>()),
            )
            .saturating_add(
                self.buckets
                    .capacity()
                    .saturating_mul(TERM_BUCKET_BYTES_ESTIMATE),
            )
            .saturating_add(self.key_scratch.capacity())
            .saturating_add(self.running_collision_bytes_reserved)
    }

    /// Reset for the next flush cycle: clears terms, retains arena chunk and
    /// container capacity (steady-state cycles allocate nothing).
    pub fn reset(&mut self) {
        self.arena.reset();
        self.spans.clear();
        self.buckets.clear();
        self.key_scratch.clear();
        self.running_bytes_used = 0;
        self.running_collision_bytes_reserved = 0;
    }

    /// Arena diagnostics for flush-cycle tracing:
    /// `(bytes_used, bytes_reserved, chunk_count)`.
    #[must_use]
    pub fn arena_stats(&self) -> (usize, usize, usize) {
        (
            self.arena.bytes_used(),
            self.arena.bytes_reserved(),
            self.arena.chunk_count(),
        )
    }
}
