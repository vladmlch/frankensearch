use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::traits::IdentityBoundEmbedding;

/// Statistics snapshot from a [`CachedEmbedder`](super::CachedEmbedder).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    /// Number of cache hits since creation (or last clear).
    pub hits: u64,
    /// Number of cache misses since creation (or last clear).
    pub misses: u64,
    /// Current number of entries in the cache.
    pub entries: usize,
    /// Maximum capacity before FIFO eviction kicks in.
    pub capacity: usize,
}

pub(super) struct CacheEntry<T> {
    value: T,
    /// Access frequency, saturating at [`FREQ_CAP`]. Drives S3-FIFO promotion
    /// (Small→Main) and the Main second-chance.
    freq: u8,
}

const FREQ_CAP: u8 = 3;

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct BoundCacheKey {
    pub(super) identity: Arc<EmbeddingIdentityBundleV1>,
    pub(super) text: String,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) enum CacheKey {
    Raw(String),
    Bound(BoundCacheKey),
}

/// S3-FIFO query-embedding cache (Yang et al., SOSP 2023), entry-count form.
///
/// Three queues over both entry maps: **Small** (new/one-hit-wonder admissions,
/// ~10% of capacity), **Main** (proven-reused, ~90%), and **Ghost** (keys recently
/// evicted from Small, metadata-only). Unlike the previous plain FIFO, a key
/// re-requested while resident is promoted to Main and survives the scan churn that
/// evicts cold one-hit-wonders from Small — measurably fewer embed misses on skewed
/// / scan-heavy query streams (see the `cache_replay` bench + `PERF_LEDGER` 2026-06-29).
/// Raw lookups borrow `&str` (no per-get allocation). Both namespaces share one
/// capacity, eviction policy, statistics and clear epoch; storing a bound response
/// never gives raw callers an entry or allocates a second independent budget.
pub(super) struct CacheState {
    pub(super) entries: HashMap<String, CacheEntry<Vec<f32>>>,
    pub(super) bound_entries: HashMap<BoundCacheKey, CacheEntry<IdentityBoundEmbedding>>,
    small: VecDeque<CacheKey>,
    main: VecDeque<CacheKey>,
    pub(super) ghost: VecDeque<CacheKey>,
    pub(super) ghost_set: HashSet<CacheKey>,
    capacity: usize,
    small_cap: usize,
    ghost_cap: usize,
    hits: u64,
    misses: u64,
    // Retained by each miss until admission. Pointer identity cannot wrap or
    // be reused while an older in-flight operation still owns its token.
    pub(super) epoch: Arc<()>,
}

impl CacheState {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            bound_entries: HashMap::new(),
            small: VecDeque::new(),
            main: VecDeque::new(),
            ghost: VecDeque::new(),
            ghost_set: HashSet::new(),
            capacity,
            // Small queue ~10% of capacity (≥1 when caching is enabled).
            small_cap: (capacity / 10).max(1),
            ghost_cap: capacity,
            hits: 0,
            misses: 0,
            epoch: Arc::new(()),
        }
    }

    pub(super) fn get(&mut self, key: &str) -> Option<Vec<f32>> {
        if let Some(entry) = self.entries.get_mut(key) {
            self.hits += 1;
            entry.freq = entry.freq.saturating_add(1).min(FREQ_CAP);
            Some(entry.value.clone())
        } else {
            self.misses += 1;
            None
        }
    }

    /// Record a hit that was served without a fresh lookup — used for an in-batch
    /// duplicate query that folds onto another slot's just-computed embedding, so
    /// batch stats match the old per-text loop (dup = hit, not a second miss).
    pub(super) fn record_hit(&mut self) {
        self.hits += 1;
    }

    pub(super) fn lookup_bound(&mut self, key: &BoundCacheKey) -> Option<IdentityBoundEmbedding> {
        if let Some(entry) = self.bound_entries.get(key) {
            // A bound hit is counted only after producer/cancellation revalidation.
            Some(entry.value.clone())
        } else {
            self.misses += 1;
            None
        }
    }

    pub(super) fn record_bound_hit(&mut self, key: &BoundCacheKey) {
        self.hits += 1;
        if let Some(entry) = self.bound_entries.get_mut(key) {
            entry.freq = entry.freq.saturating_add(1).min(FREQ_CAP);
        }
    }

    pub(super) fn insert(&mut self, key: String, value: Vec<f32>) {
        // capacity == 0 means caching is disabled.
        if self.capacity == 0 || self.entries.contains_key(&key) {
            return;
        }
        self.entries
            .insert(key.clone(), CacheEntry { value, freq: 0 });
        self.admit(CacheKey::Raw(key));
    }

    pub(super) fn insert_bound(&mut self, key: BoundCacheKey, value: IdentityBoundEmbedding) {
        if self.capacity == 0 || self.bound_entries.contains_key(&key) {
            return;
        }
        self.bound_entries
            .insert(key.clone(), CacheEntry { value, freq: 0 });
        self.admit(CacheKey::Bound(key));
    }

    fn admit(&mut self, key: CacheKey) {
        // A key seen recently (in Ghost) re-enters straight into Main; a fresh key
        // starts in Small so a scan of one-hit-wonders can't displace the hot set.
        if self.ghost_set.remove(&key) {
            self.main.push_back(key);
        } else {
            self.small.push_back(key);
        }
        while self.entries.len() + self.bound_entries.len() > self.capacity {
            self.evict_one();
        }
    }

    fn frequency_mut(&mut self, key: &CacheKey) -> Option<&mut u8> {
        match key {
            CacheKey::Raw(key) => self.entries.get_mut(key).map(|entry| &mut entry.freq),
            CacheKey::Bound(key) => self.bound_entries.get_mut(key).map(|entry| &mut entry.freq),
        }
    }

    fn remove(&mut self, key: &CacheKey) {
        match key {
            CacheKey::Raw(key) => {
                self.entries.remove(key);
            }
            CacheKey::Bound(key) => {
                self.bound_entries.remove(key);
            }
        }
    }

    /// Free exactly one slot: evict from Small when it's over its target (promoting
    /// reused keys to Main, demoting cold ones to Ghost), otherwise give Main keys a
    /// frequency-decremented second chance until one with `freq == 0` is dropped.
    fn evict_one(&mut self) {
        loop {
            // Evict from Small whenever it is at-or-over its target (`>=`): a fresh
            // key on probation survives only if re-accessed (freq>0 → promoted to
            // Main) before the next eviction; cold one-hit-wonders are dropped. Using
            // `>=` (not `>`) keeps this correct at the degenerate `small_cap ==
            // capacity` size (e.g. capacity 1), where `>` would evict a just-promoted
            // Main entry instead of the cold Small one.
            if !self.small.is_empty() && self.small.len() >= self.small_cap {
                let Some(k) = self.small.pop_front() else {
                    continue;
                };
                if let Some(freq) = self.frequency_mut(&k).filter(|freq| **freq > 0) {
                    *freq = 0;
                    self.main.push_back(k); // promote — no slot freed, keep going
                } else {
                    self.remove(&k);
                    self.push_ghost(k);
                    return;
                }
            } else if let Some(k) = self.main.pop_front() {
                if let Some(freq) = self.frequency_mut(&k).filter(|freq| **freq > 0) {
                    *freq -= 1;
                    self.main.push_back(k); // second chance — keep going
                } else {
                    self.remove(&k);
                    return;
                }
            } else {
                return; // defensive: both queues drained
            }
        }
    }

    fn push_ghost(&mut self, key: CacheKey) {
        if self.ghost_cap == 0 {
            return;
        }
        self.ghost.push_back(key.clone());
        self.ghost_set.insert(key);
        while self.ghost.len() > self.ghost_cap {
            if let Some(old) = self.ghost.pop_front() {
                self.ghost_set.remove(&old);
            }
        }
    }

    pub(super) fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            entries: self.entries.len() + self.bound_entries.len(),
            capacity: self.capacity,
        }
    }

    pub(super) fn clear(&mut self) {
        self.epoch = Arc::new(());
        self.entries.clear();
        self.bound_entries.clear();
        self.small.clear();
        self.main.clear();
        self.ghost.clear();
        self.ghost_set.clear();
        self.hits = 0;
        self.misses = 0;
    }
}
