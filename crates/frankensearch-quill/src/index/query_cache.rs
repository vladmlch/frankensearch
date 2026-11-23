#[cfg(any(test, feature = "bench-internals"))]
use crate::query::{BooleanOperator, Occur, Query, QueryNode, QueryNodeId, QueryValue};
use arc_swap::ArcSwapOption;
#[cfg(any(test, feature = "bench-internals"))]
use std::ops::Bound;
use std::sync::Arc;
use xxhash_rust::xxh3::Xxh3;

use super::error_results::QuillSearchResult;

const RANKED_QUERY_CACHE_SETS: usize = 64;
const RANKED_QUERY_CACHE_WAYS: usize = 4;
const RANKED_QUERY_CACHE_SLOTS: usize = RANKED_QUERY_CACHE_SETS * RANKED_QUERY_CACHE_WAYS;

pub(super) enum RankedQueryCacheKey {
    Raw(Arc<str>),
    #[cfg(any(test, feature = "bench-internals"))]
    Preparsed(Arc<Query>),
}

pub(super) struct CachedRankedQuery {
    pub(super) fingerprint: u64,
    pub(super) snapshot_epoch: u64,
    pub(super) limit: usize,
    pub(super) offset: usize,
    pub(super) exact_count: bool,
    pub(super) key: RankedQueryCacheKey,
    pub(super) result: Arc<QuillSearchResult>,
}

/// Small lock-free memo table for immutable-snapshot searches.
///
/// Exact keys are retained alongside their fingerprints, so collisions only
/// cost a miss. Snapshot epochs make publication the invalidation boundary.
/// The fixed four-way table bounds memory without adding reader coordination.
pub(super) struct RankedQueryCache {
    pub(super) slots: [ArcSwapOption<CachedRankedQuery>; RANKED_QUERY_CACHE_SLOTS],
}

impl Default for RankedQueryCache {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| ArcSwapOption::empty()),
        }
    }
}

impl RankedQueryCache {
    pub(super) fn get_raw(
        &self,
        fingerprint: u64,
        snapshot_epoch: u64,
        query: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Option<QuillSearchResult> {
        self.find(fingerprint, |cached| {
            cached.snapshot_epoch == snapshot_epoch
                && cached.limit == limit
                && cached.offset == offset
                && cached.exact_count == exact_count
                && matches!(&cached.key, RankedQueryCacheKey::Raw(key) if key.as_ref() == query)
        })
    }

    #[cfg(any(test, feature = "bench-internals"))]
    pub(super) fn get_preparsed(
        &self,
        snapshot_epoch: u64,
        query: &Query,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Option<QuillSearchResult> {
        let fingerprint =
            preparsed_query_cache_fingerprint(snapshot_epoch, query, limit, offset, exact_count);
        self.find(fingerprint, |cached| {
            cached.snapshot_epoch == snapshot_epoch
                && cached.limit == limit
                && cached.offset == offset
                && cached.exact_count == exact_count
                && matches!(&cached.key, RankedQueryCacheKey::Preparsed(key) if key.as_ref() == query)
        })
    }

    pub(super) fn find(
        &self,
        fingerprint: u64,
        exact_match: impl Fn(&CachedRankedQuery) -> bool,
    ) -> Option<QuillSearchResult> {
        let start = ranked_query_cache_set_start(fingerprint);
        let set = self
            .slots
            .get(start..start.checked_add(RANKED_QUERY_CACHE_WAYS)?)?;
        for slot in set {
            let Some(cached) = slot.load_full() else {
                continue;
            };
            if cached.fingerprint == fingerprint && exact_match(cached.as_ref()) {
                return Some(cached.result.as_ref().clone());
            }
        }
        None
    }

    pub(super) fn insert_raw(
        &self,
        fingerprint: u64,
        snapshot_epoch: u64,
        query: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
        result: &QuillSearchResult,
    ) {
        self.insert(Arc::new(CachedRankedQuery {
            fingerprint,
            snapshot_epoch,
            limit,
            offset,
            exact_count,
            key: RankedQueryCacheKey::Raw(Arc::from(query)),
            result: Arc::new(result.clone()),
        }));
    }

    #[cfg(any(test, feature = "bench-internals"))]
    pub(super) fn insert_preparsed(
        &self,
        snapshot_epoch: u64,
        query: &Query,
        limit: usize,
        offset: usize,
        exact_count: bool,
        result: &QuillSearchResult,
    ) {
        let fingerprint =
            preparsed_query_cache_fingerprint(snapshot_epoch, query, limit, offset, exact_count);
        self.insert(Arc::new(CachedRankedQuery {
            fingerprint,
            snapshot_epoch,
            limit,
            offset,
            exact_count,
            key: RankedQueryCacheKey::Preparsed(Arc::new(query.clone())),
            result: Arc::new(result.clone()),
        }));
    }

    pub(super) fn insert(&self, cached: Arc<CachedRankedQuery>) {
        let start = ranked_query_cache_set_start(cached.fingerprint);
        let Some(end) = start.checked_add(RANKED_QUERY_CACHE_WAYS) else {
            return;
        };
        let Some(set) = self.slots.get(start..end) else {
            return;
        };
        if let Some(slot) = set.iter().find(|slot| slot.load().is_none()) {
            slot.store(Some(cached));
            return;
        }
        let [_, replacement_byte, ..] = cached.fingerprint.to_le_bytes();
        let replacement = usize::from(replacement_byte) & (RANKED_QUERY_CACHE_WAYS - 1);
        if let Some(slot) = set.get(replacement) {
            slot.store(Some(cached));
        }
    }
}

fn ranked_query_cache_set_start(fingerprint: u64) -> usize {
    let [set_byte, ..] = fingerprint.to_le_bytes();
    let set = usize::from(set_byte) & (RANKED_QUERY_CACHE_SETS - 1);
    set * RANKED_QUERY_CACHE_WAYS
}

pub(super) fn raw_query_cache_fingerprint(
    snapshot_epoch: u64,
    query: &str,
    limit: usize,
    offset: usize,
    exact_count: bool,
) -> u64 {
    let mut hasher = ranked_query_cache_hasher(snapshot_epoch, limit, offset, exact_count);
    hasher.update(&[0]);
    hash_query_cache_bytes(&mut hasher, query.as_bytes());
    hasher.digest()
}

#[cfg(any(test, feature = "bench-internals"))]
fn preparsed_query_cache_fingerprint(
    snapshot_epoch: u64,
    query: &Query,
    limit: usize,
    offset: usize,
    exact_count: bool,
) -> u64 {
    let mut hasher = ranked_query_cache_hasher(snapshot_epoch, limit, offset, exact_count);
    hasher.update(&[1]);
    hash_exact_query(&mut hasher, query);
    hasher.digest()
}

fn ranked_query_cache_hasher(
    snapshot_epoch: u64,
    limit: usize,
    offset: usize,
    exact_count: bool,
) -> Xxh3 {
    let mut hasher = Xxh3::new();
    hasher.update(b"frankensearch.quill.ranked-query-cache.v1\0");
    hasher.update(&snapshot_epoch.to_le_bytes());
    hasher.update(&u64::try_from(limit).unwrap_or(u64::MAX).to_le_bytes());
    hasher.update(&u64::try_from(offset).unwrap_or(u64::MAX).to_le_bytes());
    hasher.update(&[u8::from(exact_count)]);
    hasher
}

fn hash_query_cache_bytes(hasher: &mut Xxh3, bytes: &[u8]) {
    hasher.update(&u64::try_from(bytes.len()).unwrap_or(u64::MAX).to_le_bytes());
    hasher.update(bytes);
}

#[cfg(any(test, feature = "bench-internals"))]
fn hash_query_cache_len(hasher: &mut Xxh3, len: usize) {
    hasher.update(&u64::try_from(len).unwrap_or(u64::MAX).to_le_bytes());
}

#[cfg(any(test, feature = "bench-internals"))]
fn hash_query_cache_value(hasher: &mut Xxh3, value: &QueryValue) {
    match value {
        QueryValue::I64(value) => {
            hasher.update(&[0]);
            hasher.update(&value.to_le_bytes());
        }
        QueryValue::U64(value) => {
            hasher.update(&[1]);
            hasher.update(&value.to_le_bytes());
        }
        QueryValue::Str(value) => {
            hasher.update(&[2]);
            hash_query_cache_bytes(hasher, value.as_bytes());
        }
    }
}

#[cfg(any(test, feature = "bench-internals"))]
fn hash_query_cache_bound(hasher: &mut Xxh3, bound: &Bound<QueryValue>) {
    match bound {
        Bound::Included(value) => {
            hasher.update(&[0]);
            hash_query_cache_value(hasher, value);
        }
        Bound::Excluded(value) => {
            hasher.update(&[1]);
            hash_query_cache_value(hasher, value);
        }
        Bound::Unbounded => hasher.update(&[2]),
    }
}

#[cfg(any(test, feature = "bench-internals"))]
fn hash_exact_query(hasher: &mut Xxh3, query: &Query) {
    enum HashTask {
        Node(QueryNodeId),
        Occur(Occur),
    }

    let Some(task_count) = query.node_count().checked_mul(2) else {
        hasher.update(&[u8::MAX]);
        return;
    };
    let mut pending = Vec::new();
    if pending.try_reserve_exact(task_count).is_err() {
        hasher.update(&[u8::MAX]);
        return;
    }
    pending.push(HashTask::Node(query.root_id()));
    while let Some(task) = pending.pop() {
        let node_id = match task {
            HashTask::Node(node_id) => node_id,
            HashTask::Occur(occur) => {
                hasher.update(&[match occur {
                    Occur::Must => 0,
                    Occur::Should => 1,
                    Occur::MustNot => 2,
                }]);
                continue;
            }
        };
        match query.node(node_id) {
            QueryNode::Empty => hasher.update(&[0]),
            QueryNode::All => hasher.update(&[1]),
            QueryNode::AllocationFailure { requested } => {
                hasher.update(&[9]);
                hasher.update(&requested.to_le_bytes());
            }
            QueryNode::Term { fields, text } => {
                hasher.update(&[2]);
                hash_query_cache_len(hasher, fields.len());
                for field in fields {
                    hasher.update(&field.field_id.to_le_bytes());
                    hasher.update(&field.boost.to_bits().to_le_bytes());
                }
                hash_query_cache_bytes(hasher, text.as_bytes());
            }
            QueryNode::Phrase {
                fields,
                terms,
                slop,
                prefix,
            } => {
                hasher.update(&[3]);
                hash_query_cache_len(hasher, fields.len());
                for field in fields {
                    hasher.update(&field.field_id.to_le_bytes());
                    hasher.update(&field.boost.to_bits().to_le_bytes());
                }
                hash_query_cache_len(hasher, terms.len());
                for term in terms {
                    hasher.update(&term.position.to_le_bytes());
                    hash_query_cache_bytes(hasher, term.text.as_bytes());
                }
                hasher.update(&slop.to_le_bytes());
                hasher.update(&[u8::from(*prefix)]);
            }
            QueryNode::Boolean { clauses, operator } => {
                hasher.update(&[4]);
                hasher.update(&[match operator {
                    None => 0,
                    Some(BooleanOperator::And) => 1,
                    Some(BooleanOperator::Or) => 2,
                }]);
                hash_query_cache_len(hasher, clauses.len());
                for clause in clauses.iter().rev() {
                    pending.push(HashTask::Node(clause.query));
                    pending.push(HashTask::Occur(clause.occur));
                }
            }
            QueryNode::Range {
                field_id,
                lower,
                upper,
            } => {
                hasher.update(&[5]);
                hasher.update(&field_id.to_le_bytes());
                hash_query_cache_bound(hasher, lower);
                hash_query_cache_bound(hasher, upper);
            }
            QueryNode::Set { field_id, values } => {
                hasher.update(&[6]);
                hasher.update(&field_id.to_le_bytes());
                hash_query_cache_len(hasher, values.len());
                for value in values {
                    hash_query_cache_value(hasher, value);
                }
            }
            QueryNode::Glob { field_ids, pattern } => {
                hasher.update(&[7]);
                hash_query_cache_len(hasher, field_ids.len());
                for field_id in field_ids {
                    hasher.update(&field_id.to_le_bytes());
                }
                hash_query_cache_bytes(hasher, pattern.as_bytes());
            }
            QueryNode::Boost {
                query: child,
                factor,
            } => {
                hasher.update(&[8]);
                hasher.update(&factor.to_bits().to_le_bytes());
                pending.push(HashTask::Node(*child));
            }
        }
    }
}
