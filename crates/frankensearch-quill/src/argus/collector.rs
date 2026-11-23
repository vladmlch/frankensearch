use std::cmp::Ordering;
use std::collections::BinaryHeap;

use frankensearch_core::DocId;

use super::ArgusError;
use super::reference_scorer::ReferenceScorer;

/// Live-document membership used only at result collection.
pub trait LiveDocs {
    /// Return whether a physical posting is visible in the current snapshot.
    fn is_live(&self, global_docid: u32) -> bool;
}

impl<F> LiveDocs for F
where
    F: Fn(u32) -> bool,
{
    fn is_live(&self, global_docid: u32) -> bool {
        self(global_docid)
    }
}

/// Membership policy for snapshots without tombstones.
#[derive(Clone, Copy, Debug, Default)]
pub struct AllLiveDocs;

impl LiveDocs for AllLiveDocs {
    fn is_live(&self, _global_docid: u32) -> bool {
        true
    }
}

/// Phase-one exhaustive result carrying no stored metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoredDoc {
    /// Stable global Quill document id.
    pub global_docid: u32,
    /// Exact contract-mode f32 score.
    pub score: f32,
}

/// Final globally merged top-doc page.
///
/// `total_count` is present only when the collector was created with
/// [`TopDocsCollector::with_exact_count`]. Hits are already offset-sliced and
/// remain in the canonical `(score desc, global_docid asc)` order.
#[derive(Clone, Debug, PartialEq)]
pub struct CollectedTopDocs {
    /// Page-local ranked hits.
    pub hits: Vec<ScoredDoc>,
    /// Exact number of live matches across every collected scorer.
    pub total_count: Option<u64>,
}

/// Cross-segment exhaustive top-doc accumulator.
///
/// Feed every per-segment scorer into one instance, then call [`Self::finish`]
/// exactly once. Offset slicing happens only after the global heap is sorted,
/// preventing per-segment pagination from discarding valid winners.
#[derive(Debug)]
pub struct TopDocsCollector {
    limit: usize,
    offset: usize,
    pub(in crate::argus) retained: usize,
    pub(in crate::argus) exact_count: bool,
    total_count: u64,
    /// Bounded max-heap of packed selection keys (`bd-y1ab`). Each key folds
    /// the entry's score and docid so the whole heap comparison is one native
    /// `u64` compare instead of an f32 `total_cmp` plus a conditional docid
    /// tiebreak: high 32 bits = `!sortable(score)` (worst score = greatest key
    /// = heap top, evicted first), low 32 bits = the global docid (higher docid
    /// = greater = evicted first on a score tie, so the lower docid is
    /// retained). Order-identical to the former per-entry `total_cmp` heap; see
    /// [`packed_selection_key`].
    heap: BinaryHeap<u64>,
}

/// Monotonic f32 -> u32 bijection: ascending `u32` order equals ascending f32
/// order for every finite value (and both signed zeros). Collector scores are
/// [`finite_score`](super::boolean_scorers::finite_score)-guaranteed finite, so the round-trip through
/// [`score_from_sortable`] is exact.
#[inline]
fn score_to_sortable(score: f32) -> u32 {
    let bits = score.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// Inverse of [`score_to_sortable`].
#[inline]
fn score_from_sortable(sortable: u32) -> f32 {
    if sortable & 0x8000_0000 != 0 {
        f32::from_bits(sortable & 0x7fff_ffff)
    } else {
        f32::from_bits(!sortable)
    }
}

/// Badness-ordered packed key: greatest for the worst entry (lowest score;
/// highest docid on a score tie), so a max-heap retains the best entries.
#[inline]
fn packed_selection_key(global_docid: u32, score: f32) -> u64 {
    (u64::from(!score_to_sortable(score)) << 32) | u64::from(global_docid)
}

/// Reconstruct the winner encoded by a packed selection key.
#[inline]
fn scored_doc_from_key(key: u64) -> ScoredDoc {
    let global_docid = (key & 0xffff_ffff) as u32;
    let score = score_from_sortable(!((key >> 32) as u32));
    ScoredDoc {
        global_docid,
        score,
    }
}

/// Score of the packed key currently at the heap top (the worst retained
/// entry), used as the competitive pruning cutoff.
#[inline]
fn cutoff_score_from_key(key: u64) -> f32 {
    score_from_sortable(!((key >> 32) as u32))
}

impl TopDocsCollector {
    /// Build a count-free top-k collector with Tantivy-compatible offset.
    ///
    /// # Errors
    ///
    /// Rejects `limit + offset` overflow and bounded heap allocation failure.
    pub fn new(limit: usize, offset: usize) -> Result<Self, ArgusError> {
        Self::build(limit, offset, false)
    }

    /// Build a top-k collector that also exhaustively counts every live match.
    ///
    /// `limit == 0` retains no scores but still counts all matches.
    ///
    /// # Errors
    ///
    /// Rejects `limit + offset` overflow and bounded heap allocation failure.
    pub fn with_exact_count(limit: usize, offset: usize) -> Result<Self, ArgusError> {
        Self::build(limit, offset, true)
    }

    fn build(limit: usize, offset: usize, exact_count: bool) -> Result<Self, ArgusError> {
        let retained = if limit == 0 {
            0
        } else {
            limit
                .checked_add(offset)
                .ok_or(ArgusError::CollectorWindowOverflow { limit, offset })?
        };
        let mut heap = BinaryHeap::new();
        heap.try_reserve_exact(retained)
            .map_err(|_| ArgusError::Allocation {
                resource: "top-k heap",
                count: retained,
            })?;
        Ok(Self {
            limit,
            offset,
            retained,
            exact_count,
            total_count: 0,
            heap,
        })
    }

    /// Build an empty collector sharing this collector's limit, offset, and
    /// count-mode shape.
    ///
    /// Segment-parallel fan-out uses this to give each rayon task a private
    /// accumulator whose retained window matches the fold target exactly.
    ///
    /// # Errors
    ///
    /// Rejects bounded heap allocation failure.
    pub fn empty_like(&self) -> Result<Self, ArgusError> {
        Self::build(self.limit, self.offset, self.exact_count)
    }

    /// Fold another collector's retained state into this one.
    ///
    /// The packed selection key's total order (score via the monotonic
    /// `score_to_sortable` transform, then ascending docid) makes the
    /// retained top-`retained` set unique regardless of insertion order, so
    /// folding per-segment partials in any order yields exactly the set a
    /// single serial collector retains.
    ///
    /// # Errors
    ///
    /// Rejects collectors with mismatched limit/offset/count shape and exact
    /// match-count overflow.
    pub fn merge(&mut self, other: Self) -> Result<(), ArgusError> {
        if self.limit != other.limit
            || self.offset != other.offset
            || self.exact_count != other.exact_count
        {
            return Err(ArgusError::CursorInvariant(
                "merged top-k collectors must share limit, offset, and count mode",
            ));
        }
        if self.exact_count {
            self.total_count = self
                .total_count
                .checked_add(other.total_count)
                .ok_or(ArgusError::MatchCountOverflow)?;
        }
        for key in other.heap {
            if self.heap.len() < self.retained {
                self.heap.push(key);
            } else if self.heap.peek().is_some_and(|&cutoff| key < cutoff) {
                let _ = self.heap.pop();
                self.heap.push(key);
            }
        }
        Ok(())
    }

    /// Consume one segment scorer into this global accumulator.
    ///
    /// # Errors
    ///
    /// Propagates scorer, visibility, count, and allocation failures.
    pub fn collect<L>(
        &mut self,
        scorer: &mut ReferenceScorer<'_>,
        live_docs: &L,
    ) -> Result<(), ArgusError>
    where
        L: LiveDocs + ?Sized,
    {
        if self.retained == 0 && !self.exact_count {
            return Ok(());
        }
        let mut state = CollectorState::TopDocs(self);
        scorer.collect_with(&mut state, live_docs)
    }

    pub(in crate::argus) fn record_live(
        &mut self,
        global_docid: u32,
        score: Option<f32>,
    ) -> Result<(), ArgusError> {
        if self.exact_count {
            self.total_count = self
                .total_count
                .checked_add(1)
                .ok_or(ArgusError::MatchCountOverflow)?;
        }
        if self.retained == 0 {
            return Ok(());
        }
        let score = score.ok_or(ArgusError::CursorInvariant(
            "top-k collector did not score a retained match",
        ))?;
        let key = packed_selection_key(global_docid, score);
        if self.heap.len() < self.retained {
            self.heap.push(key);
        } else if self.heap.peek().is_some_and(|&cutoff| key < cutoff) {
            let _ = self.heap.pop();
            self.heap.push(key);
        }
        Ok(())
    }

    pub(in crate::argus) fn competitive_cutoff_score(&self) -> Option<f32> {
        if self.retained != 0 && self.heap.len() == self.retained {
            self.heap.peek().map(|&key| cutoff_score_from_key(key))
        } else {
            None
        }
    }

    /// Sort the global heap, apply offset once, and return page winners.
    ///
    /// # Errors
    ///
    /// Returns a typed allocation failure when the final compact hit vector
    /// cannot be reserved.
    pub fn finish(self) -> Result<CollectedTopDocs, ArgusError> {
        let winner_count = self.heap.len();
        let mut hits = Vec::new();
        hits.try_reserve_exact(winner_count)
            .map_err(|_| ArgusError::Allocation {
                resource: "top-k winners",
                count: winner_count,
            })?;
        hits.extend(self.heap.into_iter().map(scored_doc_from_key));
        hits.sort_unstable_by(|left, right| compare_scored_best_first(*left, *right));
        let start = self.offset.min(hits.len());
        let end = start.saturating_add(self.limit).min(hits.len());
        if start != 0 {
            hits.copy_within(start..end, 0);
        }
        hits.truncate(end - start);
        Ok(CollectedTopDocs {
            hits,
            total_count: self.exact_count.then_some(self.total_count),
        })
    }
}

/// Cross-segment scoreless global-document set accumulator.
#[derive(Debug, Default)]
pub struct DocSetCollector {
    docs: Vec<u32>,
}

impl DocSetCollector {
    /// Build an empty scoreless collector.
    #[must_use]
    pub const fn new() -> Self {
        Self { docs: Vec::new() }
    }

    /// Consume one recursively unscored segment scorer.
    ///
    /// # Errors
    ///
    /// Rejects scored buffered unions and propagates cursor or allocation
    /// failures.
    pub fn collect<L>(
        &mut self,
        scorer: &mut ReferenceScorer<'_>,
        live_docs: &L,
    ) -> Result<(), ArgusError>
    where
        L: LiveDocs + ?Sized,
    {
        let mut state = CollectorState::DocSet(self);
        scorer.collect_with(&mut state, live_docs)
    }

    pub(in crate::argus) fn push(&mut self, global_docid: u32) -> Result<(), ArgusError> {
        if self.docs.len() == self.docs.capacity() {
            self.docs
                .try_reserve(1)
                .map_err(|_| ArgusError::Allocation {
                    resource: "doc-set matches",
                    count: self.docs.len().saturating_add(1),
                })?;
        }
        self.docs.push(global_docid);
        Ok(())
    }

    /// Fold another collector's raw matches into this one.
    ///
    /// Order does not matter: [`Self::finish`] sorts and dedups, so folding
    /// per-segment partials in any order yields the identical final set.
    ///
    /// # Errors
    ///
    /// Rejects allocation failure while growing the match vector.
    pub fn merge(&mut self, other: Self) -> Result<(), ArgusError> {
        self.docs
            .try_reserve(other.docs.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "doc-set matches",
                count: self.docs.len().saturating_add(other.docs.len()),
            })?;
        self.docs.extend(other.docs);
        Ok(())
    }

    /// Return deterministic sorted unique global docids.
    #[must_use]
    pub fn finish(mut self) -> Vec<u32> {
        self.docs.sort_unstable();
        self.docs.dedup();
        self.docs
    }
}

pub(in crate::argus) enum CollectorState<'a> {
    TopDocs(&'a mut TopDocsCollector),
    DocSet(&'a mut DocSetCollector),
}

/// Winner after phase-two external-id materialization.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterializedHit {
    /// Stable global Quill document id.
    pub global_docid: u32,
    /// External document id resolved only after top-k selection.
    pub document_id: DocId,
    /// Exact contract-mode f32 score.
    pub score: f32,
}

/// Resolve external document ids for already-selected winners only.
///
/// # Errors
///
/// Propagates the resolver's typed [`ArgusError`] or a bounded output
/// allocation failure.
pub fn materialize_doc_ids(
    winners: &[ScoredDoc],
    mut resolve: impl FnMut(u32) -> Result<DocId, ArgusError>,
) -> Result<Vec<MaterializedHit>, ArgusError> {
    let mut hits = Vec::new();
    hits.try_reserve_exact(winners.len())
        .map_err(|_| ArgusError::Allocation {
            resource: "materialized hits",
            count: winners.len(),
        })?;
    for winner in winners {
        hits.push(MaterializedHit {
            global_docid: winner.global_docid,
            document_id: resolve(winner.global_docid)?,
            score: winner.score,
        });
    }
    Ok(hits)
}

fn compare_scored_best_first(left: ScoredDoc, right: ScoredDoc) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.global_docid.cmp(&right.global_docid))
}
