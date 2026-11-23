use thiserror::Error;

use super::DOC_ORDS_PER_LEASE;

// ---------------------------------------------------------------------------
// Q1 docid allocation and shard routing (bd-quill-e1-scribe-bejd.6)
// ---------------------------------------------------------------------------

/// Width of one docid lease block in global docids (Q1 §2).
///
/// Exactly one tombstone `chunk_id` of 65,536 contiguous docids. A lease never
/// crosses a block boundary, and its ordinals are relative to the block base.
pub const DOCID_LEASE_BLOCK: u64 = DOC_ORDS_PER_LEASE as u64;

/// One recorded lease grant in the allocator's append-only session log.
///
/// The log is the deterministic witness replay suites compare: two sessions
/// fed the same routed batch sequence must produce identical grant vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeaseGrant {
    /// Zero-based shard that received the lease.
    pub shard: usize,
    /// Global docid at ordinal zero of the lease's block (always
    /// 65,536-aligned); the value spans report as [`DocIdSpan::lease_base`].
    pub lease_base: u64,
    /// First ordinal of the block the lease covers: the session watermark's
    /// offset inside the block, zero for every lease after a block boundary.
    pub first_ord: u32,
    /// Monotone grant sequence within the allocator session.
    pub grant_seq: u64,
}

impl LeaseGrant {
    /// Global docid of the first document the lease can issue.
    #[must_use]
    pub const fn first_docid(&self) -> u64 {
        self.lease_base + self.first_ord as u64
    }
}

/// One contiguous run of docids allocated inside a single lease block.
///
/// A span never crosses a lease boundary (Q1 R1); when a batch exhausts a
/// lease mid-batch the allocator ends the span, cuts, and continues in a
/// freshly granted lease. Every span after the first in one batch therefore
/// starts at ordinal zero of a new lease, and the ingest driver MUST seal the
/// accumulator before starting it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DocIdSpan {
    /// Global docid of the lease block's ordinal zero. This is the value the
    /// flush stage consumes as [`FlushSegmentInput::lease_docid_base`](super::FlushSegmentInput::lease_docid_base).
    pub lease_base: u64,
    /// First allocated lease-relative ordinal.
    pub ord_start: u32,
    /// Number of consecutive ordinals allocated.
    pub len: u32,
}

impl DocIdSpan {
    /// Global docid of the first document in the span.
    #[must_use]
    pub const fn global_first(&self) -> u64 {
        self.lease_base + self.ord_start as u64
    }

    /// Global docid one past the last document in the span.
    #[must_use]
    pub const fn global_end(&self) -> u64 {
        self.lease_base + self.ord_start as u64 + self.len as u64
    }
}

/// Docids allocated for one routed batch, as one or more lease-local spans.
///
/// Most batches fit the shard's live lease and yield exactly one span. A
/// batch that crosses a lease boundary yields multiple spans; the driver must
/// perform the R1 segment cut between consecutive spans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchDocIds {
    spans: Vec<DocIdSpan>,
}

impl BatchDocIds {
    /// Allocated spans in allocation order; never empty.
    #[must_use]
    pub fn spans(&self) -> &[DocIdSpan] {
        &self.spans
    }

    /// Whether the batch crossed a lease boundary (an R1 cut is required
    /// before each span after the first).
    #[must_use]
    pub fn crossed_lease(&self) -> bool {
        self.spans.len() > 1
    }

    /// Total docids allocated across all spans.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.spans.iter().map(|span| u64::from(span.len)).sum()
    }

    /// Whether the batch allocated no docids. Always false for a successful
    /// allocation; provided for exhaustive match ergonomics.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
}

/// Per-grant burn accounting emitted when an allocator session ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeaseBurnRecord {
    /// Zero-based shard that held the lease.
    pub shard: usize,
    /// Global docid at ordinal zero of the lease's block.
    pub lease_base: u64,
    /// Docids consumed by documents before the session ended.
    pub used: u32,
    /// Claimed docids never issued (never reusable — Q1-d). Zero for the
    /// frontier lease, which reserves only what it issues.
    pub burned: u32,
}

/// Whole-session lease accounting returned by [`DocIdAllocator::end_session`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LeaseBurnReport {
    /// One record per grant in grant order.
    pub records: Vec<LeaseBurnRecord>,
    /// Total docids the session's leases reserved (`total_used + total_burned`).
    pub total_leased: u64,
    /// Total docids consumed by documents.
    pub total_used: u64,
    /// Total docids burned: reserved lease tails never issued (Q1-d).
    pub total_burned: u64,
    /// Watermark the next session must open from; never decreases.
    pub final_watermark: u64,
}

/// Typed failures from [`DocIdAllocator`].
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DocIdAllocatorError {
    /// An allocator must serve at least one shard.
    #[error("shard count must be at least one")]
    ZeroShards,
    /// The shard id is outside the session's configured shard count.
    #[error("shard {shard} is outside the configured shard count {shard_count}")]
    UnknownShard {
        /// Rejected shard id.
        shard: usize,
        /// Session shard count.
        shard_count: usize,
    },
    /// Empty batches must not reach docid allocation.
    #[error("cannot allocate docids for an empty batch")]
    EmptyAllocation,
    /// The u32 payload domain is exhausted; deep compaction (reserved, not
    /// implemented at 1.0) is the documented escape hatch (Q1 §2).
    #[error("docid space exhausted: lease block base {base} would leave the u32 payload domain")]
    DocIdSpaceExhausted {
        /// Block base that could not be granted.
        base: u64,
    },
    /// The session was ended; its lease tails are burned and no further
    /// docids may issue from it (Q1-d).
    #[error("allocator session is closed; its lease tails are burned")]
    SessionClosed,
}

/// Per-shard live lease state.
#[derive(Debug, Clone, Copy)]
struct ShardLease {
    base_docid: u64,
    next_ord: u32,
}

/// Keeper-side monotone docid allocator implementing the Q1 lease discipline
/// (`docs/contracts/quill-q1-docid-discipline.md` §2).
///
/// One allocator serves one writer session. It opens from the manifest's
/// persisted `docid_high_watermark` and never issues a docid below it. A
/// shard's lease starts at the watermark (or the next block boundary, see
/// below), may grow to the end of its [`DOCID_LEASE_BLOCK`], and is reused
/// across watch-mode batches.
///
/// Only the *frontier* lease — the most recently granted one — reserves
/// lazily: nothing lies above it, so it reserves exactly what it issues and
/// the watermark is the exact allocation frontier. Before another shard's
/// lease may start, the frontier lease claims the rest of its block, so
/// concurrently live leases stay disjoint. When the session ends, the
/// claimed but unissued tail of every other live lease is burned, never
/// reused (Q1-d); the frontier lease burns nothing. A session that writes
/// through one shard therefore burns nothing at all. Short sessions (one per
/// fsfs watch publication, delete, or append; an incremental library indexer)
/// used to burn up to a block each, which merges stored as dense DOCLEN/IDMAP
/// holes (bd-k07zw, #41) and which exhausts the u32 docid space after 65,536
/// sessions.
///
/// The watermark advances when docids are reserved, so a crash mid-session
/// lands recovery on a watermark that covers every docid reserved before the
/// last publish — no end-of-session manifest write is required for safety.
/// Docids reserved after the last publish appear in no published artifact.
/// The value to persist at the next manifest publish is [`Self::watermark`].
#[derive(Debug)]
pub struct DocIdAllocator {
    shard_count: usize,
    next_unreserved: u64,
    /// Shard holding the frontier lease, which ends at `next_unreserved`.
    frontier_shard: Option<usize>,
    grant_seq: u64,
    leases: Vec<Option<ShardLease>>,
    grants: Vec<LeaseGrant>,
    open: bool,
    trace_grants: bool,
}

impl DocIdAllocator {
    pub(crate) fn speculative_clone(&self) -> Self {
        Self {
            shard_count: self.shard_count,
            next_unreserved: self.next_unreserved,
            frontier_shard: self.frontier_shard,
            grant_seq: self.grant_seq,
            leases: self.leases.clone(),
            grants: self.grants.clone(),
            open: self.open,
            trace_grants: false,
        }
    }

    pub(crate) fn commit_speculative_grants(&mut self, prior_grant_count: usize) {
        debug_assert!(prior_grant_count <= self.grants.len());
        self.trace_grants = true;
        if let Some(committed) = self.grants.get(prior_grant_count..) {
            for grant in committed {
                Self::trace_grant(*grant);
            }
        }
    }

    /// Open a session allocator at the manifest's persisted watermark.
    ///
    /// The first lease starts exactly at the watermark, inside its block when
    /// the watermark is not block-aligned; nothing is skipped. The Keeper
    /// manifest validator guarantees the watermark is at or above every
    /// published docid, so the lease can collide with no live document.
    ///
    /// # Errors
    ///
    /// Returns [`DocIdAllocatorError::ZeroShards`] for a zero shard count.
    pub fn open(manifest_watermark: u64, shard_count: usize) -> Result<Self, DocIdAllocatorError> {
        if shard_count == 0 {
            return Err(DocIdAllocatorError::ZeroShards);
        }
        Ok(Self {
            shard_count,
            next_unreserved: manifest_watermark,
            frontier_shard: None,
            grant_seq: 0,
            leases: vec![None; shard_count],
            grants: Vec::new(),
            open: true,
            trace_grants: true,
        })
    }

    /// Number of shards this session allocates for.
    #[must_use]
    pub const fn shard_count(&self) -> usize {
        self.shard_count
    }

    /// Watermark to persist in the manifest at the next publish.
    ///
    /// The first docid no lease has reserved. It advances as the frontier
    /// lease issues docids, and to a block end when the frontier lease claims
    /// its block for another shard's grant. It never decreases, so persisting
    /// it at any point preserves Q1-d across crashes.
    #[must_use]
    pub const fn watermark(&self) -> u64 {
        self.next_unreserved
    }

    /// Append-only log of every lease grant this session, in grant order.
    #[must_use]
    pub fn lease_grants(&self) -> &[LeaseGrant] {
        &self.grants
    }

    /// Live lease state of one shard as `(lease_base, next_free_ordinal)`, or
    /// `None` when the shard holds no lease yet.
    #[must_use]
    pub fn live_lease(&self, shard: usize) -> Option<(u64, u32)> {
        self.leases
            .get(shard)
            .and_then(|lease| lease.map(|live| (live.base_docid, live.next_ord)))
    }

    /// Docids [`Self::alloc_batch`] can issue to `shard` as one span, without
    /// an R1 cut: the rest of the shard's live lease, or of the block a fresh
    /// lease would start in. Zero once the docid space is exhausted.
    #[must_use]
    pub fn contiguous_capacity(&self, shard: usize) -> u32 {
        if let Some(lease) = self.leases.get(shard).copied().flatten()
            && lease.next_ord < DOC_ORDS_PER_LEASE
        {
            return DOC_ORDS_PER_LEASE - lease.next_ord;
        }
        let start = self.next_grant_start(shard);
        if start > u64::from(u32::MAX) {
            return 0;
        }
        u32::try_from(DOCID_LEASE_BLOCK - start % DOCID_LEASE_BLOCK).unwrap_or(0)
    }

    /// Allocate `count` docids for one batch routed to `shard`.
    ///
    /// Docids issue from the shard's live lease when one exists (watch-mode
    /// batch reuse); an exhausted or missing lease triggers a fresh grant.
    /// Spans never cross a lease boundary (R1): a batch larger than the
    /// remaining lease tail yields multiple spans, each after the first
    /// starting at ordinal zero of a newly granted lease.
    ///
    /// # Errors
    ///
    /// Returns [`DocIdAllocatorError`] for an unknown shard, an empty batch,
    /// a closed session, or u32 docid-space exhaustion.
    pub fn alloc_batch(
        &mut self,
        shard: usize,
        count: u32,
    ) -> Result<BatchDocIds, DocIdAllocatorError> {
        self.ensure_usable(shard)?;
        if count == 0 {
            return Err(DocIdAllocatorError::EmptyAllocation);
        }
        let mut remaining = count;
        let mut spans = Vec::new();
        while remaining > 0 {
            let lease = self.ensure_lease(shard)?;
            let free = DOC_ORDS_PER_LEASE - lease.next_ord;
            let take = free.min(remaining);
            let span = DocIdSpan {
                lease_base: lease.base_docid,
                ord_start: lease.next_ord,
                len: take,
            };
            lease.next_ord += take;
            remaining -= take;
            spans.push(span);
            if self.frontier_shard == Some(shard) {
                self.next_unreserved = span.global_end();
            }
        }
        Ok(BatchDocIds { spans })
    }

    /// Allocate one fresh global docid for `shard` (the upsert path: allocate
    /// a new docid, then tombstone the old one found via the IDHASH probe).
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::alloc_batch`].
    pub fn alloc_one(&mut self, shard: usize) -> Result<u64, DocIdAllocatorError> {
        let batch = self.alloc_batch(shard, 1)?;
        Ok(batch.spans[0].global_first())
    }

    /// End the session: burn the claimed but unissued tail of every live
    /// lease except the frontier lease (Q1-d) and return whole-session lease
    /// accounting. The allocator is closed; further allocation returns
    /// [`DocIdAllocatorError::SessionClosed`].
    #[must_use]
    pub fn end_session(&mut self) -> LeaseBurnReport {
        self.open = false;
        let mut report = LeaseBurnReport {
            final_watermark: self.next_unreserved,
            ..LeaseBurnReport::default()
        };
        for grant in &self.grants {
            let live = self.leases[grant.shard].filter(|live| live.base_docid == grant.lease_base);
            // A grant that is not the shard's live lease was necessarily
            // exhausted in place before its successor was granted.
            let issued_end = live.map_or(DOC_ORDS_PER_LEASE, |live| live.next_ord);
            let used = issued_end - grant.first_ord;
            let burned = if live.is_some() && self.frontier_shard != Some(grant.shard) {
                DOC_ORDS_PER_LEASE - issued_end
            } else {
                0
            };
            report.records.push(LeaseBurnRecord {
                shard: grant.shard,
                lease_base: grant.lease_base,
                used,
                burned,
            });
            report.total_leased += u64::from(used) + u64::from(burned);
            report.total_used += u64::from(used);
            report.total_burned += u64::from(burned);
        }
        report
    }

    /// First docid of the next lease granted to `shard`. A frontier lease
    /// held by another shard first claims the rest of its block, so the new
    /// lease starts at the next block boundary; otherwise it starts exactly
    /// at the watermark.
    const fn next_grant_start(&self, shard: usize) -> u64 {
        let offset = self.next_unreserved % DOCID_LEASE_BLOCK;
        match self.frontier_shard {
            Some(frontier) if frontier != shard && offset != 0 => {
                self.next_unreserved - offset + DOCID_LEASE_BLOCK
            }
            _ => self.next_unreserved,
        }
    }

    fn ensure_usable(&self, shard: usize) -> Result<(), DocIdAllocatorError> {
        if !self.open {
            return Err(DocIdAllocatorError::SessionClosed);
        }
        if shard >= self.shard_count {
            return Err(DocIdAllocatorError::UnknownShard {
                shard,
                shard_count: self.shard_count,
            });
        }
        Ok(())
    }

    fn ensure_lease(&mut self, shard: usize) -> Result<&mut ShardLease, DocIdAllocatorError> {
        let needs_grant =
            self.leases[shard].is_none_or(|lease| lease.next_ord == DOC_ORDS_PER_LEASE);
        if needs_grant {
            self.grant_lease(shard)?;
        }
        Ok(self.leases[shard].as_mut().expect("lease just granted"))
    }

    fn grant_lease(&mut self, shard: usize) -> Result<(), DocIdAllocatorError> {
        let start = self.next_grant_start(shard);
        let offset = start % DOCID_LEASE_BLOCK;
        let base = start - offset;
        if base > u64::from(u32::MAX) - (DOCID_LEASE_BLOCK - 1) {
            return Err(DocIdAllocatorError::DocIdSpaceExhausted { base });
        }
        let first_ord =
            u32::try_from(offset).map_err(|_| DocIdAllocatorError::DocIdSpaceExhausted { base })?;
        let grant = LeaseGrant {
            shard,
            lease_base: base,
            first_ord,
            grant_seq: self.grant_seq,
        };
        self.grant_seq += 1;
        self.next_unreserved = start;
        self.frontier_shard = Some(shard);
        self.leases[shard] = Some(ShardLease {
            base_docid: base,
            next_ord: first_ord,
        });
        self.grants.push(grant);
        if self.trace_grants {
            Self::trace_grant(grant);
        }
        Ok(())
    }

    fn trace_grant(grant: LeaseGrant) {
        tracing::trace!(
            target: crate::tracing_conventions::TARGET,
            phase = "scribe.docid_lease_grant",
            shard_id = grant.shard,
            grant_seq = grant.grant_seq,
            lease_base = grant.lease_base,
            first_ord = grant.first_ord,
            "docid lease granted"
        );
    }
}

/// Typed failures from [`ShardRouter`].
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ShardRouterError {
    /// A router must have at least one shard.
    #[error("shard count must be at least one")]
    ZeroShards,
    /// The requested shard count exceeds the configured clamp.
    #[error("shard count {requested} exceeds the configured maximum {max}")]
    TooManyShards {
        /// Requested shard count.
        requested: usize,
        /// Configured `max_ingest_shards` clamp.
        max: usize,
    },
}

/// Round-robin-by-batch shard router (Q1 §7, plan §5).
///
/// Whole batches of `IndexableDocument`s fan across `W` shard workers —
/// routing is by batch, never by document hash, so each batch keeps source
/// locality and its docids stay contiguous inside the shard's lease. In
/// `deterministic_ingest` mode the config resolves the shard count to one, so
/// docid assignment is a pure function of ingest order (`LabRuntime` suites and
/// replay rely on this).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardRouter {
    shard_count: usize,
    next_shard: usize,
}

impl ShardRouter {
    /// Build a router over `shard_count` workers, clamped by the configured
    /// `max_ingest_shards`.
    ///
    /// # Errors
    ///
    /// Returns [`ShardRouterError::ZeroShards`] for zero shards and
    /// [`ShardRouterError::TooManyShards`] above the configured clamp.
    pub fn new(shard_count: usize, max_ingest_shards: usize) -> Result<Self, ShardRouterError> {
        if shard_count == 0 {
            return Err(ShardRouterError::ZeroShards);
        }
        if shard_count > max_ingest_shards {
            return Err(ShardRouterError::TooManyShards {
                requested: shard_count,
                max: max_ingest_shards,
            });
        }
        Ok(Self {
            shard_count,
            next_shard: 0,
        })
    }

    /// Build a router from engine config plus an externally detected
    /// parallelism count. `deterministic_ingest` resolves to a single shard;
    /// otherwise the count is clamped to `1..=max_ingest_shards`.
    #[must_use]
    pub fn from_config(config: &crate::config::QuillConfig, detected_parallelism: usize) -> Self {
        Self {
            shard_count: config.resolved_ingest_shards(detected_parallelism),
            next_shard: 0,
        }
    }

    /// Shard that receives the next batch; advances the round-robin cursor.
    pub fn route_batch(&mut self) -> usize {
        let shard = self.next_shard;
        self.next_shard = (self.next_shard + 1) % self.shard_count;
        shard
    }

    /// Number of shard workers batches fan across.
    #[must_use]
    pub const fn shard_count(&self) -> usize {
        self.shard_count
    }

    /// Whether routing is deterministic-by-construction (single shard), as
    /// resolved from `deterministic_ingest`.
    #[must_use]
    pub const fn is_deterministic(&self) -> bool {
        self.shard_count == 1
    }
}
