use super::*;

use crate::segment::SegmentReader;

// ------------------------------------------------------------------
// Q1 docid allocator + shard router (bd-quill-e1-scribe-bejd.6)
// ------------------------------------------------------------------

/// Tiny deterministic xorshift64* PRNG for property-style tests (no dev
/// dependency; the seed pins the schedule).
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

#[test]
fn allocator_grants_aligned_disjoint_leases_and_tracks_watermark() {
    let mut allocator = DocIdAllocator::open(0, 2).expect("two-shard session opens");
    assert_eq!(allocator.watermark(), 0);
    let first = allocator
        .alloc_batch(0, 10)
        .expect("shard 0 batch allocates");
    let second = allocator
        .alloc_batch(1, 20)
        .expect("shard 1 batch allocates");
    assert_eq!(first.spans().len(), 1);
    assert_eq!(second.spans().len(), 1);
    let (a, b) = (first.spans()[0], second.spans()[0]);
    assert_eq!(a.lease_base, 0);
    assert_eq!(b.lease_base, DOCID_LEASE_BLOCK);
    assert_eq!(a.ord_start, 0);
    assert_eq!(b.ord_start, 0);
    // Shard 1's grant made shard 0 claim the rest of block 0, so the
    // concurrently live leases are disjoint. Shard 1 holds the frontier
    // lease, which reserves only what it issued.
    assert!(a.global_end() <= b.lease_base);
    assert_eq!(allocator.watermark(), DOCID_LEASE_BLOCK + 20);
    assert_eq!(
        allocator.lease_grants(),
        &[
            LeaseGrant {
                shard: 0,
                lease_base: 0,
                first_ord: 0,
                grant_seq: 0,
            },
            LeaseGrant {
                shard: 1,
                lease_base: DOCID_LEASE_BLOCK,
                first_ord: 0,
                grant_seq: 1,
            },
        ]
    );
}

#[test]
fn single_shard_session_opens_at_unaligned_watermark_without_burning() {
    // 70_000 sits inside block 1. The lease starts there, inside the
    // block, and reserves only what it issues.
    let mut allocator = DocIdAllocator::open(70_000, 1).expect("unaligned watermark opens");
    assert_eq!(allocator.watermark(), 70_000);
    assert_eq!(
        allocator.contiguous_capacity(0),
        2 * DOC_ORDS_PER_LEASE - 70_000
    );
    let batch = allocator.alloc_batch(0, 3).expect("batch allocates");
    let span = batch.spans()[0];
    assert_eq!(span.lease_base, DOCID_LEASE_BLOCK);
    assert_eq!(span.global_first(), 70_000);
    assert_eq!(allocator.watermark(), 70_003);
    let report = allocator.end_session();
    assert_eq!(report.total_burned, 0);
    assert_eq!(report.total_used, 3);
    assert_eq!(report.total_leased, 3);
    assert_eq!(report.final_watermark, 70_003);
    assert_eq!(
        report.records,
        [LeaseBurnRecord {
            shard: 0,
            lease_base: DOCID_LEASE_BLOCK,
            used: 3,
            burned: 0,
        }]
    );
}

#[test]
fn frontier_lease_claims_its_block_before_another_shard_grants() {
    let mut allocator = DocIdAllocator::open(70_000, 2).expect("unaligned watermark opens");
    let first = allocator.alloc_batch(0, 3).expect("shard 0 allocates");
    assert_eq!(first.spans()[0].global_first(), 70_000);
    // Shard 0 holds the frontier lease: it reserved only what it issued.
    assert_eq!(allocator.watermark(), 70_003);
    assert_eq!(allocator.contiguous_capacity(1), DOC_ORDS_PER_LEASE);
    let second = allocator.alloc_batch(1, 3).expect("shard 1 allocates");
    // Shard 1's grant made shard 0 claim the rest of block 1.
    assert_eq!(second.spans()[0].global_first(), 2 * DOCID_LEASE_BLOCK);
    assert_eq!(allocator.watermark(), 2 * DOCID_LEASE_BLOCK + 3);
    // Shard 0 keeps issuing inside its claimed block without moving the
    // watermark past shard 1's frontier lease.
    let third = allocator.alloc_batch(0, 4).expect("shard 0 continues");
    assert_eq!(third.spans()[0].global_first(), 70_003);
    assert_eq!(allocator.watermark(), 2 * DOCID_LEASE_BLOCK + 3);
    assert_eq!(
        allocator.lease_grants()[0],
        LeaseGrant {
            shard: 0,
            lease_base: DOCID_LEASE_BLOCK,
            first_ord: 70_000 - DOC_ORDS_PER_LEASE,
            grant_seq: 0,
        }
    );
    let report = allocator.end_session();
    // Only the non-frontier lease burns its claimed tail.
    assert_eq!(report.records[0].used, 7);
    assert_eq!(
        u64::from(report.records[0].burned),
        2 * DOCID_LEASE_BLOCK - 70_007
    );
    assert_eq!((report.records[1].used, report.records[1].burned), (3, 0));
    assert_eq!(report.total_used + report.total_burned, report.total_leased);
    assert_eq!(report.final_watermark, 2 * DOCID_LEASE_BLOCK + 3);
}

#[test]
fn short_multi_shard_session_through_one_shard_burns_nothing() {
    // #41: an incremental indexer publishing a few documents per session
    // used to move the watermark past a whole lease every time.
    let mut watermark = 0;
    for session in 1..=4_u64 {
        let mut allocator = DocIdAllocator::open(watermark, 8).expect("session opens");
        allocator.alloc_batch(0, 3).expect("one small batch");
        let report = allocator.end_session();
        assert_eq!(report.total_burned, 0);
        assert_eq!(report.final_watermark, 3 * session);
        watermark = report.final_watermark;
    }
}

#[test]
fn consecutive_single_shard_sessions_issue_contiguous_docids() {
    // bd-k07zw: every short writer session (watch publication, delete,
    // append) used to start a fresh 65,536-docid block, and merges
    // materialized each burned tail as dense DOCLEN/IDMAP holes.
    let mut watermark = 2_017;
    for _ in 0..5 {
        let mut allocator = DocIdAllocator::open(watermark, 1).expect("session opens");
        let docid = allocator.alloc_one(0).expect("one upsert");
        assert_eq!(
            docid, watermark,
            "the session must continue at the frontier"
        );
        let report = allocator.end_session();
        assert_eq!(report.total_burned, 0);
        assert_eq!(report.final_watermark, watermark + 1);
        watermark = report.final_watermark;
    }
    assert_eq!(watermark, 2_022);
}

#[test]
fn allocator_reuses_live_lease_across_watch_mode_batches() {
    let mut allocator = DocIdAllocator::open(0, 1).expect("session opens");
    let first = allocator.alloc_batch(0, 7).expect("first batch");
    let second = allocator.alloc_batch(0, 9).expect("second batch");
    assert_eq!(first.spans()[0].lease_base, second.spans()[0].lease_base);
    assert_eq!(second.spans()[0].ord_start, 7);
    assert_eq!(allocator.lease_grants().len(), 1);
    assert_eq!(allocator.watermark(), 16);
}

#[test]
fn r1_batch_crossing_lease_boundary_cuts_into_fresh_lease() {
    let mut allocator = DocIdAllocator::open(0, 1).expect("session opens");
    let fill = DOC_ORDS_PER_LEASE - 6;
    let head = allocator.alloc_batch(0, fill).expect("head fills lease");
    assert_eq!(head.spans()[0].len, fill);
    let crossing = allocator
        .alloc_batch(0, 10)
        .expect("crossing batch allocates");
    assert!(crossing.crossed_lease());
    assert_eq!(crossing.len(), 10);
    let spans = crossing.spans();
    assert_eq!(spans.len(), 2);
    // Span 1 finishes the current lease; span 2 opens a fresh lease at
    // ordinal zero, so the driver can seal between them (R1).
    assert_eq!(spans[0].lease_base, 0);
    assert_eq!((spans[0].ord_start, spans[0].len), (fill, 6));
    assert_eq!(spans[0].global_end(), DOCID_LEASE_BLOCK);
    assert_eq!(spans[1].lease_base, DOCID_LEASE_BLOCK);
    assert_eq!((spans[1].ord_start, spans[1].len), (0, 4));
    // No span ever straddles a block boundary.
    for span in spans {
        assert_eq!(
            span.lease_base / DOCID_LEASE_BLOCK,
            (span.global_end() - 1) / DOCID_LEASE_BLOCK
        );
    }
}

#[test]
fn lease_disjointness_under_concurrent_sessions_property() {
    let mut rng = XorShift(0x5EED_5EED_5EED_5EED);
    let mut watermark = 0;
    let mut handed_out = std::collections::HashSet::new();
    let mut total_allocated = 0_u64;
    for session in 0..8_u64 {
        let shard_count = 1 + usize::try_from(rng.below(4)).unwrap();
        let mut allocator = DocIdAllocator::open(watermark, shard_count).expect("session opens");
        let mut session_allocated = 0_u64;
        // Simulate concurrent shard sessions holding leases at once:
        // interleave batches across shards without ending any lease.
        for _round in 0..40 {
            let shard = usize::try_from(rng.below(shard_count as u64)).unwrap();
            let count = 1 + u32::try_from(rng.below(2_000)).unwrap();
            let batch = allocator
                .alloc_batch(shard, count)
                .expect("batch allocates");
            for span in batch.spans() {
                for offset in 0..u64::from(span.len) {
                    let docid = span.global_first() + offset;
                    assert!(
                        handed_out.insert(docid),
                        "session {session} reissued docid {docid}"
                    );
                    total_allocated += 1;
                    session_allocated += 1;
                }
            }
        }
        // Every grant inside one session lies in its own block, at or
        // above the watermark the session opened from.
        let grants = allocator.lease_grants();
        for (i, a) in grants.iter().enumerate() {
            assert!(
                a.first_docid() >= watermark,
                "session {session} grant {a:?}"
            );
            for b in &grants[i + 1..] {
                assert_ne!(
                    a.lease_base, b.lease_base,
                    "session {session} grants overlap: {a:?} vs {b:?}"
                );
            }
        }
        // Simultaneously live leases never share a block.
        let live: Vec<u64> = (0..shard_count)
            .filter_map(|shard| allocator.live_lease(shard).map(|(base, _)| base))
            .collect();
        let mut unique = live.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(live.len(), unique.len(), "live leases overlap");
        let report = allocator.end_session();
        assert_eq!(report.total_used, session_allocated);
        assert_eq!(report.total_used + report.total_burned, report.total_leased);
        // Only non-frontier live leases burn, at most one tail each.
        let burning_leases = live.len().saturating_sub(1) as u64;
        assert!(report.total_burned <= burning_leases * (DOCID_LEASE_BLOCK - 1));
        if shard_count == 1 {
            assert_eq!(
                report.total_burned, 0,
                "the frontier lease reserves on issue"
            );
            assert_eq!(report.final_watermark, watermark + session_allocated);
        }
        assert!(report.final_watermark >= watermark);
        watermark = report.final_watermark;
    }
    assert_eq!(handed_out.len() as u64, total_allocated);
}

#[test]
fn burn_on_end_accounting_and_burned_tail_never_reused() {
    let mut allocator = DocIdAllocator::open(0, 2).expect("session opens");
    allocator.alloc_batch(0, 10).expect("shard 0 partial lease");
    allocator
        .alloc_batch(1, DOC_ORDS_PER_LEASE)
        .expect("shard 1 fills a lease");
    allocator
        .alloc_batch(1, 5)
        .expect("shard 1 opens a second lease");
    let report = allocator.end_session();
    assert_eq!(report.records.len(), 3);
    assert_eq!(
        (report.records[0].used, report.records[0].burned),
        (10, DOC_ORDS_PER_LEASE - 10)
    );
    assert_eq!(
        (report.records[1].used, report.records[1].burned),
        (DOC_ORDS_PER_LEASE, 0)
    );
    // Shard 1's second lease is the frontier lease: it burns nothing.
    assert_eq!((report.records[2].used, report.records[2].burned), (5, 0));
    assert_eq!(report.total_leased, 2 * DOCID_LEASE_BLOCK + 5);
    assert_eq!(report.total_used, DOCID_LEASE_BLOCK + 15);
    assert_eq!(report.total_burned, u64::from(DOC_ORDS_PER_LEASE - 10));
    assert_eq!(report.final_watermark, 2 * DOCID_LEASE_BLOCK + 5);

    // The next session opens at the watermark: shard 0's burned tail is
    // gone, and every docid it issues lies above everything issued here.
    let mut next = DocIdAllocator::open(report.final_watermark, 2).expect("next session");
    let batch = next.alloc_batch(0, 1).expect("next session allocates");
    assert_eq!(batch.spans()[0].global_first(), 2 * DOCID_LEASE_BLOCK + 5);
}

#[test]
fn upsert_path_allocates_fresh_monotone_docids() {
    let mut allocator = DocIdAllocator::open(0, 1).expect("session opens");
    let original = allocator.alloc_one(0).expect("original docid");
    let replacement = allocator.alloc_one(0).expect("upsert docid");
    assert!(replacement > original);
    assert_eq!(replacement, original + 1);
}

#[test]
fn closed_session_and_invalid_requests_are_typed_errors() {
    let mut allocator = DocIdAllocator::open(0, 1).expect("session opens");
    assert_eq!(
        allocator.alloc_batch(0, 0),
        Err(DocIdAllocatorError::EmptyAllocation)
    );
    assert_eq!(
        allocator.alloc_batch(7, 1),
        Err(DocIdAllocatorError::UnknownShard {
            shard: 7,
            shard_count: 1,
        })
    );
    let _report = allocator.end_session();
    assert_eq!(
        allocator.alloc_batch(0, 1),
        Err(DocIdAllocatorError::SessionClosed)
    );
    assert_eq!(
        DocIdAllocator::open(0, 0).map(|a| a.watermark()),
        Err(DocIdAllocatorError::ZeroShards)
    );
}

#[test]
fn docid_space_exhaustion_is_a_typed_error() {
    // Largest legal aligned base: block starting at u32::MAX - 65_535.
    let last_base = u64::from(u32::MAX) - (DOCID_LEASE_BLOCK - 1);
    assert_eq!(last_base % DOCID_LEASE_BLOCK, 0);
    let mut allocator = DocIdAllocator::open(last_base, 2).expect("final block opens");
    allocator
        .alloc_batch(0, 1)
        .expect("last legal lease grants");
    assert_eq!(
        allocator.alloc_batch(1, 1),
        Err(DocIdAllocatorError::DocIdSpaceExhausted {
            base: last_base + DOCID_LEASE_BLOCK,
        })
    );
    assert_eq!(allocator.contiguous_capacity(1), 0);

    // A single shard issues up to the very last u32 docid.
    let last_docid = u64::from(u32::MAX);
    let mut single = DocIdAllocator::open(last_docid, 1).expect("last docid opens");
    assert_eq!(single.contiguous_capacity(0), 1);
    assert_eq!(single.alloc_one(0), Ok(last_docid));
    assert_eq!(single.contiguous_capacity(0), 0);
    assert_eq!(
        single.alloc_one(0),
        Err(DocIdAllocatorError::DocIdSpaceExhausted {
            base: last_docid + 1,
        })
    );
}

#[test]
fn router_cycles_batches_round_robin() {
    let mut router = ShardRouter::new(3, 8).expect("three-shard router");
    let order: Vec<usize> = (0..7).map(|_| router.route_batch()).collect();
    assert_eq!(order, [0, 1, 2, 0, 1, 2, 0]);
    assert!(!router.is_deterministic());
}

#[test]
fn router_config_resolution_and_clamps() {
    let config = crate::config::QuillConfig::default();
    let router = ShardRouter::from_config(&config, 128);
    assert_eq!(router.shard_count(), config.max_ingest_shards);

    let deterministic = crate::config::QuillConfig {
        deterministic_ingest: true,
        ..crate::config::QuillConfig::default()
    };
    let mut router = ShardRouter::from_config(&deterministic, 16);
    assert!(router.is_deterministic());
    assert_eq!(router.shard_count(), 1);
    assert_eq!(router.route_batch(), 0);
    assert_eq!(router.route_batch(), 0);

    assert_eq!(
        ShardRouter::new(0, 8).map(|router| router.shard_count()),
        Err(ShardRouterError::ZeroShards)
    );
    assert_eq!(
        ShardRouter::new(9, 8).map(|router| router.shard_count()),
        Err(ShardRouterError::TooManyShards {
            requested: 9,
            max: 8,
        })
    );
}

/// Drive a fixed document stream through router + allocator + accumulator
/// + flush, returning the sealed bytes and the lease-grant witness log.
fn replay_deterministic_segments() -> (Vec<u8>, Vec<LeaseGrant>) {
    let docs: Vec<(String, String)> = (0..48)
        .map(|i| {
            (
                format!("doc-{i:03}"),
                format!("alpha beta gamma delta document number {i}"),
            )
        })
        .collect();
    let config = crate::config::QuillConfig {
        deterministic_ingest: true,
        ..crate::config::QuillConfig::default()
    };
    let mut router = ShardRouter::from_config(&config, 8);
    let mut allocator = DocIdAllocator::open(0, router.shard_count()).expect("allocator opens");
    let mut accumulator = ColumnarAccumulator::new(MIXED_POSITION_SCHEMA).expect("valid schema");
    let mut identities = Vec::new();
    // Arbitrary batching must not perturb docid assignment in
    // deterministic mode: route three uneven batches.
    for batch in [&docs[..10], &docs[10..37], &docs[37..]] {
        let shard = router.route_batch();
        let ids = allocator
            .alloc_batch(shard, u32::try_from(batch.len()).unwrap())
            .expect("batch docids allocate");
        assert_eq!(ids.spans().len(), 1);
        let span = ids.spans()[0];
        for (offset, (id, content)) in batch.iter().enumerate() {
            let doc_ord = span.ord_start + u32::try_from(offset).unwrap();
            accumulator
                .add_document_with_stored(
                    doc_ord,
                    &[
                        IndexedFieldValue::new(0, id.as_str()),
                        IndexedFieldValue::new(1, content.as_str()),
                    ],
                    &[StoredFieldValue::new(3, content.as_bytes())],
                )
                .expect("document accumulates");
            identities.push(FlushDocumentInput::from_canonical_content(
                doc_ord,
                id,
                content.as_bytes(),
            ));
        }
    }
    let input = FlushSegmentInput {
        segment_id: 42,
        lease_docid_base: 0,
        created_unix_s: 1_700_000_000,
        engine_version: 0,
        documents: &identities,
    };
    let segment = flush_accumulator(&accumulator, input).expect("segment seals");
    (segment.into_bytes(), allocator.lease_grants().to_vec())
}

#[test]
fn deterministic_replay_produces_byte_identical_segments_and_grant_logs() {
    let (first_bytes, first_grants) = replay_deterministic_segments();
    let (second_bytes, second_grants) = replay_deterministic_segments();
    assert_eq!(first_grants, second_grants);
    assert_eq!(first_bytes, second_bytes);
    // Single-shard deterministic ingest assigns docids as a pure function
    // of ingest order: 48 docs from block zero.
    let reader =
        SegmentReader::from_bytes(&first_bytes, MIXED_POSITION_SCHEMA).expect("segment reopens");
    assert_eq!(reader.header().docid_lo, 0);
    assert_eq!(reader.header().docid_hi, 48);
}
