use frankensearch_core::filter::SearchFilter;
use frankensearch_core::{SearchError, VectorHit};

use crate::search::{PARALLEL_CHUNK_SIZE, SearchParams};

use super::super::*;
use super::make_normalized_vec;

#[test]
fn search_top_k_correctness() {
    let dim = 16;
    let n = 50;
    let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| make_normalized_vec(dim, i as f32 * 0.7))
        .collect();
    let query = make_normalized_vec(dim, 0.7); // should match doc-1 best

    let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();
    let hits = index.search_top_k(&query, 5, None).unwrap();

    assert_eq!(hits.len(), 5);
    // Scores should be descending
    for w in hits.windows(2) {
        assert!(w[0].score >= w[1].score, "scores not descending");
    }
    // Top hit should be doc-1 (same seed as query)
    assert_eq!(hits[0].doc_id, "doc-1");
}

#[test]
fn int8_two_pass_matches_exact_topk() {
    let dim = 32;
    let n = 200;
    let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| make_normalized_vec(dim, i as f32 * 0.31))
        .collect();
    let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();

    for qseed in [0.31_f32, 3.0, 17.5, 99.9] {
        let query = make_normalized_vec(dim, qseed);
        let exact = index.search_top_k(&query, 10, None).unwrap();
        // mult=10 -> 100 candidates of 200; pass-1 recall is 1 here, so the
        // two-pass result must be bit-identical to the exact top-k.
        let two_pass = index.search_top_k_int8_two_pass(&query, 10, 10).unwrap();

        assert_eq!(two_pass.len(), exact.len(), "qseed={qseed}");
        for w in two_pass.windows(2) {
            assert!(w[0].score >= w[1].score, "two-pass not descending");
        }
        let exact_ids: Vec<&str> = exact.iter().map(|h| h.doc_id.as_str()).collect();
        let tp_ids: Vec<&str> = two_pass.iter().map(|h| h.doc_id.as_str()).collect();
        assert_eq!(
            tp_ids, exact_ids,
            "int8 two-pass should match exact top-k at mult=10 (qseed={qseed})"
        );
        for (a, b) in two_pass.iter().zip(exact.iter()) {
            assert!((a.score - b.score).abs() < 1e-6, "scores differ");
        }
    }
}

#[test]
fn four_bit_two_pass_keep_all_matches_exact() {
    // With a multiplier large enough to retain every record, the exact f16
    // rescore must reproduce `search_top_k` bit-for-bit — verifying the nibble
    // pack/unpack, offsets, rescore, and resolve.
    let dim = 34; // odd-ish, > 32, exercises a partial last packed byte
    let n = 200;
    let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| make_normalized_vec(dim, i as f32 * 0.31))
        .collect();
    let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();

    for qseed in [0.31_f32, 3.0, 17.5, 99.9] {
        let query = make_normalized_vec(dim, qseed);
        let exact = index.search_top_k(&query, 10, None).unwrap();
        // mult=20 → candidate_count clamps to n → pass-1 retains all → identical.
        let two_pass = index.search_top_k_4bit_two_pass(&query, 10, 20).unwrap();
        let exact_ids: Vec<&str> = exact.iter().map(|h| h.doc_id.as_str()).collect();
        let tp_ids: Vec<&str> = two_pass.iter().map(|h| h.doc_id.as_str()).collect();
        assert_eq!(
            tp_ids, exact_ids,
            "4bit two-pass (keep-all) should match exact top-k (qseed={qseed})"
        );
    }
}

#[test]
fn int8_two_pass_dimension_mismatch() {
    let dim = 8;
    let doc_ids: Vec<String> = (0..4).map(|i| format!("doc-{i}")).collect();
    let vectors: Vec<Vec<f32>> = (0..4).map(|i| make_normalized_vec(dim, i as f32)).collect();
    let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();
    let err = index
        .search_top_k_int8_two_pass(&[1.0; 7], 3, 4)
        .expect_err("dimension mismatch");
    assert!(matches!(err, SearchError::DimensionMismatch { .. }));
}

#[test]
fn search_top_k_with_filter() {
    let dim = 8;
    let doc_ids: Vec<String> = (0..10).map(|i| format!("doc-{i}")).collect();
    let vectors: Vec<Vec<f32>> = (0..10)
        .map(|i| make_normalized_vec(dim, i as f32))
        .collect();
    let query = make_normalized_vec(dim, 0.0); // matches doc-0

    struct OddFilter;
    impl SearchFilter for OddFilter {
        fn matches(&self, doc_id: &str, _metadata: Option<&serde_json::Value>) -> bool {
            // Only allow odd-numbered docs
            doc_id
                .strip_prefix("doc-")
                .and_then(|n| n.parse::<usize>().ok())
                .is_some_and(|n| n % 2 == 1)
        }
        fn name(&self) -> &str {
            "odd"
        }
    }

    let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();
    let hits = index.search_top_k(&query, 5, Some(&OddFilter)).unwrap();
    assert_eq!(hits.len(), 5);
    for hit in &hits {
        let num: usize = hit.doc_id.strip_prefix("doc-").unwrap().parse().unwrap();
        assert!(num % 2 == 1, "filter should exclude even docs");
    }
}

#[test]
fn search_with_bitset_filter_uses_precomputed_hash_path() {
    // BitsetFilter resolves via matches_doc_id_hash (the precomputed-hash
    // prescreen). Result must equal the allowed doc-id set's top-k — i.e. the
    // precomputed hashes match BitsetFilter's own hashing.
    use frankensearch_core::filter::BitsetFilter;
    let dim = 8;
    let doc_ids: Vec<String> = (0..20).map(|i| format!("doc-{i}")).collect();
    let vectors: Vec<Vec<f32>> = (0..20)
        .map(|i| make_normalized_vec(dim, i as f32))
        .collect();
    let allowed: Vec<String> = doc_ids.iter().step_by(3).cloned().collect(); // doc-0,3,6,...
    let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();
    let filter = BitsetFilter::from_doc_ids(allowed.iter().cloned());

    let query = make_normalized_vec(dim, 6.0);
    let hits = index.search_top_k(&query, 20, Some(&filter)).unwrap();

    assert!(!hits.is_empty());
    for hit in &hits {
        assert!(
            allowed.iter().any(|a| a.as_str() == hit.doc_id.as_str()),
            "bitset filter must only return allowed doc-ids; got {}",
            hit.doc_id
        );
    }
    // Every allowed doc should be returned (limit 20 ≥ allowed count).
    assert_eq!(hits.len(), allowed.len());
}

#[test]
fn int8_two_pass_filtered_matches_exact_filtered() {
    // The filtered int8 two-pass must return the same top-k as the exact
    // filtered scan (pass-1 pre-screens by the same doc_id hash; lossless when
    // pass-1 retains the true filtered top-k at a generous multiplier).
    use frankensearch_core::filter::BitsetFilter;
    let dim = 16;
    let doc_ids: Vec<String> = (0..200).map(|i| format!("doc-{i:04}")).collect();
    let vectors: Vec<Vec<f32>> = (0..200)
        .map(|i| make_normalized_vec(dim, i as f32))
        .collect();
    let allowed: Vec<String> = doc_ids.iter().step_by(2).cloned().collect();
    let filter = BitsetFilter::from_doc_ids(allowed.iter().cloned());
    let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();

    for qseed in [3.0_f32, 17.0, 88.0] {
        let query = make_normalized_vec(dim, qseed);
        let exact: Vec<String> = index
            .search_top_k(&query, 10, Some(&filter))
            .unwrap()
            .into_iter()
            .map(|h| h.doc_id.to_string())
            .collect();
        let two_pass: Vec<String> = index
            .search_top_k_int8_two_pass_filtered(&query, 10, 10, Some(&filter))
            .unwrap()
            .into_iter()
            .map(|h| h.doc_id.to_string())
            .collect();
        // Only allowed docs, and identical to the exact filtered top-k.
        for id in &two_pass {
            assert!(allowed.contains(id), "two-pass returned filtered-out {id}");
        }
        assert_eq!(
            two_pass, exact,
            "filtered two-pass != exact (qseed={qseed})"
        );
    }
}

#[test]
fn selective_filter_gather_matches_scan() {
    // A selective hash-addressable filter takes the gather fast-path through
    // `search_top_k`/the two-pass filtered fns; it must be bit-identical to the
    // forced per-document filtered scan (same passing set, `(score,index)` order).
    use frankensearch_core::filter::BitsetFilter;
    let dim = 16;
    let doc_ids: Vec<String> = (0..500).map(|i| format!("doc-{i:04}")).collect();
    let vectors: Vec<Vec<f32>> = (0..500)
        .map(|i| make_normalized_vec(dim, i as f32))
        .collect();
    // ~5% allow-set (well under the selectivity gate) → gather path is taken.
    let allowed: Vec<String> = doc_ids.iter().step_by(20).cloned().collect();
    let filter = BitsetFilter::from_doc_ids(allowed.iter().cloned());
    let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();

    let ids = |hits: Vec<VectorHit>| -> Vec<String> {
        hits.into_iter().map(|h| h.doc_id.to_string()).collect()
    };
    for qseed in [1.0_f32, 42.0, 313.0] {
        let query = make_normalized_vec(dim, qseed);
        let scan = ids(index
            .bench_scan_filtered(&query, 10, Some(&filter))
            .unwrap());
        let gather = ids(index.bench_gather_filtered(&query, 10, &filter).unwrap());
        let public = ids(index.search_top_k(&query, 10, Some(&filter)).unwrap());
        let int8 = ids(index
            .search_top_k_int8_two_pass_filtered(&query, 10, 3, Some(&filter))
            .unwrap());
        let fourbit = ids(index
            .search_top_k_4bit_two_pass_filtered(&query, 10, 3, Some(&filter))
            .unwrap());
        for id in &gather {
            assert!(allowed.contains(id), "gather returned filtered-out {id}");
        }
        assert_eq!(gather, scan, "gather != scan (qseed={qseed})");
        assert_eq!(public, scan, "search_top_k gather != scan (qseed={qseed})");
        assert_eq!(
            int8, scan,
            "int8 two-pass gather != exact scan (qseed={qseed})"
        );
        assert_eq!(
            fourbit, scan,
            "4bit two-pass gather != exact scan (qseed={qseed})"
        );
    }
}

#[test]
fn parallel_gather_matches_scan() {
    // Allow-set larger than PARALLEL_CHUNK_SIZE → the gather runs its parallel
    // per-chunk-heap + merge path, which must stay bit-identical to the scan.
    use frankensearch_core::filter::BitsetFilter;
    let dim = 16;
    let n = 3000; // > PARALLEL_CHUNK_SIZE allow-set below forces the parallel path
    let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i:05}")).collect();
    let vectors: Vec<Vec<f32>> = (0..n).map(|i| make_normalized_vec(dim, i as f32)).collect();
    // ~half the corpus (1500 > 1024 chunk size) → parallel gather.
    let allowed: Vec<String> = doc_ids.iter().step_by(2).cloned().collect();
    assert!(
        allowed.len() > PARALLEL_CHUNK_SIZE,
        "must exceed chunk size"
    );
    let filter = BitsetFilter::from_doc_ids(allowed.iter().cloned());
    let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();
    let ids = |hits: Vec<VectorHit>| -> Vec<String> {
        hits.into_iter().map(|h| h.doc_id.to_string()).collect()
    };
    for qseed in [2.0_f32, 99.0, 1234.0] {
        let query = make_normalized_vec(dim, qseed);
        let scan = ids(index
            .bench_scan_filtered(&query, 25, Some(&filter))
            .unwrap());
        let gather = ids(index.bench_gather_filtered(&query, 25, &filter).unwrap());
        assert_eq!(gather, scan, "parallel gather != scan (qseed={qseed})");
    }
}

#[test]
fn search_empty_index() {
    let index = InMemoryVectorIndex::from_vectors(Vec::new(), Vec::new(), 4).unwrap();
    let hits = index.search_top_k(&[0.0, 0.0, 0.0, 0.0], 10, None).unwrap();
    assert!(hits.is_empty());
}

#[test]
fn search_dimension_mismatch() {
    let index =
        InMemoryVectorIndex::from_vectors(vec!["a".into()], vec![make_normalized_vec(4, 1.0)], 4)
            .unwrap();
    let result = index.search_top_k(&[1.0, 0.0], 10, None); // dim 2 != 4
    assert!(result.is_err());
}

#[test]
fn f16_precision_tolerance() {
    let dim = 256;
    let v = make_normalized_vec(dim, 42.0);
    let index =
        InMemoryVectorIndex::from_vectors(vec!["test".into()], vec![v.clone()], dim).unwrap();

    // Self-similarity should be ~1.0 (within f16 precision)
    let hits = index.search_top_k(&v, 1, None).unwrap();
    assert_eq!(hits.len(), 1);
    assert!(
        (hits[0].score - 1.0).abs() < 0.001,
        "f16 self-similarity should be within 0.001 of 1.0, got {}",
        hits[0].score
    );
}

#[test]
fn vector_at_f32_roundtrip() {
    let dim = 8;
    let original = make_normalized_vec(dim, 5.0);
    let index =
        InMemoryVectorIndex::from_vectors(vec!["a".into()], vec![original.clone()], dim).unwrap();
    let recovered = index.vector_at_f32(0).unwrap();
    assert_eq!(recovered.len(), dim);
    for (orig, rec) in original.iter().zip(recovered.iter()) {
        assert!((orig - rec).abs() < 0.002, "f16 round-trip error too large");
    }
}

#[test]
fn two_tier_search_fast() {
    let dim = 8;
    let n = 20;
    let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
    let vectors: Vec<Vec<f32>> = (0..n).map(|i| make_normalized_vec(dim, i as f32)).collect();
    let query = make_normalized_vec(dim, 5.0);

    let fast = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();
    let two_tier = InMemoryTwoTierIndex::new(fast, None);

    assert!(!two_tier.has_quality_index());
    assert_eq!(two_tier.doc_count(), 20);

    let hits = two_tier.search_fast(&query, 5).unwrap();
    assert_eq!(hits.len(), 5);
    assert_eq!(hits[0].doc_id, "doc-5");
}

#[test]
fn two_tier_quality_scores() {
    let dim_fast = 8;
    let dim_quality = 16;
    let n = 10;

    let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
    let fast_vecs: Vec<Vec<f32>> = (0..n)
        .map(|i| make_normalized_vec(dim_fast, i as f32))
        .collect();
    let quality_vecs: Vec<Vec<f32>> = (0..n)
        .map(|i| make_normalized_vec(dim_quality, i as f32 * 0.5))
        .collect();

    let fast = InMemoryVectorIndex::from_vectors(doc_ids.clone(), fast_vecs, dim_fast).unwrap();
    let quality = InMemoryVectorIndex::from_vectors(doc_ids, quality_vecs, dim_quality).unwrap();

    let two_tier = InMemoryTwoTierIndex::new(fast, Some(quality));
    assert!(two_tier.has_quality_index());

    let fast_query = make_normalized_vec(dim_fast, 3.0);
    let hits = two_tier.search_fast(&fast_query, 5).unwrap();

    let quality_query = make_normalized_vec(dim_quality, 1.5);
    let scores = two_tier
        .quality_scores_for_hits(&quality_query, &hits)
        .unwrap();
    assert_eq!(scores.len(), 5);
    // All scores should be Some and finite
    for s in &scores {
        assert!(
            s.is_some_and(|v| v.is_finite()),
            "quality score should be Some and finite"
        );
    }
}

#[test]
fn two_tier_no_quality_returns_nones() {
    let dim = 4;
    let fast = InMemoryVectorIndex::from_vectors(
        vec!["a".into()],
        vec![make_normalized_vec(dim, 1.0)],
        dim,
    )
    .unwrap();
    let two_tier = InMemoryTwoTierIndex::new(fast, None);

    let hits = two_tier
        .search_fast(&make_normalized_vec(dim, 1.0), 1)
        .unwrap();
    let scores = two_tier
        .quality_scores_for_hits(&make_normalized_vec(dim, 1.0), &hits)
        .unwrap();
    assert_eq!(scores, vec![None]);
}

#[test]
fn parallel_search_matches_sequential() {
    let dim = 16;
    let n = 200;
    let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| make_normalized_vec(dim, i as f32 * 0.3))
        .collect();
    let query = make_normalized_vec(dim, 7.0);

    let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();

    let seq_params = SearchParams {
        parallel_enabled: false,
        parallel_threshold: 1,
        parallel_chunk_size: 32,
    };
    let par_params = SearchParams {
        parallel_enabled: true,
        parallel_threshold: 1, // force parallel even for small index
        parallel_chunk_size: 32,
    };

    let seq_hits = index
        .search_top_k_with_params(&query, 10, None, seq_params)
        .unwrap();
    let par_hits = index
        .search_top_k_with_params(&query, 10, None, par_params)
        .unwrap();

    assert_eq!(seq_hits.len(), par_hits.len());
    for (s, p) in seq_hits.iter().zip(par_hits.iter()) {
        assert_eq!(s.doc_id, p.doc_id);
        assert!(
            (s.score - p.score).abs() < 1e-6,
            "parallel vs sequential score mismatch"
        );
    }
}
