use frankensearch_core::filter::SearchFilter;
use half::f16;
use proptest::prelude::*;
use sha2::{Digest, Sha256};

use crate::search::SearchParams;
use crate::simd::dot_product_f16_f32;

#[cfg(target_os = "linux")]
use super::super::residual_io::exact_residual_sidecar_stream_matches_reader;
use super::super::residual_sidecar::{
    ExactResidualLayout, ExactResidualSidecar, ResidualPruningCensus, ResidualQueryTransform,
    ResidualSourceBinding,
};
use super::super::*;
use super::{
    assert_residual_bounds_cover_exact_scores, bind_test_residual_source, bounded_finite_f32_bits,
    finite_f16_bits, make_normalized_vec,
};
#[cfg(target_os = "linux")]
use super::{owned_temp_dir, write_new_owned_file};

#[test]
fn exact_residual_sidecar_is_exact_on_odd_tail_ties_filters_and_repeated_queries() {
    use frankensearch_core::PredicateFilter;

    let dimension = 35; // exercises the final partial 32-dimension block.
    let count = 17; // exercises the final partial eight-candidate group.
    let doc_ids: Vec<String> = (0..count).map(|index| format!("doc-{index:02}")).collect();
    let mut vectors: Vec<Vec<f32>> = (0..count)
        .map(|index| make_normalized_vec(dimension, index as f32 * 0.37))
        .collect();
    // Exact score ties must retain the incumbent index tie-break; a sidecar
    // may only abandon on a strict proven upper-bound inequality.
    vectors[1] = vectors[0].clone();
    let mut index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dimension)
        .expect("finite source vectors");
    assert!(
        index.build_exact_residual_sidecar().is_err(),
        "caller-built vectors have no FSVI v2 generation witness"
    );
    index.residual_source_binding = Some(ResidualSourceBinding {
        generation_fingerprint: [0x11; 32],
        vector_content_digest: [0x22; 32],
        ordered_live_docset_digest: [0x33; 32],
        space_fingerprint: [0x44; 32],
    });
    let sidecar = index
        .build_exact_residual_sidecar()
        .expect("test-only witnessed source builds sidecar");
    let encoded = sidecar.encode().expect("encode sidecar");
    let decoded = ExactResidualSidecar::decode(&encoded).expect("sidecar round trip");
    assert!(
        decoded.is_bound_to(
            index
                .residual_source_binding
                .as_ref()
                .expect("test witness"),
            count,
            dimension,
        )
    );
    let mut corrupt = encoded.clone();
    corrupt[8] ^= 0x01; // corrupt the fixed-width schema version.
    assert!(
        ExactResidualSidecar::decode(&corrupt).is_err(),
        "a corrupted sidecar is rejected before it can affect search"
    );

    let flat = index.clone();
    assert!(
        index
            .admit_exact_residual_sidecar(&decoded)
            .expect("admit source-derived sidecar")
    );
    let filter = PredicateFilter::new("even ids", |id| {
        id.strip_prefix("doc-")
            .and_then(|suffix| suffix.parse::<usize>().ok())
            .is_some_and(|value| value % 2 == 0)
    });
    let queries = [
        make_normalized_vec(dimension, 0.0),
        make_normalized_vec(dimension, 3.1),
        make_normalized_vec(dimension, 7.9),
    ];
    for query in &queries {
        for &limit in &[0, 1, count, count + 7] {
            for filter in [None, Some(&filter as &dyn SearchFilter)] {
                let expected = flat
                    .search_top_k(query, limit, filter)
                    .expect("flat search");
                let actual = index
                    .search_top_k(query, limit, filter)
                    .expect("sidecar search");
                assert_eq!(actual.len(), expected.len());
                for (actual_hit, expected_hit) in actual.iter().zip(&expected) {
                    assert_eq!(actual_hit.doc_id, expected_hit.doc_id);
                    assert_eq!(actual_hit.index, expected_hit.index);
                    assert_eq!(actual_hit.score.to_bits(), expected_hit.score.to_bits());
                }
            }
        }
        let first = index.search_top_k(query, 5, None).expect("first search");
        let second = index.search_top_k(query, 5, None).expect("repeat search");
        assert_eq!(first, second, "sidecar search is repeatable");
    }
    let parallel = SearchParams {
        parallel_enabled: true,
        parallel_threshold: 1,
        parallel_chunk_size: 4,
    };
    let query = make_normalized_vec(dimension, 5.5);
    let expected = flat
        .search_top_k_with_params(&query, 5, None, parallel)
        .expect("parallel flat search");
    let actual = index
        .search_top_k_with_params(&query, 5, None, parallel)
        .expect("parallel sidecar fallback");
    assert_eq!(actual.len(), expected.len());
    for (actual_hit, expected_hit) in actual.iter().zip(&expected) {
        assert_eq!(actual_hit.doc_id, expected_hit.doc_id);
        assert_eq!(actual_hit.index, expected_hit.index);
        assert_eq!(actual_hit.score.to_bits(), expected_hit.score.to_bits());
    }
}

#[test]
fn exact_residual_preparation_failures_keep_public_search_flat_exact() {
    let dimension = 35;
    let count = 17;
    let mut cached = InMemoryVectorIndex::from_vectors(
        (0..count).map(|row| format!("fallback-{row}")).collect(),
        (0..count)
            .map(|row| make_normalized_vec(dimension, row as f32 + 0.75))
            .collect(),
        dimension,
    )
    .expect("create finite source");
    bind_test_residual_source(&mut cached);
    let uncached = cached.clone();
    let sidecar = cached
        .build_exact_residual_sidecar()
        .expect("build witnessed sidecar");
    assert!(
        cached
            .admit_exact_residual_sidecar(&sidecar)
            .expect("admit verified sidecar")
    );
    let query = make_normalized_vec(dimension, 4.25);
    let modes = [
        SearchParams {
            parallel_enabled: false,
            parallel_threshold: 1,
            parallel_chunk_size: 8,
        },
        SearchParams {
            parallel_enabled: true,
            parallel_threshold: 1,
            parallel_chunk_size: 8,
        },
    ];

    // A named allocation-failure injection site: the hook that arms the
    // next failure, paired with the predicate that reports whether it is
    // still pending.
    type AllocationInjection = (&'static str, fn(), fn() -> bool);

    let injections: [AllocationInjection; 2] = [
        (
            "transform",
            fail_next_residual_query_transform_allocation,
            residual_query_transform_allocation_failure_is_pending,
        ),
        (
            "suffix",
            fail_next_residual_query_suffix_allocation,
            residual_query_suffix_allocation_failure_is_pending,
        ),
    ];
    for params in modes {
        let expected = uncached
            .search_top_k_with_params(&query, 3, None, params)
            .expect("uncached public exact search");
        for (site, inject, is_pending) in injections {
            inject();
            let actual = cached
                .search_top_k_with_params(&query, 3, None, params)
                .expect("cache-only preparation failure must not escape public search");
            assert!(
                !is_pending(),
                "{site} injection must be consumed by the public cached route"
            );
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(&expected) {
                assert_eq!(actual.doc_id, expected.doc_id, "site={site}");
                assert_eq!(actual.index, expected.index, "site={site}");
                assert_eq!(
                    actual.score.to_bits(),
                    expected.score.to_bits(),
                    "site={site}"
                );
            }
        }
    }
}

#[test]
fn exact_residual_sidecar_rejects_finite_transformed_payload_mutations() {
    // The first group establishes a low exact cutoff.  The second group
    // contains the true winner, so a forged zero residual envelope would
    // incorrectly abandon it if transformed data were admitted on header
    // identity alone.
    let dimension = 2;
    let mut vectors = vec![vec![0.1, 0.0]; 16];
    vectors[8] = vec![1.0, 0.0];
    let mut index = InMemoryVectorIndex::from_vectors(
        (0..16).map(|row| format!("doc-{row:02}")).collect(),
        vectors,
        dimension,
    )
    .expect("finite source vectors");
    index.residual_source_binding = Some(ResidualSourceBinding {
        generation_fingerprint: [0x71; 32],
        vector_content_digest: [0x72; 32],
        ordered_live_docset_digest: [0x73; 32],
        space_fingerprint: [0x74; 32],
    });
    let expected_sidecar = index
        .build_exact_residual_sidecar()
        .expect("build source-derived sidecar");
    let layout = ExactResidualLayout::for_shape(index.record_count(), dimension)
        .expect("bounded test layout");
    let encoded = expected_sidecar.encode().expect("encode sidecar");

    // A finite corruption in every transformed payload family must fail the
    // whole-sidecar SHA-256 check before decode allocates its vectors.
    let permutation_start = EXACT_RESIDUAL_SIDECAR_HEADER_BYTES;
    let centroids_start = permutation_start + layout.permutation_len * 4;
    let residuals_start = centroids_start + layout.centroid_len * 4;
    let suffixes_start = residuals_start + layout.residual_len * 4;
    let second_group_residual = residuals_start + dimension * EXACT_RESIDUAL_LANES * 4;
    let second_group_suffix = suffixes_start + EXACT_RESIDUAL_LANES * 2 * 4;
    for offset in [
        permutation_start,
        second_group_residual,
        second_group_suffix,
    ] {
        let mut corrupt = encoded.clone();
        if offset == permutation_start {
            corrupt.swap(offset, offset + 4);
        } else {
            corrupt[offset..offset + 4].copy_from_slice(&0.0_f32.to_le_bytes());
        }
        assert!(
            ExactResidualSidecar::decode(&corrupt).is_err(),
            "digest must reject finite transformed corruption at byte {offset}"
        );
    }
    let mut oversized_header = encoded[..EXACT_RESIDUAL_SIDECAR_HEADER_BYTES].to_vec();
    let dimension_offset = 8 + 4 + 4 * 32 + 8;
    oversized_header[dimension_offset..dimension_offset + 8].copy_from_slice(
        &u64::try_from(EXACT_RESIDUAL_SIDECAR_MAX_DIMENSION + 1)
            .expect("test dimension fits u64")
            .to_le_bytes(),
    );
    assert!(
        ExactResidualSidecar::decode(&oversized_header).is_err(),
        "an oversized declared transform is rejected before decode allocation"
    );

    // A writer who recomputes the digest still cannot attach a finite
    // forged transform: admission deterministically re-derives it from the
    // admitted f16 source and compares every float bit and permutation.
    let mut forged = expected_sidecar.clone();
    forged.permutation.swap(0, 1);
    for value in &mut forged.centroids[dimension..] {
        *value = 0.0;
    }
    for value in &mut forged.residuals[dimension * EXACT_RESIDUAL_LANES..] {
        *value = 0.0;
    }
    for value in &mut forged.suffix_norms[EXACT_RESIDUAL_LANES * 2..] {
        *value = 0.0;
    }
    for value in &mut forged.correction_norms[EXACT_RESIDUAL_LANES..] {
        *value = 0.0;
    }
    let forged = ExactResidualSidecar::decode(
        &forged
            .encode()
            .expect("re-encode a finite, self-consistent forgery"),
    )
    .expect("a rehashed forgery is structurally decodable");
    assert!(
        !forged.exactly_matches_derived(&expected_sidecar),
        "finite transformed contents differ from the admitted derivation"
    );
    let query = [1.0, 0.0];
    let expected = index.search_top_k(&query, 1, None).expect("flat result");
    let forged_outcome = index
        .scan_exact_residual_sidecar(&query, 1, None, &forged)
        .expect("demonstrate the otherwise-dangerous forged scan");
    let forged_hits = index
        .resolve_heap(forged_outcome.heap)
        .expect("resolve forged scan");
    assert_eq!(
        forged_hits, expected,
        "a transformed forgery must not change the exact winner: preparation failure falls back to the incumbent flat scan"
    );
    assert!(
        !index
            .admit_exact_residual_sidecar(&forged)
            .expect("reject rehashed transformed forgery")
    );
    assert!(index.exact_residual_sidecar.get().is_none());
}

#[test]
fn exact_residual_sidecar_keeps_empty_and_nonfinite_query_contracts() {
    let dimension = 3;
    let mut empty = InMemoryVectorIndex::from_vectors(Vec::new(), Vec::new(), dimension)
        .expect("empty index is valid");
    empty.residual_source_binding = Some(ResidualSourceBinding {
        generation_fingerprint: [0x81; 32],
        vector_content_digest: [0x82; 32],
        ordered_live_docset_digest: [0x83; 32],
        space_fingerprint: [0x84; 32],
    });
    let sidecar = empty
        .build_exact_residual_sidecar()
        .expect("empty exact sidecar is structurally valid");
    assert!(
        empty
            .admit_exact_residual_sidecar(&sidecar)
            .expect("admit empty source-derived sidecar")
    );
    assert!(
        empty
            .search_top_k(&[1.0, 0.0, 0.0], 1, None)
            .expect("empty sidecar search")
            .is_empty()
    );
    let nonfinite = [f32::NAN, 0.0, 0.0];
    assert!(
        empty.search_top_k_classified(&nonfinite, 1, None).is_err(),
        "the classified entry point rejects non-finite queries before any sidecar route"
    );
}

#[test]
fn exact_residual_sidecar_corruption_and_source_mismatch_leave_flat_fallback_selected() {
    let dimension = 9;
    let mut index = InMemoryVectorIndex::from_vectors(
        vec!["a".into(), "b".into(), "c".into()],
        vec![
            make_normalized_vec(dimension, 1.0),
            make_normalized_vec(dimension, 2.0),
            make_normalized_vec(dimension, 3.0),
        ],
        dimension,
    )
    .expect("finite vectors");
    let query = make_normalized_vec(dimension, 1.5);
    let expected = index.search_top_k(&query, 2, None).expect("flat fallback");
    index.residual_source_binding = Some(ResidualSourceBinding {
        generation_fingerprint: [0x51; 32],
        vector_content_digest: [0x52; 32],
        ordered_live_docset_digest: [0x53; 32],
        space_fingerprint: [0x54; 32],
    });
    let mut foreign = index
        .build_exact_residual_sidecar()
        .expect("build sidecar")
        .encode()
        .expect("encode sidecar");
    // The first source digest starts immediately after magic/version and the
    // four source digests are admission-critical, not advisory metadata.
    foreign[12 + 32] ^= 0x80;
    let digest_start = foreign.len() - EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES;
    let digest = Sha256::digest(&foreign[..digest_start]);
    foreign[digest_start..].copy_from_slice(&digest);
    let foreign = ExactResidualSidecar::decode(&foreign).expect("well-formed foreign sidecar");
    assert!(
        !foreign.is_bound_to(
            index
                .residual_source_binding
                .as_ref()
                .expect("source binding"),
            index.record_count(),
            dimension,
        ),
        "a sidecar from another generation cannot be attached"
    );
    assert!(
        !index
            .admit_exact_residual_sidecar(&foreign)
            .expect("foreign sidecar is rejected after re-derivation")
    );
    assert!(index.exact_residual_sidecar.get().is_none());
    let actual = index
        .search_top_k(&query, 2, None)
        .expect("flat remains selected");
    assert_eq!(actual, expected);
}

#[test]
fn exact_residual_upper_bound_covers_f16_extremes_partial_tails_and_signed_zero() {
    // Two partial shapes (35 dimensions and 17 rows) force both a tail
    // residual block and a tail lane group.  The bit patterns pin the
    // f16-specific boundary families that are easy to lose in a generic
    // floating-point differential: max finite values, signed zeros, and
    // both signs of the smallest subnormal.
    let dimension = 35;
    let count = 17;
    let special = [
        f16::MAX.to_bits(),
        (-f16::MAX).to_bits(),
        0x0000,
        0x8000,
        0x0001,
        0x8001,
        f16::from_f32(1.0).to_bits(),
        f16::from_f32(-1.0).to_bits(),
    ];
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|row| {
            (0..dimension)
                .map(|column| finite_f16_bits(special[(row * 3 + column) % special.len()]))
                .collect()
        })
        .collect();
    let query: Vec<f32> = (0..dimension)
        .map(|column| finite_f16_bits(special[(column * 5 + 1) % special.len()]))
        .collect();
    let mut index = InMemoryVectorIndex::from_vectors(
        (0..count).map(|row| format!("edge-{row}")).collect(),
        vectors,
        dimension,
    )
    .expect("finite f16 edge source");
    assert_eq!(index.vector_slice(0)[3].to_bits(), 0x8000);
    assert_eq!(index.vector_slice(0)[4].to_bits(), 0x0001);
    assert_eq!(index.vector_slice(0)[5].to_bits(), 0x8001);
    bind_test_residual_source(&mut index);
    let sidecar = index
        .build_exact_residual_sidecar()
        .expect("bounded source-derived sidecar");
    assert_residual_bounds_cover_exact_scores(&index, &sidecar, &query);
}

#[test]
fn exact_residual_upper_bound_covers_large_finite_queries_and_tail_construction_error() {
    // A five-lane tail forces a non-power-of-two centroid division. Its
    // f32 residuals therefore exercise the f64 construction remainder
    // persisted in `correction_norms`, while the query magnitude stays
    // just inside the analytically admitted f32-flat envelope.
    let dimension = 35;
    let tail = [
        f16::MAX.to_f32(),
        1.0,
        -1.0,
        0.0,
        f16::from_bits(1).to_f32(),
    ];
    let vectors: Vec<Vec<f32>> = (0..13)
        .map(|row| {
            (0..dimension)
                .map(|column| tail[(row + column * 3) % tail.len()])
                .collect()
        })
        .collect();
    let large_component =
        f32::MAX / (f16::MAX.to_f32() * dimension as f32 * EXACT_RESIDUAL_LANES as f32);
    let query: Vec<f32> = (0..dimension)
        .map(|column| {
            if column % 2 == 0 {
                large_component
            } else {
                -large_component * 0.5
            }
        })
        .collect();
    let mut index = InMemoryVectorIndex::from_vectors(
        (0..vectors.len())
            .map(|row| format!("construction-{row}"))
            .collect(),
        vectors,
        dimension,
    )
    .expect("finite f16 construction source");
    bind_test_residual_source(&mut index);
    let sidecar = index
        .build_exact_residual_sidecar()
        .expect("construct a tail-aware sidecar");
    let transformed =
        ResidualQueryTransform::from_query(&query, &sidecar).expect("finite large transform");
    assert!(
        transformed.f32_flat_envelope_is_finite,
        "the large finite query must exercise the bound rather than its flat fallback"
    );
    assert_residual_bounds_cover_exact_scores(&index, &sidecar, &query);
}

#[test]
fn exact_residual_near_cutoff_simd_differential_keeps_the_true_winner() {
    // The first group establishes a cutoff one f16 ULP below the winner.
    // The query is scaled so the authoritative f16×f32 SIMD score is near
    // the top finite f32 range while remaining inside the sidecar's flat
    // envelope. This makes the distinction only a few hundred output ULPs
    // and exercises the real SIMD accumulation/reduction route, not a
    // scalar reference sum.
    let dimension = 35;
    let high = f16::MAX.to_f32();
    let one_f16_ulp_lower = f16::from_bits(f16::MAX.to_bits() - 1).to_f32();
    let query_component = f32::MAX / (high * dimension as f32 * 8.0);
    let query = vec![query_component; dimension];
    let mut near_cutoff = vec![high; dimension];
    near_cutoff[0] = one_f16_ulp_lower;
    let mut vectors = vec![near_cutoff.clone(); EXACT_RESIDUAL_LANES];
    vectors.push(vec![high; dimension]); // row 8, the true winner.
    vectors.extend(vec![near_cutoff; EXACT_RESIDUAL_LANES - 1]);
    // A uniform negative group gives the eliminator a real safe prune while
    // the previous group tests that the near-cutoff winner survives.
    vectors.extend(vec![vec![-high; dimension]; EXACT_RESIDUAL_LANES]);
    let mut index = InMemoryVectorIndex::from_vectors(
        (0..vectors.len())
            .map(|row| format!("simd-near-cutoff-{row}"))
            .collect(),
        vectors,
        dimension,
    )
    .expect("finite f16 SIMD differential source");
    bind_test_residual_source(&mut index);
    let flat = index.clone();
    let sidecar = index
        .build_exact_residual_sidecar()
        .expect("build witnessed SIMD sidecar");
    assert_residual_bounds_cover_exact_scores(&index, &sidecar, &query);
    assert!(
        index
            .admit_exact_residual_sidecar(&sidecar)
            .expect("admit exact SIMD sidecar")
    );

    let incumbent = dot_product_f16_f32(flat.vector_slice(0), &query)
        .expect("authoritative SIMD incumbent score");
    let winner = dot_product_f16_f32(flat.vector_slice(EXACT_RESIDUAL_LANES), &query)
        .expect("authoritative SIMD winner score");
    assert!(winner.is_finite() && winner > f32::MAX * 0.1);
    assert!(winner > incumbent, "one f16 ULP must make row 8 win");
    assert!(
        winner - incumbent <= (winner.next_up() - winner) * 512.0,
        "fixture must remain near the authoritative SIMD cutoff"
    );

    let expected = flat
        .search_top_k(&query, 1, None)
        .expect("flat SIMD exact result");
    let outcome = index
        .scan_exact_residual_sidecar(
            &query,
            1,
            None,
            index
                .exact_residual_sidecar
                .get()
                .expect("admitted SIMD sidecar"),
        )
        .expect("sidecar SIMD differential");
    assert!(
        outcome.census.lanes_pruned > 0,
        "the fixture must exercise a real sidecar prune as well as the near cutoff"
    );
    let actual = index
        .resolve_heap(outcome.heap)
        .expect("resolve sidecar SIMD result");
    assert_eq!(actual, expected);
    assert_eq!(
        actual[0].index,
        u32::try_from(EXACT_RESIDUAL_LANES).expect("residual lane count fits u32")
    );
}

proptest! {
    #[test]
    fn exact_residual_upper_bound_property_never_underestimates_across_shapes(
        dimension in 1_usize..129,
        row_bits in proptest::collection::vec(any::<u16>(), 1..65),
        query_bits in proptest::collection::vec(any::<u32>(), 128),
    ) {
        let vectors: Vec<Vec<f32>> = row_bits
            .iter()
            .enumerate()
            .map(|(row, &seed)| {
                (0..dimension)
                    .map(|column| {
                        finite_f16_bits(
                            seed.rotate_left(
                                u32::try_from((row + column) % 16)
                                    .expect("a modulo-16 rotation fits u32"),
                            )
                            .wrapping_add(
                                u16::try_from(column * 211)
                                    .expect("a bounded property column product fits u16"),
                            ),
                        )
                    })
                    .collect()
            })
            .collect();
        let query: Vec<f32> = query_bits
            .iter()
            .take(dimension)
            .enumerate()
            .map(|(column, &bits)| {
                bounded_finite_f32_bits(bits.rotate_right(
                    u32::try_from(column % 32).expect("a modulo-32 rotation fits u32"),
                ))
            })
            .collect();
        let mut index = InMemoryVectorIndex::from_vectors(
            (0..row_bits.len()).map(|row| format!("property-{row}")).collect(),
            vectors,
            dimension,
        )
        .expect("finite f16 property source");
        bind_test_residual_source(&mut index);
        let sidecar = index
            .build_exact_residual_sidecar()
            .expect("bounded property sidecar");
        assert_residual_bounds_cover_exact_scores(&index, &sidecar, &query);
    }
}

#[test]
fn exact_residual_sidecar_pruning_census_proves_honest_lane_elimination() {
    let dimension = 35;
    let count = 17;
    let mut vectors = vec![vec![0.0_f32; dimension]; count];
    for vector in vectors.iter_mut().take(EXACT_RESIDUAL_LANES) {
        vector[0] = 1.0;
    }
    for vector in vectors.iter_mut().skip(EXACT_RESIDUAL_LANES) {
        vector[0] = -1.0;
    }
    let mut index = InMemoryVectorIndex::from_vectors(
        (0..count).map(|row| format!("prune-{row}")).collect(),
        vectors,
        dimension,
    )
    .expect("finite pruning source");
    bind_test_residual_source(&mut index);
    let flat = index.clone();
    let sidecar = index
        .build_exact_residual_sidecar()
        .expect("build sidecar from admitted test source");
    assert!(
        index
            .admit_exact_residual_sidecar(&sidecar)
            .expect("admit exact sidecar")
    );
    let query = {
        let mut query = vec![0.0_f32; dimension];
        query[0] = 1.0;
        query
    };
    let outcome = index
        .scan_exact_residual_sidecar(
            &query,
            1,
            None,
            index
                .exact_residual_sidecar
                .get()
                .expect("admitted sidecar"),
        )
        .expect("exact residual scan");
    assert!(
        outcome.census.lanes_pruned > 0,
        "the honest cache prunes lanes"
    );
    assert!(
        outcome.census.exact_sidecar_scores < outcome.census.eligible_lanes,
        "pruned lanes must avoid exact f16 rescoring"
    );
    let actual = index
        .resolve_heap(outcome.heap)
        .expect("resolve pruned result");
    let expected = flat.search_top_k(&query, 1, None).expect("flat result");
    assert_eq!(actual, expected, "pruning preserves the exact result");
}

#[test]
fn exact_residual_sidecar_adaptive_fallback_is_censused_and_exact() {
    let dimension = 35;
    let count = (EXACT_RESIDUAL_ADAPTIVE_PROBE_GROUPS + 2) * EXACT_RESIDUAL_LANES;
    let mut vectors = vec![vec![0.0_f32; dimension]; count];
    for vector in &mut vectors {
        vector[0] = 1.0;
    }
    let mut index = InMemoryVectorIndex::from_vectors(
        (0..count).map(|row| format!("adaptive-{row}")).collect(),
        vectors,
        dimension,
    )
    .expect("finite non-selective source");
    bind_test_residual_source(&mut index);
    let flat = index.clone();
    let sidecar = index.build_exact_residual_sidecar().expect("build sidecar");
    assert!(
        index
            .admit_exact_residual_sidecar(&sidecar)
            .expect("admit sidecar")
    );
    let query = {
        let mut query = vec![0.0_f32; dimension];
        query[0] = 1.0;
        query
    };
    let outcome = index
        .scan_exact_residual_sidecar(
            &query,
            1,
            None,
            index
                .exact_residual_sidecar
                .get()
                .expect("admitted sidecar"),
        )
        .expect("adaptive exact scan");
    assert_eq!(outcome.census.lanes_pruned, 0, "tied rows cannot prune");
    assert_eq!(outcome.census.adaptive_fallbacks, 1);
    assert!(outcome.census.flat_fallback_rows > 0);
    assert_eq!(
        outcome.census.groups_scanned,
        EXACT_RESIDUAL_ADAPTIVE_PROBE_GROUPS
    );
    let actual = index
        .resolve_heap(outcome.heap)
        .expect("resolve adaptive fallback");
    let expected = flat.search_top_k(&query, 1, None).expect("flat result");
    assert_eq!(actual, expected);
}

#[test]
fn exact_residual_sidecar_caps_extreme_k_at_the_available_rows() {
    let dimension = 35;
    let count = 17;
    let mut index = InMemoryVectorIndex::from_vectors(
        (0..count).map(|row| format!("k-{row}")).collect(),
        (0..count)
            .map(|row| make_normalized_vec(dimension, row as f32 + 0.25))
            .collect(),
        dimension,
    )
    .expect("finite source");
    bind_test_residual_source(&mut index);
    let flat = index.clone();
    let sidecar = index.build_exact_residual_sidecar().expect("build sidecar");
    assert!(
        index
            .admit_exact_residual_sidecar(&sidecar)
            .expect("admit sidecar")
    );
    let query = make_normalized_vec(dimension, 4.0);
    let outcome = index
        .scan_exact_residual_sidecar(
            &query,
            usize::MAX,
            None,
            index
                .exact_residual_sidecar
                .get()
                .expect("admitted sidecar"),
        )
        .expect("extreme k uses a count-bounded heap");
    let actual = index
        .resolve_heap(outcome.heap)
        .expect("resolve sidecar heap");
    let expected = flat
        .search_top_k(&query, usize::MAX, None)
        .expect("count-bounded flat heap");
    assert_eq!(actual, expected);
    assert_eq!(actual.len(), count);
}

#[test]
fn exact_residual_sidecar_nonselective_parallel_probe_is_global_at_100k() {
    const COUNT: usize = 100_000;
    let dimension = 3;
    let mut index = InMemoryVectorIndex::from_vectors(
        (0..COUNT)
            .map(|row| format!("nonselective-{row}"))
            .collect(),
        vec![vec![1.0, 0.0, 0.0]; COUNT],
        dimension,
    )
    .expect("create 100k nonselective source");
    bind_test_residual_source(&mut index);
    let flat = index.clone();
    let sidecar = index
        .build_exact_residual_sidecar()
        .expect("build nonselective sidecar");
    assert!(
        index
            .admit_exact_residual_sidecar(&sidecar)
            .expect("admit nonselective sidecar")
    );
    let params = SearchParams {
        parallel_enabled: true,
        ..SearchParams::default()
    };
    let query = [1.0, 0.0, 0.0];
    let outcome = index
        .scan_exact_residual_sidecar_parallel(
            &query,
            1,
            None,
            index
                .exact_residual_sidecar
                .get()
                .expect("admitted sidecar"),
            params.parallel_chunk_size,
        )
        .expect("run globally adaptive parallel scan");
    assert_eq!(
        outcome.census.groups_scanned, EXACT_RESIDUAL_ADAPTIVE_PROBE_GROUPS,
        "one query-level probe replaces per-chunk residual work"
    );
    assert_eq!(outcome.census.lanes_pruned, 0);
    assert_eq!(outcome.census.adaptive_fallbacks, 1);
    assert_eq!(outcome.census.flat_fallback_rows, COUNT);
    let actual = index
        .resolve_heap(outcome.heap)
        .expect("resolve globally adaptive result");
    let expected = flat
        .search_top_k_with_params(&query, 1, None, params)
        .expect("flat exact parallel result");
    assert_eq!(actual, expected);
}

#[test]
fn exact_residual_sidecar_f32_max_queries_fail_closed_to_exact_sequential_and_parallel() {
    let dimension = 35;
    let count = 17;
    let mut index = InMemoryVectorIndex::from_vectors(
        (0..count).map(|row| format!("max-query-{row}")).collect(),
        vec![vec![f16::MAX.to_f32(); dimension]; count],
        dimension,
    )
    .expect("finite f16 source for max-query fallback");
    bind_test_residual_source(&mut index);
    let flat = index.clone();
    let sidecar = index.build_exact_residual_sidecar().expect("build sidecar");
    assert!(
        index
            .admit_exact_residual_sidecar(&sidecar)
            .expect("admit sidecar")
    );
    let query = vec![f32::MAX; dimension];
    let sequential = index
        .scan_exact_residual_sidecar(
            &query,
            3,
            None,
            index
                .exact_residual_sidecar
                .get()
                .expect("admitted sidecar"),
        )
        .expect("sequential max-query fallback");
    let parallel = index
        .scan_exact_residual_sidecar_parallel(
            &query,
            3,
            None,
            index
                .exact_residual_sidecar
                .get()
                .expect("admitted sidecar"),
            8,
        )
        .expect("parallel max-query fallback");
    assert_eq!(sequential.census, ResidualPruningCensus::default());
    assert_eq!(parallel.census, ResidualPruningCensus::default());
    let sequential = index
        .resolve_heap(sequential.heap)
        .expect("resolve sequential fallback");
    let parallel = index
        .resolve_heap(parallel.heap)
        .expect("resolve parallel fallback");
    let expected_sequential = flat
        .search_top_k_with_params(
            &query,
            3,
            None,
            SearchParams {
                parallel_enabled: false,
                ..SearchParams::default()
            },
        )
        .expect("flat sequential max-query result");
    let expected_parallel = flat
        .search_top_k_with_params(
            &query,
            3,
            None,
            SearchParams {
                parallel_enabled: true,
                parallel_threshold: 1,
                parallel_chunk_size: 8,
            },
        )
        .expect("flat parallel max-query result");
    assert_eq!(sequential, expected_sequential);
    assert_eq!(parallel, expected_parallel);
    assert_eq!(sequential, parallel);
}

#[test]
fn exact_residual_sidecar_parallel_uses_the_default_10k_to_100k_target_scales() {
    let dimension = 3;
    let params = SearchParams {
        parallel_enabled: true,
        ..SearchParams::default()
    };
    assert_eq!(params.parallel_threshold, crate::search::PARALLEL_THRESHOLD);
    let query = [1.0, 0.0, 0.0];
    for count in [crate::search::PARALLEL_THRESHOLD + 1, 100_000] {
        let mut index = InMemoryVectorIndex::from_vectors(
            (0..count).map(|row| format!("parallel-{row}")).collect(),
            (0..count)
                .map(|row| {
                    vec![
                        if row % EXACT_RESIDUAL_LANES == 0 {
                            1.0
                        } else {
                            -1.0
                        },
                        0.0,
                        0.0,
                    ]
                })
                .collect(),
            dimension,
        )
        .expect("target-scale finite source");
        bind_test_residual_source(&mut index);
        let flat = index.clone();
        let sidecar = index
            .build_exact_residual_sidecar()
            .expect("build target-scale sidecar");
        assert!(
            index
                .admit_exact_residual_sidecar(&sidecar)
                .expect("admit sidecar")
        );
        let census_outcome = index
            .scan_exact_residual_sidecar_parallel(
                &query,
                10,
                None,
                index
                    .exact_residual_sidecar
                    .get()
                    .expect("admitted sidecar"),
                params.parallel_chunk_size,
            )
            .expect("parallel sidecar route");
        assert!(census_outcome.census.groups_scanned > 0);
        assert!(
            census_outcome.census.lanes_pruned > 0,
            "{count}-row target must exercise per-lane pruning"
        );
        assert!(
            census_outcome.census.exact_sidecar_scores < census_outcome.census.eligible_lanes,
            "{count}-row target must avoid exact scores for proven lanes"
        );
        let expected = flat
            .search_top_k_with_params(&query, 10, None, params)
            .expect("flat parallel result");
        assert_eq!(
            index
                .resolve_heap(census_outcome.heap)
                .expect("resolve censused parallel result"),
            expected,
            "{count}-row censused route preserves exact ordering"
        );
        let actual = index
            .search_top_k_with_params(&query, 10, None, params)
            .expect("sidecar parallel result");
        assert_eq!(
            actual, expected,
            "{count}-row sidecar path preserves exact ordering"
        );
    }
}

#[cfg(target_os = "linux")]
#[test]
fn exact_residual_sidecar_public_io_rejects_snapshot_changes_symlinks_and_overwrite_races() {
    use std::os::unix::fs::symlink;

    let dimension = 35;
    let mut index = InMemoryVectorIndex::from_vectors(
        (0..17).map(|row| format!("io-{row}")).collect(),
        (0..17)
            .map(|row| make_normalized_vec(dimension, row as f32 * 0.75))
            .collect(),
        dimension,
    )
    .expect("finite sidecar source");
    bind_test_residual_source(&mut index);
    let dir = owned_temp_dir("exact_residual_public_io");
    let sidecar_path = dir.join("sidecar.fsrs");
    let occupied_path = dir.join("occupied.fsrs");
    let raced_path = dir.join("raced.fsrs");
    let symlink_path = dir.join("symlink.fsrs");
    let target_path = dir.join("target.fsrs");

    index
        .write_exact_residual_sidecar(&sidecar_path)
        .expect("public write publishes an owned sidecar");
    let reader = index.clone();
    assert!(
        reader
            .try_open_exact_residual_sidecar(&sidecar_path)
            .expect("public no-follow open")
    );
    let expected = index
        .build_exact_residual_sidecar()
        .expect("derive test sidecar without touching the published file");
    let encoded = expected.encode().expect("encode test sidecar in memory");
    let mut truncated_after_stat = std::io::Cursor::new(encoded[..encoded.len() - 1].to_vec());
    assert!(
        !exact_residual_sidecar_stream_matches_reader(&mut truncated_after_stat, &expected)
            .expect("bounded reader comparison"),
        "the production reader seam rejects a stream truncated after fstat"
    );
    let mut grown_after_stat = encoded.clone();
    grown_after_stat.push(0x5a);
    let mut grown_after_stat = std::io::Cursor::new(grown_after_stat);
    assert!(
        !exact_residual_sidecar_stream_matches_reader(&mut grown_after_stat, &expected)
            .expect("bounded reader comparison"),
        "the production one-byte probe rejects a stream that grows after fstat"
    );

    write_new_owned_file(&occupied_path, b"incumbent destination");
    for _ in 0..3 {
        assert!(
            index.write_exact_residual_sidecar(&occupied_path).is_err(),
            "linkat publication refuses an occupied destination"
        );
    }
    assert_eq!(
        std::fs::read(&occupied_path).expect("read incumbent"),
        b"incumbent destination"
    );
    assert_eq!(
        std::fs::read_dir(&dir)
            .expect("inspect owned public-I/O directory")
            .count(),
        2,
        "failed no-replace publications must leave no visible temporary artifact"
    );

    let start = std::sync::Arc::new(std::sync::Barrier::new(3));
    let left_start = std::sync::Arc::clone(&start);
    let left_index = index.clone();
    let left_path = raced_path.clone();
    let left = std::thread::spawn(move || {
        left_start.wait();
        left_index.write_exact_residual_sidecar(&left_path).is_ok()
    });
    let right_start = std::sync::Arc::clone(&start);
    let right_index = index.clone();
    let right_path = raced_path.clone();
    let right = std::thread::spawn(move || {
        right_start.wait();
        right_index
            .write_exact_residual_sidecar(&right_path)
            .is_ok()
    });
    start.wait();
    let published = [
        left.join().expect("left writer did not panic"),
        right.join().expect("right writer did not panic"),
    ]
    .into_iter()
    .filter(|published| *published)
    .count();
    assert_eq!(
        published, 1,
        "concurrent public writers publish exactly one immutable destination"
    );
    let race_reader = index.clone();
    assert!(
        race_reader
            .try_open_exact_residual_sidecar(&raced_path)
            .expect("winner remains descriptor-admissible")
    );
    assert!(
        std::fs::read_dir(&dir)
            .expect("read test parent")
            .flatten()
            .count()
            == 3,
        "anonymous O_TMPFILE failures create no visible temporary paths"
    );

    write_new_owned_file(&target_path, b"symlink target");
    symlink(&target_path, &symlink_path).expect("create final-component symlink");
    assert!(
        !index
            .try_open_exact_residual_sidecar(&symlink_path)
            .expect("no-follow symlink open is an optional miss")
    );
    assert!(
        index.write_exact_residual_sidecar(&symlink_path).is_err(),
        "atomic no-replace publication refuses a symlink destination"
    );
    assert_eq!(
        std::fs::read(&target_path).expect("read symlink target"),
        b"symlink target"
    );
}
