use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use frankensearch_core::BoundQueryEmbedding;
use frankensearch_core::SearchError;
use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, QuantizationFormat,
};

use crate::{FsviV2IdentityBinding, Quantization};

use super::super::*;
#[cfg(target_os = "linux")]
use super::{bind_test_residual_source, write_new_owned_file};
use super::{cleanup, make_normalized_vec, owned_temp_dir, temp_index_path};

#[test]
fn from_vectors_basic() {
    let dim = 8;
    let doc_ids = vec!["a".into(), "b".into(), "c".into()];
    let vectors = vec![
        make_normalized_vec(dim, 1.0),
        make_normalized_vec(dim, 2.0),
        make_normalized_vec(dim, 3.0),
    ];
    let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();
    assert_eq!(index.record_count(), 3);
    assert_eq!(index.dimension(), 8);
    assert_eq!(index.doc_id_at(0).unwrap(), "a");
    assert_eq!(index.doc_id_at(2).unwrap(), "c");
}

#[test]
fn from_vectors_dimension_mismatch() {
    let doc_ids = vec!["a".into()];
    let vectors = vec![vec![1.0, 2.0, 3.0]]; // dim 3 != expected 4
    let result = InMemoryVectorIndex::from_vectors(doc_ids, vectors, 4);
    assert!(result.is_err());
}

#[test]
fn from_vectors_count_mismatch() {
    let doc_ids = vec!["a".into(), "b".into()];
    let vectors = vec![vec![1.0, 2.0]]; // 1 vector, 2 doc_ids
    let result = InMemoryVectorIndex::from_vectors(doc_ids, vectors, 2);
    assert!(result.is_err());
}

#[test]
fn from_vectors_non_finite_rejected() {
    let doc_ids = vec!["a".into()];
    let vectors = vec![vec![1.0, f32::NAN]];
    let result = InMemoryVectorIndex::from_vectors(doc_ids, vectors, 2);
    assert!(result.is_err());
}

#[test]
fn from_fsvi_matches_file_backed_search() {
    let path = temp_index_path("from_fsvi");
    cleanup(&path);

    let dim = 32;
    let docs = 64usize;
    let mut writer = crate::VectorIndex::create_with_revision(
        &path,
        "test-embedder",
        "rev-a",
        dim,
        Quantization::F16,
    )
    .unwrap();

    for i in 0..docs {
        let vector = make_normalized_vec(dim, i as f32 * 0.73);
        writer.write_record(&format!("doc-{i}"), &vector).unwrap();
    }
    writer.finish().unwrap();

    let file_index = crate::VectorIndex::open_read_only(&path).unwrap();
    let memory_index = InMemoryVectorIndex::from_fsvi(&path).unwrap();
    assert_eq!(memory_index.record_count(), docs);
    assert_eq!(memory_index.dimension(), dim);

    let query = make_normalized_vec(dim, 12.4);
    let file_hits = file_index.search_top_k(&query, 10, None).unwrap();
    let memory_hits = memory_index.search_top_k(&query, 10, None).unwrap();
    assert_eq!(file_hits.len(), memory_hits.len());

    for (file, memory) in file_hits.iter().zip(memory_hits.iter()) {
        assert_eq!(file.doc_id, memory.doc_id);
        assert!(
            (file.score - memory.score).abs() < 0.001,
            "score mismatch for {}: file={} memory={}",
            file.doc_id,
            file.score,
            memory.score
        );
    }

    // Verify vectors were loaded in quantized form and still round-trip.
    let recovered = memory_index.vector_at_f32(0).unwrap();
    assert_eq!(recovered.len(), dim);

    cleanup(&path);
}

// ─── Embedding-space identity (bd-9xuj T2-C3) ───────────────────────────

/// Write a real FSVI v2 file at `path` whose identity binds
/// `explicit_test_model(model_id, dimension)` with fsvi-v2 storage.
/// Returns the exact identity binding (needed for admission) and the
/// core-side space fingerprint the file's header preserves.
fn write_fsvi_v2_fixture(
    path: &Path,
    model_id: &str,
    dimension: usize,
    generation_sequence: u64,
    rows: &[(String, Vec<f32>)],
) -> (FsviV2IdentityBinding, String) {
    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(
        model_id,
        u32::try_from(dimension).expect("test dimension fits u32"),
    );
    identity.storage.format = "fsvi-v2".to_owned();
    identity.storage.quantization = QuantizationFormat::F16;
    identity.storage.endianness = "little-endian".to_owned();
    let space_fingerprint = identity.space.fingerprint();
    let generation = ArtifactGenerationIdentityV1::new(generation_sequence, [0x5c; 16])
        .expect("valid test generation");
    let binding =
        FsviV2IdentityBinding::new(generation, identity.freeze().expect("freeze identity"))
            .expect("valid FSVI v2 identity binding");
    let mut writer =
        crate::VectorIndex::create_v2(path, binding.clone()).expect("create v2 writer");
    for (doc_id, vector) in rows {
        writer.write_record(doc_id, vector).expect("write v2 row");
    }
    writer.finish().expect("finish v2 fixture");
    (binding, space_fingerprint)
}

fn identity_rows(dim: usize, count: usize) -> (Vec<String>, Vec<Vec<f32>>) {
    let doc_ids = (0..count).map(|i| format!("doc-{i}")).collect();
    let vectors = (0..count)
        .map(|i| make_normalized_vec(dim, (i + 1) as f32))
        .collect();
    (doc_ids, vectors)
}

#[test]
fn from_vectors_with_identity_exposes_space_fingerprint() {
    let dim = 8;
    let bundle = EmbeddingIdentityBundleV1::explicit_test_model("mem-space-a", 8);
    let (doc_ids, vectors) = identity_rows(dim, 3);
    let index =
        InMemoryVectorIndex::from_vectors_with_identity(doc_ids, vectors, dim, &bundle.space)
            .expect("build identified index");
    assert_eq!(
        index.space_fingerprint_hex(),
        Some(bundle.space.fingerprint().as_str()),
        "the index must expose exactly the space fingerprint it was built with"
    );
    assert_eq!(index.record_count(), 3);
    assert!(
        !index.space_identity_is_attested(),
        "a caller-declared space is DECLARED, never header-attested (C4-write guards 2+8)"
    );

    // The identity-less constructor stays typed-absent: no fabrication.
    let (doc_ids, vectors) = identity_rows(dim, 3);
    let legacy = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).expect("build");
    assert_eq!(
        legacy.space_fingerprint_hex(),
        None,
        "an identity-less build must never fabricate a space fingerprint"
    );
    assert!(!legacy.space_identity_is_attested());
}

#[test]
fn from_vectors_with_identity_rejects_dimension_mismatch() -> Result<(), String> {
    // A 16-dim space cannot describe an 8-dim index: the identity claim
    // is checked before it is stored, never trusted on assertion alone.
    let bundle = EmbeddingIdentityBundleV1::explicit_test_model("mem-space-mismatch", 16);
    let (doc_ids, vectors) = identity_rows(8, 2);
    let error = InMemoryVectorIndex::from_vectors_with_identity(doc_ids, vectors, 8, &bundle.space)
        .expect_err("16-dim space must not bind an 8-dim index");
    let rendered = format!("{error:?}");
    let SearchError::InvalidConfig { field, value, .. } = error else {
        return Err(format!("expected InvalidConfig, got {rendered}"));
    };
    assert_eq!(field, "space_identity.dimension");
    assert_eq!(value, "16");
    Ok(())
}

#[test]
fn admitted_v2_load_preserves_space_identity() {
    // Readiness map §3.3(3), adapted to the only v2 open path that
    // exists: `VectorIndex::open` is v1-only by design, so an
    // identity-complete v2 artifact reaches memory through exact
    // admission. The in-memory index must preserve the admitted space
    // fingerprint, and its hex must equal the core-side
    // `EmbeddingSpaceIdentityV1::fingerprint()` that produced the file —
    // one join key across crates and codecs.
    //
    // The fixture lives in its OWN directory: since 58726e26 admission
    // snapshots the containing directory and fails closed with
    // `DirectoryChangedDuringRead` when sibling test files churn it, so
    // the shared test temp dir is not a legal admission site.
    let dir = owned_temp_dir("admitted_v2_space_identity");
    let path = dir.join("admitted_v2_space_identity.fsvi");
    let dim = 8;
    let (doc_ids, vectors) = identity_rows(dim, 4);
    let rows: Vec<(String, Vec<f32>)> = doc_ids.into_iter().zip(vectors).collect();
    let (binding, expected) = write_fsvi_v2_fixture(&path, "fsvi-space-model", dim, 11, &rows);

    let admitted = crate::VectorIndex::open_admitted_v2(&path, &binding)
        .expect("exact admission of the v2 fixture");
    let index = InMemoryVectorIndex::from_admitted_v2(&admitted)
        .expect("load admitted v2 artifact into memory");
    assert_eq!(index.record_count(), 4);
    assert_eq!(
        index.space_fingerprint_hex(),
        Some(expected.as_str()),
        "the in-memory index must preserve the admitted space identity"
    );
    assert!(
        index.space_identity_is_attested(),
        "an identity read out of a validated v2 header through exact admission is ATTESTED \
             (C4-write guards 2+8)"
    );
    let two_tier = InMemoryTwoTierIndex::new(index, None);
    assert!(two_tier.fast_identity_is_attested());
    assert!(
        !two_tier.quality_identity_is_attested(),
        "no quality tier means no attested quality identity"
    );

    // Sanity pin: the legacy pathname loader cannot read v2 at all, so
    // the admitted path above is not optional for identified artifacts.
    let error =
        InMemoryVectorIndex::from_fsvi(&path).expect_err("VectorIndex::open must reject v2 bytes");
    assert!(matches!(error, SearchError::IndexVersionMismatch { .. }));
}

#[cfg(target_os = "linux")]
#[test]
fn admitted_v2_two_tier_shipping_cache_skips_corrupt_entries_without_overwrite() {
    // Use a private directory because v2 admission snapshots the parent;
    // sidecar publication and discovery must share that same held-directory
    // model without test-suite sibling churn or cross-process path reuse.
    let dir = owned_temp_dir("admitted_v2_residual_product");
    let path = dir.join("index.fsvi");
    let cache_dir = dir.join("residual-cache");
    std::fs::create_dir(&cache_dir).expect("create owned cache directory");
    let dimension = 35;
    let (doc_ids, vectors) = identity_rows(dimension, 17);
    let rows: Vec<(String, Vec<f32>)> = doc_ids.into_iter().zip(vectors).collect();
    let (binding, _) = write_fsvi_v2_fixture(&path, "residual-product", dimension, 29, &rows);
    let admitted =
        crate::VectorIndex::open_admitted_v2(&path, &binding).expect("admit private v2 source");
    let query = make_normalized_vec(dimension, 3.25);
    let flat = InMemoryVectorIndex::from_admitted_v2(&admitted).expect("flat admitted index");
    let cache_prefix = flat
        .exact_residual_generation_cache_prefix()
        .expect("derive admitted generation cache key");
    let mut header_valid_corrupt = flat
        .build_exact_residual_sidecar()
        .expect("derive a cache-shaped sidecar")
        .encode()
        .expect("encode a cache-shaped sidecar");
    header_valid_corrupt[EXACT_RESIDUAL_SIDECAR_HEADER_BYTES] ^= 0x01;
    let mut corrupt_paths = Vec::new();
    corrupt_paths
        .try_reserve_exact(EXACT_RESIDUAL_CACHE_ATTEMPTS + 1)
        .expect("reserve bounded corrupt-cache fixture paths");
    for sequence in 0..=EXACT_RESIDUAL_CACHE_ATTEMPTS {
        let corrupt_path = cache_dir.join(format!("{cache_prefix}corrupt-{sequence}.fsrs"));
        write_new_owned_file(&corrupt_path, &header_valid_corrupt);
        corrupt_paths.push(corrupt_path);
    }
    let indexed = InMemoryTwoTierIndex::from_admitted_v2_with_residual_sidecar_cache(
        &admitted, &cache_dir, None,
    )
    .expect("shipping two-tier constructor keeps the admitted index usable");
    assert!(
        indexed.fast_index.exact_residual_sidecar.get().is_some(),
        "the shipping route publishes and attaches the source-derived sidecar"
    );
    assert_eq!(
        indexed
            .search_fast(&query, 5)
            .expect("sidecar product search"),
        flat.search_top_k(&query, 5, None)
            .expect("flat product search")
    );

    let entry_count_after_publish = std::fs::read_dir(&cache_dir)
        .expect("read owned cache after publication")
        .flatten()
        .count();
    assert_eq!(
        entry_count_after_publish,
        corrupt_paths.len() + 1,
        "publication retains every corrupt immutable entry and adds one generation-matched artifact"
    );
    for corrupt_path in &corrupt_paths {
        assert_eq!(
            std::fs::read(corrupt_path).expect("read planted corrupt cache entry"),
            header_valid_corrupt,
            "the shipping cache route never overwrites a stale/corrupt artifact"
        );
    }

    let reopened = InMemoryTwoTierIndex::from_admitted_v2_with_residual_sidecar_cache(
        &admitted, &cache_dir, None,
    )
    .expect("cache reader scans past corrupt entries to the valid generation artifact");
    assert!(reopened.fast_index.exact_residual_sidecar.get().is_some());
    assert_eq!(
        reopened
            .search_fast(&query, 5)
            .expect("reopened product search"),
        flat.search_top_k(&query, 5, None)
            .expect("baseline flat search")
    );
    let entry_count_after_reopen = std::fs::read_dir(&cache_dir)
        .expect("read owned cache after reopening")
        .flatten()
        .count();
    assert_eq!(
        entry_count_after_reopen, entry_count_after_publish,
        "reopening selects the existing valid sidecar instead of writing another artifact"
    );

    let unavailable_cache = dir.join("unavailable-cache");
    let fallback = InMemoryTwoTierIndex::from_admitted_v2_with_residual_sidecar_cache(
        &admitted,
        &unavailable_cache,
        None,
    )
    .expect("unavailable optional cache retains the admitted flat tier");
    assert!(fallback.fast_index.exact_residual_sidecar.get().is_none());
}

#[cfg(target_os = "linux")]
#[test]
fn exact_residual_capability_failure_precedes_public_writer_derivation() {
    let dimension = 35;
    let mut index = InMemoryVectorIndex::from_vectors(
        (0..17)
            .map(|row| format!("writer-capability-{row}"))
            .collect(),
        (0..17)
            .map(|row| make_normalized_vec(dimension, row as f32 + 0.25))
            .collect(),
        dimension,
    )
    .expect("finite public-writer source");
    bind_test_residual_source(&mut index);
    let directory = owned_temp_dir("exact_residual_writer_capability");
    let destination = directory.join("sidecar.fsrs");

    reset_exact_residual_sidecar_build_count();
    fail_next_exact_residual_publication_acquisition();
    assert!(
        index.write_exact_residual_sidecar(&destination).is_err(),
        "a failed create-only capability must leave public writing unavailable"
    );
    assert!(
        !exact_residual_publication_acquisition_failure_is_pending(),
        "the public writer must consume the acquisition-failure seam"
    );
    assert_eq!(
        exact_residual_sidecar_build_count(),
        0,
        "public writer must acquire publication capability before deriving a sidecar"
    );
    assert_eq!(
        std::fs::read_dir(&directory)
            .expect("inspect private writer directory")
            .count(),
        0,
        "failed capability acquisition must leave no visible temporary artifact"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn valid_exact_residual_candidate_attaches_and_prunes_without_publication_capability() {
    let directory = owned_temp_dir("exact_residual_read_only_candidate");
    let source_path = directory.join("index.fsvi");
    let cache_dir = directory.join("residual-cache");
    std::fs::create_dir(&cache_dir).expect("create private cache directory");
    let dimension = 35;
    let count = 17;
    let mut vectors = vec![vec![0.0_f32; dimension]; count];
    for vector in vectors.iter_mut().take(EXACT_RESIDUAL_LANES) {
        vector[0] = 1.0;
    }
    for vector in vectors.iter_mut().skip(EXACT_RESIDUAL_LANES) {
        vector[0] = -1.0;
    }
    let doc_ids = (0..count).map(|row| format!("read-only-candidate-{row}"));
    let rows: Vec<(String, Vec<f32>)> = doc_ids.zip(vectors).collect();
    let (binding, _) =
        write_fsvi_v2_fixture(&source_path, "residual-read-only", dimension, 57, &rows);
    let admitted = crate::VectorIndex::open_admitted_v2(&source_path, &binding)
        .expect("admit private v2 source");
    let flat = InMemoryVectorIndex::from_admitted_v2(&admitted).expect("load flat admitted source");
    let cache_prefix = flat
        .exact_residual_generation_cache_prefix()
        .expect("derive generation cache prefix");
    let valid = flat
        .build_exact_residual_sidecar()
        .expect("derive cache-shaped valid fixture")
        .encode()
        .expect("encode cache-shaped valid fixture");
    let valid_path = cache_dir.join(format!("{cache_prefix}valid.fsrs"));
    write_new_owned_file(&valid_path, &valid);

    reset_exact_residual_sidecar_build_count();
    fail_next_exact_residual_publication_acquisition();
    let cached =
        InMemoryVectorIndex::from_admitted_v2_with_residual_sidecar_cache(&admitted, &cache_dir)
            .expect("read-only reusable cache candidate must retain the exact source");
    assert!(
        exact_residual_publication_acquisition_failure_is_pending(),
        "a reusable immutable candidate must attach without acquiring publication capability"
    );
    assert_eq!(
        exact_residual_sidecar_build_count(),
        1,
        "the sidecar must be derived exactly once to authenticate the read-only candidate"
    );
    assert!(
        cached.has_exact_residual_sidecar(),
        "a valid immutable candidate must remain attached when publication is unavailable"
    );
    let mut query = vec![0.0_f32; dimension];
    query[0] = 1.0;
    let outcome = cached
        .scan_exact_residual_sidecar(
            &query,
            1,
            None,
            cached
                .exact_residual_sidecar
                .get()
                .expect("read-only candidate is attached"),
        )
        .expect("scan attached read-only candidate");
    assert!(
        outcome.census.lanes_pruned > 0,
        "the reused candidate must retain real exact-residual pruning"
    );
    assert_eq!(
        cached
            .resolve_heap(outcome.heap)
            .expect("resolve read-only candidate result"),
        flat.search_top_k(&query, 1, None)
            .expect("flat exact baseline"),
        "read-only candidate pruning preserves the exact result"
    );
    assert!(
        take_exact_residual_publication_acquisition_failure(),
        "clear the planted unavailable-publication seam after proving reuse bypassed it"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn exact_residual_capability_failure_precedes_empty_cache_derivation() {
    let directory = owned_temp_dir("exact_residual_empty_capability");
    let source_path = directory.join("index.fsvi");
    let cache_dir = directory.join("residual-cache");
    std::fs::create_dir(&cache_dir).expect("create empty private cache directory");
    let dimension = 35;
    let (doc_ids, vectors) = identity_rows(dimension, 17);
    let rows: Vec<(String, Vec<f32>)> = doc_ids.into_iter().zip(vectors).collect();
    let (binding, _) =
        write_fsvi_v2_fixture(&source_path, "residual-capability", dimension, 57, &rows);
    let admitted = crate::VectorIndex::open_admitted_v2(&source_path, &binding)
        .expect("admit private v2 source");

    reset_exact_residual_sidecar_build_count();
    fail_next_exact_residual_publication_acquisition();
    let fallback =
        InMemoryVectorIndex::from_admitted_v2_with_residual_sidecar_cache(&admitted, &cache_dir)
            .expect("optional cache failure must retain an exact flat source");
    assert!(
        !exact_residual_publication_acquisition_failure_is_pending(),
        "an empty cache must consume the publication-capability failure seam"
    );
    assert_eq!(
        exact_residual_sidecar_build_count(),
        0,
        "an unavailable empty cache must acquire capability before deriving a sidecar"
    );
    assert!(
        !fallback.has_exact_residual_sidecar(),
        "an unavailable optional empty cache must stay on the exact flat route"
    );
    assert_eq!(
        std::fs::read_dir(&cache_dir)
            .expect("inspect private empty cache directory")
            .count(),
        0,
        "failed capability acquisition must leave no visible temporary artifact"
    );
}

/// A header-matching candidate whose body is corrupt must be rejected from
/// its own trailing digest, never by deriving the source sidecar and
/// discovering the byte-exact comparison fails.
///
/// Discovery filters on the header alone, so before the cheap admission
/// this exact input forced a derivation of up to
/// `EXACT_RESIDUAL_SIDECAR_MAX_BYTES` and then discarded it the moment the
/// unavailable optional cache refused publication. The build counter is the
/// discriminating assertion: it reads 1 without the admission gate and 0
/// with it.
#[cfg(target_os = "linux")]
#[test]
fn exact_residual_corrupt_candidate_derives_nothing_when_publication_is_unavailable() {
    let directory = owned_temp_dir("exact_residual_corrupt_capability");
    let source_path = directory.join("index.fsvi");
    let cache_dir = directory.join("residual-cache");
    std::fs::create_dir(&cache_dir).expect("create private corrupt-candidate cache directory");
    let dimension = 35;
    let (doc_ids, vectors) = identity_rows(dimension, 17);
    let rows: Vec<(String, Vec<f32>)> = doc_ids.into_iter().zip(vectors).collect();
    let (binding, _) = write_fsvi_v2_fixture(
        &source_path,
        "residual-corrupt-capability",
        dimension,
        57,
        &rows,
    );
    let admitted = crate::VectorIndex::open_admitted_v2(&source_path, &binding)
        .expect("admit private v2 source");
    let flat = InMemoryVectorIndex::from_admitted_v2(&admitted).expect("load flat admitted source");
    let cache_prefix = flat
        .exact_residual_generation_cache_prefix()
        .expect("derive generation cache prefix");
    let mut header_valid_corrupt = flat
        .build_exact_residual_sidecar()
        .expect("derive cache-shaped fixture")
        .encode()
        .expect("encode cache-shaped fixture");
    // Leave the header byte-identical so discovery still admits this as a
    // candidate, and corrupt the first payload byte so nothing but the
    // self-covering trailing digest can reject it.
    header_valid_corrupt[EXACT_RESIDUAL_SIDECAR_HEADER_BYTES] ^= 0x01;
    write_new_owned_file(
        &cache_dir.join(format!("{cache_prefix}corrupt.fsrs")),
        &header_valid_corrupt,
    );
    let entries_before = std::fs::read_dir(&cache_dir)
        .expect("read private corrupt-candidate cache directory")
        .count();

    reset_exact_residual_sidecar_build_count();
    fail_next_exact_residual_publication_acquisition();
    let fallback =
        InMemoryVectorIndex::from_admitted_v2_with_residual_sidecar_cache(&admitted, &cache_dir)
            .expect("a corrupt optional candidate must retain an exact flat source");

    assert_eq!(
        exact_residual_sidecar_build_count(),
        0,
        "a corrupt header-matching candidate must be rejected from its own digest \
             before any source derivation"
    );
    assert!(
        !exact_residual_publication_acquisition_failure_is_pending(),
        "the no-reusable-candidate path must still reserve publication before deriving"
    );
    assert!(
        !fallback.has_exact_residual_sidecar(),
        "a corrupt candidate must never attach"
    );
    assert_eq!(
        std::fs::read_dir(&cache_dir)
            .expect("re-read private corrupt-candidate cache directory")
            .count(),
        entries_before,
        "an unavailable optional cache must leave no visible artifact"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn admitted_v2_cache_discovery_budget_skips_unrelated_and_corrupt_work() {
    // Each hostile directory is private and create-only. When either
    // deterministic budget is reached, cache use and publication are both
    // skipped; the exact flat product remains available.
    let dir = owned_temp_dir("admitted_v2_residual_cache_budget");
    let source_path = dir.join("index.fsvi");
    let dimension = 35;
    let (doc_ids, vectors) = identity_rows(dimension, 17);
    let rows: Vec<(String, Vec<f32>)> = doc_ids.into_iter().zip(vectors).collect();
    let (binding, _) =
        write_fsvi_v2_fixture(&source_path, "residual-cache-budget", dimension, 41, &rows);
    let admitted = crate::VectorIndex::open_admitted_v2(&source_path, &binding)
        .expect("admit private v2 source");
    let flat = InMemoryVectorIndex::from_admitted_v2(&admitted).expect("open exact baseline");
    let query = make_normalized_vec(dimension, 6.5);
    let expected = flat
        .search_top_k(&query, 5, None)
        .expect("flat exact baseline");

    let unrelated_dir = dir.join("many-unrelated");
    std::fs::create_dir(&unrelated_dir).expect("create private unrelated cache directory");
    for entry in 0..=EXACT_RESIDUAL_CACHE_DIRECTORY_ENTRY_LIMIT {
        write_new_owned_file(
            &unrelated_dir.join(format!("unrelated-{entry:04}")),
            b"not a sidecar",
        );
    }
    let unrelated_entries_before = std::fs::read_dir(&unrelated_dir)
        .expect("read private unrelated cache directory")
        .count();
    let unrelated = InMemoryVectorIndex::from_admitted_v2_with_residual_sidecar_cache(
        &admitted,
        &unrelated_dir,
    )
    .expect("entry-budget exhaustion remains an exact open");
    assert!(
        !unrelated.has_exact_residual_sidecar(),
        "entry-budget exhaustion must skip optional cache attachment"
    );
    assert_eq!(
        unrelated
            .search_top_k(&query, 5, None)
            .expect("unrelated-cache exact search"),
        expected,
        "entry-budget exhaustion must retain the exact flat result"
    );
    assert_eq!(
        std::fs::read_dir(&unrelated_dir)
            .expect("re-read private unrelated cache directory")
            .count(),
        unrelated_entries_before,
        "entry-budget exhaustion skips cache publication"
    );

    let corrupt_dir = dir.join("many-corrupt");
    std::fs::create_dir(&corrupt_dir).expect("create private corrupt cache directory");
    let cache_prefix = flat
        .exact_residual_generation_cache_prefix()
        .expect("derive generation-keyed cache prefix");
    let mut header_valid_corrupt = flat
        .build_exact_residual_sidecar()
        .expect("derive cache-shaped sidecar")
        .encode()
        .expect("encode cache-shaped sidecar");
    header_valid_corrupt[EXACT_RESIDUAL_SIDECAR_HEADER_BYTES] ^= 0x01;
    for entry in 0..=EXACT_RESIDUAL_CACHE_COMPARISON_CANDIDATE_LIMIT {
        write_new_owned_file(
            &corrupt_dir.join(format!("{cache_prefix}corrupt-{entry:04}.fsrs")),
            &header_valid_corrupt,
        );
    }
    let corrupt_entries_before = std::fs::read_dir(&corrupt_dir)
        .expect("read private corrupt cache directory")
        .count();
    let corrupt =
        InMemoryVectorIndex::from_admitted_v2_with_residual_sidecar_cache(&admitted, &corrupt_dir)
            .expect("comparison-budget exhaustion remains an exact open");
    assert!(
        !corrupt.has_exact_residual_sidecar(),
        "comparison-budget exhaustion must skip optional cache attachment"
    );
    assert_eq!(
        corrupt
            .search_top_k(&query, 5, None)
            .expect("corrupt-cache exact search"),
        expected,
        "comparison-budget exhaustion must retain the exact flat result"
    );
    assert_eq!(
        std::fs::read_dir(&corrupt_dir)
            .expect("re-read private corrupt cache directory")
            .count(),
        corrupt_entries_before,
        "comparison-budget exhaustion skips cache publication"
    );
}

#[test]
fn from_fsvi_legacy_v1_stays_unidentified() {
    // Legacy v1 files carry no identity bundle. Absence must survive the
    // load as the typed None state (the C4 seams route it as
    // LegacyUnidentified) — never a fabricated fingerprint.
    let path = temp_index_path("from_fsvi_legacy_v1");
    cleanup(&path);
    let dim = 8;
    let mut writer =
        crate::VectorIndex::create(&path, "legacy-embedder", dim).expect("create legacy v1 writer");
    writer
        .write_record("doc-0", &make_normalized_vec(dim, 1.0))
        .expect("write v1 row");
    writer.finish().expect("finish v1 file");

    let index = InMemoryVectorIndex::from_fsvi(&path).expect("load v1 file into memory");
    assert_eq!(
        index.space_fingerprint_hex(),
        None,
        "legacy v1 absence must stay typed, never fabricated"
    );
    assert!(
        !index.space_identity_is_attested(),
        "a v1 header attests nothing (C4-write guards 2+8)"
    );
    cleanup(&path);
}

#[test]
fn from_fsvi_preserves_embedder_identity_strings() {
    // bd-9xuj T2-C2: from_fsvi used to keep only the dimension and drop
    // the header's embedder id/revision on the floor. Both must survive
    // the load verbatim — including a v1 header's EMPTY revision, which
    // stays `Some("")` (what the header says), distinct from the `None`
    // of a bare-vector build (no header at all).
    let path = temp_index_path("from_fsvi_embedder_identity");
    cleanup(&path);
    let dim = 8;
    let mut writer = crate::VectorIndex::create_with_revision(
        &path,
        "kept-embedder",
        "kept-revision-v7",
        dim,
        Quantization::F16,
    )
    .expect("create v1 writer with revision");
    writer
        .write_record("doc-0", &make_normalized_vec(dim, 1.0))
        .expect("write v1 row");
    writer.finish().expect("finish v1 file");

    let index = InMemoryVectorIndex::from_fsvi(&path).expect("load v1 file into memory");
    assert_eq!(index.embedder_id(), Some("kept-embedder"));
    assert_eq!(index.embedder_revision(), Some("kept-revision-v7"));
    assert_eq!(
        index.space_fingerprint_hex(),
        None,
        "id strings never synthesize a space identity"
    );
    cleanup(&path);

    // Empty-revision v1 header: kept verbatim as Some("").
    let path = temp_index_path("from_fsvi_empty_revision");
    cleanup(&path);
    let mut writer =
        crate::VectorIndex::create(&path, "legacy-embedder", dim).expect("create legacy v1 writer");
    writer
        .write_record("doc-0", &make_normalized_vec(dim, 1.0))
        .expect("write v1 row");
    writer.finish().expect("finish v1 file");
    let index = InMemoryVectorIndex::from_fsvi(&path).expect("load v1 file into memory");
    assert_eq!(index.embedder_id(), Some("legacy-embedder"));
    assert_eq!(
        index.embedder_revision(),
        Some(""),
        "an empty header revision is the header's content, not absence"
    );
    cleanup(&path);
}

#[test]
fn constructor_embedder_identity_rules() {
    // Bare vectors: no source header, no declared space — both stay the
    // typed None (nothing to retain, nothing fabricated).
    let dim = 8;
    let (doc_ids, vectors) = identity_rows(dim, 2);
    let bare = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).expect("build");
    assert_eq!(bare.embedder_id(), None);
    assert_eq!(bare.embedder_revision(), None);

    // Declared space: the same id/revision rule the production v2 writer
    // uses (`create_v2` writes logical_model_id / immutable_revision).
    let bundle = EmbeddingIdentityBundleV1::explicit_test_model("declared-model", 8);
    let (doc_ids, vectors) = identity_rows(dim, 2);
    let declared =
        InMemoryVectorIndex::from_vectors_with_identity(doc_ids, vectors, dim, &bundle.space)
            .expect("build identified index");
    assert_eq!(declared.embedder_id(), Some("declared-model"));
    assert_eq!(
        declared.embedder_revision(),
        Some(bundle.space.immutable_revision.as_str())
    );
}

#[test]
fn admitted_v2_load_preserves_embedder_identity_strings() {
    // The admitted-v2 in-memory load must retain the artifact header's
    // id/revision strings alongside the space fingerprint C3 already
    // preserves. Own directory: admission fails closed with
    // DirectoryChangedDuringRead when sibling test files churn the dir.
    let dir = std::env::temp_dir()
        .join("frankensearch_in_memory_tests")
        .join("admitted_v2_embedder_identity_dir");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create isolated admission dir");
    let path = dir.join("admitted_v2_embedder_identity.fsvi");
    let dim = 8;
    let (doc_ids, vectors) = identity_rows(dim, 3);
    let rows: Vec<(String, Vec<f32>)> = doc_ids.into_iter().zip(vectors).collect();
    let (binding, expected_space) =
        write_fsvi_v2_fixture(&path, "v2-embedder-model", dim, 13, &rows);

    let admitted = crate::VectorIndex::open_admitted_v2(&path, &binding)
        .expect("exact admission of the v2 fixture");
    let index = InMemoryVectorIndex::from_admitted_v2(&admitted)
        .expect("load admitted v2 artifact into memory");
    // create_v2 writes logical_model_id / immutable_revision as the
    // header strings; explicit_test_model pins the revision value.
    assert_eq!(index.embedder_id(), Some("v2-embedder-model"));
    assert_eq!(index.embedder_revision(), Some("explicit-test-v1"));
    assert_eq!(index.space_fingerprint_hex(), Some(expected_space.as_str()));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn in_memory_two_tier_exposes_per_tier_embedder_identity() {
    let dim = 8;
    let fast_bundle = EmbeddingIdentityBundleV1::explicit_test_model("tier-fast-model", 8);
    let (doc_ids, vectors) = identity_rows(dim, 2);
    let fast =
        InMemoryVectorIndex::from_vectors_with_identity(doc_ids, vectors, dim, &fast_bundle.space)
            .expect("build fast tier");
    let (doc_ids, vectors) = identity_rows(dim, 2);
    let quality = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).expect("build");

    // Identified fast + identity-less quality: each tier reports its own
    // state; the identified tier never bleeds into the absent one.
    let two_tier = InMemoryTwoTierIndex::new(fast, Some(quality));
    assert_eq!(two_tier.fast_embedder_id(), Some("tier-fast-model"));
    assert_eq!(
        two_tier.fast_embedder_revision(),
        Some(fast_bundle.space.immutable_revision.as_str())
    );
    assert_eq!(two_tier.quality_embedder_id(), None);
    assert_eq!(two_tier.quality_embedder_revision(), None);

    // No quality tier at all: same typed absence.
    let (doc_ids, vectors) = identity_rows(dim, 2);
    let fast_only = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).expect("build");
    let composed = InMemoryTwoTierIndex::new(fast_only, None);
    assert_eq!(composed.fast_embedder_id(), None);
    assert_eq!(composed.quality_embedder_id(), None);
    assert_eq!(composed.quality_embedder_revision(), None);
}

/// First production-shaped consumer of T2-C1's `verify_space_identity`:
/// an in-memory index built WITH identity supplies the index-side join
/// key, a query embedding bound through `BoundQueryEmbedding` supplies
/// the query side, and the verifier decides admission before the real
/// search path runs.
#[test]
fn bound_query_joins_in_memory_space_identity_through_verifier() -> Result<(), String> {
    let dim = 8;
    let fast_bundle = EmbeddingIdentityBundleV1::explicit_test_model("consumer-fast-model", 8);
    let other_bundle = EmbeddingIdentityBundleV1::explicit_test_model("consumer-quality-model", 8);

    let (doc_ids, vectors) = identity_rows(dim, 3);
    let index =
        InMemoryVectorIndex::from_vectors_with_identity(doc_ids, vectors, dim, &fast_bundle.space)
            .expect("build fast-space index");
    let (doc_ids, vectors) = identity_rows(dim, 3);
    let wrong_space_index =
        InMemoryVectorIndex::from_vectors_with_identity(doc_ids, vectors, dim, &other_bundle.space)
            .expect("build quality-space index");

    // Query side: bind the vector to the space that produced it (C1).
    let query = make_normalized_vec(dim, 1.0);
    let bound = BoundQueryEmbedding::new(query, fast_bundle).expect("bind query embedding");

    // Matching space: the verifier admits, then the REAL search path runs.
    let fingerprint = index
        .space_fingerprint_hex()
        .expect("identified index exposes its space");
    bound
        .verify_space_identity(fingerprint, "fast")
        .expect("same space must verify");
    let hits = index
        .search_top_k(bound.vector(), 1, None)
        .expect("search admitted query");
    assert_eq!(hits[0].doc_id, "doc-0", "query equals doc-0's vector");

    // The trap this slice exists for: the raw path CANNOT detect a
    // same-dimension wrong-space vector — it searches happily...
    let raw_hits = wrong_space_index
        .search_top_k(bound.vector(), 1, None)
        .expect("raw path accepts same-dimension wrong-space silently");
    assert!(!raw_hits.is_empty());

    // ...and the space-fingerprint join is what closes exactly that hole.
    let wrong_fingerprint = wrong_space_index
        .space_fingerprint_hex()
        .expect("identified index exposes its space");
    let error = bound
        .verify_space_identity(wrong_fingerprint, "quality")
        .expect_err("wrong space at equal dimension must reject");
    let rendered = format!("{error:?}");
    let SearchError::InvalidConfig { field, .. } = error else {
        return Err(format!("expected InvalidConfig, got {rendered}"));
    };
    assert_eq!(field, "query_embedding.quality.space_identity");
    Ok(())
}

#[test]
fn in_memory_two_tier_exposes_per_tier_space_identity() {
    let dim = 8;

    // Composed: each tier built with its own space identity exposes its
    // own, distinct fingerprint through the two-tier wrapper.
    let fast_bundle = EmbeddingIdentityBundleV1::explicit_test_model("two-tier-fast-model", 8);
    let quality_bundle =
        EmbeddingIdentityBundleV1::explicit_test_model("two-tier-quality-model", 8);
    let (doc_ids, vectors) = identity_rows(dim, 3);
    let fast =
        InMemoryVectorIndex::from_vectors_with_identity(doc_ids, vectors, dim, &fast_bundle.space)
            .expect("build fast tier");
    let (doc_ids, vectors) = identity_rows(dim, 3);
    let quality = InMemoryVectorIndex::from_vectors_with_identity(
        doc_ids,
        vectors,
        dim,
        &quality_bundle.space,
    )
    .expect("build quality tier");
    let two_tier = InMemoryTwoTierIndex::new(fast, Some(quality));
    assert_eq!(
        two_tier.fast_space_fingerprint_hex(),
        Some(fast_bundle.space.fingerprint().as_str())
    );
    assert_eq!(
        two_tier.quality_space_fingerprint_hex(),
        Some(quality_bundle.space.fingerprint().as_str())
    );
    assert_ne!(
        two_tier.fast_space_fingerprint_hex(),
        two_tier.quality_space_fingerprint_hex(),
        "distinct models must expose distinct per-tier spaces"
    );

    // from_dir loads through the v1-only pathname loader: absence stays
    // typed through the directory path — no tier identity is fabricated.
    static DIR_NONCE: AtomicU64 = AtomicU64::new(0);
    let nonce = DIR_NONCE.fetch_add(1, AtomicOrdering::Relaxed);
    let dir = std::env::temp_dir()
        .join("frankensearch_in_memory_tests")
        .join(format!("two_tier_space_identity-{nonce}"));
    // Scrub any stale state a previously interrupted run left behind.
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create two-tier dir");
    let fast_path = dir.join(crate::two_tier::VECTOR_INDEX_FAST_FILENAME);
    let mut writer = crate::VectorIndex::create(&fast_path, "legacy-fast", dim)
        .expect("create legacy v1 fast tier");
    writer
        .write_record("doc-0", &make_normalized_vec(dim, 1.0))
        .expect("write v1 row");
    writer.finish().expect("finish v1 fast tier");
    let loaded = InMemoryTwoTierIndex::from_dir(&dir).expect("load v1 two-tier dir");
    assert_eq!(loaded.fast_space_fingerprint_hex(), None);
    assert_eq!(loaded.quality_space_fingerprint_hex(), None);
    let _ = std::fs::remove_dir_all(&dir);

    // new(): a composed two-tier without a quality index has no
    // quality-tier space to verify against — typed absence again.
    let (doc_ids, vectors) = identity_rows(dim, 2);
    let fast_only = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).expect("build");
    let composed = InMemoryTwoTierIndex::new(fast_only, None);
    assert_eq!(composed.fast_space_fingerprint_hex(), None);
    assert_eq!(composed.quality_space_fingerprint_hex(), None);
}
