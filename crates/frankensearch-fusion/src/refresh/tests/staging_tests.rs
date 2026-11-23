use super::*;

/// Finding-1 regression + composite-authority guard: an attested,
/// same-producer, space-matching v2 canonical generation is recognized
/// through inspection/admission (never plain-opened), the identity gates
/// PASS, and the cycle still refuses pre-drain with the typed
/// composite-generation-authority reason — with ZERO side effects: no
/// drain, no embed, no write, no staging.
#[test]
fn live_v2_same_producer_cycle_refuses_composite_authority_with_zero_side_effects() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let embedder = Arc::new(StubEmbedder::new("v2-stub", V2_DIM));
        let (dir, cache, _binding, fast_path) = v2_canonical_fixture(
            "v2-composite-guard",
            embedder.identity_bundle(),
            &[("doc-old", normalized(V2_DIM, 0.5))],
            7,
        );
        let canonical_bytes = std::fs::read(&fast_path).expect("read canonical");

        let queue = make_queue(100);
        submit(&queue, "doc-new", "new document");
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            embedder.clone(),
            cache,
        );

        let error = worker
            .run_cycle(&cx)
            .await
            .expect_err("canonical publication must be refused until composite authority");
        assert_invalid_config(
            &error,
            "refresh.canonical_publication",
            "composite-generation-authority-unavailable",
        );
        assert!(
            invalid_config_reason(&error).contains("bd-xomn"),
            "reason must name the dependency"
        );

        // Zero embed/drain/write/stage side effects. The spy is the
        // direct proof; the docs_embedded metric below corroborates.
        assert_no_embed_invocations_since(&embedder, 0);
        assert_eq!(queue.pending_count(), 1, "refusal must fire before drain");
        let snap = worker.metrics().snapshot();
        assert_eq!(snap.docs_embedded, 0);
        assert_eq!(snap.index_rebuilds, 0);
        assert_eq!(snap.rebuild_failures, 0);
        assert_eq!(
            std::fs::read(&fast_path).expect("reread canonical"),
            canonical_bytes,
            "admission/inspection must not mutate the canonical artifact"
        );
        assert!(
            !dir.join(STAGED_V2_DIR_NAME).exists(),
            "run_cycle must not stage as a side effect"
        );
        let drained = queue.drain_batch();
        assert_eq!(drained[0].retry_count, 0, "no retry budget consumed");
    });
}

/// bd-ctzo C5: the typed activation is consumed OUTSIDE the index crate,
/// on the one production object that holds all three of its inputs — the
/// retained per-tier owners and the exact bindings the generation was
/// proven with.
///
/// Paired in one test: the staged generation's own producing space
/// activates and serves, and a same-dimension foreign space is refused
/// before any read.
#[test]
fn staged_generation_activates_typed_owner_backed_search() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let embedder = Arc::new(StubEmbedder::new("v2-activation-stub", V2_DIM));
        let (dir, cache, _binding, _fast_path) = v2_canonical_fixture(
            "v2-ctzo-activation",
            embedder.identity_bundle(),
            &[
                ("old-1", normalized(V2_DIM, 0.25)),
                ("old-2", normalized(V2_DIM, 0.75)),
            ],
            7,
        );
        let queue = make_queue(100);
        submit(&queue, "doc-3", "a brand new document");
        let jobs = queue.drain_batch();
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            embedder.clone(),
            cache,
        );
        let staged = worker
            .stage_identity_bound_generation(&cx, &jobs)
            .await
            .expect("staging must succeed");

        // A query bound in the STAGED GENERATION'S OWN producing space.
        // Storage is deliberately the in-process f32 contract: a query is
        // a Vec<f32> and can never carry the artifact's persisted f16
        // storage identity, which is why activation joins space and
        // producer rather than whole bundles.
        let mut query_identity = embedder.identity_bundle().clone();
        query_identity.storage.quantization = QuantizationFormat::F32;
        "in-memory-f32".clone_into(&mut query_identity.storage.format);
        "native-f32-values".clone_into(&mut query_identity.storage.endianness);
        let query = BoundQueryEmbedding::new(normalized(V2_DIM, 0.25), query_identity)
            .expect("bind a query in the staged generation's space");
        let embeddings = TieredQueryEmbeddings::fast_only(query);

        let activated = staged
            .activate_search(&embeddings)
            .expect("the generation's own space must activate");
        let hits = activated.search_fast(1).expect("owner-backed search");
        assert_eq!(hits.len(), 1, "the activated capability serves reads");
        assert_eq!(
            activated
                .fast()
                .expect("fast tier activated")
                .generation_sequence(),
            staged.fast_binding.generation().sequence,
            "coverage is read from the retained owner witness"
        );

        // Same dimension, foreign space: refused before any read.
        let foreign = Arc::new(StubEmbedder::new("v2-foreign-stub", V2_DIM));
        let mut foreign_identity = foreign.identity_bundle().clone();
        foreign_identity.storage.quantization = QuantizationFormat::F32;
        "in-memory-f32".clone_into(&mut foreign_identity.storage.format);
        "native-f32-values".clone_into(&mut foreign_identity.storage.endianness);
        let foreign_query = BoundQueryEmbedding::new(normalized(V2_DIM, 0.25), foreign_identity)
            .expect("bind a foreign-space query");
        let foreign_embeddings = TieredQueryEmbeddings::fast_only(foreign_query);
        let error = staged
            .activate_search(&foreign_embeddings)
            .expect_err("a same-dimension foreign space must be refused");
        assert!(
            matches!(
                error,
                SearchError::InvalidConfig { ref field, .. }
                    if field == "query_embedding.fast.space_identity"
            ),
            "got {error:?}"
        );
    });
}

/// Red proof (a), positive re-enablement: the same-producer, attested,
/// space-matching merge the containment refused is now staged
/// successfully — carried rows plus updates plus new documents — and the
/// replacement's identity is verified from the republished artifact's
/// OWN header bytes through `open_admitted_v2` (reviewer caution: never
/// only against the in-memory bundle).
#[test]
fn staging_same_producer_merges_and_republishes_attested_v2() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let embedder = Arc::new(StubEmbedder::new("v2-stub", V2_DIM));
        let (dir, cache, canonical_binding, fast_path) = v2_canonical_fixture(
            "v2-stage-positive",
            embedder.identity_bundle(),
            &[
                ("old-1", normalized(V2_DIM, 0.25)),
                ("old-2", normalized(V2_DIM, 0.75)),
            ],
            7,
        );
        let canonical_bytes = std::fs::read(&fast_path).expect("read canonical");

        let queue = make_queue(100);
        submit(&queue, "old-2", "updated content for old-2");
        submit(&queue, "doc-3", "a brand new document");
        let jobs = queue.drain_batch();
        assert_eq!(jobs.len(), 2);

        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            embedder.clone(),
            cache,
        );
        let staged = worker
            .stage_identity_bound_generation(&cx, &jobs)
            .await
            .expect("same-producer attested merge must stage successfully");

        // The staged two-tier view is attested and carries all rows.
        assert_eq!(staged.index.doc_count(), 3);
        assert!(staged.index.fast_identity_is_attested());
        assert_eq!(
            staged.index.fast_space_fingerprint_hex(),
            Some(embedder.identity_bundle().space.fingerprint().as_str())
        );
        assert_eq!(
            staged.fast_binding.generation().sequence,
            8,
            "replacement generation must succeed the attested sequence 7"
        );

        // REVIEWER CAUTION: prove the identity from the artifact's own
        // header bytes via the only legitimate v2 open path.
        let staged_owner = VectorIndex::open_admitted_v2(&staged.fast_path, &staged.fast_binding)
            .expect("staged artifact must admit exactly");
        assert_eq!(
            fingerprint_hex(&staged_owner.identity_v2().space_fingerprint),
            embedder.identity_bundle().space.fingerprint(),
            "the header of the republished artifact must carry the producing space \
             fingerprint bit-for-bit"
        );
        let staged_vectors = admitted_vectors_by_doc(&staged_owner);
        assert_eq!(staged_vectors.len(), 3);

        // Carried row is byte-stable; updated row actually changed.
        let canonical_owner = VectorIndex::open_admitted_v2(&fast_path, &canonical_binding)
            .expect("canonical artifact still admits");
        let canonical_vectors = admitted_vectors_by_doc(&canonical_owner);
        assert_eq!(
            staged_vectors.get("old-1"),
            canonical_vectors.get("old-1"),
            "carried rows must be preserved exactly"
        );
        assert_ne!(
            staged_vectors.get("old-2"),
            canonical_vectors.get("old-2"),
            "updated rows must carry the NEW embedding"
        );
        assert!(staged_vectors.contains_key("doc-3"));

        // Staging is non-canonical and queue-neutral.
        assert!(staged.fast_path.starts_with(dir.join(STAGED_V2_DIR_NAME)));
        assert_eq!(
            std::fs::read(&fast_path).expect("reread canonical"),
            canonical_bytes,
            "staging must not touch the canonical generation"
        );
        assert_eq!(queue.pending_count(), 0);
    });
}

/// Red proof (b): a cross-space attempt is rejected with the NEW typed
/// space-identity reason — not the generic republication refusal — both
/// at the staging seam and pre-drain in the canonical cycle.
#[test]
fn cross_space_attempt_rejects_with_typed_space_identity() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        // Canonical generation from model alpha; executing embedder is
        // model beta at the SAME dimension (the case dimension checks
        // can never catch).
        let alpha = EmbeddingIdentityBundleV1::explicit_test_model(
            "canonical-model-alpha",
            u32::try_from(V2_DIM).expect("dim"),
        );
        let (dir, cache, _binding, fast_path) = v2_canonical_fixture(
            "v2-cross-space",
            &alpha,
            &[("doc-old", normalized(V2_DIM, 0.5))],
            3,
        );
        let canonical_bytes = std::fs::read(&fast_path).expect("read canonical");
        let beta = Arc::new(StubEmbedder::new("beta-stub", V2_DIM));
        let beta_space_fingerprint = beta.identity_bundle().space.fingerprint();
        assert_ne!(beta_space_fingerprint, alpha.space.fingerprint());

        let queue = make_queue(100);
        submit(&queue, "doc-new", "new document");
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            beta.clone(),
            cache,
        );

        // Canonical cycle: typed cross-space refusal, pre-drain.
        let error = worker
            .run_cycle(&cx)
            .await
            .expect_err("cross-space merge must be refused");
        assert_invalid_config(
            &error,
            "refresh.fast_space_identity",
            &beta_space_fingerprint,
        );
        assert!(
            invalid_config_reason(&error).contains(&alpha.space.fingerprint()),
            "reason must carry the attested space fingerprint"
        );
        assert_eq!(queue.pending_count(), 1);
        assert_no_embed_invocations_since(&beta, 0);

        // Staging seam: same typed refusal.
        let jobs = queue.drain_batch();
        let error = worker
            .stage_identity_bound_generation(&cx, &jobs)
            .await
            .expect_err("cross-space staging must be refused");
        assert_invalid_config(
            &error,
            "refresh.fast_space_identity",
            &beta_space_fingerprint,
        );
        assert_no_embed_invocations_since(&beta, 0);
        assert_eq!(
            std::fs::read(&fast_path).expect("reread canonical"),
            canonical_bytes
        );
    });
}

/// Red proof (d): a producer that IS conformance-certified against the
/// attested producer (same space, byte-identical golden certificate,
/// different attested backend) — the exact
/// `ConformanceCompatibleProducer` shape — is refused with the typed
/// `executing-producer-attestation-unavailable` reason carrying both
/// producer fingerprints (truthful telemetry).
#[test]
fn certified_sibling_producer_refused_attestation_unavailable() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let base = EmbeddingIdentityBundleV1::explicit_test_model(
            "shared-space-model",
            u32::try_from(V2_DIM).expect("dim"),
        );
        let mut sibling = base.clone();
        "alternate-conformant-backend".clone_into(&mut sibling.producer.backend);
        sibling.validate().expect("sibling must validate");
        assert!(
            conformance_compatible(&sibling, &base) && conformance_compatible(&base, &sibling),
            "fixture must be the certified conformance-compatible shape"
        );
        assert_ne!(base.producer.fingerprint(), sibling.producer.fingerprint());
        assert_eq!(base.space.fingerprint(), sibling.space.fingerprint());

        let (dir, cache, _binding, fast_path) = v2_canonical_fixture(
            "v2-certified-sibling",
            &base,
            &[("doc-old", normalized(V2_DIM, 0.5))],
            3,
        );
        let canonical_bytes = std::fs::read(&fast_path).expect("read canonical");
        let executing = Arc::new(StubEmbedder::with_identity(
            "sibling-stub",
            V2_DIM,
            sibling.clone(),
        ));

        let queue = make_queue(100);
        submit(&queue, "doc-new", "new document");
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            executing.clone(),
            cache,
        );

        let error = worker
            .run_cycle(&cx)
            .await
            .expect_err("certified sibling must still be refused (SameProducer only)");
        assert_invalid_config(
            &error,
            "refresh.fast_producer_conformance",
            "executing-producer-attestation-unavailable",
        );
        let reason = invalid_config_reason(&error);
        assert!(
            reason.contains(&base.producer.fingerprint())
                && reason.contains(&sibling.producer.fingerprint()),
            "truthful telemetry must carry both producer fingerprints: {reason}"
        );
        assert_no_embed_invocations_since(&executing, 0);

        let jobs = queue.drain_batch();
        let error = worker
            .stage_identity_bound_generation(&cx, &jobs)
            .await
            .expect_err("staging must refuse the certified sibling too");
        assert_invalid_config(
            &error,
            "refresh.fast_producer_conformance",
            "executing-producer-attestation-unavailable",
        );
        assert_no_embed_invocations_since(&executing, 0);
        assert_eq!(
            std::fs::read(&fast_path).expect("reread canonical"),
            canonical_bytes
        );
    });
}

/// Red proof (e), reviewer addition (carry-forward-#1 attack shape): a
/// same-space, DIFFERENT-producer bundle whose golden-vector certificate
/// is a byte-identical COPY of the attested producer's certificate must
/// REJECT. Trivial under SameProducer-only — and exactly the regression
/// guard that must stay red when the conformance-compatible lane opens:
/// certificate bytes can be copied; executing-producer attestation
/// cannot.
#[test]
fn copied_certificate_foreign_producer_rejects() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let victim = EmbeddingIdentityBundleV1::explicit_test_model(
            "victim-space-model",
            u32::try_from(V2_DIM).expect("dim"),
        );
        let mut attacker = victim.clone();
        "attacker-backend".clone_into(&mut attacker.producer.backend);
        "attacker-rev-1".clone_into(&mut attacker.producer.implementation_revision);
        attacker.validate().expect("attacker bundle validates");
        assert_eq!(
            attacker.producer.golden_vectors, victim.producer.golden_vectors,
            "the certificate is a byte-identical copy"
        );
        assert_ne!(
            attacker.producer.fingerprint(),
            victim.producer.fingerprint()
        );
        assert_eq!(attacker.space.fingerprint(), victim.space.fingerprint());

        let (dir, cache, _binding, fast_path) = v2_canonical_fixture(
            "v2-copied-cert",
            &victim,
            &[("doc-old", normalized(V2_DIM, 0.5))],
            3,
        );
        let canonical_bytes = std::fs::read(&fast_path).expect("read canonical");
        let executing = Arc::new(StubEmbedder::with_identity(
            "attacker-stub",
            V2_DIM,
            attacker,
        ));

        let queue = make_queue(100);
        submit(&queue, "doc-new", "new document");
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            executing.clone(),
            cache,
        );

        let error = worker
            .run_cycle(&cx)
            .await
            .expect_err("copied certificate must never admit a foreign producer");
        assert_invalid_config(
            &error,
            "refresh.fast_producer_conformance",
            "executing-producer-attestation-unavailable",
        );
        assert_no_embed_invocations_since(&executing, 0);

        let jobs = queue.drain_batch();
        let error = worker
            .stage_identity_bound_generation(&cx, &jobs)
            .await
            .expect_err("staging must reject the copied certificate too");
        assert_invalid_config(
            &error,
            "refresh.fast_producer_conformance",
            "executing-producer-attestation-unavailable",
        );
        assert_no_embed_invocations_since(&executing, 0);
        assert_eq!(
            std::fs::read(&fast_path).expect("reread canonical"),
            canonical_bytes
        );
    });
}

/// The per-record seam half of guard 7: an embedder whose BOUND outputs
/// carry a conformance-certified sibling producer while its `identity()`
/// attests another is caught at the bound-record seam
/// (`verify_producer_conformance` per embedding), not just at the
/// artifact gate.
#[test]
fn two_faced_embedder_refused_at_bound_record_seam() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let public = EmbeddingIdentityBundleV1::explicit_test_model(
            "two-faced-model",
            u32::try_from(V2_DIM).expect("dim"),
        );
        let mut bound_side = public.clone();
        "alternate-conformant-backend".clone_into(&mut bound_side.producer.backend);
        bound_side.validate().expect("bound-side validates");
        assert!(conformance_compatible(&bound_side, &public));

        let dir = temp_index_dir("two-faced");
        let queue = make_queue(100);
        let cache = make_cache(&dir, V2_DIM);
        let embedder = Arc::new(TwoFacedEmbedder {
            public_identity: public,
            bound_identity: bound_side,
            dimension: V2_DIM,
        });
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            embedder,
            cache,
        );

        submit(&queue, "doc-1", "some document");
        let jobs = queue.drain_batch();
        let error = worker
            .stage_identity_bound_generation(&cx, &jobs)
            .await
            .expect_err("bound records from a different producer must be refused");
        assert_invalid_config(
            &error,
            "refresh.fast_producer_conformance",
            "executing-producer-attestation-unavailable",
        );
    });
}

/// Finding-2 pin: canonical installation of a staged (proven) generation
/// is refused with the typed composite-generation-authority reason, and
/// the canonical directory is untouched by both staging and the refusal.
#[test]
fn publish_staged_canonical_refuses_split_generation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("staged-publish-guard");
        let queue = make_queue(100);
        let cache = make_cache(&dir, V2_DIM);
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let seed_bytes = std::fs::read(&fast_path).expect("read v1 seed");
        let embedder = Arc::new(StubEmbedder::new("v2-stub", V2_DIM));
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            embedder.clone(),
            cache,
        );

        submit(&queue, "doc-1", "some document");
        let jobs = queue.drain_batch();
        let staged = worker
            .stage_identity_bound_generation(&cx, &jobs)
            .await
            .expect("staging over an empty seed must succeed");
        assert!(staged.fast_path.exists());
        assert!(staged.index.fast_identity_is_attested());

        let embed_baseline = embedder.embed_invocations();
        let error = worker
            .publish_staged_canonical(&staged)
            .expect_err("split canonical publication must be refused in this slice");
        assert_invalid_config(
            &error,
            "refresh.canonical_publication",
            "composite-generation-authority-unavailable",
        );
        assert!(invalid_config_reason(&error).contains("bd-xomn"));
        assert_no_embed_invocations_since(&embedder, embed_baseline);

        assert_eq!(
            std::fs::read(&fast_path).expect("reread canonical seed"),
            seed_bytes,
            "neither staging nor the publication refusal may touch the canonical files"
        );
    });
}

/// Tombstoned rows in an attested generation stay dead through the merge.
#[test]
fn staging_never_resurrects_tombstoned_rows() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let embedder = Arc::new(StubEmbedder::new("v2-stub", V2_DIM));
        let dir = temp_index_dir("v2-tombstones");
        let cache = make_cache_with_fallback_seed(&dir, V2_DIM);
        let binding = v2_binding(embedder.identity_bundle(), 4);
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        {
            let mut writer =
                VectorIndex::create_v2(&fast_path, binding.clone()).expect("create_v2");
            writer
                .write_record("doc-keep", &normalized(V2_DIM, 0.5))
                .expect("live row");
            writer
                .write_tombstone_record("doc-dead", &normalized(V2_DIM, 0.7))
                .expect("tombstone row");
            writer.finish().expect("finish");
        }

        let queue = make_queue(100);
        submit(&queue, "doc-b", "a new document");
        let jobs = queue.drain_batch();
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            embedder,
            cache,
        );
        let staged = worker
            .stage_identity_bound_generation(&cx, &jobs)
            .await
            .expect("staging must succeed");

        let owner = VectorIndex::open_admitted_v2(&staged.fast_path, &staged.fast_binding)
            .expect("admit staged");
        let vectors = admitted_vectors_by_doc(&owner);
        assert!(vectors.contains_key("doc-keep"));
        assert!(vectors.contains_key("doc-b"));
        assert!(
            !vectors.contains_key("doc-dead"),
            "a tombstoned document must never be resurrected by the merge"
        );
    });
}

/// Quality tier end-to-end: carried + new quality rows are staged under
/// the quality embedder's identity and admit exactly.
#[test]
fn staging_quality_tier_carries_and_binds() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Arc::new(StubEmbedder::new("v2-fast", V2_DIM));
        let quality = Arc::new(StubEmbedder::new("v2-quality", 16));
        let (dir, cache, _fast_binding, _fast_path) = v2_canonical_fixture(
            "v2-quality-carry",
            fast.identity_bundle(),
            &[("old-1", normalized(V2_DIM, 0.25))],
            2,
        );
        let quality_binding = v2_binding(quality.identity_bundle(), 2);
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
        write_v2_tier(
            &quality_path,
            &quality_binding,
            &[("old-1", normalized(16, 0.5))],
        );

        let queue = make_queue(100);
        submit(&queue, "doc-2", "a second document");
        let jobs = queue.drain_batch();
        let worker = RefreshWorker::new(RefreshWorkerConfig::new(&dir), queue.clone(), fast, cache)
            .with_quality_embedder(quality.clone());

        let staged = worker
            .stage_identity_bound_generation(&cx, &jobs)
            .await
            .expect("two-tier staging must succeed");
        assert!(staged.quality_path.is_some());
        assert!(staged.index.quality_identity_is_attested());

        let quality_owner = VectorIndex::open_admitted_v2(
            staged.quality_path.as_ref().expect("quality path"),
            staged.quality_binding.as_ref().expect("quality binding"),
        )
        .expect("staged quality admits");
        assert_eq!(
            fingerprint_hex(&quality_owner.identity_v2().space_fingerprint),
            quality.identity_bundle().space.fingerprint(),
            "staged quality header must carry the quality producer's space"
        );
        let vectors = admitted_vectors_by_doc(&quality_owner);
        assert!(vectors.contains_key("old-1"), "carried quality row");
        assert!(vectors.contains_key("doc-2"), "new quality row");
    });
}

/// The staged v2 pair analog of the v1
/// `finish_stamps_both_tiers_with_one_nonzero_publication_nonce`: both
/// staged tiers must carry one shared NON-ZERO publication nonce
/// (bd-miio8), so the pair is bound as a single publication instead of
/// reading 0 == 0 and slipping past the mixed-generation check.
#[test]
fn staging_stamps_both_tiers_with_one_nonzero_publication_nonce() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Arc::new(StubEmbedder::new("v2-fast", V2_DIM));
        let quality = Arc::new(StubEmbedder::new("v2-quality", 16));
        let (dir, cache, fast_binding, fast_path) = v2_canonical_fixture(
            "v2-pair-nonce",
            fast.identity_bundle(),
            &[("old-1", normalized(V2_DIM, 0.25))],
            2,
        );
        let quality_binding = v2_binding(quality.identity_bundle(), 2);
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
        // Stamp the CANONICAL pair with nonce 7 so the successor
        // assertion below distinguishes real derivation (7 -> 8) from a
        // hard-coded 1 (round-2 hardening).
        write_v2_tier_with_nonce(
            &fast_path,
            &fast_binding,
            &[("old-1", normalized(V2_DIM, 0.25))],
            7,
        );
        write_v2_tier_with_nonce(
            &quality_path,
            &quality_binding,
            &[("old-1", normalized(16, 0.5))],
            7,
        );

        let queue = make_queue(100);
        submit(&queue, "doc-2", "a second document");
        let jobs = queue.drain_batch();
        let worker = RefreshWorker::new(RefreshWorkerConfig::new(&dir), queue.clone(), fast, cache)
            .with_quality_embedder(quality.clone());

        let staged = worker
            .stage_identity_bound_generation(&cx, &jobs)
            .await
            .expect("two-tier staging must succeed");

        let fast_owner = VectorIndex::open_admitted_v2(&staged.fast_path, &staged.fast_binding)
            .expect("staged fast admits");
        let quality_owner = VectorIndex::open_admitted_v2(
            staged.quality_path.as_ref().expect("quality path"),
            staged.quality_binding.as_ref().expect("quality binding"),
        )
        .expect("staged quality admits");
        let fast_nonce = fast_owner.metadata().publication_nonce;
        let quality_nonce = quality_owner.metadata().publication_nonce;
        assert_ne!(
            fast_nonce, 0,
            "a staged v2 pair must carry a real publication identity"
        );
        assert_eq!(
            fast_nonce, quality_nonce,
            "both staged tiers share one publication"
        );
        assert_eq!(
            fast_nonce, 8,
            "the staged nonce must be the canonical pair's successor (7 -> 8)"
        );
    });
}

/// A canonical pair whose tiers carry DISAGREEING publication nonces is a
/// mixed-generation pair (crash between tier installs); admission must
/// refuse it rather than launder both tiers into one freshly-stamped
/// staged generation (round-2 hardening of bd-miio8).
#[test]
fn staging_refuses_a_mixed_nonce_canonical_pair() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Arc::new(StubEmbedder::new("v2-fast", V2_DIM));
        let quality = Arc::new(StubEmbedder::new("v2-quality", 16));
        let (dir, cache, fast_binding, fast_path) = v2_canonical_fixture(
            "v2-pair-nonce-mismatch",
            fast.identity_bundle(),
            &[("old-1", normalized(V2_DIM, 0.25))],
            2,
        );
        let quality_binding = v2_binding(quality.identity_bundle(), 2);
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
        write_v2_tier_with_nonce(
            &fast_path,
            &fast_binding,
            &[("old-1", normalized(V2_DIM, 0.25))],
            7,
        );
        write_v2_tier_with_nonce(
            &quality_path,
            &quality_binding,
            &[("old-1", normalized(16, 0.5))],
            3,
        );

        let queue = make_queue(100);
        submit(&queue, "doc-2", "a second document");
        let jobs = queue.drain_batch();
        let worker = RefreshWorker::new(RefreshWorkerConfig::new(&dir), queue.clone(), fast, cache)
            .with_quality_embedder(quality.clone());

        let error = worker
            .stage_identity_bound_generation(&cx, &jobs)
            .await
            .expect_err("a mixed-nonce canonical pair must refuse admission");
        assert_invalid_config(
            &error,
            "refresh.canonical_pair_publication_nonce",
            "fast=7 quality=3",
        );
    });
}

/// Carry-forward regression: an admitted v2 QUALITY tier with no
/// admitted fast tier (`AttestedV2 { fast: None, quality: Some(_) }`) is
/// reachable, and its live quality rows must still be carried into the
/// staged generation — landing in the staged quality tier only, never as
/// fabricated fast rows. The former merge nested the quality carry
/// inside the fast-tier `if let`, silently dropping every carried
/// quality row while still stamping successor lineage.
#[test]
fn staging_quality_only_generation_carries_quality_rows() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Arc::new(StubEmbedder::new("v2-fast", V2_DIM));
        let quality = Arc::new(StubEmbedder::new("v2-quality", 16));
        let dir = temp_index_dir("v2-quality-only-carry");
        let cache = make_cache_with_fallback_seed(&dir, V2_DIM);
        // Attested v2 quality tier; the fast tier is only the empty v1
        // fallback seed, so admission yields fast: None, quality: Some.
        let quality_binding = v2_binding(quality.identity_bundle(), 4);
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
        write_v2_tier(
            &quality_path,
            &quality_binding,
            &[("old-q", normalized(16, 0.5))],
        );

        let queue = make_queue(100);
        submit(&queue, "doc-2", "a second document");
        let jobs = queue.drain_batch();
        let worker = RefreshWorker::new(RefreshWorkerConfig::new(&dir), queue.clone(), fast, cache)
            .with_quality_embedder(quality.clone());

        let staged = worker
            .stage_identity_bound_generation(&cx, &jobs)
            .await
            .expect("quality-only generation must stage successfully");

        let quality_owner = VectorIndex::open_admitted_v2(
            staged.quality_path.as_ref().expect("quality path"),
            staged.quality_binding.as_ref().expect("quality binding"),
        )
        .expect("staged quality admits");
        let quality_vectors = admitted_vectors_by_doc(&quality_owner);
        assert!(
            quality_vectors.contains_key("old-q"),
            "carried quality rows must survive without an admitted fast tier"
        );
        assert!(quality_vectors.contains_key("doc-2"), "new quality row");
        assert_eq!(
            staged
                .quality_binding
                .as_ref()
                .expect("quality binding")
                .generation()
                .sequence,
            5,
            "successor lineage must build on the attested quality sequence 4"
        );

        // The quality-only carried row must not fabricate a fast row.
        let fast_owner = VectorIndex::open_admitted_v2(&staged.fast_path, &staged.fast_binding)
            .expect("staged fast admits");
        let fast_vectors = admitted_vectors_by_doc(&fast_owner);
        assert!(
            !fast_vectors.contains_key("old-q"),
            "a carried quality-only row must never fabricate a fast row"
        );
        assert!(fast_vectors.contains_key("doc-2"), "new fast row");
    });
}

/// An attested quality tier with no quality embedder cannot be
/// republished identity-bound: typed refusal, pre-drain.
#[test]
fn live_v2_quality_without_quality_embedder_refuses_republication() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Arc::new(StubEmbedder::new("v2-fast", V2_DIM));
        let quality_bundle = EmbeddingIdentityBundleV1::explicit_test_model("v2-quality-model", 16);
        let (dir, cache, _fast_binding, _fast_path) = v2_canonical_fixture(
            "v2-quality-missing-embedder",
            fast.identity_bundle(),
            &[("old-1", normalized(V2_DIM, 0.25))],
            2,
        );
        let quality_binding = v2_binding(&quality_bundle, 2);
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
        write_v2_tier(
            &quality_path,
            &quality_binding,
            &[("old-1", normalized(16, 0.5))],
        );

        let queue = make_queue(100);
        submit(&queue, "doc-2", "a second document");
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            fast.clone(),
            cache,
        );

        let error = worker
            .run_cycle(&cx)
            .await
            .expect_err("attested quality without a quality embedder must refuse");
        assert_invalid_config(
            &error,
            "refresh.index_publication",
            "identity-bound-republication-unavailable",
        );
        assert_eq!(queue.pending_count(), 1, "pre-drain refusal");
        assert_no_embed_invocations_since(&fast, 0);
    });
}
