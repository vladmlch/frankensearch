use super::*;

/// Tantivy requires a per-writer floor, so an eight-thread pool needs at
/// least `8 * 15_000_000` bytes. Ask for that plus headroom, or the widest
/// case silently exercises a narrower pool than it claims.
const HEAP: usize = 8 * 15_000_000 + 64 * 1024 * 1024;

#[test]
fn shipping_auto_uses_the_pinned_selection_path_and_reports_no_width() {
    let mut index = TantivyIndex::in_memory_with_shipping_auto_writer(HEAP, true)
        .expect("shipping-auto writer");
    let attestation = index
        .take_benchmark_writer_attestation()
        .expect("successful benchmark writer mints one live attestation");
    let receipt = index
        .benchmark_writer_receipt()
        .cloned()
        .expect("shipping-auto stamps a receipt");

    assert_eq!(attestation.receipt(), &receipt);
    assert!(
        index.take_benchmark_writer_attestation().is_none(),
        "the live attestation is a one-shot capability"
    );
    assert_eq!(receipt.mode, BenchmarkWriterMode::ShippingAuto);
    assert_eq!(
        receipt.materialized_width,
        BenchmarkMaterializedWidth::Unobservable {
            reason: BenchmarkWidthUnobservableReason::EngineSelectedWidthNotExposed,
        }
    );
    // The unknown must stay unknown everywhere it is read.
    assert_eq!(receipt.materialized_width.authenticated(), None);
    assert_eq!(index.benchmark_materialized_writer_threads(), None);
    assert_eq!(receipt.writer_heap_bytes, HEAP);
    assert!(!receipt.writer_rearmed);
}

#[test]
fn identical_fixed_constructors_mint_distinct_live_attestations() {
    let mut first =
        TantivyIndex::in_memory_with_benchmark_config(HEAP, 4, true).expect("first fixed writer");
    let mut second =
        TantivyIndex::in_memory_with_benchmark_config(HEAP, 4, true).expect("second fixed writer");
    let first_attestation = first
        .take_benchmark_writer_attestation()
        .expect("first live attestation");
    let second_attestation = second
        .take_benchmark_writer_attestation()
        .expect("second live attestation");

    assert_ne!(
        first_attestation.construction_id(),
        second_attestation.construction_id(),
        "identical benchmark requests still construct separate live writers"
    );
    assert_eq!(
        first_attestation.receipt(),
        first.benchmark_writer_receipt().expect("first receipt")
    );
    assert_eq!(
        second_attestation.receipt(),
        second.benchmark_writer_receipt().expect("second receipt")
    );
    assert_eq!(
        first_attestation.receipt(),
        second_attestation.receipt(),
        "the diagnostic receipt may agree for identical real constructors"
    );
    assert_eq!(
        first_attestation.receipt().mode,
        BenchmarkWriterMode::Fixed { threads: 4 }
    );
    assert_eq!(
        first_attestation.receipt().materialized_width,
        BenchmarkMaterializedWidth::Authenticated(4)
    );
}

#[test]
fn fixed_width_is_authenticated_exactly() {
    for threads in [1usize, 3, 8] {
        let index = TantivyIndex::in_memory_with_benchmark_config(HEAP, threads, true)
            .expect("fixed writer");
        let receipt = index.benchmark_writer_receipt().expect("fixed receipt");

        assert_eq!(receipt.mode, BenchmarkWriterMode::Fixed { threads });
        assert_eq!(
            receipt.materialized_width,
            BenchmarkMaterializedWidth::Authenticated(threads)
        );
        assert_eq!(receipt.materialized_width.authenticated(), Some(threads));
        assert_eq!(index.benchmark_materialized_writer_threads(), Some(threads));
    }
}

#[test]
fn invalid_width_and_insufficient_heap_fail_closed() {
    assert!(matches!(
        TantivyIndex::in_memory_with_benchmark_config(HEAP, 0, true),
        Err(SearchError::InvalidConfig { .. })
    ));
    // A heap below Tantivy's per-writer floor must surface its typed
    // construction error, never a silently downgraded pool.
    assert!(TantivyIndex::in_memory_with_benchmark_config(1, 4, true).is_err());
    assert!(TantivyIndex::in_memory_with_shipping_auto_writer(1, true).is_err());
}

#[test]
fn shipping_and_fixed_receipts_are_distinguishable() {
    let auto = TantivyIndex::in_memory_with_shipping_auto_writer(HEAP, true)
        .expect("shipping-auto writer");
    let fixed = TantivyIndex::in_memory_with_benchmark_config(HEAP, 4, true).expect("fixed writer");
    let auto = auto
        .benchmark_writer_receipt()
        .expect("auto receipt")
        .clone();
    let fixed = fixed.benchmark_writer_receipt().expect("fixed receipt");

    assert_ne!(
        &auto, fixed,
        "the two modes must not produce equal receipts"
    );
    assert_ne!(auto.mode, fixed.mode);
    assert_ne!(auto.materialized_width, fixed.materialized_width);
    // Identity that is genuinely shared must still match, or the receipt
    // would be distinguishing on incidental noise instead of on mode.
    assert_eq!(auto.schema_fields, fixed.schema_fields);
    assert_eq!(auto.tokenizer_name, fixed.tokenizer_name);
}

#[test]
fn schema_and_oracle_identity_track_the_index_that_was_built() {
    let with_positions_index =
        TantivyIndex::in_memory_with_shipping_auto_writer(HEAP, true).expect("positions-on writer");
    let without_positions_index = TantivyIndex::in_memory_with_shipping_auto_writer(HEAP, false)
        .expect("positions-off writer");
    let with_positions = with_positions_index
        .benchmark_writer_receipt()
        .expect("receipt")
        .clone();
    let without_positions = without_positions_index
        .benchmark_writer_receipt()
        .expect("receipt");

    assert!(with_positions.positions);
    assert!(!without_positions.positions);
    assert_ne!(
        &with_positions, without_positions,
        "a positions mutation must change the receipt"
    );
    assert_eq!(with_positions.tokenizer_name, TOKENIZER_NAME);
    // The receipt's positions value must be the live indexing option, not a
    // remembered argument, so read the schema back independently.
    assert_eq!(
        positions_from_live_schema(&with_positions_index.index.schema()),
        Some(true)
    );
    assert_eq!(
        positions_from_live_schema(&without_positions_index.index.schema()),
        Some(false)
    );
    // Pin the exact field set this crate's schema builds, so the receipt is
    // read from the live schema rather than reporting a plausible shape.
    // A field added, renamed, or dropped must fail here.
    assert_eq!(
        with_positions.schema_fields,
        vec![
            "id".to_owned(),
            "content".to_owned(),
            "title".to_owned(),
            "metadata_json".to_owned(),
            "ord".to_owned(),
        ],
        "schema identity must match the fields build_schema_with_positions creates"
    );
    assert_eq!(
        without_positions.schema_fields, with_positions.schema_fields,
        "positions is an index option, not a field, so the field set must not move"
    );
}

#[test]
fn a_fresh_writer_never_claims_a_rearm() {
    for index in [
        TantivyIndex::in_memory_with_shipping_auto_writer(HEAP, true).expect("auto"),
        TantivyIndex::in_memory_with_benchmark_config(HEAP, 2, true).expect("fixed"),
    ] {
        assert!(
            !index
                .benchmark_writer_receipt()
                .expect("receipt")
                .writer_rearmed,
            "a freshly constructed writer must not report a rearm"
        );
    }
}

#[test]
fn the_construction_receipt_flips_after_a_rearm() {
    let mut index =
        TantivyIndex::in_memory_with_shipping_auto_writer(HEAP, true).expect("auto writer");
    let original_attestation = index
        .take_benchmark_writer_attestation()
        .expect("original live attestation");
    assert_eq!(
        index.benchmark_writer_receipt().expect("receipt").mode,
        BenchmarkWriterMode::ShippingAuto
    );

    let (mut rearmed, _join) = index
        .benchmark_join_workers_and_rearm(HEAP, 2)
        .expect("join and rearm");
    let rearmed_attestation = rearmed
        .take_benchmark_writer_attestation()
        .expect("replacement writer mints a fresh live attestation");
    let receipt = rearmed.benchmark_writer_receipt().expect("rearm receipt");

    // A rearm replaces the writer, so the construction receipt must stop
    // describing the original one: the mode moves to the fixed constructor
    // the rearm actually called, and the rearm flag is set.
    assert!(receipt.writer_rearmed);
    assert_eq!(receipt.mode, BenchmarkWriterMode::Fixed { threads: 2 });
    assert_eq!(
        receipt.materialized_width,
        BenchmarkMaterializedWidth::Authenticated(2)
    );
    assert_eq!(rearmed_attestation.receipt(), receipt);
    assert_ne!(
        original_attestation.construction_id(),
        rearmed_attestation.construction_id(),
        "rearming replaces the writer and must mint a new live identity"
    );
    // The receipt states the rearm's intent; this states the call it made.
    // Carrying the old writer's observation forward instead would leave a
    // rearmed index reporting Auto, which this rejects.
    assert_eq!(
        rearmed.observed_writer_call,
        super::WriterCall::Fixed(2),
        "the rearm must observe its own writer_with_num_threads call"
    );
}

#[test]
fn the_receipt_seed_cannot_drift_from_the_constructor_branch() {
    // Discriminating observer: the two factories differ only in which
    // Tantivy constructor they reach, and the receipt is seeded inside that
    // branch. If the seed were passed in beside the call instead, these two
    // could report the same mode while calling different constructors.
    type Factory = (&'static str, fn() -> TantivyIndex, BenchmarkWriterMode);
    let factories: [Factory; 2] = [
        (
            "Index::writer",
            || TantivyIndex::in_memory_with_shipping_auto_writer(HEAP, true).expect("auto"),
            BenchmarkWriterMode::ShippingAuto,
        ),
        (
            "Index::writer_with_num_threads",
            || TantivyIndex::in_memory_with_benchmark_config(HEAP, 8, true).expect("fixed"),
            BenchmarkWriterMode::Fixed { threads: 8 },
        ),
    ];
    for (constructor, factory, expected_mode) in factories {
        let index = factory();
        let receipt = index.benchmark_writer_receipt().expect("receipt");
        assert_eq!(
            receipt.mode, expected_mode,
            "receipt mode must name the constructor that ran: {constructor}"
        );
        assert_eq!(
            receipt.writer_heap_bytes, HEAP,
            "heap must come from the same branch as the mode: {constructor}"
        );
        assert_eq!(
            receipt.materialized_width.authenticated(),
            match expected_mode {
                BenchmarkWriterMode::ShippingAuto => None,
                BenchmarkWriterMode::Fixed { threads } => Some(threads),
            },
            "width authentication must follow the branch: {constructor}"
        );
    }
}

/// Heap that is exactly one thread's floor.
///
/// This is the label-independent discriminator. `writer_with_num_threads`
/// rejects a width whose per-thread share falls under Tantivy's floor,
/// while `Index::writer` *clamps* the width instead of failing. So at this
/// budget the two constructors disagree observably: a fixed eight-wide
/// construction must fail and a shipping-auto construction must succeed.
const ONE_THREAD_FLOOR_HEAP: usize = 15_000_000;

#[test]
fn the_constructor_is_discriminated_by_behaviour_not_by_its_label() {
    // If ShippingAuto secretly called writer_with_num_threads(8, ..) this
    // would fail; if Fixed{8} secretly called Index::writer it would
    // succeed. Neither can be rescued by a matching label.
    let auto = TantivyIndex::in_memory_with_shipping_auto_writer(ONE_THREAD_FLOOR_HEAP, true)
        .expect("Index::writer clamps the width instead of rejecting this budget");
    assert_eq!(
        auto.benchmark_writer_receipt().expect("receipt").mode,
        BenchmarkWriterMode::ShippingAuto
    );

    assert!(
        TantivyIndex::in_memory_with_benchmark_config(ONE_THREAD_FLOOR_HEAP, 8, true).is_err(),
        "writer_with_num_threads must reject eight writers sharing one writer's floor"
    );
}

#[test]
fn the_rearm_constructor_is_discriminated_the_same_way() {
    let index = TantivyIndex::in_memory_with_shipping_auto_writer(HEAP, true)
        .expect("auto writer at a comfortable budget");
    // The rearm must reconstruct through writer_with_num_threads, so it
    // inherits that constructor's floor rejection. Succeeding here would
    // prove it had fallen back to Index::writer.
    assert!(
        index
            .benchmark_join_workers_and_rearm(ONE_THREAD_FLOOR_HEAP, 8)
            .is_err(),
        "rearm must reconstruct at a pinned width, not through the auto path"
    );
}

#[test]
fn ordinary_constructors_claim_no_benchmark_receipt() {
    // Only an explicit benchmark plan may produce a receipt. An ordinary
    // index that happens to reach the same Tantivy call must not be able to
    // present itself as a screened candidate.
    let mut ordinary = TantivyIndex::in_memory().expect("ordinary in-memory index");
    assert!(ordinary.benchmark_writer_receipt().is_none());
    assert!(ordinary.benchmark_materialized_writer_threads().is_none());
    assert!(
        ordinary.take_benchmark_writer_attestation().is_none(),
        "ordinary construction must not mint a live benchmark capability"
    );

    // The pinned oracle only exists behind `tantivy-oracle`; this module is
    // gated on `bench-internals` alone, so its negative coverage has to be
    // conditional or `bench-internals` by itself stops compiling.
    #[cfg(feature = "tantivy-oracle")]
    {
        let mut oracle =
            TantivyIndex::in_memory_single_threaded_oracle().expect("single-threaded oracle");
        assert!(
            oracle.benchmark_writer_receipt().is_none(),
            "pinning a width is not the same as claiming a screening receipt"
        );
        assert!(
            oracle.benchmark_materialized_writer_threads().is_none(),
            "an unscreened oracle authenticates no width to a screening consumer"
        );
        assert!(
            oracle.take_benchmark_writer_attestation().is_none(),
            "an ordinary pinned oracle must not mint a live benchmark capability"
        );
    }
}

#[cfg(feature = "tantivy-oracle")]
#[test]
fn the_ordinary_pinned_oracle_invokes_the_fixed_constructor() {
    use super::WriterCall;

    // Asserting "no receipt" here proves nothing: reverting this oracle to
    // WriterPlan::Shipping keeps the receipt absent while silently moving
    // the writer onto Tantivy's auto selection, which is exactly the
    // regression this test exists to catch. So observe the call itself,
    // recorded by the helper that performed it, on this instance — not a
    // global counter that parallel tests would race.
    let oracle = TantivyIndex::in_memory_single_threaded_oracle().expect("single-threaded oracle");

    assert_eq!(
        oracle.observed_writer_call,
        WriterCall::Fixed(1),
        "the oracle must invoke writer_with_num_threads(1, ..), not the auto path"
    );
    assert!(
        oracle.benchmark_writer_receipt().is_none(),
        "invoking the fixed constructor still claims no screening identity"
    );
}

#[test]
fn the_observed_call_tracks_the_constructor_each_plan_reaches() {
    use super::WriterCall;

    // The same observer over the benchmark plans, so a plan that reached
    // the wrong Tantivy entry point cannot hide behind a matching label.
    let auto = TantivyIndex::in_memory_with_shipping_auto_writer(HEAP, true).expect("auto writer");
    assert_eq!(auto.observed_writer_call, WriterCall::Auto);

    let fixed = TantivyIndex::in_memory_with_benchmark_config(HEAP, 4, true).expect("fixed writer");
    assert_eq!(fixed.observed_writer_call, WriterCall::Fixed(4));

    let ordinary = TantivyIndex::in_memory().expect("ordinary in-memory index");
    assert_eq!(ordinary.observed_writer_call, WriterCall::Auto);
}

#[test]
fn a_reopened_index_rejects_a_disagreeing_positions_claim() {
    let directory = tempfile::tempdir().expect("temporary index directory");
    let path = directory.path().join("oracle");
    TantivyIndex::create_with_benchmark_config(&path, HEAP, 2, true)
        .expect("create with positions");

    // Reopening while claiming the opposite positions setting is an
    // assertion about bytes that already exist, and must fail closed rather
    // than writing a false value into every downstream receipt.
    assert!(matches!(
        TantivyIndex::open_with_benchmark_config(&path, HEAP, 2, false),
        Err(SearchError::InvalidConfig { .. })
    ));

    let reopened = TantivyIndex::open_with_benchmark_config(&path, HEAP, 2, true)
        .expect("reopen with the true positions setting");
    assert!(
        reopened
            .benchmark_writer_receipt()
            .expect("receipt")
            .positions,
        "the reopened receipt must carry the live schema's positions option"
    );
}

#[test]
fn a_reopened_positions_off_index_rejects_the_mirrored_claim() {
    // Mirror of the case above: the rejection must be symmetric, or a
    // positions-off index could be reopened while claiming positions-on and
    // carry that false claim into every downstream receipt.
    let directory = tempfile::tempdir().expect("temporary index directory");
    let path = directory.path().join("oracle");
    TantivyIndex::create_with_benchmark_config(&path, HEAP, 2, false)
        .expect("create without positions");

    assert!(matches!(
        TantivyIndex::open_with_benchmark_config(&path, HEAP, 2, true),
        Err(SearchError::InvalidConfig { .. })
    ));

    let reopened = TantivyIndex::open_with_benchmark_config(&path, HEAP, 2, false)
        .expect("reopen with the true positions setting");
    assert!(
        !reopened
            .benchmark_writer_receipt()
            .expect("receipt")
            .positions,
        "the reopened receipt must carry the live schema's positions option"
    );
}

#[test]
fn qg4_directory_sync_requires_fresh_no_merge_preparation() {
    let directory = tempfile::tempdir().expect("temporary index directory");
    let path = directory.path().join("qg4");
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = TantivyIndex::create_with_benchmark_config(&path, HEAP, 1, true)
            .expect("create QG-4 index");

        assert!(matches!(
            index.benchmark_arm_qg4_directory_sync(),
            Err(SearchError::InvalidConfig { value, .. }) if value == "not_prepared"
        ));

        index
            .index_document(&cx, &IndexableDocument::new("warm", "warm document"))
            .await
            .expect("stage warm document");
        assert!(matches!(
            index.benchmark_prepare_qg4_no_merge_directory_sync(&cx).await,
            Err(SearchError::InvalidConfig { value, .. }) if value == "documents_already_indexed"
        ));
    });
}

#[test]
fn qg4_directory_sync_receipt_binds_real_sync_opstamp_and_reopen() {
    let directory = tempfile::tempdir().expect("temporary index directory");
    let path = directory.path().join("qg4");
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = TantivyIndex::create_with_benchmark_config(&path, HEAP, 1, true)
            .expect("create QG-4 index");
        index
            .benchmark_prepare_qg4_no_merge_directory_sync(&cx)
            .await
            .expect("prepare no-merge QG-4 fixture");

        index
            .index_document(&cx, &IndexableDocument::new("warm", "warm document"))
            .await
            .expect("stage warm document");
        index.commit(&cx).await.expect("commit warm document");

        index
            .index_document(&cx, &IndexableDocument::new("staged", "staged document"))
            .await
            .expect("stage probe document");
        let mut arm = index
            .benchmark_arm_qg4_directory_sync()
            .expect("arm QG-4 directory observation");
        index
            .benchmark_commit_qg4_directory_sync(&cx, &mut arm)
            .await
            .expect("commit staged QG-4 document");
        let receipt = arm.finish().expect("one real directory sync");
        assert_eq!(receipt.root, path);
        assert_ne!(receipt.commit_opstamp, 0);
        assert_ne!(receipt.session_nonce, 0);
        assert_eq!(receipt.observed_sync_count, 1);

        drop(index);
        let reopened =
            TantivyIndex::open_with_benchmark_config(&path, HEAP, 1, true).expect("fresh reopen");
        assert_eq!(reopened.doc_count().expect("reopened count"), 2);
        let staged_hits = reopened
            .search_doc_ids(&cx, "id:staged", 2)
            .expect("explicit staged-id query");
        assert_eq!(
            staged_hits.len(),
            1,
            "fresh reopen must retain exactly the staged ID"
        );
        assert_eq!(staged_hits[0].doc_id.as_str(), "staged");
    });
}

#[test]
fn qg4_directory_sync_rejects_a_second_real_armed_sync() {
    let directory = tempfile::tempdir().expect("temporary index directory");
    let path = directory.path().join("qg4");
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = TantivyIndex::create_with_benchmark_config(&path, HEAP, 1, true)
            .expect("create QG-4 index");
        index
            .benchmark_prepare_qg4_no_merge_directory_sync(&cx)
            .await
            .expect("prepare no-merge QG-4 fixture");
        index
            .index_document(&cx, &IndexableDocument::new("staged", "staged document"))
            .await
            .expect("stage probe document");

        let mut arm = index
            .benchmark_arm_qg4_directory_sync()
            .expect("arm QG-4 directory observation");
        index
            .benchmark_test_second_armed_directory_sync()
            .expect("first real wrapped sync");
        index
            .benchmark_commit_qg4_directory_sync(&cx, &mut arm)
            .await
            .expect("commit staged QG-4 document");
        assert!(matches!(
            arm.finish(),
            Err(SearchError::InvalidConfig { value, .. }) if value == "multiple_syncs"
        ));
    });
}
