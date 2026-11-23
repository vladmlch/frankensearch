#[cfg(any(feature = "profile-internals", feature = "pruning-conformance"))]
use super::*;

#[cfg(feature = "profile-internals")]
#[test]
fn profile_session_is_invocation_local_one_shot_and_counter_exact() {
    let session = QuillProfileSession::new(17, 23, 5, 2);
    session
        .bind_cache(QuillProfileCacheDisposition::Miss)
        .expect("bind cache disposition once");
    session
        .bind_fanout_eligibility(true)
        .expect("bind shipping fan-out eligibility once");
    session
        .bind_execution(QuillProfileExecutionMode::Rayon)
        .expect("bind actual shipping branch once");
    session.record_snapshot_doc_freq(5);
    session.record_global_doc_freq(25);
    session.record_term_dictionary_view(30);
    session.record_segment_lowered(7);
    session.record_fuel_units(41);

    let receipt = session
        .complete(QuillProfileOutcome::Completed)
        .expect("complete one invocation-local receipt");
    assert_eq!(receipt.snapshot_epoch(), 17);
    assert_eq!(receipt.keeper_generation(), 23);
    assert_eq!(receipt.segment_counts(), (5, 2));
    assert_eq!(receipt.cache(), QuillProfileCacheDisposition::Miss);
    assert_eq!(receipt.fanout_eligible(), Some(true));
    assert_eq!(receipt.execution(), Some(QuillProfileExecutionMode::Rayon));
    assert_eq!(receipt.counters(), (5, 25, 30, 7, 41));
    assert!(!receipt.overflowed());
    assert_eq!(receipt.outcome(), QuillProfileOutcome::Completed);
    assert!(matches!(
        session.complete(QuillProfileOutcome::Completed),
        Err(QuillIndexError::InvalidState { detail })
            if detail.contains("already completed")
    ));

    let conflicting_cache = QuillProfileSession::new(1, 1, 0, 0);
    conflicting_cache
        .bind_cache(QuillProfileCacheDisposition::Disabled)
        .expect("bind first cache fact");
    conflicting_cache
        .bind_fanout_eligibility(false)
        .expect("bind first fan-out fact");
    assert!(matches!(
        conflicting_cache.bind_fanout_eligibility(true),
        Err(QuillIndexError::InvalidState { detail })
            if detail.contains("already bound")
    ));
    assert!(matches!(
        conflicting_cache.bind_cache(QuillProfileCacheDisposition::Hit),
        Err(QuillIndexError::InvalidState { detail })
            if detail.contains("already bound")
    ));
    let retained = conflicting_cache
        .complete(QuillProfileOutcome::Cancelled)
        .expect("conflicting bind cannot overwrite the original fact");
    assert_eq!(retained.cache(), QuillProfileCacheDisposition::Disabled);
    assert_eq!(retained.fanout_eligible(), Some(false));
    assert_eq!(retained.outcome(), QuillProfileOutcome::Cancelled);

    let incomplete = QuillProfileSession::new(1, 1, 0, 0);
    assert!(matches!(
        incomplete.complete(QuillProfileOutcome::OtherError),
        Err(QuillIndexError::InvalidState { detail })
            if detail.contains("never bound")
    ));
    assert!(matches!(
        incomplete.bind_cache(QuillProfileCacheDisposition::NotChecked),
        Err(QuillIndexError::InvalidState { detail })
            if detail.contains("already bound")
    ));
}

#[cfg(feature = "profile-internals")]
#[test]
fn profile_session_rejects_counter_overflow() {
    let session = QuillProfileSession::new(1, 1, 0, 0);
    session
        .bind_cache(QuillProfileCacheDisposition::NotChecked)
        .expect("bind explicit no-lookup fact");
    session.record_fuel_units(u64::MAX);
    session.record_fuel_units(1);
    assert!(matches!(
        session.complete(QuillProfileOutcome::FuelExhausted),
        Err(QuillIndexError::InvalidState { detail })
            if detail.contains("counter overflowed")
    ));
}

#[cfg(all(feature = "profile-internals", not(feature = "conformance-internals")))]
#[test]
fn profiled_checkpoint_distinguishes_admitted_and_refused_work() {
    let cx = Cx::for_testing();
    let session = QuillProfileSession::new(1, 1, 0, 0);
    session
        .bind_cache(QuillProfileCacheDisposition::Miss)
        .expect("bind checkpoint test cache fact");
    let checkpoint = QueryCheckpoint::new_with_profile(&cx, "profile_test", 1, 2, &session);
    checkpoint
        .admit(QueryWorkKind::Segment, 1)
        .expect("admit first fuel unit");
    assert!(matches!(
        checkpoint.admit(QueryWorkKind::DictionaryBlock, 1),
        Err(ArgusError::QueryFuelExhausted { .. })
    ));

    let receipt = session
        .complete(QuillProfileOutcome::FuelExhausted)
        .expect("complete checkpoint work receipt");
    assert_eq!(receipt.work_units().requested(), [1, 1, 0, 0]);
    assert_eq!(receipt.work_units().admitted(), [1, 0, 0, 0]);
    assert_eq!(receipt.work_units().refused(), [0, 1, 0, 0]);
    assert_eq!(receipt.counters().4, 1);
    assert_eq!(receipt.cancellation_observations(), 0);
    assert_eq!(receipt.outcome(), QuillProfileOutcome::FuelExhausted);
}

#[cfg(feature = "pruning-conformance")]
#[test]
fn pruning_conformance_session_is_dense_one_shot_and_fail_closed() {
    let completed = ConformancePruningTraceSession::new(2);
    completed
        .bind_execution_mode(ConformancePruningExecutionMode::Rayon)
        .expect("bind exact shipping branch");
    completed
        .record(1, 7, &[])
        .expect("record second segment first");
    completed
        .record(0, 5, &[])
        .expect("record first segment second");
    let receipt = completed.complete().expect("complete dense trace");
    assert_eq!(
        receipt.execution_mode(),
        ConformancePruningExecutionMode::Rayon
    );
    assert_eq!(
        receipt
            .segments()
            .iter()
            .map(ConformanceSegmentPruningReceipt::segment_ordinal)
            .collect::<Vec<_>>(),
        vec![0, 1],
    );
    assert_eq!(
        receipt
            .segments()
            .iter()
            .map(ConformanceSegmentPruningReceipt::segment_doc_count)
            .collect::<Vec<_>>(),
        vec![5, 7],
    );
    assert!(matches!(
        completed.complete(),
        Err(QuillIndexError::InvalidState { detail })
            if detail.contains("already completed")
    ));

    let record_before_bind = ConformancePruningTraceSession::new(1);
    assert!(matches!(
        record_before_bind.record(0, 1, &[]),
        Err(QuillIndexError::InvalidState { detail })
            if detail.contains("was not bound before recording")
    ));
    assert!(matches!(
        record_before_bind.bind_execution_mode(ConformancePruningExecutionMode::Serial),
        Err(QuillIndexError::InvalidState { detail })
            if detail.contains("has failed")
    ));

    let double_bind = ConformancePruningTraceSession::new(1);
    double_bind
        .bind_execution_mode(ConformancePruningExecutionMode::Serial)
        .expect("bind one exact execution mode");
    assert!(matches!(
        double_bind.bind_execution_mode(ConformancePruningExecutionMode::Rayon),
        Err(QuillIndexError::InvalidState { detail })
            if detail.contains("already bound")
    ));
    assert!(matches!(
        double_bind.record(0, 1, &[]),
        Err(QuillIndexError::InvalidState { detail })
            if detail.contains("has failed")
    ));

    let out_of_range = ConformancePruningTraceSession::new(1);
    out_of_range
        .bind_execution_mode(ConformancePruningExecutionMode::Serial)
        .expect("bind out-of-range session");
    assert!(matches!(
        out_of_range.record(1, 1, &[]),
        Err(QuillIndexError::InvalidState { detail })
            if detail.contains("exceeds expected count")
    ));
    assert!(matches!(
        out_of_range.record(0, 1, &[]),
        Err(QuillIndexError::InvalidState { detail })
            if detail.contains("has failed")
    ));

    let duplicate = ConformancePruningTraceSession::new(1);
    duplicate
        .bind_execution_mode(ConformancePruningExecutionMode::Serial)
        .expect("bind serial path");
    duplicate.record(0, 1, &[]).expect("first receipt");
    assert!(matches!(
        duplicate.record(0, 1, &[]),
        Err(QuillIndexError::InvalidState { detail })
            if detail.contains("recorded twice")
    ));
    assert!(matches!(
        duplicate.complete(),
        Err(QuillIndexError::InvalidState { detail })
            if detail.contains("has failed")
    ));

    let incomplete = ConformancePruningTraceSession::new(2);
    incomplete
        .bind_execution_mode(ConformancePruningExecutionMode::Serial)
        .expect("bind incomplete serial path");
    incomplete
        .record(0, 1, &[])
        .expect("record partial receipt");
    assert!(matches!(
        incomplete.complete(),
        Err(QuillIndexError::InvalidState { detail })
            if detail.contains("trace is incomplete")
    ));
    assert!(matches!(
        incomplete.record(1, 1, &[]),
        Err(QuillIndexError::InvalidState { detail })
            if detail.contains("has failed")
    ));

    let dropped_controller = Arc::new(ConformanceCancellationController::default());
    dropped_controller
        .arm(
            ConformanceCancellationStage::PruningTraceSegmentRecorded,
            u64::MAX,
        )
        .expect("arm dropped-session accounting");
    let dropped_guard = ConformancePruningTraceGuard::new(1, Arc::clone(&dropped_controller));
    let dropped_session = Arc::clone(&dropped_guard.session);
    dropped_session
        .bind_execution_mode(ConformancePruningExecutionMode::Serial)
        .expect("bind guard-owned session");
    dropped_session
        .record(0, 1, &[])
        .expect("record partial guard-owned receipt");
    drop(dropped_guard);
    assert!(matches!(
        dropped_session.complete(),
        Err(QuillIndexError::InvalidState { detail })
            if detail.contains("has failed")
    ));
    assert_eq!(
        dropped_controller.discarded_pruning_trace_sessions(),
        1,
        "dropping an incomplete invocation records one discarded session"
    );

    let generation_controller = Arc::new(ConformanceCancellationController::default());
    generation_controller
        .arm(
            ConformanceCancellationStage::PruningTraceSegmentRecorded,
            u64::MAX,
        )
        .expect("arm stale generation");
    let stale_guard = ConformancePruningTraceGuard::new(1, Arc::clone(&generation_controller));
    stale_guard
        .session()
        .bind_execution_mode(ConformancePruningExecutionMode::Serial)
        .expect("bind stale guard");
    generation_controller.disarm();
    generation_controller
        .arm(
            ConformanceCancellationStage::PruningTraceSegmentRecorded,
            u64::MAX,
        )
        .expect("arm successor generation");
    drop(stale_guard);
    assert_eq!(
        generation_controller.discarded_pruning_trace_sessions(),
        0,
        "a stale guard must not contaminate a later arm generation"
    );
    let current_guard = ConformancePruningTraceGuard::new(1, Arc::clone(&generation_controller));
    current_guard
        .session()
        .bind_execution_mode(ConformancePruningExecutionMode::Serial)
        .expect("bind current guard");
    drop(current_guard);
    assert_eq!(
        generation_controller.discarded_pruning_trace_sessions(),
        1,
        "the current arm generation must retain its own discarded-session count"
    );

    let exhausted_controller = ConformanceCancellationController::default();
    exhausted_controller
        .arm_generation
        .store(u64::MAX, Ordering::Relaxed);
    assert!(matches!(
        exhausted_controller.arm(
            ConformanceCancellationStage::PruningTraceSegmentRecorded,
            1,
        ),
        Err(QuillIndexError::InvalidState { detail })
            if detail.contains("arm generation is exhausted")
    ));
    assert_eq!(
        exhausted_controller.stage.load(Ordering::Acquire),
        ConformanceCancellationController::DISARMED,
        "generation exhaustion must leave the controller disarmed"
    );
}
