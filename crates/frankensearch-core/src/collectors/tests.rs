use std::path::PathBuf;
use std::time::Duration;

use super::*;
use crate::error::SearchError;
use crate::query_class::QueryClass;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../schemas/fixtures")
        .join(name)
}

fn parse_fixture(name: &str) -> TelemetryEnvelope {
    let path = fixture_path(name);
    let body = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("read fixture {}", path.display()));
    serde_json::from_str::<TelemetryEnvelope>(&body)
        .unwrap_or_else(|_| panic!("deserialize fixture {}", path.display()))
}

fn instance() -> TelemetryInstance {
    TelemetryInstance {
        instance_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
        project_key: "/data/projects/frankensearch".to_owned(),
        host_name: "atlas".to_owned(),
        pid: Some(4242),
    }
}

fn correlation() -> TelemetryCorrelation {
    TelemetryCorrelation {
        event_id: "01JAH9A2X1K2M3N4P5Q6R7S8T9".to_owned(),
        root_request_id: "01JAH9A2WZZZZZZZZZZZZZZZZZ".to_owned(),
        parent_event_id: None,
    }
}

fn search_event(query_text: &str, latency_us: u64) -> TelemetryEnvelope {
    RuntimeMetricsCollector::default().emit_search(
        "2026-02-14T00:45:00Z",
        instance(),
        correlation(),
        SearchCollectorSample {
            query_text: query_text.to_owned(),
            query_class: QueryClass::NaturalLanguage,
            phase: SearchEventPhase::Initial,
            result_count: 10,
            lexical_count: 18,
            semantic_count: 30,
            hash_control_count: 0,
            latency_us,
            memory_bytes: Some(1_835_008),
        },
    )
}

#[test]
fn config_default_matches_ops_contract() {
    let cfg = CollectorConfig::default();
    assert_eq!(cfg.collection_interval_ms, DEFAULT_COLLECTION_INTERVAL_MS);
    assert_eq!(cfg.interval(), Duration::from_secs(1));
    cfg.validate().expect("default should be valid");
}

#[test]
fn config_rejects_too_fast_interval() {
    let cfg = CollectorConfig {
        collection_interval_ms: 99,
    };
    let err = cfg.validate().expect_err("interval below min must fail");
    assert!(matches!(err, SearchError::InvalidConfig { .. }));
}

#[test]
fn collector_emits_search_event_with_canonical_fields() {
    let collector = RuntimeMetricsCollector::default();
    let sample = SearchCollectorSample {
        query_text: "distributed consensus".to_owned(),
        query_class: QueryClass::NaturalLanguage,
        phase: SearchEventPhase::Initial,
        result_count: 10,
        lexical_count: 18,
        semantic_count: 30,
        hash_control_count: 0,
        latency_us: 8_421,
        memory_bytes: Some(1_835_008),
    };

    let envelope = collector.emit_search("2026-02-14T00:45:00Z", instance(), correlation(), sample);
    let value = serde_json::to_value(&envelope).expect("serialize");

    assert_eq!(value["v"], 1);
    assert_eq!(value["event"]["type"], "search");
    assert_eq!(value["event"]["query"]["class"], "natural_language");
    assert_eq!(value["event"]["query"]["phase"], "initial");
    assert_eq!(value["event"]["metrics"]["latency_us"], 8_421);

    let snapshot = collector.snapshot();
    assert_eq!(snapshot.search_events_emitted, 1);
    assert_eq!(snapshot.total_events_emitted, 1);
}

#[test]
fn collector_emits_embedding_index_resource_events() {
    let collector = RuntimeMetricsCollector::default();
    let ts = "2026-02-14T00:45:00Z";
    let base_instance = instance();
    let base_correlation = correlation();

    let embedding = collector.emit_embedding(
        ts,
        base_instance.clone(),
        base_correlation.clone(),
        EmbeddingCollectorSample {
            job_id: "01JAH9A31K2M3N4P5Q6R7S8T9W".to_owned(),
            queue_depth: 12,
            doc_count: 32,
            stage: EmbeddingStage::Background,
            embedder_id: "potion-multilingual-128M".to_owned(),
            tier: EmbedderTier::Fast,
            dimension: 256,
            status: EmbeddingStatus::Completed,
            duration_ms: 14,
        },
    );
    assert!(matches!(embedding.event, TelemetryEvent::Embedding { .. }));

    let index = collector.emit_index(
        ts,
        base_instance.clone(),
        base_correlation.clone(),
        IndexCollectorSample {
            operation: IndexOperation::Rebuild,
            inventory: IndexInventory {
                words: 245_781,
                tokens: 331_992,
                lines: 58_740,
                bytes: 19_833_421,
                docs: 12_034,
            },
            dimension: 384,
            quantization: QuantizationMode::F16,
            status: IndexStatus::Completed,
            duration_ms: 18_423,
        },
    );
    assert!(matches!(index.event, TelemetryEvent::Index { .. }));

    let resource = collector.emit_resource(
        ts,
        base_instance,
        base_correlation,
        ResourceCollectorSample {
            cpu_pct: 142.0,
            rss_bytes: 73_400_320,
            io_read_bytes: 993_421,
            io_write_bytes: 341_204,
            interval_ms: 0,
            load_avg_1m: Some(1.43),
            pressure_profile: Some(PressureProfile::Performance),
        },
    );

    let value = serde_json::to_value(resource).expect("serialize");
    assert_eq!(value["event"]["type"], "resource");
    assert_eq!(value["event"]["sample"]["cpu_pct"], 100.0);
    assert_eq!(value["event"]["sample"]["interval_ms"], 1);

    let snapshot = collector.snapshot();
    assert_eq!(snapshot.embedding_events_emitted, 1);
    assert_eq!(snapshot.index_events_emitted, 1);
    assert_eq!(snapshot.resource_events_emitted, 1);
    assert_eq!(snapshot.total_events_emitted, 3);
}

#[test]
fn empty_query_is_normalized_for_schema() {
    let collector = RuntimeMetricsCollector::default();
    let envelope = collector.emit_search(
        "2026-02-14T00:45:00Z",
        instance(),
        correlation(),
        SearchCollectorSample {
            query_text: "   ".to_owned(),
            query_class: QueryClass::Empty,
            phase: SearchEventPhase::Initial,
            result_count: 0,
            lexical_count: 0,
            semantic_count: 0,
            hash_control_count: 0,
            latency_us: 0,
            memory_bytes: None,
        },
    );
    let value = serde_json::to_value(envelope).expect("serialize");
    assert_eq!(value["event"]["query"]["text"], "<empty>");
    assert_eq!(value["event"]["query"]["class"], "empty");
}

#[test]
fn search_stream_lossy_mode_drops_oldest_with_accounting() {
    let emitter = LiveSearchStreamEmitter::new(SearchStreamConfig {
        capacity: 1,
        mode: SearchStreamMode::Lossy,
    })
    .expect("valid stream config");

    let first = emitter
        .publish_search(search_event("first query", 111))
        .expect("first publish should succeed");
    assert_eq!(first.sequence, 0);
    assert_eq!(first.dropped_since_last, 0);

    let second = emitter
        .publish_search(search_event("second query", 222))
        .expect("second publish should succeed in lossy mode");
    assert_eq!(second.sequence, 1);
    assert_eq!(second.dropped_since_last, 1);

    let drained = emitter.drain(10);
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0].sequence, 1);
    assert_eq!(drained[0].dropped_since_last, 1);

    let health = emitter.health();
    assert_eq!(health.mode, SearchStreamMode::Lossy);
    assert_eq!(health.capacity, 1);
    assert_eq!(health.emitted_total, 2);
    assert_eq!(health.dropped_total, 1);
    assert_eq!(health.backpressure_rejections, 0);
}

#[test]
fn search_stream_non_lossy_mode_reports_backpressure() {
    let emitter = LiveSearchStreamEmitter::new(SearchStreamConfig {
        capacity: 1,
        mode: SearchStreamMode::NonLossy,
    })
    .expect("valid stream config");

    emitter
        .publish_search(search_event("first query", 111))
        .expect("first publish should succeed");

    let err = emitter
        .publish_search(search_event("second query", 222))
        .expect_err("second publish should fail in non-lossy mode");
    assert!(matches!(
        err,
        SearchError::QueueFull {
            pending: 1,
            capacity: 1
        }
    ));

    let health = emitter.health();
    assert_eq!(health.emitted_total, 1);
    assert_eq!(health.dropped_total, 0);
    assert_eq!(health.backpressure_rejections, 1);
}

#[test]
fn search_stream_rejects_non_search_events() {
    let emitter = LiveSearchStreamEmitter::default();
    let non_search = RuntimeMetricsCollector::default().emit_index(
        "2026-02-14T00:45:05Z",
        instance(),
        correlation(),
        IndexCollectorSample {
            operation: IndexOperation::Rebuild,
            inventory: IndexInventory {
                words: 1,
                tokens: 1,
                lines: 1,
                bytes: 1,
                docs: 1,
            },
            dimension: 384,
            quantization: QuantizationMode::F16,
            status: IndexStatus::Completed,
            duration_ms: 10,
        },
    );

    let err = emitter
        .publish_search(non_search)
        .expect_err("non-search event should be rejected");
    assert!(matches!(err, SearchError::InvalidConfig { .. }));
}

#[test]
fn search_stream_rejects_mismatched_schema_version() {
    let emitter = LiveSearchStreamEmitter::default();
    let mut event = search_event("schema mismatch query", 777);
    event.v = TELEMETRY_SCHEMA_VERSION.saturating_add(1);

    let err = emitter
        .publish_search(event)
        .expect_err("schema version mismatch should be rejected");

    assert!(
        matches!(err, SearchError::InvalidConfig { .. }),
        "expected invalid config error for schema mismatch"
    );

    if let SearchError::InvalidConfig {
        field,
        value,
        reason,
    } = err
    {
        assert_eq!(field, "search_stream.event.schema_version");
        assert_eq!(
            value,
            TELEMETRY_SCHEMA_VERSION.saturating_add(1).to_string()
        );
        assert!(
            reason.contains(&TELEMETRY_SCHEMA_VERSION.to_string()),
            "reason should mention expected schema version"
        );
    }
}

#[test]
fn search_stream_preserves_search_metrics_and_correlation() {
    let emitter = LiveSearchStreamEmitter::default();
    let event = search_event("correlation query", 8421);
    emitter
        .publish_search(event)
        .expect("publish should succeed in default mode");

    let drained = emitter.drain(1);
    assert_eq!(drained.len(), 1);
    let frame = &drained[0];
    assert_eq!(frame.sequence, 0);

    if let TelemetryEvent::Search {
        correlation,
        metrics,
        ..
    } = &frame.event.event
    {
        assert_eq!(correlation.event_id, "01JAH9A2X1K2M3N4P5Q6R7S8T9");
        assert_eq!(metrics.latency_us, 8421);
        assert_eq!(metrics.memory_bytes, Some(1_835_008));
    } else {
        panic!("expected search event");
    }
}

#[test]
fn search_stream_config_rejects_zero_capacity() {
    let cfg = SearchStreamConfig {
        capacity: 0,
        mode: SearchStreamMode::Lossy,
    };
    let err = cfg.validate().expect_err("zero capacity must fail");
    assert!(matches!(err, SearchError::InvalidConfig { .. }));
}

#[test]
fn telemetry_event_fixtures_deserialize() {
    for fixture in [
        "telemetry-search-v1.json",
        "telemetry-embedding-v1.json",
        "telemetry-index-v1.json",
        "telemetry-resource-v1.json",
        "telemetry-lifecycle-v1.json",
    ] {
        let envelope = parse_fixture(fixture);
        assert_eq!(envelope.v, TELEMETRY_SCHEMA_VERSION);
    }
}

// ── sanitize_query_text edge cases ──────────────────────────────────

#[test]
fn sanitize_query_text_empty_string() {
    assert_eq!(sanitize_query_text(String::new()), EMPTY_QUERY_PLACEHOLDER);
}

#[test]
fn sanitize_query_text_whitespace_only() {
    assert_eq!(
        sanitize_query_text("   \t\n  ".to_owned()),
        EMPTY_QUERY_PLACEHOLDER
    );
}

#[test]
fn sanitize_query_text_normal_passthrough() {
    assert_eq!(sanitize_query_text("hello world".to_owned()), "hello world");
}

#[test]
fn sanitize_query_text_trims_leading_trailing() {
    assert_eq!(sanitize_query_text("  spaced  ".to_owned()), "spaced");
}

#[test]
fn sanitize_query_text_truncates_long_input() {
    let long = "a".repeat(600);
    let result = sanitize_query_text(long);
    assert_eq!(result.len(), MAX_QUERY_TEXT_CHARS);
    assert!(result.chars().all(|c| c == 'a'));
}

#[test]
fn sanitize_query_text_exact_limit_not_truncated() {
    let exact = "b".repeat(MAX_QUERY_TEXT_CHARS);
    assert_eq!(sanitize_query_text(exact.clone()), exact);
}

#[test]
fn sanitize_query_text_unicode_boundaries_are_character_counted() {
    let expected = "é".repeat(MAX_QUERY_TEXT_CHARS);
    let result = sanitize_query_text("é".repeat(MAX_QUERY_TEXT_CHARS + 1));
    assert_eq!(result, expected);
}

#[test]
fn sanitize_query_text_unicode_whitespace_is_trimmed() {
    assert_eq!(
        sanitize_query_text("\u{2003}search café 東京\u{2003}".to_owned()),
        "search café 東京"
    );
}

#[test]
fn sanitize_query_text_reuses_clean_short_input_buffer() {
    let mut query_text = String::with_capacity(64);
    query_text.push_str("find authentication error");
    let input_ptr = query_text.as_ptr();

    let result = sanitize_query_text(query_text);

    assert_eq!(result, "find authentication error");
    assert_eq!(result.as_ptr(), input_ptr);
}

// ── sanitize_cpu_pct edge cases ─────────────────────────────────────

#[test]
fn sanitize_cpu_pct_nan_becomes_zero() {
    assert!(sanitize_cpu_pct(f64::NAN).abs() < f64::EPSILON);
}

#[test]
fn sanitize_cpu_pct_positive_infinity_becomes_zero() {
    assert!(sanitize_cpu_pct(f64::INFINITY).abs() < f64::EPSILON);
}

#[test]
fn sanitize_cpu_pct_negative_infinity_becomes_zero() {
    assert!(sanitize_cpu_pct(f64::NEG_INFINITY).abs() < f64::EPSILON);
}

#[test]
fn sanitize_cpu_pct_negative_clamped_to_zero() {
    assert!(sanitize_cpu_pct(-5.0).abs() < f64::EPSILON);
}

#[test]
fn sanitize_cpu_pct_over_100_clamped() {
    assert!((sanitize_cpu_pct(142.0) - 100.0).abs() < f64::EPSILON);
}

#[test]
fn sanitize_cpu_pct_zero_passthrough() {
    assert!(sanitize_cpu_pct(0.0).abs() < f64::EPSILON);
}

#[test]
fn sanitize_cpu_pct_normal_passthrough() {
    assert!((sanitize_cpu_pct(55.5) - 55.5).abs() < f64::EPSILON);
}

#[test]
fn sanitize_cpu_pct_exactly_100_passthrough() {
    assert!((sanitize_cpu_pct(100.0) - 100.0).abs() < f64::EPSILON);
}

// ── normalize_non_negative_f64 edge cases ───────────────────────────

#[test]
fn normalize_non_negative_positive() {
    assert_eq!(normalize_non_negative_f64(1.5), Some(1.5));
}

#[test]
fn normalize_non_negative_zero() {
    assert_eq!(normalize_non_negative_f64(0.0), Some(0.0));
}

#[test]
fn normalize_non_negative_negative() {
    assert_eq!(normalize_non_negative_f64(-0.1), None);
}

#[test]
fn normalize_non_negative_nan() {
    assert_eq!(normalize_non_negative_f64(f64::NAN), None);
}

#[test]
fn normalize_non_negative_positive_infinity() {
    assert_eq!(normalize_non_negative_f64(f64::INFINITY), None);
}

#[test]
fn normalize_non_negative_negative_infinity() {
    assert_eq!(normalize_non_negative_f64(f64::NEG_INFINITY), None);
}

// ── TelemetryQueryClass::from coverage ──────────────────────────────

#[test]
fn telemetry_query_class_from_all_variants() {
    assert_eq!(
        TelemetryQueryClass::from(QueryClass::Empty),
        TelemetryQueryClass::Empty
    );
    assert_eq!(
        TelemetryQueryClass::from(QueryClass::Identifier),
        TelemetryQueryClass::Identifier
    );
    assert_eq!(
        TelemetryQueryClass::from(QueryClass::ShortKeyword),
        TelemetryQueryClass::ShortKeyword
    );
    assert_eq!(
        TelemetryQueryClass::from(QueryClass::NaturalLanguage),
        TelemetryQueryClass::NaturalLanguage
    );
}

// ── CollectorConfig boundary and interval ───────────────────────────

#[test]
fn config_at_exact_minimum_interval_passes() {
    let cfg = CollectorConfig {
        collection_interval_ms: MIN_COLLECTION_INTERVAL_MS,
    };
    cfg.validate().expect("exact minimum should pass");
    assert_eq!(cfg.interval(), Duration::from_millis(100));
}

#[test]
fn config_one_below_minimum_fails() {
    let cfg = CollectorConfig {
        collection_interval_ms: MIN_COLLECTION_INTERVAL_MS - 1,
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn config_serde_roundtrip() {
    let cfg = CollectorConfig {
        collection_interval_ms: 500,
    };
    let json = serde_json::to_string(&cfg).unwrap();
    let decoded: CollectorConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg, decoded);
}

// ── SearchStreamConfig defaults ─────────────────────────────────────

#[test]
fn search_stream_config_default_values() {
    let cfg = SearchStreamConfig::default();
    assert_eq!(cfg.capacity, DEFAULT_SEARCH_STREAM_CAPACITY);
    assert_eq!(cfg.mode, SearchStreamMode::Lossy);
    cfg.validate().expect("default config should be valid");
}

// ── RuntimeMetricsCollector construction ─────────────────────────────

#[test]
fn collector_new_with_valid_config() {
    let collector = RuntimeMetricsCollector::new(CollectorConfig {
        collection_interval_ms: 200,
    })
    .expect("valid config");
    assert_eq!(collector.config().collection_interval_ms, 200);
}

#[test]
fn collector_new_rejects_invalid_config() {
    let err = RuntimeMetricsCollector::new(CollectorConfig {
        collection_interval_ms: 50,
    })
    .expect_err("must reject too-fast interval");
    assert!(matches!(err, SearchError::InvalidConfig { .. }));
}

#[test]
fn collector_default_snapshot_all_zero() {
    let collector = RuntimeMetricsCollector::default();
    let snap = collector.snapshot();
    assert_eq!(snap.search_events_emitted, 0);
    assert_eq!(snap.embedding_events_emitted, 0);
    assert_eq!(snap.index_events_emitted, 0);
    assert_eq!(snap.resource_events_emitted, 0);
    assert_eq!(snap.total_events_emitted, 0);
    assert_eq!(snap.collection_interval_ms, DEFAULT_COLLECTION_INTERVAL_MS);
}

// ── TelemetryEnvelope construction ──────────────────────────────────

#[test]
fn envelope_new_sets_schema_version() {
    let envelope = TelemetryEnvelope::new(
        "2026-02-14T12:00:00Z",
        TelemetryEvent::Lifecycle {
            instance: instance(),
            correlation: correlation(),
            state: LifecycleState::Started,
            severity: LifecycleSeverity::Info,
            reason: None,
            uptime_ms: None,
        },
    );
    assert_eq!(envelope.v, TELEMETRY_SCHEMA_VERSION);
    assert_eq!(envelope.ts, "2026-02-14T12:00:00Z");
}

// ── Stream emitter drain edge cases ─────────────────────────────────

#[test]
fn drain_zero_returns_empty() {
    let emitter = LiveSearchStreamEmitter::default();
    emitter.publish_search(search_event("test", 100)).unwrap();
    let drained = emitter.drain(0);
    assert!(drained.is_empty());
    // Item still in buffer
    assert_eq!(emitter.health().buffered, 1);
}

#[test]
fn drain_more_than_available() {
    let emitter = LiveSearchStreamEmitter::default();
    emitter.publish_search(search_event("test", 100)).unwrap();
    let drained = emitter.drain(1000);
    assert_eq!(drained.len(), 1);
    assert_eq!(emitter.health().buffered, 0);
}

#[test]
fn drain_empty_emitter_returns_empty() {
    let emitter = LiveSearchStreamEmitter::default();
    let drained = emitter.drain(10);
    assert!(drained.is_empty());
}

// ── Stream emitter health initial state ─────────────────────────────

#[test]
fn health_initial_state() {
    let emitter = LiveSearchStreamEmitter::default();
    let health = emitter.health();
    assert_eq!(health.mode, SearchStreamMode::Lossy);
    assert_eq!(health.capacity, DEFAULT_SEARCH_STREAM_CAPACITY);
    assert_eq!(health.buffered, 0);
    assert_eq!(health.emitted_total, 0);
    assert_eq!(health.dropped_total, 0);
    assert_eq!(health.backpressure_rejections, 0);
}

// ── Cursor format ───────────────────────────────────────────────────

#[test]
fn cursor_format_is_zero_padded() {
    let emitter = LiveSearchStreamEmitter::default();
    let outcome = emitter.publish_search(search_event("test", 100)).unwrap();
    assert_eq!(outcome.cursor, "search-00000000000000000000");
    assert_eq!(outcome.sequence, 0);
}

// ── Resource event sanitization through emit_resource ───────────────

#[test]
fn emit_resource_nan_cpu_becomes_zero() {
    let collector = RuntimeMetricsCollector::default();
    let envelope = collector.emit_resource(
        "2026-02-14T00:00:00Z",
        instance(),
        correlation(),
        ResourceCollectorSample {
            cpu_pct: f64::NAN,
            rss_bytes: 0,
            io_read_bytes: 0,
            io_write_bytes: 0,
            interval_ms: 1000,
            load_avg_1m: None,
            pressure_profile: None,
        },
    );
    let value = serde_json::to_value(&envelope).unwrap();
    assert_eq!(value["event"]["sample"]["cpu_pct"], 0.0);
}

#[test]
fn emit_resource_negative_cpu_clamped() {
    let collector = RuntimeMetricsCollector::default();
    let envelope = collector.emit_resource(
        "2026-02-14T00:00:00Z",
        instance(),
        correlation(),
        ResourceCollectorSample {
            cpu_pct: -10.0,
            rss_bytes: 0,
            io_read_bytes: 0,
            io_write_bytes: 0,
            interval_ms: 1000,
            load_avg_1m: None,
            pressure_profile: None,
        },
    );
    let value = serde_json::to_value(&envelope).unwrap();
    assert_eq!(value["event"]["sample"]["cpu_pct"], 0.0);
}

#[test]
fn emit_resource_interval_zero_clamped_to_one() {
    let collector = RuntimeMetricsCollector::default();
    let envelope = collector.emit_resource(
        "2026-02-14T00:00:00Z",
        instance(),
        correlation(),
        ResourceCollectorSample {
            cpu_pct: 50.0,
            rss_bytes: 0,
            io_read_bytes: 0,
            io_write_bytes: 0,
            interval_ms: 0,
            load_avg_1m: None,
            pressure_profile: None,
        },
    );
    let value = serde_json::to_value(&envelope).unwrap();
    assert_eq!(value["event"]["sample"]["interval_ms"], 1);
}

#[test]
fn emit_resource_nan_load_avg_filtered() {
    let collector = RuntimeMetricsCollector::default();
    let envelope = collector.emit_resource(
        "2026-02-14T00:00:00Z",
        instance(),
        correlation(),
        ResourceCollectorSample {
            cpu_pct: 50.0,
            rss_bytes: 0,
            io_read_bytes: 0,
            io_write_bytes: 0,
            interval_ms: 1000,
            load_avg_1m: Some(f64::NAN),
            pressure_profile: None,
        },
    );
    let value = serde_json::to_value(&envelope).unwrap();
    assert!(value["event"]["sample"]["load_avg_1m"].is_null());
}

#[test]
fn emit_resource_negative_load_avg_filtered() {
    let collector = RuntimeMetricsCollector::default();
    let envelope = collector.emit_resource(
        "2026-02-14T00:00:00Z",
        instance(),
        correlation(),
        ResourceCollectorSample {
            cpu_pct: 50.0,
            rss_bytes: 0,
            io_read_bytes: 0,
            io_write_bytes: 0,
            interval_ms: 1000,
            load_avg_1m: Some(-1.0),
            pressure_profile: None,
        },
    );
    let value = serde_json::to_value(&envelope).unwrap();
    assert!(value["event"]["sample"]["load_avg_1m"].is_null());
}

// ── Serde roundtrips for all enums ──────────────────────────────────

#[test]
fn search_stream_mode_serde_roundtrip() {
    for mode in [SearchStreamMode::Lossy, SearchStreamMode::NonLossy] {
        let json = serde_json::to_string(&mode).unwrap();
        let decoded: SearchStreamMode = serde_json::from_str(&json).unwrap();
        assert_eq!(mode, decoded);
    }
}

#[test]
fn telemetry_query_class_serde_roundtrip() {
    for class in [
        TelemetryQueryClass::Empty,
        TelemetryQueryClass::Identifier,
        TelemetryQueryClass::ShortKeyword,
        TelemetryQueryClass::NaturalLanguage,
    ] {
        let json = serde_json::to_string(&class).unwrap();
        let decoded: TelemetryQueryClass = serde_json::from_str(&json).unwrap();
        assert_eq!(class, decoded);
    }
}

#[test]
fn search_event_phase_serde_roundtrip() {
    for phase in [
        SearchEventPhase::Initial,
        SearchEventPhase::Refined,
        SearchEventPhase::RefinementFailed,
    ] {
        let json = serde_json::to_string(&phase).unwrap();
        let decoded: SearchEventPhase = serde_json::from_str(&json).unwrap();
        assert_eq!(phase, decoded);
    }
}

#[test]
fn embedding_stage_serde_roundtrip() {
    for stage in [
        EmbeddingStage::Fast,
        EmbeddingStage::Quality,
        EmbeddingStage::Background,
    ] {
        let json = serde_json::to_string(&stage).unwrap();
        let decoded: EmbeddingStage = serde_json::from_str(&json).unwrap();
        assert_eq!(stage, decoded);
    }
}

#[test]
fn embedder_tier_serde_roundtrip() {
    for tier in [
        EmbedderTier::Hash,
        EmbedderTier::Fast,
        EmbedderTier::Quality,
    ] {
        let json = serde_json::to_string(&tier).unwrap();
        let decoded: EmbedderTier = serde_json::from_str(&json).unwrap();
        assert_eq!(tier, decoded);
    }
}

#[test]
fn embedding_status_serde_roundtrip() {
    for status in [
        EmbeddingStatus::Queued,
        EmbeddingStatus::Running,
        EmbeddingStatus::Completed,
        EmbeddingStatus::Failed,
        EmbeddingStatus::Cancelled,
    ] {
        let json = serde_json::to_string(&status).unwrap();
        let decoded: EmbeddingStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(status, decoded);
    }
}

#[test]
fn index_operation_serde_roundtrip() {
    for op in [
        IndexOperation::Build,
        IndexOperation::Rebuild,
        IndexOperation::Append,
        IndexOperation::Compact,
        IndexOperation::Repair,
        IndexOperation::Snapshot,
    ] {
        let json = serde_json::to_string(&op).unwrap();
        let decoded: IndexOperation = serde_json::from_str(&json).unwrap();
        assert_eq!(op, decoded);
    }
}

#[test]
fn quantization_mode_serde_roundtrip() {
    for mode in [
        QuantizationMode::F32,
        QuantizationMode::F16,
        QuantizationMode::Int8,
        QuantizationMode::Pq,
    ] {
        let json = serde_json::to_string(&mode).unwrap();
        let decoded: QuantizationMode = serde_json::from_str(&json).unwrap();
        assert_eq!(mode, decoded);
    }
}

#[test]
fn index_status_serde_roundtrip() {
    for status in [
        IndexStatus::Started,
        IndexStatus::Completed,
        IndexStatus::Failed,
    ] {
        let json = serde_json::to_string(&status).unwrap();
        let decoded: IndexStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(status, decoded);
    }
}

#[test]
fn pressure_profile_serde_roundtrip() {
    for profile in [
        PressureProfile::Strict,
        PressureProfile::Performance,
        PressureProfile::Degraded,
    ] {
        let json = serde_json::to_string(&profile).unwrap();
        let decoded: PressureProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(profile, decoded);
    }
}

#[test]
fn lifecycle_state_serde_roundtrip() {
    for state in [
        LifecycleState::Started,
        LifecycleState::Stopped,
        LifecycleState::Healthy,
        LifecycleState::Degraded,
        LifecycleState::Stale,
        LifecycleState::Recovering,
    ] {
        let json = serde_json::to_string(&state).unwrap();
        let decoded: LifecycleState = serde_json::from_str(&json).unwrap();
        assert_eq!(state, decoded);
    }
}

#[test]
fn lifecycle_severity_serde_roundtrip() {
    for severity in [
        LifecycleSeverity::Info,
        LifecycleSeverity::Warn,
        LifecycleSeverity::Error,
    ] {
        let json = serde_json::to_string(&severity).unwrap();
        let decoded: LifecycleSeverity = serde_json::from_str(&json).unwrap();
        assert_eq!(severity, decoded);
    }
}

// ── Full envelope serde roundtrip ───────────────────────────────────

#[test]
fn search_envelope_serde_roundtrip() {
    let envelope = search_event("roundtrip test", 500);
    let json = serde_json::to_string(&envelope).unwrap();
    let decoded: TelemetryEnvelope = serde_json::from_str(&json).unwrap();
    assert_eq!(envelope, decoded);
}

#[test]
fn lifecycle_envelope_serde_roundtrip() {
    let envelope = TelemetryEnvelope::new(
        "2026-02-14T12:00:00Z",
        TelemetryEvent::Lifecycle {
            instance: instance(),
            correlation: correlation(),
            state: LifecycleState::Degraded,
            severity: LifecycleSeverity::Warn,
            reason: Some("memory pressure".to_owned()),
            uptime_ms: Some(86_400_000),
        },
    );
    let json = serde_json::to_string(&envelope).unwrap();
    let decoded: TelemetryEnvelope = serde_json::from_str(&json).unwrap();
    assert_eq!(envelope, decoded);
}

// ── CollectorSnapshot serde roundtrip ────────────────────────────────

#[test]
fn collector_snapshot_serde_roundtrip() {
    let snap = CollectorSnapshot {
        collection_interval_ms: 500,
        search_events_emitted: 10,
        embedding_events_emitted: 20,
        index_events_emitted: 5,
        resource_events_emitted: 30,
        total_events_emitted: 65,
    };
    let json = serde_json::to_string(&snap).unwrap();
    let decoded: CollectorSnapshot = serde_json::from_str(&json).unwrap();
    assert_eq!(snap, decoded);
}

// ── Snapshot total is sum of families ────────────────────────────────

#[test]
fn snapshot_total_is_sum_of_families() {
    let collector = RuntimeMetricsCollector::default();
    let ts = "2026-02-14T00:00:00Z";
    let inst = instance();
    let corr = correlation();

    // Emit 2 search, 1 embedding, 1 index, 1 resource
    let _ = collector.emit_search(
        ts,
        inst.clone(),
        corr.clone(),
        SearchCollectorSample {
            query_text: "a".to_owned(),
            query_class: QueryClass::ShortKeyword,
            phase: SearchEventPhase::Initial,
            result_count: 0,
            lexical_count: 0,
            semantic_count: 0,
            hash_control_count: 0,
            latency_us: 0,
            memory_bytes: None,
        },
    );
    let _ = collector.emit_search(
        ts,
        inst.clone(),
        corr.clone(),
        SearchCollectorSample {
            query_text: "b".to_owned(),
            query_class: QueryClass::Identifier,
            phase: SearchEventPhase::Refined,
            result_count: 5,
            lexical_count: 3,
            semantic_count: 2,
            hash_control_count: 0,
            latency_us: 100,
            memory_bytes: None,
        },
    );
    let _ = collector.emit_embedding(
        ts,
        inst.clone(),
        corr.clone(),
        EmbeddingCollectorSample {
            job_id: "j1".to_owned(),
            queue_depth: 0,
            doc_count: 1,
            stage: EmbeddingStage::Fast,
            embedder_id: "potion".to_owned(),
            tier: EmbedderTier::Fast,
            dimension: 256,
            status: EmbeddingStatus::Completed,
            duration_ms: 1,
        },
    );
    let _ = collector.emit_index(
        ts,
        inst.clone(),
        corr.clone(),
        IndexCollectorSample {
            operation: IndexOperation::Build,
            inventory: IndexInventory {
                words: 0,
                tokens: 0,
                lines: 0,
                bytes: 0,
                docs: 0,
            },
            dimension: 256,
            quantization: QuantizationMode::F16,
            status: IndexStatus::Started,
            duration_ms: 0,
        },
    );
    let _ = collector.emit_resource(
        ts,
        inst,
        corr,
        ResourceCollectorSample {
            cpu_pct: 10.0,
            rss_bytes: 0,
            io_read_bytes: 0,
            io_write_bytes: 0,
            interval_ms: 1000,
            load_avg_1m: None,
            pressure_profile: None,
        },
    );

    let snap = collector.snapshot();
    assert_eq!(snap.search_events_emitted, 2);
    assert_eq!(snap.embedding_events_emitted, 1);
    assert_eq!(snap.index_events_emitted, 1);
    assert_eq!(snap.resource_events_emitted, 1);
    assert_eq!(snap.total_events_emitted, 5);
}

// ── SearchStreamHealth serde roundtrip ───────────────────────────────

#[test]
fn search_stream_health_serde_roundtrip() {
    let health = SearchStreamHealth {
        mode: SearchStreamMode::NonLossy,
        capacity: 512,
        buffered: 10,
        emitted_total: 100,
        dropped_total: 5,
        backpressure_rejections: 3,
    };
    let json = serde_json::to_string(&health).unwrap();
    let decoded: SearchStreamHealth = serde_json::from_str(&json).unwrap();
    assert_eq!(health, decoded);
}

// ── LiveSearchFrame cursor monotonicity ─────────────────────────────

#[test]
fn stream_cursors_are_monotonically_increasing() {
    let emitter = LiveSearchStreamEmitter::new(SearchStreamConfig {
        capacity: 10,
        mode: SearchStreamMode::Lossy,
    })
    .unwrap();

    for i in 0..5u64 {
        let outcome = emitter
            .publish_search(search_event("query", i * 100))
            .unwrap();
        assert_eq!(outcome.sequence, i);
    }
    let drained = emitter.drain(10);
    assert_eq!(drained.len(), 5);
    for (i, frame) in drained.iter().enumerate() {
        assert_eq!(frame.sequence, i as u64);
    }
}

// ── Emitter new with zero capacity rejected ─────────────────────────

#[test]
fn emitter_new_rejects_zero_capacity() {
    let err = LiveSearchStreamEmitter::new(SearchStreamConfig {
        capacity: 0,
        mode: SearchStreamMode::Lossy,
    })
    .expect_err("zero capacity must fail");
    assert!(matches!(err, SearchError::InvalidConfig { .. }));
}

// ── TelemetryCorrelation with parent event ──────────────────────────

#[test]
fn correlation_with_parent_event_serializes() {
    let corr = TelemetryCorrelation {
        event_id: "ev1".to_owned(),
        root_request_id: "root1".to_owned(),
        parent_event_id: Some("parent1".to_owned()),
    };
    let json = serde_json::to_string(&corr).unwrap();
    let decoded: TelemetryCorrelation = serde_json::from_str(&json).unwrap();
    assert_eq!(corr, decoded);
    assert_eq!(decoded.parent_event_id, Some("parent1".to_owned()));
}

// ── Schema version constant ─────────────────────────────────────────

#[test]
fn schema_version_is_one() {
    assert_eq!(TELEMETRY_SCHEMA_VERSION, 1);
}

// ── SearchStreamPublishOutcome serde roundtrip ───────────────────────

#[test]
fn publish_outcome_serde_roundtrip() {
    let outcome = SearchStreamPublishOutcome {
        cursor: "search-00000000000000000042".to_owned(),
        sequence: 42,
        buffered: 3,
        dropped_since_last: 1,
    };
    let json = serde_json::to_string(&outcome).unwrap();
    let decoded: SearchStreamPublishOutcome = serde_json::from_str(&json).unwrap();
    assert_eq!(outcome, decoded);
}

// ── IndexInventory serde roundtrip ───────────────────────────────────

#[test]
fn index_inventory_serde_roundtrip() {
    let inv = IndexInventory {
        words: 100_000,
        tokens: 150_000,
        lines: 20_000,
        bytes: 5_000_000,
        docs: 500,
    };
    let json = serde_json::to_string(&inv).unwrap();
    let decoded: IndexInventory = serde_json::from_str(&json).unwrap();
    assert_eq!(inv, decoded);
}

// ── LiveSearchFrame serde roundtrip ──────────────────────────────────

#[test]
fn live_search_frame_serde_roundtrip() {
    let frame = LiveSearchFrame {
        cursor: "search-00000000000000000007".to_owned(),
        sequence: 7,
        dropped_since_last: 0,
        event: search_event("frame test", 300),
    };
    let json = serde_json::to_string(&frame).unwrap();
    let decoded: LiveSearchFrame = serde_json::from_str(&json).unwrap();
    assert_eq!(frame, decoded);
}

// ── Multiple lossy drops accumulate correctly ────────────────────────

#[test]
fn lossy_multiple_drops_accumulate() {
    let emitter = LiveSearchStreamEmitter::new(SearchStreamConfig {
        capacity: 1,
        mode: SearchStreamMode::Lossy,
    })
    .unwrap();

    // Fill the single slot
    emitter.publish_search(search_event("first", 100)).unwrap();
    // Three more publishes, each drops the oldest
    emitter.publish_search(search_event("second", 200)).unwrap();
    emitter.publish_search(search_event("third", 300)).unwrap();
    let outcome = emitter.publish_search(search_event("fourth", 400)).unwrap();

    // dropped_since_last resets on each publish frame, so only 1 for this frame
    assert_eq!(outcome.dropped_since_last, 1);

    let health = emitter.health();
    assert_eq!(health.dropped_total, 3);
    assert_eq!(health.emitted_total, 4);
}
