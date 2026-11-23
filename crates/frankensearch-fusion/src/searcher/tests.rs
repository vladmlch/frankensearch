#![allow(
    clippy::unnecessary_literal_bound,
    clippy::cast_precision_loss,
    clippy::significant_drop_tightening,
    clippy::wildcard_imports
)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ahash::AHashMap;
use asupersync::Cx;
use asupersync::time::timeout;

#[cfg(feature = "graph")]
use frankensearch_core::DocumentGraph;
use frankensearch_core::ParsedQuery;
use frankensearch_core::config::{TwoTierConfig, ZeroSignalReason};
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::explanation::{ExplainedSource, ExplanationPhase};
use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, QuantizationFormat,
};
use frankensearch_core::query_class::QueryClass;
use frankensearch_core::traits::{
    Embedder, IdentityBoundEmbedding, LexicalHydrationContext, LexicalRead, MetricsExporter,
    ModelCategory, Reranker, SearchFuture,
};
use frankensearch_core::types::{
    EmbeddingMetrics, IndexMetrics, PhaseMetrics, ScoreSource, ScoredResult, SearchMetrics,
    SearchPhase, VectorHit,
};
use frankensearch_core::{
    AdapterIdentity, AdapterLifecycleEvent, EmbedderTier, EmbeddingStage, EmbeddingStatus,
    HostAdapter, LifecycleSeverity, LifecycleState, LiveSearchStreamEmitter,
    RuntimeMetricsCollector, SearchEventPhase, TelemetryEnvelope, TelemetryEvent,
};
use frankensearch_embed::CachedEmbedder;
use frankensearch_index::{FsviV2IdentityBinding, SearchParams, TwoTierIndex};

use crate::normalize::NqcDenseWeight;
use crate::phase_gate::{PhaseGate, PhaseGateConfig, PhaseObservation};
use crate::prf::PrfConfig;

use super::TwoTierSearcher;
use super::admission::admit_legacy_tier_embedder;
use super::results::{
    NormalizedExclusions, contains_negative_term, contains_term_with_word_boundaries,
    filter_scored_results_by_negations, filter_vector_hits_by_negations, find_negative_match,
    fused_hits_to_scored_results, is_word_char, normalize_for_negation_match,
    promoted_quality_source, scaled_budget, should_exclude_document, should_lexical_short_circuit,
    source_can_promote_to_quality, term_is_word_like, vector_exclusion_source,
    vector_hits_to_scored_results,
};
use super::telemetry::{
    CpuJiffiesSnapshot, TELEMETRY_TIMESTAMP_FALLBACK_RFC3339, cpu_pct_from_jiffies,
    embedder_tier_for_stage, next_telemetry_identifier, parse_proc_io_bytes,
    parse_proc_load_avg_1m, parse_proc_process_jiffies, parse_proc_status_rss_bytes,
    parse_proc_total_jiffies, telemetry_timestamp_now,
};

mod support;
use support::*;

mod admission_tests;
mod exclusion_telemetry_tests;
mod native_tests;
mod orchestration_tests;
mod phase_gate_graph_tests;
mod unit_tests;
