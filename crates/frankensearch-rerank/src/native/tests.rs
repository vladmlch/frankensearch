use std::path::Path;
use std::sync::Arc;

use asupersync::sync::OwnedMutexGuard;
use wide::f32x8;

use frankensearch_core::error::SearchError;
use frankensearch_core::traits::{RerankDocument, SyncRerank};

use super::kernels::{exp_vec8_non_fused, scalar_exp_fma};
use super::reranker::NativeReranker;
use super::weights::{LinearPrecision, LinearWeights, decode_f32_le};
use super::{SAFETENSORS_FALLBACK, SAFETENSORS_PRIMARY, TOKENIZER_JSON};

// Explicit producer qualification, not a promise about every GNU libc or
// x86 CPU. Run on the retained GNU x86 producer with its resolved expf
// implementation and executable provenance recorded by the RCH driver.
#[cfg(all(target_os = "linux", target_env = "gnu", target_arch = "x86_64"))]
#[test]
#[ignore = "requires the qualified GNU x86 live expf producer"]
fn scalar_exp_matches_live_qualified_gnu_x86_producer() {
    fn compare(input_bits: u32, sample: &str) {
        let input = f32::from_bits(input_bits);
        if !input.is_finite() {
            return;
        }
        // Keep the oracle a runtime library operation, independent of
        // the candidate implementation and its table/constants.
        let expected = std::hint::black_box(input).exp().to_bits();
        let observed = scalar_exp_fma(input).to_bits();
        assert_eq!(
            observed, expected,
            "sample={sample} input_bits={input_bits:#010x} input={input:?} \
             candidate_bits={observed:#010x} oracle_bits={expected:#010x}",
        );
    }

    // The conversion intentionally selects the nearest representable
    // f32 input around a mathematical grid or reduction boundary.
    #[allow(clippy::cast_possible_truncation)]
    fn input_bits(value: f64) -> u32 {
        (value as f32).to_bits()
    }

    // An odd additive stride traverses all u32 patterns before repeating.
    // This bounded prefix spans signs, exponents and significands; the
    // comparison helper excludes only non-finite inputs.
    let mut bits = 0_u32;
    for _ in 0..65_536 {
        compare(bits, "finite-bit-stride");
        bits = bits.wrapping_add(0x9e37_79b9);
    }

    // Dense practical stable-softmax range: [-128, 0] at exact 1/256
    // spacing, including the subnormal-output and zero-output region.
    for step in -32_768..=0 {
        compare(input_bits(f64::from(step) / 256.0), "softmax-grid");
    }

    // Neighbors on both sides of every half-integer reduction boundary
    // throughout the finite expf output range. Use a separate literal
    // here so this sampling does not read the candidate's constants.
    let reduction_scale = 32.0 / std::f64::consts::LN_2;
    for integer in -4_800..=4_096 {
        let center = input_bits((f64::from(integer) + 0.5) / reduction_scale);
        for offset in -2..=2 {
            compare(center.wrapping_add_signed(offset), "reduction-neighbor");
        }
    }

    // Directly cover representable neighbors at both range thresholds,
    // the normal/subnormal input boundary, signed zero, and the captured
    // hard model input where promotion to f64 exp chooses another bit.
    for center in [
        0x42b1_7217_u32,
        0xc2cf_f1b4,
        0x0080_0000,
        0x8080_0000,
        0x0000_0000,
        0x8000_0000,
        0xc088_10e8,
    ] {
        for offset in -8..=8 {
            compare(center.wrapping_add_signed(offset), "edge-neighbor");
        }
    }
}

#[test]
fn scalar_exp_preserves_recorded_rounding_and_extremes() {
    // Retained GNU FMA producer outputs, including a captured model input
    // for which exp(f64(input)) rounded to f32 chooses the other adjacent bit.
    for (input, expected) in [
        (0x0000_0000, 0x3f80_0000),
        (0x8000_0000, 0x3f80_0000),
        (0x3f80_0000, 0x402d_f854),
        (0xbf80_0000, 0x3ebc_5ab2),
        (0xc088_10e8, 0x3c69_3931),
        (0x42b1_7217, 0x7f7f_ff84),
        (0x42b1_7218, 0x7f80_0000),
        (0xc2cf_f1b4, 0x0000_0001),
        (0xc2cf_f1b5, 0x0000_0000),
        (0x7f80_0000, 0x7f80_0000),
        (0xff80_0000, 0x0000_0000),
    ] {
        assert_eq!(scalar_exp_fma(f32::from_bits(input)).to_bits(), expected);
    }
    for input in [0x7fc0_0001, 0xffc0_0001, 0x7fa0_0001] {
        assert!(scalar_exp_fma(f32::from_bits(input)).is_nan());
    }
}

#[test]
fn non_fused_exp_preserves_exceptional_lanes() {
    let input = f32x8::new([
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_1234),
        f32::from_bits(0xff80_0001),
        f32::MAX,
        -f32::MAX,
    ]);
    assert_eq!(
        exp_vec8_non_fused(input).to_array().map(f32::to_bits),
        [
            1.0_f32.to_bits(),
            1.0_f32.to_bits(),
            f32::INFINITY.to_bits(),
            0,
            0x7fc0_0101,
            0x7fc0_0101,
            f32::INFINITY.to_bits(),
            0,
        ]
    );
    assert_eq!(
        exp_vec8_non_fused(f32x8::splat(-104.0))
            .to_array()
            .map(f32::to_bits),
        [0; 8]
    );
}

// The oracle is the existing producer's actual separately rounded wide
// implementation. FMA-enabled builds use a different oracle and must not
// silently replace this comparison with a tolerance or another polynomial.
#[cfg(all(target_arch = "x86_64", not(target_feature = "fma")))]
#[test]
fn non_fused_exp_matches_existing_x86_producer_bits() {
    fn check(lanes: [f32; 8]) {
        let input = f32x8::new(lanes);
        assert_eq!(
            exp_vec8_non_fused(input).to_array().map(f32::to_bits),
            input.exp().to_array().map(f32::to_bits),
            "input bits: {:?}",
            lanes.map(f32::to_bits)
        );
    }

    for sample in 0_u16..=u16::MAX {
        let x = -104.0 + f32::from(sample) * (193.0 / 65_535.0);
        check([x, x.next_down(), x.next_up(), -x, 0.0, -0.0, 1.0, -1.0]);
    }
    check([
        (-103.63_f32).next_down(),
        -103.63,
        (-103.63_f32).next_up(),
        88.723_f32.next_down(),
        88.723,
        88.723_f32.next_up(),
        f32::INFINITY,
        f32::NEG_INFINITY,
    ]);
    for exponent in -150_i16..=128 {
        let boundary = (f32::from(exponent) + 0.5) / std::f32::consts::LOG2_E;
        let center = f32::from(exponent) / std::f32::consts::LOG2_E;
        check([
            boundary.next_down(),
            boundary,
            boundary.next_up(),
            center.next_down(),
            center,
            center.next_up(),
            f32::INFINITY,
            f32::NEG_INFINITY,
        ]);
    }
    let mut seed = 0x6d2b_79f5_u32;
    for _ in 0..8192 {
        let lanes = std::array::from_fn(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32::from_bits(seed)
        });
        check(lanes);
        let mut reversed = lanes;
        reversed.reverse();
        check(reversed);
    }
}

fn verified_async_fixture() -> NativeReranker {
    let dir = std::path::PathBuf::from(
        std::env::var("FRANKENSEARCH_RERANK_MODEL_DIR").expect("reranker fixture required"),
    );
    frankensearch_embed::model_manifest::ModelManifest::ms_marco_reranker()
        .verify_dir(&dir)
        .expect("registered cross-encoder artifacts");
    assert!(
        !dir.join(SAFETENSORS_PRIMARY).exists(),
        "unregistered preferred weights must not replace the verified fixture"
    );
    NativeReranker::load(dir).expect("actual native reranker")
}

#[test]
#[ignore = "requires verified FRANKENSEARCH_RERANK_MODEL_DIR; run optimized"]
fn async_fixture_preserves_scores_and_requires_explicit_pool() {
    use frankensearch_core::traits::Reranker;

    let native = verified_async_fixture();
    let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .blocking_threads(0, 2)
        .build()
        .unwrap();
    let cx = runtime.request_cx_with_budget(asupersync::types::Budget::INFINITE);
    let query = "how to retry failed requests";
    let documents = [
        doc("bread", "A recipe for baking sourdough bread."),
        doc(
            "retry",
            "Retry failed network requests with exponential backoff and jitter.",
        ),
        doc(
            "duplicate",
            "Retry failed network requests with exponential backoff and jitter.",
        ),
    ];
    assert!(!Reranker::is_available(&native));
    let error = runtime
        .block_on(native.rerank(&cx, query, &documents))
        .expect_err("no inline inference without a pool");
    assert!(matches!(error, SearchError::RerankFailed { .. }));
    assert!(error.to_string().contains("caller-owned blocking pool"));
    assert!(matches!(
        runtime.block_on(native.rerank(&cx, query, &[])),
        Err(SearchError::RerankFailed { .. })
    ));

    let expected = native.rerank_sync(query, &documents).unwrap();
    let native = native.with_blocking_pool(runtime.blocking_handle().unwrap());
    let cloned = native.clone();
    assert!(Reranker::is_available(&cloned));
    assert_eq!(Reranker::id(&cloned), SyncRerank::id(&native));
    assert_eq!(
        Reranker::model_name(&cloned),
        SyncRerank::model_name(&native)
    );
    assert_eq!(
        Reranker::max_length(&cloned),
        SyncRerank::max_length(&native)
    );
    let actual = runtime
        .block_on(cloned.rerank(&cx, query, &documents))
        .unwrap();
    assert_eq!(actual.len(), documents.len());
    assert_eq!(
        actual
            .iter()
            .map(|score| score.doc_id.as_str())
            .collect::<Vec<_>>(),
        ["retry", "duplicate", "bread"]
    );
    assert!(actual[0].score > actual[2].score);
    assert_eq!(actual[0].score, actual[1].score);
    for score in &actual {
        let reference = &expected[score.original_rank];
        assert_eq!(score.doc_id, reference.doc_id);
        assert_eq!(score.score.to_bits(), reference.score.to_bits());
        assert_eq!(
            score.raw_logit.map(f32::to_bits),
            reference.raw_logit.map(f32::to_bits)
        );
    }
    assert!(
        runtime
            .block_on(native.rerank(&cx, query, &[]))
            .unwrap()
            .is_empty()
    );
    let cancelled = runtime.request_cx_with_budget(asupersync::types::Budget::INFINITE);
    cancelled.cancel_fast(asupersync::CancelKind::User);
    for input in [documents.as_slice(), &[]] {
        assert!(matches!(
            runtime.block_on(native.rerank(&cancelled, query, input)),
            Err(SearchError::Cancelled { .. })
        ));
    }
    assert!(native.admission.try_lock().is_ok());
    let stopped_pool = asupersync::runtime::blocking_pool::BlockingPool::new(0, 1);
    let stopped = native.clone().with_blocking_pool(stopped_pool.handle());
    assert!(stopped_pool.shutdown_and_wait(std::time::Duration::from_secs(2)));
    assert!(!Reranker::is_available(&stopped));
    let error = runtime
        .block_on(stopped.rerank(&cx, query, &documents))
        .unwrap_err();
    assert!(matches!(error, SearchError::RerankFailed { .. }));
    assert!(error.to_string().contains("blocking pool is shut down"));
    assert!(native.admission.try_lock().is_ok());

    // Shutdown after the initial pool check but before admission tests the
    // runtime's inline recovery path with an otherwise live scheduler.
    let racing_pool = asupersync::runtime::blocking_pool::BlockingPool::new(0, 1);
    let racing = native.clone().with_blocking_pool(racing_pool.handle());
    runtime.block_on(async {
        let admission = OwnedMutexGuard::lock(Arc::clone(&native.admission), &cx)
            .await
            .unwrap();
        let mut pending = racing.rerank(&cx, query, &documents);
        std::future::poll_fn(|task_cx| {
            assert!(pending.as_mut().poll(task_cx).is_pending());
            std::task::Poll::Ready(())
        })
        .await;
        assert!(racing_pool.shutdown_and_wait(std::time::Duration::from_secs(2)));
        drop(admission);
        let error = pending.await.unwrap_err();
        assert!(matches!(error, SearchError::RerankFailed { .. }));
        assert!(
            error
                .to_string()
                .contains("dispatch fell back to an async executor")
        );
    });
    assert!(native.admission.try_lock().is_ok());
    assert!(runtime.shutdown_timeout(std::time::Duration::from_secs(5)));
}

#[test]
#[ignore = "requires verified FRANKENSEARCH_RERANK_MODEL_DIR; run optimized"]
fn async_fixture_request_cancellation_after_admission_never_publishes_scores() {
    use frankensearch_core::traits::Reranker;
    use std::time::{Duration, Instant};

    let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .blocking_threads(0, 1)
        .build()
        .unwrap();
    let pool = runtime.blocking_handle().unwrap();
    let native = verified_async_fixture().with_blocking_pool(pool.clone());
    let cx = runtime.request_cx_with_budget(asupersync::types::Budget::INFINITE);
    let documents = [doc(
        "retry",
        "Retry failed requests with exponential backoff.",
    )];
    let (held_tx, held_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let model = Arc::clone(&native.inner);
        scope.spawn(move || {
            let _model = model.lock().unwrap();
            held_tx.send(()).unwrap();
            let _ = release_rx.recv_timeout(Duration::from_secs(5));
        });
        held_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        runtime.block_on(async {
            let mut running = native.rerank(&cx, "retry", &documents);
            std::future::poll_fn(|task_cx| {
                assert!(running.as_mut().poll(task_cx).is_pending());
                std::task::Poll::Ready(())
            })
            .await;
            let started = Instant::now();
            while pool.busy_threads() != 1 {
                assert!(
                    started.elapsed() < Duration::from_secs(2),
                    "worker did not start"
                );
                asupersync::time::sleep(cx.now(), Duration::from_millis(1)).await;
            }
            cx.cancel_fast(asupersync::CancelKind::User);
            release_tx.send(()).unwrap();
            assert!(
                matches!(running.await, Err(SearchError::Cancelled { .. })),
                "request cancellation after admission must not return successful scores"
            );
        });
    });
    assert!(native.admission.try_lock().is_ok());
    let fresh = runtime.request_cx_with_budget(asupersync::types::Budget::INFINITE);
    let scores = runtime
        .block_on(native.rerank(&fresh, "retry", &documents))
        .unwrap();
    assert_eq!(scores.len(), 1);
    assert!(scores[0].score.is_finite());
    assert!(runtime.shutdown_timeout(Duration::from_secs(5)));
}

#[test]
#[ignore = "requires verified FRANKENSEARCH_RERANK_MODEL_DIR; run optimized"]
fn async_fixture_timeout_retains_admission_and_cancels_queued_waiter() {
    use frankensearch_core::traits::Reranker;
    use std::future::poll_fn;
    use std::task::Poll;
    use std::time::{Duration, Instant};

    let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .blocking_threads(0, 2)
        .build()
        .unwrap();
    let pool = runtime.blocking_handle().unwrap();
    let native = verified_async_fixture().with_blocking_pool(pool.clone());
    let cx = runtime.request_cx_with_budget(asupersync::types::Budget::INFINITE);
    let waiting_cx = runtime.request_cx_with_budget(asupersync::types::Budget::INFINITE);
    let documents = [doc(
        "retry",
        "Retry failed requests with exponential backoff.",
    )];
    let (held_tx, held_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        // Hold the actual model. The fuse turns accidental inline execution
        // into a failed responsiveness assertion rather than a test hang.
        let model = Arc::clone(&native.inner);
        scope.spawn(move || {
            let _model = model.lock().unwrap();
            held_tx.send(()).unwrap();
            let _ = release_rx.recv_timeout(Duration::from_secs(5));
        });
        held_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        runtime.block_on(async {
            let started = Instant::now();
            let mut running = native.rerank(&cx, "retry", &documents);
            poll_fn(|task_cx| {
                assert!(running.as_mut().poll(task_cx).is_pending());
                Poll::Ready(())
            })
            .await;
            while pool.busy_threads() != 1 {
                assert!(
                    started.elapsed() < Duration::from_secs(2),
                    "worker did not start"
                );
                asupersync::time::sleep(cx.now(), Duration::from_millis(1)).await;
            }
            assert!(native.admission.try_lock().is_err());
            assert!(
                asupersync::time::timeout(cx.now(), Duration::from_millis(20), running)
                    .await
                    .is_err()
            );
            assert!(
                started.elapsed() < Duration::from_secs(2),
                "executor was blocked"
            );
            assert_eq!(pool.busy_threads(), 1);
            assert!(
                native.admission.try_lock().is_err(),
                "dropped future must leave admission with its worker"
            );

            let cloned = native.clone();
            let mut waiting = cloned.rerank(&waiting_cx, "retry", &documents);
            poll_fn(|task_cx| {
                assert!(waiting.as_mut().poll(task_cx).is_pending());
                Poll::Ready(())
            })
            .await;
            waiting_cx.cancel_fast(asupersync::CancelKind::User);
            assert!(matches!(waiting.await, Err(SearchError::Cancelled { .. })));
            assert_eq!(
                pool.busy_threads(),
                1,
                "cancelled admission used another worker"
            );
            assert_eq!(pool.pending_count(), 0);
            release_tx.send(()).unwrap();
            let drain_started = Instant::now();
            while pool.busy_threads() != 0 {
                assert!(
                    drain_started.elapsed() < Duration::from_secs(2),
                    "worker did not drain"
                );
                asupersync::time::sleep(cx.now(), Duration::from_millis(1)).await;
            }
            assert!(native.admission.try_lock().is_ok());
            let healthy = cloned.rerank(&cx, "retry", &documents).await.unwrap();
            let sync = native.rerank_sync("retry", &documents).unwrap();
            assert_eq!(healthy.len(), 1);
            assert_eq!(healthy[0].doc_id, "retry");
            assert_eq!(healthy[0].raw_logit, sync[0].raw_logit);
            assert_eq!(healthy[0].score, sync[0].score);
        });
    });
    assert!(runtime.shutdown_timeout(Duration::from_secs(5)));
}

#[test]
fn f32_weight_decode_preserves_bits_for_aligned_and_unaligned_storage() {
    // Signed zero, subnormals, finite extremes, infinities, and NaN payloads
    // must survive loading without any floating-point arithmetic.
    let bits = [
        0_u32,
        0x8000_0000,
        1,
        0x8000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_1234,
        0x7f80_0001,
        0xffc0_4321,
    ];
    let bytes = bits
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    // u32 storage supplies a known aligned base; the offsets exercise every
    // possible alignment, including the scalar fallback, on either endian.
    let mut storage = vec![0_u32; bits.len() + 1];
    for offset in 0..4 {
        let storage_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut storage);
        storage_bytes[offset..offset + bytes.len()].copy_from_slice(&bytes);
        let input = &storage_bytes[offset..offset + bytes.len()];
        assert_eq!(
            decode_f32_le(input)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            bits,
            "byte offset {offset}"
        );
    }
    assert!(decode_f32_le(&[]).is_empty());
    for len in 1..4 {
        assert!(decode_f32_le(&bytes[..len]).is_empty());
        assert_eq!(decode_f32_le(&bytes[..4 + len])[0].to_bits(), bits[0]);
        assert_eq!(decode_f32_le(&bytes[..4 + len]).len(), 1);
    }
}

#[test]
fn f32_linear_matches_nonsquare_scalar_oracle_and_rejects_bad_storage() {
    let weights = [1.0, -2.0, 0.5, 3.0, 4.0, -1.0];
    let bias = [0.25, -0.75];
    let input = [2.0, 3.0, -1.0, -4.0, 0.5, 2.0];
    let layer = LinearWeights::from_f32(&weights, bias.to_vec(), 2, 3, LinearPrecision::F32)
        .expect("load 2x3 weight matrix");
    let actual = layer.forward(&input, 2).expect("two rows");
    let mut expected = Vec::new();
    for row in input.as_chunks::<3>().0 {
        for (output, &offset) in weights.as_chunks::<3>().0.iter().zip(&bias) {
            expected.push(row.iter().zip(output).map(|(a, b)| a * b).sum::<f32>() + offset);
        }
    }
    assert_eq!(actual, expected);
    assert!(layer.forward(&[], 0).expect("empty batch").is_empty());
    assert!(layer.forward(&input[..5], 2).is_err());
    assert!(
        layer.forward(&input, 1).is_err(),
        "extra rows must not be ignored"
    );
    assert!(LinearWeights::from_f32(&weights, vec![0.0], 2, 3, LinearPrecision::F32).is_err());
    assert!(
        LinearWeights::from_f32(&weights[..5], bias.to_vec(), 2, 3, LinearPrecision::F32).is_err()
    );
}

/// Legacy hand-ported fixture location (the macOS port that produced the
/// parity reference). Kept last in the search order so a registered cache
/// on any host wins.
const LEGACY_MODEL_DIR: &str = "/private/tmp/ee-reranker-port/model";
/// Install dir of the `ms-marco-minilm-l-6-v2` manifest inside the
/// frankensearch model cache (`fsfs download-models ms-marco-minilm-l-6-v2`).
const CACHE_MODEL_DIR_NAME: &str = "ms-marco-MiniLM-L-6-v2";

fn dir_has_model(dir: &Path) -> bool {
    dir.join(TOKENIZER_JSON).is_file()
        && (dir.join(SAFETENSORS_PRIMARY).is_file() || dir.join(SAFETENSORS_FALLBACK).is_file())
}

/// Resolve the model directory the real-model tests load from:
/// `FRANKENSEARCH_RERANK_MODEL_DIR`, else the registered model cache
/// (`FRANKENSEARCH_MODEL_DIR` / XDG / `~/.local/share/frankensearch/models`),
/// else the legacy port fixture. Only a directory that actually carries the
/// tokenizer and weights is returned.
fn model_dir() -> Option<std::path::PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(explicit) = std::env::var("FRANKENSEARCH_RERANK_MODEL_DIR")
        && !explicit.trim().is_empty()
    {
        candidates.push(std::path::PathBuf::from(explicit));
    }
    candidates
        .push(frankensearch_embed::model_cache::resolve_cache_root().join(CACHE_MODEL_DIR_NAME));
    candidates.push(std::path::PathBuf::from(LEGACY_MODEL_DIR));
    candidates.into_iter().find(|dir| dir_has_model(dir))
}

// (query, document, reference logit) from the validated parity_cases.json
// (numpy reference in f64, itself validated bit-for-ranking against the real ONNX
// model). The forward runs int8 Linear matmuls on an f32 substrate, so logits track
// the f64 reference only within an int8 quantization tolerance (PARITY_TOL); the
// ranking is what must stay identical (as it did for the original int8 ONNX model).
const PARITY_TOL: f64 = 0.6;
const CASES: &[(&str, &str, f64)] = &[
    (
        "how to fix a failing release workflow",
        "the release pipeline builds cross platform binaries and uploads them to github",
        -9.808_567,
    ),
    (
        "how to fix a failing release workflow",
        "bananas are a good source of potassium and taste sweet",
        -11.332_987,
    ),
    (
        "what is the capital of france",
        "paris is the capital and most populous city of france",
        7.472_003,
    ),
    (
        "rust memory safety",
        "the borrow checker enforces ownership rules at compile time",
        -11.367_251,
    ),
];

fn model_available() -> bool {
    model_dir().is_some()
}

fn doc(id: &str, text: &str) -> RerankDocument {
    RerankDocument {
        doc_id: id.to_owned(),
        text: text.to_owned(),
    }
}

#[test]
fn parity_logits_and_ranking_match_reference() {
    if !model_available() {
        eprintln!(
            "[native_reranker] SKIP parity: no model dir (FRANKENSEARCH_RERANK_MODEL_DIR, the model cache, or {LEGACY_MODEL_DIR})"
        );
        return;
    }
    let reranker = NativeReranker::load(model_dir().expect("reranker model dir"))
        .expect("load native reranker");
    let mut logits = Vec::new();
    let mut max_diff = 0.0_f64;
    eprintln!("[native_reranker] idx |     ft_logit |    ref_logit |     diff");
    for (i, (query, document, ref_logit)) in CASES.iter().enumerate() {
        let scored = reranker
            .rerank_sync(query, &[doc("d", document)])
            .expect("rerank_sync");
        assert_eq!(scored.len(), 1, "one doc in, one score out");
        let logit = f64::from(scored[0].raw_logit.expect("raw logit present"));
        let diff = (logit - ref_logit).abs();
        max_diff = max_diff.max(diff);
        logits.push(logit);
        eprintln!("[native_reranker] {i:3} | {logit:12.6} | {ref_logit:12.6} | {diff:8.5}");
        assert!(
            diff < PARITY_TOL,
            "case {i} logit {logit} differs from reference {ref_logit} by {diff} (>{PARITY_TOL})"
        );
    }
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    eprintln!(
        "[native_reranker] ranking(desc)={order:?} expected=[2, 0, 1, 3] max_diff={max_diff:.6}"
    );
    assert_eq!(order, vec![2usize, 0, 1, 3], "ranking must match reference");
}

#[test]
fn forward_batch_matches_per_doc() {
    // The batched (varlen) forward must produce the same logits as running each
    // document through the single-pair `forward`: it reuses the same attention
    // per document and every other op is row-wise, so batching is exact, not an
    // approximation. This is the contract that lets the throughput lever ship
    // without touching the validated ranking.
    if !model_available() {
        eprintln!("[native_reranker] SKIP batch-equiv: model dir not present");
        return;
    }
    let reranker = NativeReranker::load(model_dir().expect("reranker model dir"))
        .expect("load native reranker");
    let query = CASES[0].0;
    let mut batch: Vec<(Vec<i64>, Vec<i64>)> = Vec::new();
    for (_, document, _) in CASES {
        let enc = reranker
            .tokenizer
            .encode((query, *document), true)
            .expect("tokenize");
        let ids: Vec<i64> = enc.get_ids().iter().map(|&x| i64::from(x)).collect();
        let typ: Vec<i64> = enc.get_type_ids().iter().map(|&x| i64::from(x)).collect();
        batch.push((ids, typ));
    }
    let mut model = reranker.inner.lock().expect("lock");
    let per_doc: Vec<f32> = batch
        .iter()
        .map(|(ids, typ)| model.forward(ids, typ).expect("forward"))
        .collect();
    let batched = model.forward_batch(&batch).expect("forward_batch");
    drop(model);
    assert_eq!(batched.len(), per_doc.len());
    for (i, (b, p)) in batched.iter().zip(&per_doc).enumerate() {
        let diff = (f64::from(*b) - f64::from(*p)).abs();
        eprintln!("[native_reranker] doc {i}: batched={b:.6} per_doc={p:.6} diff={diff:.2e}");
        assert!(
            diff < 1e-3,
            "doc {i}: batched {b} vs per-doc {p} diff {diff} too large"
        );
    }
}

#[test]
fn empty_documents_yield_empty_scores() {
    if !model_available() {
        eprintln!("[native_reranker] SKIP empty-docs: model dir not present");
        return;
    }
    let reranker = NativeReranker::load(model_dir().expect("reranker model dir")).expect("load");
    let scored = reranker.rerank_sync("any query", &[]).expect("empty ok");
    assert!(scored.is_empty());
    eprintln!("[native_reranker] empty-docs -> empty scores OK");
}

#[test]
fn whitespace_and_long_documents_do_not_panic() {
    if !model_available() {
        eprintln!("[native_reranker] SKIP whitespace/long: model dir not present");
        return;
    }
    let reranker = NativeReranker::load(model_dir().expect("reranker model dir")).expect("load");
    // whitespace-only doc
    let ws = reranker
        .rerank_sync("q", &[doc("ws", "   ")])
        .expect("whitespace ok");
    assert_eq!(ws.len(), 1);
    // very long doc (forces truncation well beyond max_length)
    let long_text = "memory safety ".repeat(400);
    let lng = reranker
        .rerank_sync("rust", &[doc("long", &long_text)])
        .expect("long ok");
    assert_eq!(lng.len(), 1);
    assert!(lng[0].score.is_finite());
    eprintln!(
        "[native_reranker] whitespace score={:.6}, truncated-long score={:.6} OK",
        ws[0].score, lng[0].score
    );
}

#[test]
fn ranking_is_deterministic_across_runs() {
    if !model_available() {
        eprintln!("[native_reranker] SKIP determinism: model dir not present");
        return;
    }
    let reranker = NativeReranker::load(model_dir().expect("reranker model dir")).expect("load");
    let docs: Vec<RerankDocument> = CASES
        .iter()
        .enumerate()
        .map(|(i, (_, d, _))| doc(&format!("d{i}"), d))
        .collect();
    let run1 = reranker
        .rerank_sync("what is the capital of france", &docs)
        .expect("run1");
    let run2 = reranker
        .rerank_sync("what is the capital of france", &docs)
        .expect("run2");
    assert_eq!(run1.len(), run2.len());
    for (a, b) in run1.iter().zip(run2.iter()) {
        assert_eq!(a.doc_id, b.doc_id);
        assert_eq!(a.raw_logit, b.raw_logit, "logits must be deterministic");
    }
    eprintln!("[native_reranker] determinism across 2 runs OK");
}

#[test]
fn many_documents_rerank_without_deadlock() {
    // Regression guard for the nested-rayon + Mutex deadlock. The fix runs
    // each forward on its slot's dedicated single-thread rayon pool, so no
    // frankentorch op (int8 linear, the f32 attention bmm, softmax,
    // layer_norm, ...) can fan rayon work back onto the doc-dispatch pool
    // while a worker holds a session Mutex. Reranking far more documents
    // than the pool size forces multiple rayon workers onto the same slot —
    // the exact collision that previously hung the multi-doc path. If a
    // future change re-introduces nesting (a forward spawning rayon work
    // onto the doc-dispatch pool while holding its session Mutex), this test
    // deadlocks (CI timeout) rather than silently shipping a hang.
    if !model_available() {
        eprintln!("[native_reranker] SKIP many-docs: model dir not present");
        return;
    }
    let reranker = NativeReranker::load(model_dir().expect("reranker model dir")).expect("load");
    // 24 docs >> the 8-session pool cap, so several workers share a slot.
    let docs: Vec<RerankDocument> = (0..24)
        .map(|i| doc(&format!("d{i}"), CASES[i % CASES.len()].1))
        .collect();
    let scored = reranker
        .rerank_sync("what is the capital of france", &docs)
        .expect("many-doc rerank completes (no deadlock)");
    assert_eq!(scored.len(), docs.len(), "one score per doc");
    for (i, s) in scored.iter().enumerate() {
        assert_eq!(s.original_rank, i, "original_rank preserves input order");
        assert_eq!(s.doc_id, format!("d{i}"));
        assert!(s.score.is_finite());
    }
    // Same input twice -> identical scores (the parallel path is deterministic).
    let again = reranker
        .rerank_sync("what is the capital of france", &docs)
        .expect("rerun");
    for (a, b) in scored.iter().zip(again.iter()) {
        assert_eq!(
            a.raw_logit, b.raw_logit,
            "parallel rerank must be deterministic"
        );
    }
    eprintln!(
        "[native_reranker] {}-doc concurrent rerank OK (no deadlock, deterministic)",
        docs.len()
    );
}

#[test]
fn load_missing_dir_errors() {
    let err = NativeReranker::load("/private/tmp/definitely-not-a-model-dir-xyz");
    assert!(err.is_err(), "loading a missing dir must error, not panic");
    eprintln!("[native_reranker] missing-dir load error OK: {err:?}");
}
