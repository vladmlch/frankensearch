#[cfg(all(feature = "bench-internals", feature = "conformance-internals"))]
use crate::query::QueryField;

use super::*;

/// A refresh with nothing new published reads only the MANIFEST: it
/// succeeds even when the (immutable) segment files cannot be opened, and
/// a later publication is still caught.
#[cfg(unix)]
#[test]
fn read_only_refresh_without_publication_opens_no_segment() {
    use std::os::unix::fs::PermissionsExt as _;

    let directory = tempfile::tempdir().expect("refresh directory");
    let cx = Cx::for_testing();
    let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .blocking_threads(1, 4)
        .build()
        .expect("build refresh runtime");
    runtime.block_on(async {
        let writer = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create writer");
        LexicalWrite::index_document(&writer, &cx, &IndexableDocument::new("old", "old token"))
            .await
            .expect("index first document");
        LexicalWrite::commit(&writer, &cx)
            .await
            .expect("publish first document");
        let reader = QuillSearchIndex::open(&cx, directory.path(), deterministic_config())
            .await
            .expect("open read-only reader");
        assert!(!reader.refresh(&cx).await.expect("unchanged refresh"));

        let segments = std::fs::read_dir(directory.path())
            .expect("list index directory")
            .map(|entry| entry.expect("index entry").path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "fslx"))
            .collect::<Vec<_>>();
        assert!(!segments.is_empty(), "fixture must have a sealed segment");
        let set_mode = |mode| {
            for segment in &segments {
                std::fs::set_permissions(segment, std::fs::Permissions::from_mode(mode))
                    .expect("set segment mode");
            }
        };
        set_mode(0o000);
        let unchanged = reader.refresh(&cx).await;
        set_mode(0o644);
        assert!(!unchanged.expect("an unchanged MANIFEST needs no segment"));

        LexicalWrite::index_document(&writer, &cx, &IndexableDocument::new("new", "new token"))
            .await
            .expect("index successor document");
        LexicalWrite::commit(&writer, &cx)
            .await
            .expect("publish successor");
        assert!(reader.refresh(&cx).await.expect("refresh to successor"));
        assert_eq!(
            LexicalRead::search(&reader, &cx, "new", 10)
                .await
                .expect("search successor")[0]
                .doc_id,
            "new"
        );
    });
}

#[test]
fn root_bound_reader_refreshes_to_the_current_blue_green_generation() {
    let root = tempfile::tempdir().expect("root-bound reader directory");
    let lexical_root = root.path().to_path_buf();
    let cx = Cx::for_testing();
    let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .blocking_threads(1, 4)
        .build()
        .expect("build root-bound reader runtime");

    runtime.block_on(async {
        let v1 = create_root_generation(&cx, &lexical_root, "quill-v1", "old", "old token").await;
        publish_current(&lexical_root, &root_pointer("quill-v1"))
            .expect("publish first root generation");
        let reader = RootBoundQuillSearchIndex::open(&cx, &lexical_root, deterministic_config())
            .await
            .expect("open root-bound reader");
        assert_eq!(reader.active_path(), v1);
        assert_eq!(
            LexicalRead::search(&reader, &cx, "old", 10)
                .await
                .expect("search first generation")[0]
                .doc_id,
            "old"
        );

        let v2 = create_root_generation(&cx, &lexical_root, "quill-v2", "new", "new token").await;
        publish_current(&lexical_root, &root_pointer("quill-v2"))
            .expect("publish replacement root generation");
        assert!(reader.refresh(&cx).await.expect("refresh root reader"));
        assert_eq!(reader.active_path(), v2);
        assert_eq!(
            LexicalRead::search(&reader, &cx, "new", 10)
                .await
                .expect("search replacement generation")[0]
                .doc_id,
            "new"
        );
        assert!(
            LexicalRead::search(&reader, &cx, "old", 10)
                .await
                .expect("search replacement generation")
                .is_empty(),
            "refresh must switch the root reader rather than reopening the original child"
        );
    });
}

#[test]
fn labruntime_cancel_before_root_bound_swap_retains_the_old_reader() {
    let root = tempfile::tempdir().expect("root-bound cancellation directory");
    let lexical_root = root.path().to_path_buf();
    let setup_cx = Cx::for_testing();
    let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .blocking_threads(1, 4)
        .build()
        .expect("build root-bound cancellation runtime");

    let (reader, replacement) = runtime.block_on(async {
        create_root_generation(&setup_cx, &lexical_root, "quill-v1", "old", "old token").await;
        publish_current(&lexical_root, &root_pointer("quill-v1"))
            .expect("publish first root generation");
        let reader =
            RootBoundQuillSearchIndex::open(&setup_cx, &lexical_root, deterministic_config())
                .await
                .expect("open root-bound reader");
        create_root_generation(&setup_cx, &lexical_root, "quill-v2", "new", "new token").await;
        publish_current(&lexical_root, &root_pointer("quill-v2"))
            .expect("publish replacement root generation");
        let replacement = RootBoundQuillSearchIndex::open_state(
            &setup_cx,
            &lexical_root,
            &deterministic_config(),
        )
        .await
        .expect("open and validate replacement before Lab cancellation");
        (reader, replacement)
    });
    let pause = Arc::clone(&reader.refresh_pause);
    pause
        .stage
        .store(RootRefreshPause::STAGE_SWAP, Ordering::Release);
    let refresh_cx = Arc::new(Cx::for_testing());
    let cancelled = Arc::new(AtomicBool::new(false));

    let mut lab = LabRuntime::new(LabConfig::new(0x8a02_0002).max_steps(100_000));
    let region = lab.state.create_root_region(Budget::INFINITE);
    let refresh_reader = reader.clone();
    let refresh_context = Arc::clone(&refresh_cx);
    let refresh_cancelled = Arc::clone(&cancelled);
    let (refresh, _) = lab
        .state
        .create_task(region, Budget::INFINITE, async move {
            assert!(matches!(
                refresh_reader
                    .install_validated_replacement(refresh_context.as_ref(), replacement)
                    .await,
                Err(QuillIndexError::Cancelled {
                    phase: "root-bound reader refresh swap"
                })
            ));
            refresh_cancelled.store(true, Ordering::SeqCst);
        })
        .expect("create root-bound refresh task");

    let cancel_context = Arc::clone(&refresh_cx);
    let cancel_pause = Arc::clone(&pause);
    let (cancel_task, _) = lab
        .state
        .create_task(region, Budget::INFINITE, async move {
            while !cancel_pause.arrived.load(Ordering::Acquire) {
                yield_now().await;
            }
            cancel_context.set_cancel_requested(true);
            cancel_pause.release.store(true, Ordering::Release);
        })
        .expect("create root-bound cancellation task");

    lab.scheduler.lock().schedule(refresh, 0);
    lab.scheduler.lock().schedule(cancel_task, 0);
    let report = lab.run_until_quiescent_with_report();
    assert!(cancelled.load(Ordering::SeqCst));
    assert!(report.quiescent, "root-bound refresh schedule must quiesce");
    assert!(
        report.oracle_report.all_passed(),
        "root-bound refresh oracles must pass"
    );
    assert!(report.invariant_violations.is_empty());

    runtime.block_on(async {
        assert_eq!(reader.active_path(), lexical_root.join("quill-v1"));
        assert_eq!(
            LexicalRead::search(&reader, &setup_cx, "old", 10)
                .await
                .expect("old reader remains usable after cancellation")[0]
                .doc_id,
            "old"
        );
        assert!(
            LexicalRead::search(&reader, &setup_cx, "new", 10)
                .await
                .expect("cancelled refresh leaves old generation active")
                .is_empty()
        );
    });
}

/// Park a real `refresh` at one cancellation window, cancel it there under
/// a deterministic `LabRuntime` schedule, and prove the old reader survives.
///
/// bd-8nqz.2's acceptance names "cancellation at every open/validate/swap
/// stage". The swap window had a test; the other two did not, and the
/// difference matters because they fail differently: the OPEN window has
/// no replacement reader in hand, while the VALIDATE window is holding a
/// fully opened `QuillSearchIndex` that must be dropped rather than
/// installed. A cancellation that leaked at either point would leave the
/// reader serving a generation nobody asked for, which is precisely the
/// failure this bead exists to make impossible.
///
/// The pause is what makes this a proof rather than a race: the refresh
/// task parks AT the window, the canceller waits for it to arrive, and only
/// then requests cancellation. Without it the canceller can win before the
/// refresh reaches the checkpoint and the test passes without ever
/// exercising the window it names.
fn assert_cancel_at_stage_retains_old_reader(seed: u64, stage: u8, expected_phase: &'static str) {
    let root = tempfile::tempdir().expect("root-bound cancellation directory");
    let lexical_root = root.path().to_path_buf();
    let setup_cx = Cx::for_testing();
    let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .blocking_threads(1, 4)
        .build()
        .expect("build root-bound cancellation runtime");

    let reader = runtime.block_on(async {
        create_root_generation(&setup_cx, &lexical_root, "quill-v1", "old", "old token").await;
        publish_current(&lexical_root, &root_pointer("quill-v1"))
            .expect("publish first root generation");
        let reader =
            RootBoundQuillSearchIndex::open(&setup_cx, &lexical_root, deterministic_config())
                .await
                .expect("open root-bound reader");
        // The replacement is published on disk, so a refresh that ran to
        // completion WOULD swap. That is what makes the retained old
        // generation below evidence of cancellation rather than of there
        // being nothing to move to.
        create_root_generation(&setup_cx, &lexical_root, "quill-v2", "new", "new token").await;
        publish_current(&lexical_root, &root_pointer("quill-v2"))
            .expect("publish replacement root generation");
        reader
    });

    let pause = Arc::clone(&reader.refresh_pause);
    pause.stage.store(stage, Ordering::Release);
    let refresh_cx = Arc::new(Cx::for_testing());
    let cancelled = Arc::new(AtomicBool::new(false));

    let mut lab = LabRuntime::new(LabConfig::new(seed).max_steps(100_000));
    let region = lab.state.create_root_region(Budget::INFINITE);
    let refresh_reader = reader.clone();
    let refresh_context = Arc::clone(&refresh_cx);
    let refresh_cancelled = Arc::clone(&cancelled);
    let (refresh, _) = lab
        .state
        .create_task(region, Budget::INFINITE, async move {
            let outcome = refresh_reader.refresh(refresh_context.as_ref()).await;
            assert!(
                matches!(outcome, Err(QuillIndexError::Cancelled { phase }) if phase == expected_phase),
                "refresh cancelled at {expected_phase} must report that phase, got {outcome:?}"
            );
            refresh_cancelled.store(true, Ordering::SeqCst);
        })
        .expect("create root-bound refresh task");

    let cancel_context = Arc::clone(&refresh_cx);
    let cancel_pause = Arc::clone(&pause);
    let (cancel_task, _) = lab
        .state
        .create_task(region, Budget::INFINITE, async move {
            while !cancel_pause.arrived.load(Ordering::Acquire) {
                yield_now().await;
            }
            cancel_context.set_cancel_requested(true);
            cancel_pause.release.store(true, Ordering::Release);
        })
        .expect("create root-bound cancellation task");

    lab.scheduler.lock().schedule(refresh, 0);
    lab.scheduler.lock().schedule(cancel_task, 0);
    let report = lab.run_until_quiescent_with_report();
    assert!(
        cancelled.load(Ordering::SeqCst),
        "the refresh task must have observed cancellation at {expected_phase}"
    );
    assert!(
        pause.arrived.load(Ordering::Acquire),
        "the refresh must actually have REACHED the {expected_phase} window; a test that \
         cancels before the window proves nothing about it"
    );
    assert!(report.quiescent, "root-bound refresh schedule must quiesce");
    assert!(
        report.oracle_report.all_passed(),
        "root-bound refresh oracles must pass"
    );
    assert!(report.invariant_violations.is_empty());

    runtime.block_on(async {
        assert_eq!(
            reader.active_path(),
            lexical_root.join("quill-v1"),
            "a cancelled refresh must leave the old engine directory pinned"
        );
        assert_eq!(
            LexicalRead::search(&reader, &setup_cx, "old", 10)
                .await
                .expect("old reader remains usable after cancellation")[0]
                .doc_id,
            "old"
        );
        assert!(
            LexicalRead::search(&reader, &setup_cx, "new", 10)
                .await
                .expect("cancelled refresh leaves old generation active")
                .is_empty(),
            "the replacement generation must not be observable after a cancelled refresh"
        );
    });
}

/// bd-8nqz.2 TOTAL LAYOUT INSPECTION at the READER boundary.
///
/// `inspect_lexical_layout` already classifies all eight states and
/// `inspect_lexical_layout_classifies_every_state_without_mutating` pins
/// that. What was NOT pinned is the consequence the acceptance actually
/// names: "mixed/ambiguous/corrupt layouts failing closed" and "read-only
/// open performs zero filesystem writes and acquires no writer lease".
/// Classification is an input to that decision, not the decision itself —
/// a reader could classify a root as `Mixed` and still go on to open
/// something, and the layout test would stay green.
///
/// So this walks every state a reader can meet and asserts three things
/// per state: the typed outcome, that a refusal's message NAMES the layout
/// (an unattributable refusal is not a typed one), and that the refusal
/// left the root byte-identical — no `CURRENT` published, no directory
/// created, no lease file.
///
/// Foreign engines are the interesting half. A `DirectTantivy` root and a
/// blue-green root whose `CURRENT` selects Tantivy are both perfectly
/// VALID layouts that this reader must nonetheless refuse, because it only
/// speaks Quill. Refusing them is not error handling; it is the rollback
/// contract — the facade keeps a Tantivy generation openable when
/// `lexical-tantivy` is compiled, and it can only do that if the Quill
/// reader declines rather than guesses.
#[test]
fn root_bound_open_refuses_every_foreign_or_damaged_layout_without_writing() {
    let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .blocking_threads(1, 4)
        .build()
        .expect("build total-layout runtime");
    let cx = Cx::for_testing();

    // A refusal must not leave a trace. Compared before and after the
    // attempted open, so "no writes" is measured rather than asserted.
    fn root_fingerprint(root: &Path) -> Vec<(std::ffi::OsString, u64)> {
        let Ok(entries) = std::fs::read_dir(root) else {
            return Vec::new();
        };
        let mut listing = entries
            .filter_map(Result::ok)
            .map(|entry| {
                let len = entry.metadata().map(|meta| meta.len()).unwrap_or_default();
                (entry.file_name(), len)
            })
            .collect::<Vec<_>>();
        listing.sort();
        listing
    }

    runtime.block_on(async {
        // --- states that must OPEN -------------------------------------
        let direct = tempfile::tempdir().expect("direct-quill root");
        let direct_root = direct.path().join("lexical");
        std::fs::create_dir_all(&direct_root).expect("create direct root");
        create_root_generation(&cx, direct.path(), "lexical", "d", "direct token").await;
        RootBoundQuillSearchIndex::open(&cx, &direct_root, deterministic_config())
            .await
            .expect("a direct Quill root is readable for migration compatibility");

        let blue = tempfile::tempdir().expect("blue-green root");
        let blue_root = blue.path().to_path_buf();
        create_root_generation(&cx, &blue_root, "quill-v1", "b", "blue token").await;
        publish_current(&blue_root, &root_pointer("quill-v1")).expect("publish CURRENT");
        RootBoundQuillSearchIndex::open(&cx, &blue_root, deterministic_config())
            .await
            .expect("a published Quill blue-green root is readable");

        // --- states that must FAIL CLOSED ------------------------------
        /// One refused layout: its stable label, and how to build a root
        /// that classifies as it.
        type RefusedLayout = (&'static str, Box<dyn Fn(&Path)>);
        let refusals: Vec<RefusedLayout> = vec![
            ("empty", Box::new(|_root: &Path| {})),
            (
                "direct-tantivy",
                Box::new(|root: &Path| {
                    std::fs::write(root.join("meta.json"), b"{}").expect("tantivy marker");
                }),
            ),
            (
                "mixed",
                Box::new(|root: &Path| {
                    std::fs::write(root.join("MANIFEST"), b"m").expect("quill marker");
                    std::fs::write(root.join("meta.json"), b"{}").expect("tantivy marker");
                }),
            ),
            (
                "ambiguous",
                Box::new(|root: &Path| {
                    for name in ["quill-v1", "quill-v2"] {
                        let child = root.join(name);
                        std::fs::create_dir_all(&child).expect("candidate child");
                        std::fs::write(child.join("MANIFEST"), b"m").expect("child marker");
                    }
                }),
            ),
        ];

        for (label, build) in refusals {
            let dir = tempfile::tempdir().expect("refusal root");
            let root = dir.path().to_path_buf();
            build(&root);
            let before = root_fingerprint(&root);

            let error = RootBoundQuillSearchIndex::open(&cx, &root, deterministic_config())
                .await
                .map(|_| ())
                .expect_err(&format!("{label} layout must fail closed"));
            let rendered = error.to_string();
            assert!(
                rendered.contains(label),
                "{label} refusal must name the layout it refused; got {rendered}"
            );
            assert_eq!(
                root_fingerprint(&root),
                before,
                "{label} refusal wrote to the root; read-only open must not mutate"
            );
            assert!(
                !root.join("CURRENT").exists(),
                "{label} refusal published a CURRENT pointer"
            );
        }

        // A blue-green root whose CURRENT selects a FOREIGN engine is a
        // valid layout, refused for a different reason than damage: this
        // reader only speaks Quill, and the Tantivy generation stays
        // available to the facade's rollback path.
        let foreign = tempfile::tempdir().expect("foreign-pointer root");
        let foreign_root = foreign.path().to_path_buf();
        let tantivy_child = foreign_root.join("tantivy-v1");
        std::fs::create_dir_all(&tantivy_child).expect("tantivy child");
        std::fs::write(tantivy_child.join("meta.json"), b"{}").expect("tantivy marker");
        publish_current(
            &foreign_root,
            &CurrentPointer::new(
                BlueGreenEngine::Tantivy,
                "tantivy-v1",
                crate::segment::FSLX_FORMAT_VERSION,
            )
            .expect("valid Tantivy pointer"),
        )
        .expect("publish foreign CURRENT");
        let before = root_fingerprint(&foreign_root);
        let error = RootBoundQuillSearchIndex::open(&cx, &foreign_root, deterministic_config())
            .await
            .map(|_| ())
            .expect_err("a Tantivy CURRENT must not be opened by the Quill reader");
        let rendered = error.to_string();
        assert!(
            rendered.contains("only supports Quill"),
            "a foreign-engine refusal must say so rather than report damage: {rendered}"
        );
        assert_eq!(
            root_fingerprint(&foreign_root),
            before,
            "refusing a foreign CURRENT must not mutate the root"
        );
        assert!(
            tantivy_child.join("meta.json").exists(),
            "the Tantivy generation must survive intact for rollback"
        );
    });
}

/// OPEN window: cancelled while holding only the inspected layout.
#[test]
fn labruntime_cancel_during_root_bound_open_retains_the_old_reader() {
    assert_cancel_at_stage_retains_old_reader(
        0x8a02_0003,
        RootRefreshPause::STAGE_INSPECTION,
        "root-bound reader layout inspection",
    );
}

/// VALIDATE window: cancelled while holding a fully opened replacement
/// reader that has not been installed. The opened index must be dropped,
/// not swapped in.
#[test]
fn labruntime_cancel_during_root_bound_validation_retains_the_old_reader() {
    assert_cancel_at_stage_retains_old_reader(
        0x8a02_0004,
        RootRefreshPause::STAGE_VALIDATION,
        "root-bound reader validation",
    );
}

#[cfg(all(feature = "bench-internals", feature = "conformance-internals"))]
#[derive(Clone, Debug)]
struct RankedCacheTraceWriter {
    buffer: Arc<StdMutex<Vec<u8>>>,
}

#[cfg(all(feature = "bench-internals", feature = "conformance-internals"))]
impl Write for RankedCacheTraceWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.buffer
            .lock()
            .expect("ranked-cache trace buffer lock is not poisoned")
            .extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(all(feature = "bench-internals", feature = "conformance-internals"))]
fn trace_text_field<'a>(line: &'a str, field: &str) -> Option<&'a str> {
    let prefix = format!("{field}=");
    let start = line.find(&prefix)?.saturating_add(prefix.len());
    let value = line.get(start..)?;
    if let Some(quoted) = value.strip_prefix('"') {
        return quoted.split_once('"').map(|(field_value, _)| field_value);
    }
    let end = value
        .find(|ch: char| ch.is_ascii_whitespace() || matches!(ch, ',' | '}'))
        .unwrap_or(value.len());
    value.get(..end)
}

#[cfg(all(feature = "bench-internals", feature = "conformance-internals"))]
fn is_trace_stage_close(line: &str, stage: &str) -> bool {
    if !line.contains(": close") {
        return false;
    }
    let Some(stage_position) = line.rfind(stage) else {
        return false;
    };
    crate::tracing_conventions::ALL_SPAN_NAMES
        .iter()
        .filter_map(|candidate| line.rfind(candidate))
        .all(|candidate_position| candidate_position <= stage_position)
}

#[cfg(all(feature = "bench-internals", feature = "conformance-internals"))]
fn assert_ranked_cache_trace_case(
    logs: &str,
    trace_case: &str,
    query_form: &str,
    cache_lookup: &str,
    expected_children: &[&str],
) {
    let case_lines = logs
        .lines()
        .filter(|line| trace_text_field(line, "trace_case") == Some(trace_case))
        .collect::<Vec<_>>();
    assert!(
        !case_lines.is_empty(),
        "missing trace case {trace_case}: {logs}"
    );
    let query = case_lines
        .iter()
        .copied()
        .find(|line| is_trace_stage_close(line, crate::tracing_conventions::ARGUS_QUERY))
        .expect("ranked-cache trace case must contain one ARGUS_QUERY close");
    assert_eq!(
        trace_text_field(query, "query_form"),
        Some(query_form),
        "trace case {trace_case} reported the wrong query form: {query}",
    );
    assert_eq!(
        trace_text_field(query, "cache_lookup"),
        Some(cache_lookup),
        "trace case {trace_case} reported the wrong cache disposition: {query}",
    );
    assert!(
        query.contains("duration_us="),
        "trace case {trace_case} omitted query timing: {query}",
    );
    if cache_lookup == "not_checked" {
        assert!(
            !query.contains("result_count=") && !query.contains("total_count="),
            "pre-cancelled trace case {trace_case} fabricated result evidence: {query}",
        );
    } else {
        assert!(
            query.contains("result_count=") && query.contains("total_count="),
            "successful exact-count trace case {trace_case} omitted result evidence: {query}",
        );
    }

    for stage in [
        crate::tracing_conventions::ARGUS_PARSE,
        crate::tracing_conventions::ARGUS_SCORE,
        crate::tracing_conventions::ARGUS_COLLECT,
    ] {
        let observed = case_lines
            .iter()
            .any(|line| is_trace_stage_close(line, stage));
        assert_eq!(
            observed,
            expected_children.contains(&stage),
            "trace case {trace_case} child-stage mismatch for {stage}: {logs}",
        );
    }
}

#[test]
fn ranked_query_cache_requires_an_exact_key_and_snapshot_epoch() {
    let cache = RankedQueryCache::default();
    let raw_result = QuillSearchResult {
        hits: vec![QuillHit {
            document_id: "raw-hit".to_owned(),
            global_docid: 7,
            score: 3.5,
        }]
        .into(),
        total_count: None,
        doc_count: 11,
        diagnostics: Vec::new(),
    };
    let raw_fingerprint = raw_query_cache_fingerprint(4, "rust", 10, 2, false);
    cache.insert_raw(raw_fingerprint, 4, "rust", 10, 2, false, &raw_result);
    let get_raw = |snapshot_epoch, query, limit, offset, exact_count| {
        cache.get_raw(
            raw_fingerprint,
            snapshot_epoch,
            query,
            limit,
            offset,
            exact_count,
        )
    };

    let first_hit = get_raw(4, "rust", 10, 2, false).expect("first exact cache hit");
    let second_hit = get_raw(4, "rust", 10, 2, false).expect("second exact cache hit");
    assert_eq!(first_hit, raw_result);
    assert_eq!(second_hit, raw_result);
    assert!(Arc::ptr_eq(&first_hit.hits, &second_hit.hits));
    assert!(Arc::ptr_eq(&first_hit.hits, &raw_result.hits));
    assert!(get_raw(5, "rust", 10, 2, false).is_none());
    assert!(get_raw(4, "Rust", 10, 2, false).is_none());
    assert!(get_raw(4, "rust", 11, 2, false).is_none());
    assert!(get_raw(4, "rust", 10, 3, false).is_none());
    assert!(get_raw(4, "rust", 10, 2, true).is_none());

    let query = Query::boost(Query::all(), 2.0);
    let preparsed_result = QuillSearchResult {
        hits: Arc::from([]),
        total_count: Some(11),
        doc_count: 11,
        diagnostics: Vec::new(),
    };
    cache.insert_preparsed(4, &query, 0, 0, true, &preparsed_result);
    assert_eq!(
        cache.get_preparsed(4, &query, 0, 0, true),
        Some(preparsed_result)
    );
    assert!(cache.get_preparsed(5, &query, 0, 0, true).is_none());
    assert!(
        cache
            .get_preparsed(4, &Query::boost(Query::all(), 2.5), 0, 0, true,)
            .is_none()
    );
}

#[cfg(all(feature = "bench-internals", feature = "conformance-internals"))]
#[test]
fn ranked_query_cache_trace_contract_covers_every_lookup_disposition() {
    let trace_buffer = Arc::new(StdMutex::new(Vec::<u8>::new()));
    let writer_buffer = Arc::clone(&trace_buffer);
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .without_time()
        .with_env_filter("off,frankensearch.quill=info")
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
        .with_writer(move || RankedCacheTraceWriter {
            buffer: Arc::clone(&writer_buffer),
        })
        .finish();

    tracing::subscriber::with_default(subscriber, || {
        run_with_cx(|cx| async move {
            let index =
                QuillIndex::in_memory(deterministic_config()).expect("ranked-cache trace index");
            index
                .index_document(&cx, &IndexableDocument::new("doc-1", "alpha beta"))
                .await
                .expect("index ranked-cache trace document");
            index
                .commit(&cx)
                .await
                .expect("commit ranked-cache trace index");
            let query = Query::term(Vec::from([QueryField::new(1, 1.0)]), "alpha".to_owned());

            let run_raw = |trace_case: &'static str, request_cx: &Cx| {
                let span = tracing::info_span!(
                    target: crate::tracing_conventions::TARGET,
                    "ranked_cache_trace_case",
                    trace_case,
                );
                let _entered = span.enter();
                index.search_paginated(request_cx, "alpha", 10, 0, true)
            };
            let run_preparsed = |trace_case: &'static str, request_cx: &Cx| {
                let span = tracing::info_span!(
                    target: crate::tracing_conventions::TARGET,
                    "ranked_cache_trace_case",
                    trace_case,
                );
                let _entered = span.enter();
                index.search_preparsed_paginated(request_cx, &query, 10, 0, true)
            };

            run_raw("raw-miss", &cx).expect("raw cache miss executes");
            run_raw("raw-hit", &cx).expect("raw cache hit returns");
            run_preparsed("preparsed-miss", &cx).expect("preparsed cache miss executes");
            run_preparsed("preparsed-hit", &cx).expect("preparsed cache hit returns");

            let controller = index.conformance_cancellation_controller();
            controller
                .arm(ConformanceCancellationStage::CommitPublication, 1)
                .expect("arm unrelated conformance stage to disable the cache");
            run_raw("raw-disabled", &cx).expect("disabled raw cache executes");
            run_preparsed("preparsed-disabled", &cx).expect("disabled preparsed cache executes");
            assert_eq!(controller.observed_checkpoints(), 0);
            assert!(!controller.fired());
            controller.disarm();

            let cancelled = cx.clone();
            cancelled.set_cancel_requested(true);
            assert!(matches!(
                run_raw("raw-not-checked", &cancelled),
                Err(QuillIndexError::Cancelled { phase }) if phase == "search"
            ));
            assert!(matches!(
                run_preparsed("preparsed-not-checked", &cancelled),
                Err(QuillIndexError::Cancelled { phase }) if phase == "search_preparsed"
            ));
        });
    });

    let logs = String::from_utf8(
        trace_buffer
            .lock()
            .expect("ranked-cache trace buffer lock is not poisoned")
            .clone(),
    )
    .expect("ranked-cache trace output is UTF-8");
    let raw_children = [
        crate::tracing_conventions::ARGUS_PARSE,
        crate::tracing_conventions::ARGUS_SCORE,
        crate::tracing_conventions::ARGUS_COLLECT,
    ];
    let preparsed_children = [
        crate::tracing_conventions::ARGUS_SCORE,
        crate::tracing_conventions::ARGUS_COLLECT,
    ];
    assert_ranked_cache_trace_case(&logs, "raw-miss", "raw", "miss", &raw_children);
    assert_ranked_cache_trace_case(&logs, "raw-hit", "raw", "hit", &[]);
    assert_ranked_cache_trace_case(
        &logs,
        "preparsed-miss",
        "preparsed",
        "miss",
        &preparsed_children,
    );
    assert_ranked_cache_trace_case(&logs, "preparsed-hit", "preparsed", "hit", &[]);
    assert_ranked_cache_trace_case(&logs, "raw-disabled", "raw", "disabled", &raw_children);
    assert_ranked_cache_trace_case(
        &logs,
        "preparsed-disabled",
        "preparsed",
        "disabled",
        &preparsed_children,
    );
    assert_ranked_cache_trace_case(&logs, "raw-not-checked", "raw", "not_checked", &[]);
    assert_ranked_cache_trace_case(
        &logs,
        "preparsed-not-checked",
        "preparsed",
        "not_checked",
        &[],
    );
    assert!(
        !logs.contains("cache_key=") && !logs.contains("cache_fingerprint="),
        "ranked-cache traces leaked a high-cardinality cache identity: {logs}",
    );
}
