#[cfg(feature = "tantivy-oracle")]
use super::PREVIEW_MAX_CHARS;
use super::{
    CASS_SCHEMA_HASH, CONTENT_PREFIX_MAX_BYTES, CassConversationKey, CassDerivedColumns,
    CassDocument, CassFieldValues, SchemaDocument, cass_build_preview, cass_document_identity,
    cass_prefix_source, cass_schema_hash_matches, field,
};

#[path = "tests_search.rs"]
mod search;

fn sample_document(
    source_id: &str,
    msg_idx: u64,
    agent: &str,
    title: &str,
    content: &str,
) -> CassDocument {
    CassDocument {
        agent: agent.to_owned(),
        workspace: Some("frankensearch".to_owned()),
        workspace_original: Some("FrankenSearch".to_owned()),
        source_path: format!("/transcripts/{source_id}.jsonl"),
        msg_idx,
        created_at: Some(1_700_000_000 + i64::try_from(msg_idx).expect("fixture ordinal")),
        title: Some(title.to_owned()),
        content: content.to_owned(),
        source_id: source_id.to_owned(),
        origin_kind: "local".to_owned(),
        origin_host: None,
        conversation_id: Some(42),
    }
}

/// A conversation at the incumbent's per-conversation indexing cap must be
/// admitted, not refused.
///
/// cass truncates a conversation's indexed body at 8 MiB and indexes the
/// truncated text. Quill's admission refuses rather than truncates, so if
/// its budget were tighter than that cap, migrating a real corpus would
/// fail on the first large conversation — and fail the whole batch, not
/// just that document. This pins that the incumbent's ceiling is inside
/// Quill's admission.
#[test]
fn a_conversation_at_the_incumbent_content_cap_is_admitted() {
    const CAP_BYTES: usize = 8 * 1024 * 1024;

    // Realistic token mix rather than one repeated word: a single repeated
    // term would collapse to one dictionary entry and understate the term
    // bucket pressure a real 8 MiB conversation applies.
    let mut content = String::with_capacity(CAP_BYTES + 16);
    let mut ordinal = 0_u32;
    while content.len() < CAP_BYTES {
        use std::fmt::Write as _;
        let _ = write!(content, "token{:07} ", ordinal % 100_000);
        ordinal += 1;
    }

    asupersync::test_utils::run_test_with_cx(move |cx| async move {
        let directory = tempfile::tempdir().expect("cass index directory");
        let index = crate::index::QuillIndex::create_with_schema(
            &cx,
            directory.path(),
            crate::schema::CASS_SEMANTIC_SCHEMA,
            crate::QuillConfig::default(),
        )
        .await
        .expect("create a CASS-schema index");

        let document = sample_document("bulk", 0, "claude", "a very long session", &content)
            .to_schema_document();
        index
            .index_schema_documents(&cx, std::slice::from_ref(&document))
            .await
            .expect("a conversation at the incumbent cap must be admitted");
        index.commit(&cx).await.expect("publish the large document");

        let reader = crate::index::QuillSearchIndex::open_with_schema(
            &cx,
            directory.path(),
            crate::schema::CASS_SEMANTIC_SCHEMA,
            crate::QuillConfig::default(),
        )
        .await
        .expect("open the reader");
        assert_eq!(reader.doc_count().expect("doc count"), 1);

        let parser = crate::query::CassQueryParser::new(crate::schema::CASS_SEMANTIC_SCHEMA)
            .expect("build the CASS parser");
        let parsed = parser.parse("token0000042", &crate::query::CassQueryFilters::default());
        let result = reader
            .search_preparsed_paginated(&cx, &parsed.query, 10, 0, true)
            .expect("search the large document");
        assert_eq!(
            result.total_count,
            Some(1),
            "a term from deep inside the capped body must still be findable"
        );
    });
}

/// Content for a deep-window fixture: at least a full 4 KiB window of
/// filler whose tokens neither equal nor extend the probe term, then the
/// probe's only occurrence. The returned string's filler bytes exceed
/// [`CONTENT_PREFIX_MAX_BYTES`] before the probe token starts, so the
/// probe's edge grams cannot be in `content_prefix`.
fn deep_window_content(probe: &str) -> String {
    let mut content = String::with_capacity(6 * 1024);
    while content.len() < CONTENT_PREFIX_MAX_BYTES + 1024 {
        content.push_str("benign filler text; ");
    }
    content.push_str(probe);
    content
}

/// A bare or `*`-suffixed term must find a document whose only matching
/// token sits past the 4 KiB `content_prefix` window.
///
/// `content` holds every exact token of the whole body, but the
/// edge-ngram `content_prefix` column is built over only the first
/// [`CONTENT_PREFIX_MAX_BYTES`] bytes, so the incumbent single term
/// query found `dockerfile` only inside the window: a `dockerfile` at
/// offset 5 KiB had no exact `docker` token and no in-window prefix
/// gram, and `docker` silently missed it. The two-arm lowering (fork
/// cass-prefix-expansion) adds the leaf-local `docker*` glob over the
/// unwindowed `content` field, so the deep-window document must now be
/// a hit for the bare term, the explicit prefix, and the quoted single
/// term alike — and a term absent from the document must stay absent.
#[test]
fn bare_term_finds_a_deep_window_prefix_match() {
    let content = deep_window_content("dockerfile");

    asupersync::test_utils::run_test_with_cx(move |cx| async move {
        let directory = tempfile::tempdir().expect("cass index directory");
        let index = crate::index::QuillIndex::create_with_schema(
            &cx,
            directory.path(),
            crate::schema::CASS_SEMANTIC_SCHEMA,
            crate::QuillConfig::default(),
        )
        .await
        .expect("create a CASS-schema index");

        let document =
            sample_document("deep", 0, "claude", "unrelated title", &content).to_schema_document();
        index
            .index_schema_documents(&cx, std::slice::from_ref(&document))
            .await
            .expect("ingest the deep-window document");
        index.commit(&cx).await.expect("publish the document");

        let reader = crate::index::QuillSearchIndex::open_with_schema(
            &cx,
            directory.path(),
            crate::schema::CASS_SEMANTIC_SCHEMA,
            crate::QuillConfig::default(),
        )
        .await
        .expect("open the reader");

        let parser = crate::query::CassQueryParser::new(crate::schema::CASS_SEMANTIC_SCHEMA)
            .expect("build the CASS parser");
        let filters = crate::query::CassQueryFilters::default();
        for raw in ["docker", "docker*", "\"docker\""] {
            let parsed = parser.parse(raw, &filters);
            let result = reader
                .search_preparsed_paginated(&cx, &parsed.query, 10, 0, true)
                .unwrap_or_else(|error| panic!("search {raw} must lower and run: {error}"));
            assert_eq!(
                result.total_count,
                Some(1),
                "the deep-window dockerfile must be found by {raw:?}"
            );
        }
        let parsed = parser.parse("kubernetes", &filters);
        let result = reader
            .search_preparsed_paginated(&cx, &parsed.query, 10, 0, true)
            .expect("search an absent term");
        assert_eq!(
            result.total_count,
            Some(0),
            "an absent term must not match the deep-window document"
        );
    });
}

/// The leaf-local glob expansion must cover every sealed leaf.
///
/// Two documents are sealed in two separate commits with tier merges
/// disabled, so the snapshot holds two leaves: the deep-window
/// `dockerfile` document in the first, `dockworker` in the second. One
/// bare-term query must find both documents through each leaf's own
/// `content` dictionary, and the discriminating queries must keep their
/// counts — `dockerfile` (the deep exact token), `dock*` (both), and
/// `dockw*` (the second document only).
#[test]
fn bare_term_expands_the_content_glob_on_every_leaf() {
    let deep = deep_window_content("dockerfile");
    let config = crate::QuillConfig {
        tier_fanout: usize::MAX,
        ..crate::QuillConfig::default()
    };

    asupersync::test_utils::run_test_with_cx(move |cx| async move {
        let directory = tempfile::tempdir().expect("cass index directory");
        let index = crate::index::QuillIndex::create_with_schema(
            &cx,
            directory.path(),
            crate::schema::CASS_SEMANTIC_SCHEMA,
            config.clone(),
        )
        .await
        .expect("create a CASS-schema index");

        let early =
            sample_document("early", 0, "claude", "unrelated title", &deep).to_schema_document();
        index
            .index_schema_documents(&cx, std::slice::from_ref(&early))
            .await
            .expect("ingest the deep-window document");
        index.commit(&cx).await.expect("publish the first leaf");

        let late = sample_document(
            "late",
            1,
            "codex",
            "another title",
            "the dockworker manifest",
        )
        .to_schema_document();
        index
            .index_schema_documents(&cx, std::slice::from_ref(&late))
            .await
            .expect("ingest the second leaf");
        index.commit(&cx).await.expect("publish the second leaf");

        let reader = crate::index::QuillSearchIndex::open_with_schema(
            &cx,
            directory.path(),
            crate::schema::CASS_SEMANTIC_SCHEMA,
            config,
        )
        .await
        .expect("open the reader");

        let parser = crate::query::CassQueryParser::new(crate::schema::CASS_SEMANTIC_SCHEMA)
            .expect("build the CASS parser");
        let filters = crate::query::CassQueryFilters::default();
        for (raw, expected) in [
            ("dock", 2_u64),
            ("dockerfile", 1),
            ("dockw*", 1),
            ("kubernetes", 0),
        ] {
            let parsed = parser.parse(raw, &filters);
            let result = reader
                .search_preparsed_paginated(&cx, &parsed.query, 10, 0, true)
                .unwrap_or_else(|error| panic!("search {raw} must lower and run: {error}"));
            assert_eq!(
                result.total_count,
                Some(expected),
                "query {raw:?} must see every leaf's content dictionary"
            );
        }
    });
}

/// The two-arm glob arm skips its own literal term, not its matches.
///
/// Two deep-tail documents past the 4 KiB window: one whose only
/// matching token is the query literal itself (`error`) and one whose
/// only match is a proper extension (`errors`). The bare-term query
/// must keep finding both — the literal document through the exact
/// Term arm over the unwindowed `content` field, the variant document
/// through the glob arm. Because the exact arm accompanies it, the
/// glob arm drops its own literal from the expansion
/// (`two_arm_variants_glob` in `index.rs`): the flat boost lands only
/// on the variant document, and the literal document's whole two-arm
/// score equals the exact arm alone — the same BM25 a content-only
/// Term query reports — where the arm used to add the flat boost to
/// every literal document as well.
#[test]
fn two_arm_glob_arm_skips_its_literal_not_its_matches() {
    let literal_content = deep_window_content("error");
    let variant_content = deep_window_content("errors");

    asupersync::test_utils::run_test_with_cx(move |cx| async move {
        let directory = tempfile::tempdir().expect("cass index directory");
        let index = crate::index::QuillIndex::create_with_schema(
            &cx,
            directory.path(),
            crate::schema::CASS_SEMANTIC_SCHEMA,
            crate::QuillConfig::default(),
        )
        .await
        .expect("create a CASS-schema index");

        let literal = sample_document(
            "literalhit",
            0,
            "claude",
            "unrelated title",
            &literal_content,
        )
        .to_schema_document();
        let variant = sample_document("varianthit", 1, "codex", "another title", &variant_content)
            .to_schema_document();
        index
            .index_schema_documents(&cx, &[literal, variant])
            .await
            .expect("ingest both deep-tail documents");
        index.commit(&cx).await.expect("publish the documents");

        let reader = crate::index::QuillSearchIndex::open_with_schema(
            &cx,
            directory.path(),
            crate::schema::CASS_SEMANTIC_SCHEMA,
            crate::QuillConfig::default(),
        )
        .await
        .expect("open the reader");

        let parser = crate::query::CassQueryParser::new(crate::schema::CASS_SEMANTIC_SCHEMA)
            .expect("build the CASS parser");
        let filters = crate::query::CassQueryFilters::default();
        let parsed = parser.parse("error", &filters);
        let page = reader
            .search_preparsed_paginated(&cx, &parsed.query, 10, 0, true)
            .expect("the bare term must lower and run");
        assert_eq!(
            page.total_count,
            Some(2),
            "the literal and the variant deep-tail documents must both stay hits"
        );

        let literal_score = page
            .hits
            .iter()
            .find(|hit| hit.document_id.starts_with("literalhit#"))
            .expect("the literal-only document must be a hit")
            .score;
        let variant_score = page
            .hits
            .iter()
            .find(|hit| hit.document_id.starts_with("varianthit#"))
            .expect("the variant-only document must be a hit")
            .score;
        assert_eq!(
            variant_score.to_bits(),
            1.0_f32.to_bits(),
            "the variant document matches through the glob arm alone and must carry exactly its flat boost"
        );

        // The exact arm alone — a Term over just the unwindowed `content`
        // field — reports the literal document's whole two-arm score:
        // its title and windowed prefix columns carry no `error` token
        // or gram, and the glob arm's post-drop expansion (`errors`)
        // does not contain it.
        let content_term = crate::query::Query::term(
            Vec::from([crate::query::QueryField::new(field::CONTENT, 1.0)]),
            "error".to_owned(),
        );
        let oracle = reader
            .search_preparsed_paginated(&cx, &content_term, 10, 0, true)
            .expect("the content-term oracle must run");
        let oracle_score = oracle
            .hits
            .iter()
            .find(|hit| hit.document_id.starts_with("literalhit#"))
            .expect("the content term must match the literal document")
            .score;
        assert_eq!(
            literal_score.to_bits(),
            oracle_score.to_bits(),
            "the literal-only document's two-arm score must equal the exact arm alone"
        );
    });
}

/// An over-broad prefix whose expansion crosses the per-leaf cap is a
/// typed refusal that names the glob, never a silently truncated match
/// set.
///
/// One document carries 32 distinct `docker`-prefixed tokens, and the
/// reader's `glob_expansion_limit` is 8, so the glob arm of the two-arm
/// `docker` lowering must refuse with
/// [`crate::index::QuillIndexError::Dictionary`] carrying
/// [`crate::grimoire::TermDictionaryError::GlobExpansionLimitExceeded`]
/// whose `pattern` is `docker*` — the pattern the engine exploded, not
/// just the field and the count.
#[test]
fn an_over_broad_prefix_refuses_typed_and_names_the_glob() {
    let content = (0..32)
        .map(|index| format!("docker{index:02}"))
        .collect::<Vec<_>>()
        .join(" ");
    let config = crate::QuillConfig {
        glob_expansion_limit: 8,
        ..crate::QuillConfig::default()
    };

    asupersync::test_utils::run_test_with_cx(move |cx| async move {
        let directory = tempfile::tempdir().expect("cass index directory");
        let index = crate::index::QuillIndex::create_with_schema(
            &cx,
            directory.path(),
            crate::schema::CASS_SEMANTIC_SCHEMA,
            crate::QuillConfig::default(),
        )
        .await
        .expect("create a CASS-schema index");

        let document =
            sample_document("broad", 0, "claude", "many dockers", &content).to_schema_document();
        index
            .index_schema_documents(&cx, std::slice::from_ref(&document))
            .await
            .expect("ingest the broad-prefix document");
        index.commit(&cx).await.expect("publish the document");

        let reader = crate::index::QuillSearchIndex::open_with_schema(
            &cx,
            directory.path(),
            crate::schema::CASS_SEMANTIC_SCHEMA,
            config,
        )
        .await
        .expect("open the reader");

        let parser = crate::query::CassQueryParser::new(crate::schema::CASS_SEMANTIC_SCHEMA)
            .expect("build the CASS parser");
        let filters = crate::query::CassQueryFilters::default();
        let parsed = parser.parse("docker", &filters);
        let error = reader
            .search_preparsed_paginated(&cx, &parsed.query, 10, 0, true)
            .expect_err("the over-broad prefix must refuse, not under-match");
        match error {
            crate::index::QuillIndexError::Dictionary(
                crate::grimoire::TermDictionaryError::GlobExpansionLimitExceeded {
                    pattern,
                    field_ord: _,
                    limit,
                    actual: _,
                },
            ) => {
                assert_eq!(limit, 8, "the refusal must carry the configured cap");
                assert_eq!(
                    pattern, "docker*",
                    "the refusal must name the glob it exploded"
                );
            }
            other => panic!("the refusal must be the typed glob limit, got {other}"),
        }
    });
}

/// A two-arm query that cannot afford its own work refuses with the
/// typed fuel error, and a budget that fits completes the same query —
/// so the refusal is budget-driven, not data-driven.
#[test]
fn an_unaffordable_two_arm_query_refuses_with_the_typed_fuel_error() {
    let content = (0..32)
        .map(|index| format!("docker{index:02}"))
        .collect::<Vec<_>>()
        .join(" ");

    asupersync::test_utils::run_test_with_cx(move |cx| async move {
        let directory = tempfile::tempdir().expect("cass index directory");
        let index = crate::index::QuillIndex::create_with_schema(
            &cx,
            directory.path(),
            crate::schema::CASS_SEMANTIC_SCHEMA,
            crate::QuillConfig::default(),
        )
        .await
        .expect("create a CASS-schema index");

        let document =
            sample_document("fuel", 0, "claude", "many dockers", &content).to_schema_document();
        index
            .index_schema_documents(&cx, std::slice::from_ref(&document))
            .await
            .expect("ingest the fuel document");
        index.commit(&cx).await.expect("publish the document");

        let parser = crate::query::CassQueryParser::new(crate::schema::CASS_SEMANTIC_SCHEMA)
            .expect("build the CASS parser");
        let filters = crate::query::CassQueryFilters::default();
        let parsed = parser.parse("docker", &filters);

        let starved = crate::QuillConfig {
            query_fuel_budget: 1,
            ..crate::QuillConfig::default()
        };
        let reader = crate::index::QuillSearchIndex::open_with_schema(
            &cx,
            directory.path(),
            crate::schema::CASS_SEMANTIC_SCHEMA,
            starved,
        )
        .await
        .expect("open the starved reader");
        match reader.search_preparsed_paginated(&cx, &parsed.query, 10, 0, true) {
            Err(crate::index::QuillIndexError::QueryFuelExhausted { budget, .. }) => {
                assert_eq!(budget, 1, "the refusal must carry the configured budget");
            }
            Err(other) => {
                panic!("the starved query must refuse as fuel exhaustion, got {other}")
            }
            Ok(result) => panic!(
                "a fuel budget of 1 cannot afford the two-arm query, yet it returned {} hits",
                result.total_count.unwrap_or_default()
            ),
        }

        let fed = crate::index::QuillSearchIndex::open_with_schema(
            &cx,
            directory.path(),
            crate::schema::CASS_SEMANTIC_SCHEMA,
            crate::QuillConfig::default(),
        )
        .await
        .expect("open the fed reader");
        let result = fed
            .search_preparsed_paginated(&cx, &parsed.query, 10, 0, true)
            .expect("the default budget must afford the same query");
        assert_eq!(
            result.total_count,
            Some(1),
            "the default budget must afford the same query and find the document"
        );
    });
}

/// The CJK bigram lowering and the suffix/substring regex globs keep
/// their incumbent behavior under the two-arm exact/prefix lowering.
///
/// A CJK query still decomposes to exact bigram terms over the four
/// searchable fields (no glob arm), and a `*`-suffix glob still matches
/// through the regex path over `content` and `title` — including for a
/// token past the prefix window.
#[test]
fn cjk_bigrams_and_regex_globs_keep_their_lowerings() {
    let deep = deep_window_content("containerfile");

    asupersync::test_utils::run_test_with_cx(move |cx| async move {
        let directory = tempfile::tempdir().expect("cass index directory");
        let index = crate::index::QuillIndex::create_with_schema(
            &cx,
            directory.path(),
            crate::schema::CASS_SEMANTIC_SCHEMA,
            crate::QuillConfig::default(),
        )
        .await
        .expect("create a CASS-schema index");

        let cjk = sample_document("cjk", 0, "claude", "搜索引擎", "这是一个搜索引擎测试")
            .to_schema_document();
        let deep =
            sample_document("deep", 1, "codex", "unrelated title", &deep).to_schema_document();
        index
            .index_schema_documents(&cx, &[cjk, deep])
            .await
            .expect("ingest the regression documents");
        index.commit(&cx).await.expect("publish the documents");

        let reader = crate::index::QuillSearchIndex::open_with_schema(
            &cx,
            directory.path(),
            crate::schema::CASS_SEMANTIC_SCHEMA,
            crate::QuillConfig::default(),
        )
        .await
        .expect("open the reader");

        let parser = crate::query::CassQueryParser::new(crate::schema::CASS_SEMANTIC_SCHEMA)
            .expect("build the CASS parser");
        let filters = crate::query::CassQueryFilters::default();
        for (raw, expected) in [
            ("搜索引擎", 1_u64),
            ("*file", 1),
            ("containerfile", 1),
            ("*ocker*", 0),
        ] {
            let parsed = parser.parse(raw, &filters);
            let result = reader
                .search_preparsed_paginated(&cx, &parsed.query, 10, 0, true)
                .unwrap_or_else(|error| panic!("search {raw} must lower and run: {error}"));
            assert_eq!(
                result.total_count,
                Some(expected),
                "regression query {raw:?} must keep its incumbent result"
            );
        }
    });
}

/// Ingest and query throughput for the CASS schema path.
///
/// Opt-in (`--ignored`) and release-only to be meaningful. This is a
/// measurement probe, not a gate: it asserts nothing about timing, it only
/// reports, so it can never fail the suite for being run on a loaded box.
#[test]
#[ignore = "measurement probe; run explicitly under --release --ignored"]
fn cass_schema_ingest_and_query_throughput() {
    use std::time::Instant;

    const DOCUMENTS: usize = 20_000;
    const BATCH: usize = 500;

    // Deterministic pseudo-random text: a fixed LCG so the corpus is the
    // same on every run and two measurements are comparable.
    let vocabulary: Vec<String> = (0..4096).map(|index| format!("term{index:04}")).collect();
    let mut seed = 0x2545_F491_4F6C_DD1D_u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let agents = ["claude", "codex", "gemini", "local"];
    let corpus: Vec<CassDocument> = (0..DOCUMENTS)
        .map(|index| {
            let content = (0..120)
                .map(|_| {
                    vocabulary[usize::try_from(next() % 4096).expect("vocabulary index")].as_str()
                })
                .collect::<Vec<_>>()
                .join(" ");
            sample_document(
                &format!("session{:05}", index / 50),
                u64::try_from(index).expect("ordinal"),
                agents[index % agents.len()],
                "measurement session",
                &content,
            )
        })
        .collect();
    let content_bytes: usize = corpus.iter().map(|document| document.content.len()).sum();

    asupersync::test_utils::run_test_with_cx(move |cx| async move {
        let directory = tempfile::tempdir().expect("cass index directory");
        let index = crate::index::QuillIndex::create_with_schema(
            &cx,
            directory.path(),
            crate::schema::CASS_SEMANTIC_SCHEMA,
            crate::QuillConfig::default(),
        )
        .await
        .expect("create a CASS-schema index");

        let projection_start = Instant::now();
        let projected: Vec<SchemaDocument> = corpus
            .iter()
            .map(CassDocument::to_schema_document)
            .collect();
        let projection = projection_start.elapsed();

        let ingest_start = Instant::now();
        for batch in projected.chunks(BATCH) {
            index
                .index_schema_documents(&cx, batch)
                .await
                .expect("ingest a CASS batch");
        }
        let accumulate = ingest_start.elapsed();
        index.commit(&cx).await.expect("publish the CASS corpus");
        let ingest_total = ingest_start.elapsed();

        let reader = crate::index::QuillSearchIndex::open_with_schema(
            &cx,
            directory.path(),
            crate::schema::CASS_SEMANTIC_SCHEMA,
            crate::QuillConfig::default(),
        )
        .await
        .expect("open a CASS-schema reader");
        let parser = crate::query::CassQueryParser::new(crate::schema::CASS_SEMANTIC_SCHEMA)
            .expect("build the CASS query parser");
        let filters = crate::query::CassQueryFilters::default();

        let mut latencies = Vec::new();
        let mut hits = 0_u64;
        for round in 0..500_usize {
            let raw = format!("term{:04} term{:04}", round % 4096, (round * 7) % 4096);
            let parsed = parser.parse(&raw, &filters);
            let start = Instant::now();
            let result = reader
                .search_preparsed_paginated(&cx, &parsed.query, 10, 0, false)
                .expect("query the CASS corpus");
            latencies.push(start.elapsed());
            hits += result.hits.len() as u64;
        }
        latencies.sort_unstable();

        let mib = content_bytes as f64 / (1024.0 * 1024.0);
        println!("--- CASS schema path, {DOCUMENTS} docs, {mib:.1} MiB content ---");
        println!(
            "projection  {:>8.2?}  ({:.0} docs/s)",
            projection,
            DOCUMENTS as f64 / projection.as_secs_f64()
        );
        println!(
            "accumulate  {:>8.2?}  ({:.0} docs/s, {:.1} MiB/s)",
            accumulate,
            DOCUMENTS as f64 / accumulate.as_secs_f64(),
            mib / accumulate.as_secs_f64()
        );
        println!(
            "ingest+commit {:>6.2?}  ({:.0} docs/s, {:.1} MiB/s)",
            ingest_total,
            DOCUMENTS as f64 / ingest_total.as_secs_f64(),
            mib / ingest_total.as_secs_f64()
        );
        println!(
            "query p50 {:>8.2?}  p99 {:>8.2?}  max {:>8.2?}  ({hits} hits over 500 queries)",
            latencies[latencies.len() / 2],
            latencies[latencies.len() * 99 / 100],
            latencies[latencies.len() - 1],
        );
    });
}

#[test]
fn preview_is_character_bounded_and_only_ellipsizes_when_truncating() {
    assert_eq!(cass_build_preview("abc", 400), "abc");
    assert_eq!(cass_build_preview("abcdef", 3), "abc…");
    // Exactly at the bound must NOT gain an ellipsis: a preview that
    // claims truncation it did not perform is a lie to the reader.
    assert_eq!(cass_build_preview("abc", 3), "abc");
    // Multi-byte scalars count as one character, not one byte.
    let wide = "日本語のテキスト";
    assert_eq!(cass_build_preview(wide, 3), "日本語…");
    assert_eq!(cass_build_preview("", 5), "");
}

#[test]
fn prefix_source_never_splits_a_scalar_and_is_bounded() {
    assert_eq!(cass_prefix_source("abcdef", 100), "abcdef");
    assert_eq!(cass_prefix_source("abcdef", 3), "abc");
    // "日" is three bytes, so a four-byte bound must fall back to one
    // scalar rather than slicing the second one in half.
    let wide = "日本";
    let cut = cass_prefix_source(wide, 4);
    assert_eq!(cut, "日");
    assert!(wide.is_char_boundary(cut.len()));
    // Every bound from 0..=len must land on a boundary.
    for bound in 0..=wide.len() {
        let slice = cass_prefix_source(wide, bound);
        assert!(wide.is_char_boundary(slice.len()), "bound {bound}");
        assert!(slice.len() <= bound);
    }
}

#[test]
fn derived_columns_come_from_the_document_and_title_absence_is_empty() {
    let document = CassDocument {
        agent: "claude".to_owned(),
        workspace: Some("/repo".to_owned()),
        workspace_original: Some("/repo".to_owned()),
        source_path: "/tmp/session.jsonl".to_owned(),
        msg_idx: 7,
        created_at: Some(1_700_000_000),
        title: Some("hello world".to_owned()),
        content: "searchable content here".to_owned(),
        source_id: "local".to_owned(),
        origin_kind: "local".to_owned(),
        origin_host: None,
        conversation_id: Some(3),
    };
    let derived = CassDerivedColumns::derive(document.as_ref());
    assert!(derived.title_prefix.contains("he"));
    assert!(derived.content_prefix.contains("se"));
    assert_eq!(derived.preview, "searchable content here");

    let untitled = CassDocument {
        title: None,
        ..document
    };
    let derived = CassDerivedColumns::derive(untitled.as_ref());
    assert!(
        derived.title_prefix.is_empty(),
        "an absent title must contribute no prefix terms, not the string \"None\""
    );
}

#[test]
fn schema_sentinel_rejects_the_tantivy_generation() {
    assert!(cass_schema_hash_matches(CASS_SCHEMA_HASH));
    assert!(
        !cass_schema_hash_matches(
            "tantivy-schema-v8-hyphen-cjk-bigrams-bounded-content-prefix-preview-stored-content-external"
        ),
        "the Tantivy-era sentinel must force a rebuild, never be accepted as current"
    );
    assert!(!cass_schema_hash_matches(""));
}

/// The ported derivation must be byte-identical to the incumbent.
///
/// `preview` and `content_prefix` are persisted columns: any divergence
/// silently changes what a rebuilt index matches and what a user is shown,
/// and would do so without failing anything else. Parity is checked
/// against the shipped Tantivy implementation rather than against
/// hand-written expectations, so the assertion cannot drift with it.
///
/// Only `cass_build_preview` is reachable — the incumbent keeps its prefix
/// walk private — so the prefix boundary rule is pinned separately by
/// `prefix_source_never_splits_a_scalar_and_is_bounded`, which checks every
/// bound in `0..=len` rather than sampling.
#[cfg(feature = "tantivy-oracle")]
#[test]
fn derivation_is_byte_identical_to_the_tantivy_incumbent() {
    use frankensearch_lexical::cass_compat::cass_build_preview as oracle_preview;

    let samples = [
        "",
        "a",
        "short content",
        "日本語のテキストとascii mixed",
        "hyphen-joined words and CamelCase Tokens",
        &"x".repeat(10_000),
        &"日".repeat(3_000),
    ];
    for sample in samples {
        assert_eq!(
            cass_build_preview(sample, PREVIEW_MAX_CHARS),
            oracle_preview(sample, PREVIEW_MAX_CHARS),
            "preview diverged for {} bytes",
            sample.len()
        );
    }
}

/// Every emitted ordinal must exist in the compiled schema with a matching
/// storage shape, and absent optionals must emit nothing at all.
///
/// The ordinals are hand-written constants, so nothing else stops
/// `field::TITLE` from drifting onto the `content` column after a schema
/// edit — this checks each one against the descriptor by name rather than
/// trusting the constants.
#[test]
fn field_values_match_the_compiled_schema_and_omit_absent_columns() {
    use crate::schema::{CASS_SEMANTIC_SCHEMA, FieldKind};

    let named = |ordinal: u16| {
        CASS_SEMANTIC_SCHEMA
            .fields
            .iter()
            .find(|field| field.id == ordinal)
            .unwrap_or_else(|| panic!("ordinal {ordinal} is absent from the CASS schema"))
    };
    assert_eq!(named(field::AGENT).name, "agent");
    assert_eq!(named(field::CONTENT).name, "content");
    assert_eq!(named(field::TITLE).name, "title");
    assert_eq!(named(field::TITLE_PREFIX).name, "title_prefix");
    assert_eq!(named(field::CONTENT_PREFIX).name, "content_prefix");
    assert_eq!(named(field::PREVIEW).name, "preview");
    assert_eq!(named(field::MSG_IDX).name, "msg_idx");
    assert_eq!(named(field::CREATED_AT).name, "created_at");
    assert_eq!(named(field::CONVERSATION_ID).name, "conversation_id");
    assert_eq!(named(field::SOURCE_PATH).name, "source_path");
    assert_eq!(named(field::WORKSPACE).name, "workspace");
    assert_eq!(named(field::WORKSPACE_ORIGINAL).name, "workspace_original");
    assert_eq!(named(field::SOURCE_ID).name, "source_id");
    assert_eq!(named(field::ORIGIN_KIND).name, "origin_kind");
    assert_eq!(named(field::ORIGIN_HOST).name, "origin_host");

    let full = CassDocument {
        agent: "claude".to_owned(),
        workspace: Some("/repo".to_owned()),
        workspace_original: Some("/Repo".to_owned()),
        source_path: "/tmp/s.jsonl".to_owned(),
        msg_idx: 7,
        created_at: Some(1_700_000_000),
        title: Some("a title".to_owned()),
        content: "body text".to_owned(),
        source_id: "local".to_owned(),
        origin_kind: "local".to_owned(),
        origin_host: Some("host".to_owned()),
        conversation_id: Some(-3),
    };
    let derived = CassDerivedColumns::derive(full.as_ref());
    let values = CassFieldValues::build(full.as_ref(), &derived);

    // Every emitted column must be a real, correctly-shaped schema field.
    for value in &values.indexed {
        let field = named(value.field_ord);
        assert!(
            matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. }),
            "{} is not an indexable column",
            field.name
        );
    }
    for value in &values.numeric {
        let field = named(value.field_ord);
        assert!(
            matches!(field.kind, FieldKind::I64 { .. } | FieldKind::U64 { .. }),
            "{} is not numeric",
            field.name
        );
        assert!(
            field.kind.has_numeric_column(),
            "{} owns no numeric column, so it must be written as stored bytes",
            field.name
        );
    }
    for value in &values.stored {
        assert!(named(value.field_ord).stored, "column is not stored");
    }
    // A signed conversation id must survive as readable text.
    // Canonical little-endian, which is what Scribe validates a stored
    // numeric column against; a negative value must round-trip.
    assert_eq!(derived.conversation_id_bytes, Some((-3_i64).to_le_bytes()));
    assert_eq!(
        derived.conversation_id_bytes.map(i64::from_le_bytes),
        Some(-3)
    );

    // Absent optionals emit nothing: an empty keyword is a matchable term,
    // so writing one would make "absent" indistinguishable from "empty".
    let sparse = CassDocument {
        workspace: None,
        workspace_original: None,
        title: None,
        created_at: None,
        origin_host: None,
        conversation_id: None,
        ..full
    };
    let sparse_derived = CassDerivedColumns::derive(sparse.as_ref());
    let sparse_values = CassFieldValues::build(sparse.as_ref(), &sparse_derived);
    for absent in [
        field::WORKSPACE,
        field::TITLE,
        field::TITLE_PREFIX,
        field::ORIGIN_HOST,
    ] {
        assert!(
            !sparse_values
                .indexed
                .iter()
                .any(|value| value.field_ord == absent),
            "absent optional {absent} must not be written"
        );
    }
    assert!(
        !sparse_values
            .numeric
            .iter()
            .any(|value| value.field_ord == field::CREATED_AT)
    );
    for absent in [field::WORKSPACE_ORIGINAL, field::CONVERSATION_ID] {
        assert!(
            !sparse_values
                .stored
                .iter()
                .any(|value| value.field_ord == absent)
        );
    }
    // Required columns are still present in the sparse case.
    assert!(
        sparse_values
            .indexed
            .iter()
            .any(|value| value.field_ord == field::CONTENT)
    );
    assert!(
        sparse_values
            .numeric
            .iter()
            .any(|value| value.field_ord == field::MSG_IDX)
    );
}

/// Identity must discriminate source, conversation, AND ordinal.
///
/// The conversation axis is the one that matters most here: every locally
/// discovered conversation shares one `source_id`, so an identity built
/// only from source and ordinal collides on message 0 of every local
/// conversation. Tantivy tolerated that; Quill refuses it as a duplicate
/// primary key, which is exactly the bug this shape prevents.
#[test]
fn document_identity_discriminates_source_conversation_and_ordinal() {
    use CassConversationKey as Key;
    assert_eq!(
        cass_document_identity("local", Key::new("/a.jsonl", Some(7)), 0),
        "local#/a.jsonl#7#0"
    );

    assert_ne!(
        cass_document_identity("local", Key::new("/a.jsonl", Some(1)), 1),
        cass_document_identity("remote", Key::new("/a.jsonl", Some(1)), 1),
        "identity must not collide across sources"
    );
    assert_ne!(
        cass_document_identity("local", Key::new("/a.jsonl", Some(1)), 0),
        cass_document_identity("local", Key::new("/b.jsonl", Some(1)), 0),
        "message 0 of two conversations in ONE source, sharing a reused id, must not collide"
    );
    assert_ne!(
        cass_document_identity("local", Key::new("/a.jsonl", Some(1)), 0),
        cass_document_identity("local", Key::new("/a.jsonl", Some(1)), 1),
        "identity must not collide across ordinals"
    );
    // Without an assigned conversation id the transcript path carries the
    // same discrimination.
    assert_ne!(
        cass_document_identity("local", Key::new("/a.jsonl", None), 0),
        cass_document_identity("local", Key::new("/b.jsonl", None), 0),
        "message 0 of two unassigned conversations must not collide"
    );
}

/// A batch of conversations from one source must be ingestable.
///
/// This is the regression test for the real defect: under the old
/// source-scoped identity every conversation's message 0 was `local#0`, so
/// a multi-conversation local batch was refused outright.
#[test]
fn a_multi_conversation_local_batch_has_unique_identities() {
    let documents: Vec<CassDocument> = (0..4)
        .flat_map(|conversation| {
            (0..3).map(move |msg_idx| {
                let mut document = sample_document("local", msg_idx, "claude", "t", "body");
                document.conversation_id = Some(conversation);
                document
            })
        })
        .collect();
    let identities: std::collections::BTreeSet<String> = documents
        .iter()
        .map(|document| document.to_schema_document().id)
        .collect();
    assert_eq!(
        identities.len(),
        documents.len(),
        "every message in a multi-conversation local batch needs its own identity"
    );
}
