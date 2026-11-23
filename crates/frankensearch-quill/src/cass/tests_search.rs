use crate::cass::CassDocument;

use super::sample_document;

/// A CASS-schema index must ingest and answer queries through the same
/// writer the shipping five-field shape uses.
///
/// This is the end-to-end claim the whole schema-document path exists to
/// support: an index whose schema is wider than `IndexableDocument` can
/// describe still routes through Quill's admission, accumulation, commit,
/// and search. It asserts a field-scoped match *and* a field-scoped
/// non-match, so a lowering that ignored field scope entirely would fail
/// here rather than pass by accident.
#[test]
fn cass_schema_index_ingests_and_searches_end_to_end() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("cass index directory");
        let index = crate::index::QuillIndex::create_with_schema(
            &cx,
            directory.path(),
            crate::schema::CASS_SEMANTIC_SCHEMA,
            crate::QuillConfig::default(),
        )
        .await
        .expect("create a CASS-schema index");

        let documents = [
            sample_document(
                "alpha",
                0,
                "claude",
                "Borrow checker session",
                "the borrow checker rejected this lifetime in rustc",
            ),
            sample_document(
                "beta",
                1,
                "codex",
                "Tokenizer throughput",
                "tokenizer throughput regressed on long input in rustc",
            ),
        ]
        .iter()
        .map(CassDocument::to_schema_document)
        .collect::<Vec<_>>();

        index
            .index_schema_documents(&cx, &documents)
            .await
            .expect("ingest CASS documents");
        index.commit(&cx).await.expect("publish CASS documents");

        // Read through the shipping reader rather than the writer handle:
        // that is the path a consumer actually uses, and it additionally
        // proves the CASS schema survives a reopen from disk.
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

        // `lifetime` appears in exactly one document, and in neither
        // title, so a match here is attributable to the content column.
        let parsed = parser.parse("lifetime", &filters);
        let hit = reader
            .search_preparsed_paginated(&cx, &parsed.query, 10, 0, true)
            .expect("search the CASS index");
        assert_eq!(
            hit.total_count,
            Some(1),
            "exactly one document mentions a lifetime"
        );
        assert_eq!(hit.doc_count, 2, "both documents are published and live");

        // `rustc` is in both documents, so the agent filter is the only
        // thing that can change this count. Asserting all three of
        // unfiltered / matching / non-matching is what distinguishes a
        // filter that restricts from one that is silently ignored (which
        // would answer 2 every time) or always refuses (0 every time).
        let shared = parser.parse("rustc", &filters);
        let shared_hit = reader
            .search_preparsed_paginated(&cx, &shared.query, 10, 0, true)
            .expect("search the CASS index for a shared term");
        assert_eq!(
            shared_hit.total_count,
            Some(2),
            "both documents mention rustc"
        );

        let by_agent = crate::query::CassQueryFilters {
            agents: vec!["codex".to_owned()],
            ..Default::default()
        };
        let scoped = parser.parse("rustc", &by_agent);
        let scoped_hit = reader
            .search_preparsed_paginated(&cx, &scoped.query, 10, 0, true)
            .expect("search the CASS index by agent");
        assert_eq!(
            scoped_hit.total_count,
            Some(1),
            "the agent filter must restrict a term both documents carry"
        );

        let by_absent_agent = crate::query::CassQueryFilters {
            agents: vec!["nobody".to_owned()],
            ..Default::default()
        };
        let absent = parser.parse("rustc", &by_absent_agent);
        let absent_hit = reader
            .search_preparsed_paginated(&cx, &absent.query, 10, 0, true)
            .expect("search the CASS index for an absent agent");
        assert_eq!(
            absent_hit.total_count,
            Some(0),
            "an agent that produced nothing must match nothing"
        );
    });
}

/// A date-filtered query must keep working once a sealed segment holds a
/// tombstone (frankensearch#49, cass GH #499).
///
/// cass revises a message by upserting it under its stable identity, which
/// tombstones the old row inside the segment it was sealed in. cass's
/// `--since`/`--days` window is a NUMERIC range leaf over `created_at`,
/// composed in a Boolean with the query's term leaves. Before the fix the
/// range leaf was scored on the segment's at-seal row count while the term
/// leaves used its live count, and the Boolean refused to compose them
/// ("Boolean children belong to different segment domains"). The
/// unfiltered control, the in-window counts (every live match, never the
/// tombstoned original), and the empty out-of-window result mean a fix
/// that dropped the range leaf or resurrected tombstones fails here too.
#[test]
fn date_filtered_query_survives_a_tombstone_in_a_sealed_segment() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        // cass's production settings: no engine tier merge and no
        // lag-driven publication. With the defaults a commit may merge or
        // compact the segment and drop the tombstone before any query
        // runs, which is how a first version of this test passed on the
        // unfixed engine.
        let config = crate::QuillConfig {
            tier_fanout: usize::MAX,
            max_visibility_lag_ms: u64::MAX,
            ..crate::QuillConfig::default()
        };
        let directory = tempfile::tempdir().expect("cass index directory");
        let index = crate::index::QuillIndex::create_with_schema(
            &cx,
            directory.path(),
            crate::schema::CASS_SEMANTIC_SCHEMA,
            config.clone(),
        )
        .await
        .expect("create a CASS-schema index");

        // One tombstone in 40 rows (2.5%) stays far below the default
        // compaction_tombstone_density (20%), like the reporter's large
        // segments with thousands of scattered tombstones.
        let sealed = (0..40_u64)
            .map(|msg_idx| {
                sample_document(
                    "gh499",
                    msg_idx,
                    "codex",
                    "Tombstone probe",
                    &format!("window probe original {msg_idx}"),
                )
                .to_schema_document()
            })
            .collect::<Vec<_>>();
        index
            .index_schema_documents(&cx, &sealed)
            .await
            .expect("ingest the first generation");
        index.commit(&cx).await.expect("seal the first generation");

        // Revise message 20 under its stable identity, as cass does for an
        // edited message: its sealed row becomes a tombstone.
        let revised = [sample_document(
            "gh499",
            20,
            "codex",
            "Tombstone probe",
            "window probe revised",
        )
        .to_schema_document()];
        index
            .upsert_schema_documents(&cx, &revised)
            .await
            .expect("upsert a sealed identity");
        index.commit(&cx).await.expect("publish the revision");

        // Guard the fixture itself: without a surviving tombstone in a
        // sealed segment this test cannot observe the defect.
        let snapshot = index.snapshot().expect("published snapshot");
        let tombstones: u64 = snapshot
            .segments()
            .iter()
            .map(|segment| segment.tombstone_count())
            .sum();
        assert!(
            tombstones >= 1,
            "the fixture must leave a tombstone in a sealed segment"
        );

        let reader = crate::index::QuillSearchIndex::open_with_schema(
            &cx,
            directory.path(),
            crate::schema::CASS_SEMANTIC_SCHEMA,
            config,
        )
        .await
        .expect("open a CASS-schema reader");
        let parser = crate::query::CassQueryParser::new(crate::schema::CASS_SEMANTIC_SCHEMA)
            .expect("build the CASS query parser");
        let count = |text: &str, filters: &crate::query::CassQueryFilters| {
            let parsed = parser.parse(text, filters);
            reader
                .search_preparsed_paginated(&cx, &parsed.query, 10, 0, true)
                .map(|page| page.total_count)
        };

        let unfiltered = crate::query::CassQueryFilters::default();
        assert_eq!(
            count("probe", &unfiltered).expect("unfiltered search"),
            Some(40),
            "forty live messages mention probe"
        );

        // created_at is 1_700_000_000 + msg_idx, so this window holds
        // messages 10..=39. It must cover only part of the sealed segment:
        // a range over every row is lowered as a whole-segment match and
        // never reaches the range scorer, which is how the first two
        // versions of this test passed on the unfixed engine.
        let in_window = crate::query::CassQueryFilters {
            created_from: Some(1_700_000_010),
            created_to: Some(1_800_000_000),
            ..Default::default()
        };
        assert_eq!(
            count("probe", &in_window)
                .expect("a date filter over a tombstoned sealed segment must not fail"),
            Some(30),
            "the window holds messages 10..=39"
        );
        assert_eq!(
            count("original", &in_window).expect("windowed search for originals"),
            Some(29),
            "the tombstoned original must stay invisible inside the window"
        );
        assert_eq!(
            count("revised", &in_window).expect("windowed search for the revision"),
            Some(1),
            "the revision is live inside the window"
        );

        let out_of_window = crate::query::CassQueryFilters {
            created_from: Some(1_800_000_000),
            ..Default::default()
        };
        assert_eq!(
            count("probe", &out_of_window).expect("out-of-window search"),
            Some(0),
            "a window after every message must match nothing"
        );
    });
}

/// A disjunction under a conjunction must return every match in every
/// segment layout. cass's metamorphic oracle (`tests/search_metamorphic.rs`)
/// saw `(kiwiword OR limeword) AND mangoword` miss one conversation's
/// `limeword AND mangoword` rows in one indexing run and not in another,
/// so the answer depended on insertion order or commit boundaries. The
/// corpus replays that oracle (eighteen sessions of ten messages, seeded
/// terms); every layout re-indexes it, and every nested query's hits must
/// equal the set-algebra answer.
#[test]
fn nested_disjunction_under_conjunction_matches_set_algebra_in_every_layout() {
    use std::collections::BTreeSet;

    const TERMS: [&str; 5] = ["kiwiword", "limeword", "mangoword", "plumword", "pearword"];
    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    // (session, index, term indices), exactly as the cass oracle draws them.
    let mut messages: Vec<(u64, u64, Vec<usize>)> = Vec::new();
    for session in 0..18_u64 {
        let mut rng = 0x9E37_79B9_7F4A_7C15_u64 ^ (session + 1).wrapping_mul(0x2545_F491_4F6C_DD1D);
        for index in 0..10_u64 {
            let terms = if index == 0 {
                Vec::new()
            } else {
                (0..TERMS.len())
                    .filter(|_| next(&mut rng) % 100 < 40)
                    .collect()
            };
            messages.push((session, index, terms));
        }
    }
    // A revision keeps the message's terms, so every expected answer is
    // unchanged; it only leaves a tombstone in the sealed segment.
    let document = |position: usize, revised: bool| {
        let (session, index, terms) = &messages[position];
        let words: Vec<&str> = terms.iter().map(|term| TERMS[*term]).collect();
        let suffix = if revised { " revisedcopy" } else { "" };
        sample_document(
            &format!("s{session:02}"),
            *index,
            "codex",
            &format!("msgid{} note", session * 100),
            &format!(
                "msgid{} note {}{suffix}",
                session * 100 + index,
                words.join(" ")
            ),
        )
        .to_schema_document()
    };

    // Each layout is a list of commits; each commit a list of positions.
    let all: Vec<usize> = (0..messages.len()).collect();
    let session_of = |position: &usize| messages[*position].0;
    let mut layouts: Vec<(String, Vec<Vec<usize>>)> = vec![
        ("single".to_owned(), vec![all.clone()]),
        (
            "reversed".to_owned(),
            vec![all.iter().rev().copied().collect()],
        ),
        (
            "per-session".to_owned(),
            (0..18_u64)
                .map(|session| {
                    all.iter()
                        .copied()
                        .filter(|position| session_of(position) == session)
                        .collect()
                })
                .collect(),
        ),
        (
            "chunks-64".to_owned(),
            all.chunks(64).map(<[usize]>::to_vec).collect(),
        ),
        (
            "session16-last".to_owned(),
            vec![
                all.iter()
                    .copied()
                    .filter(|p| session_of(p) != 16)
                    .collect(),
                all.iter()
                    .copied()
                    .filter(|p| session_of(p) == 16)
                    .collect(),
            ],
        ),
        (
            "session16-alone-before-17".to_owned(),
            vec![
                all.iter().copied().filter(|p| session_of(p) < 16).collect(),
                all.iter()
                    .copied()
                    .filter(|p| session_of(p) == 16)
                    .collect(),
                all.iter()
                    .copied()
                    .filter(|p| session_of(p) == 17)
                    .collect(),
            ],
        ),
    ];
    for seed in 1..=4_u64 {
        let mut shuffled = all.clone();
        let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        for i in (1..shuffled.len()).rev() {
            let bound = u64::try_from(i + 1).expect("bound");
            let j = usize::try_from(next(&mut rng) % bound).expect("index");
            shuffled.swap(i, j);
        }
        layouts.push((format!("shuffled-{seed}"), vec![shuffled.clone()]));
        layouts.push((
            format!("shuffled-{seed}-chunks-40"),
            shuffled.chunks(40).map(<[usize]>::to_vec).collect(),
        ));
    }
    // The same layouts again with revisions after the last commit: each
    // revised position is upserted under its stable identity, which
    // tombstones its sealed row (cass does this for an edited message and
    // on some reconcile paths).
    let session16_head: Vec<usize> = all
        .iter()
        .copied()
        .filter(|p| session_of(p) == 16 && messages[*p].1 < 4)
        .collect();
    let revisions: Vec<(&str, Vec<usize>)> = vec![
        ("first-1", all[..1].to_vec()),
        ("first-4", all[..4].to_vec()),
        ("first-10", all[..10].to_vec()),
        ("session16-head", session16_head),
        ("every-7th", all.iter().copied().step_by(7).collect()),
    ];
    // Each layout also runs with its committed segments concat-merged into
    // one before any revision, as cass's staged rebuild folds shards.
    let plain = layouts.clone();
    /// Name, commits (positions per commit), concat-merge, revisions.
    type Layout = (String, Vec<Vec<usize>>, bool, Vec<usize>);
    let mut layouts: Vec<Layout> = Vec::new();
    for (name, commits) in &plain {
        let merges: &[bool] = if commits.len() > 1 {
            &[false, true]
        } else {
            &[false]
        };
        for &merge in merges {
            let base = if merge {
                format!("{name}+merged")
            } else {
                name.clone()
            };
            layouts.push((base.clone(), commits.clone(), merge, Vec::new()));
            for (revision, positions) in &revisions {
                layouts.push((
                    format!("{base}+revise-{revision}"),
                    commits.clone(),
                    merge,
                    positions.clone(),
                ));
            }
        }
    }

    // Every `(a OR b) AND c` and `c AND (a OR b)` over distinct terms.
    let mut queries: Vec<(String, BTreeSet<u64>)> = Vec::new();
    for c in 0..TERMS.len() {
        for a in 0..TERMS.len() {
            for b in (a + 1)..TERMS.len() {
                if a == c || b == c {
                    continue;
                }
                let expected: BTreeSet<u64> = messages
                    .iter()
                    .filter(|(_, _, terms)| {
                        (terms.contains(&a) || terms.contains(&b)) && terms.contains(&c)
                    })
                    .map(|(session, index, _)| session * 100 + index)
                    .collect();
                let (ta, tb, tc) = (TERMS[a], TERMS[b], TERMS[c]);
                queries.push((format!("({ta} OR {tb}) AND {tc}"), expected.clone()));
                queries.push((format!("{tc} AND ({ta} OR {tb})"), expected));
            }
        }
    }

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let config = crate::QuillConfig {
            tier_fanout: usize::MAX,
            max_visibility_lag_ms: u64::MAX,
            ..crate::QuillConfig::default()
        };
        let parser = crate::query::CassQueryParser::new(crate::schema::CASS_SEMANTIC_SCHEMA)
            .expect("build the CASS query parser");
        let mut failures = Vec::new();
        for (name, commits, merge, revised) in &layouts {
            let directory = tempfile::tempdir().expect("index directory");
            let index = crate::index::QuillIndex::create_with_schema(
                &cx,
                directory.path(),
                crate::schema::CASS_SEMANTIC_SCHEMA,
                config.clone(),
            )
            .await
            .expect("create a CASS-schema index");
            for commit in commits {
                let batch: Vec<_> = commit
                    .iter()
                    .map(|position| document(*position, false))
                    .collect();
                index
                    .index_schema_documents(&cx, &batch)
                    .await
                    .expect("ingest a batch");
                index.commit(&cx).await.expect("commit a batch");
            }
            if *merge {
                let inputs: Vec<u64> = index
                    .snapshot()
                    .expect("snapshot")
                    .segments()
                    .iter()
                    .map(|segment| segment.manifest().segment_id)
                    .collect();
                let mut output = 0x0C0A_7000_u64;
                while inputs.contains(&output) {
                    output += 1;
                }
                index
                    .concat_merge(&cx, &inputs, output, 1_700_000_000)
                    .await
                    .expect("concat-merge every committed segment");
            }
            if !revised.is_empty() {
                let batch: Vec<_> = revised
                    .iter()
                    .map(|position| document(*position, true))
                    .collect();
                index
                    .upsert_schema_documents(&cx, &batch)
                    .await
                    .expect("upsert revisions");
                index.commit(&cx).await.expect("publish revisions");
            }
            let snapshot = index.snapshot().expect("snapshot");
            let segments = snapshot.segments().len();
            let tombstones: u64 = snapshot
                .segments()
                .iter()
                .map(|segment| segment.tombstone_count())
                .sum();
            let reader = crate::index::QuillSearchIndex::open_with_schema(
                &cx,
                directory.path(),
                crate::schema::CASS_SEMANTIC_SCHEMA,
                config.clone(),
            )
            .await
            .expect("open a CASS-schema reader");
            // cass fetches limit * 3 / 2 without an exact count first.
            for ((query, expected), exact) in queries
                .iter()
                .flat_map(|query| [(query, false), (query, true)])
            {
                let parsed = parser.parse(query, &crate::query::CassQueryFilters::default());
                let page = reader
                    .search_preparsed_paginated(&cx, &parsed.query, 1500, 0, exact)
                    .expect("search");
                let got: BTreeSet<u64> = page
                    .hits
                    .iter()
                    .map(|hit| {
                        let session: u64 = hit.document_id[1..3].parse().expect("session");
                        let index: u64 = hit
                            .document_id
                            .rsplit('#')
                            .next()
                            .and_then(|tail| tail.parse().ok())
                            .expect("msg_idx");
                        session * 100 + index
                    })
                    .collect();
                let expected_count = u64::try_from(expected.len()).expect("count");
                let count_ok = !exact || page.total_count == Some(expected_count);
                if &got != expected || !count_ok {
                    failures.push(format!(
                        "{name} ({segments} segments, {tombstones} tombstones, exact={exact}) {query}: missing {:?}, unexpected {:?}, total_count {:?} vs {}",
                        expected.difference(&got).collect::<Vec<_>>(),
                        got.difference(expected).collect::<Vec<_>>(),
                        page.total_count,
                        expected.len()
                    ));
                }
            }
        }
        assert!(
            failures.is_empty(),
            "{} wrong answers:\n{}",
            failures.len(),
            failures.join("\n")
        );
    });
}
