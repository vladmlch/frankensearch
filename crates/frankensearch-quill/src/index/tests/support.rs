use super::*;

/// Test-rendezvous bound: catches a hang, never races the scheduler.
/// 2s flaked when the full parallel suite saturated the blocking pool
/// (bd-zam2i).
pub(super) const PUBLISH_PAUSE_RENDEZVOUS_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(30);

pub(super) const CONCAT_MERGE_QUERIES: [&str; 4] =
    ["rust", "python", "rust OR python", "\"rust ownership\""];

const Q1_OB2A_QUERIES: [&str; 5] = [
    "shared",
    "left",
    "right",
    "left OR right",
    "\"shared left\"",
];

const Q1_OB4_QUERIES: [&str; 5] = [
    "shared",
    "left",
    "shared AND left",
    "\"shared left\"",
    "ord:[0 TO 39]",
];

#[cfg(feature = "bench-internals")]
const POSITIONLESS_QG_FIELDS: [FieldDescriptor; 5] = [
    FieldDescriptor {
        id: 0,
        name: "id",
        kind: FieldKind::Keyword,
        stored: true,
    },
    FieldDescriptor {
        id: 1,
        name: "content",
        kind: FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: false,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 2,
        name: "title",
        kind: FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: false,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 3,
        name: "metadata_json",
        kind: FieldKind::StoredOnly,
        stored: true,
    },
    FieldDescriptor {
        id: 4,
        name: "ord",
        kind: FieldKind::U64 {
            indexed: false,
            fast: true,
        },
        stored: true,
    },
];

#[cfg(feature = "bench-internals")]
pub(super) const POSITIONLESS_QG_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "qg-positionless-typed-capability-test",
    fields: &POSITIONLESS_QG_FIELDS,
};

const MIXED_BUDGET_FIELDS: [FieldDescriptor; 5] = [
    FieldDescriptor {
        id: 0,
        name: "id",
        kind: FieldKind::Keyword,
        stored: true,
    },
    FieldDescriptor {
        id: 1,
        name: "content",
        kind: FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: true,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 2,
        name: "title",
        kind: FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: false,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 3,
        name: "metadata_json",
        kind: FieldKind::StoredOnly,
        stored: true,
    },
    FieldDescriptor {
        id: 4,
        name: "ord",
        kind: FieldKind::U64 {
            indexed: false,
            fast: true,
        },
        stored: true,
    },
];

pub(super) const MIXED_BUDGET_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "parallel-budget-mixed-positions-test",
    fields: &MIXED_BUDGET_FIELDS,
};

#[cfg(feature = "bench-internals")]
const POSITIONED_QG_FIELDS: [FieldDescriptor; 5] = [
    FieldDescriptor {
        id: 0,
        name: "id",
        kind: FieldKind::Keyword,
        stored: true,
    },
    FieldDescriptor {
        id: 1,
        name: "content",
        kind: FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: true,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 2,
        name: "title",
        kind: FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: true,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 3,
        name: "metadata_json",
        kind: FieldKind::StoredOnly,
        stored: true,
    },
    FieldDescriptor {
        id: 4,
        name: "ord",
        kind: FieldKind::U64 {
            indexed: false,
            fast: true,
        },
        stored: true,
    },
];

#[cfg(feature = "bench-internals")]
pub(super) const POSITIONED_QG_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "qg-positioned-capability-control",
    fields: &POSITIONED_QG_FIELDS,
};

pub(super) const KNOWN_SECTION_KINDS: [SectionKind; 10] = [
    SectionKind::TERMDICT,
    SectionKind::POSTINGS,
    SectionKind::POSITIONS,
    SectionKind::BLOCKMAX,
    SectionKind::DOCLEN,
    SectionKind::IDMAP,
    SectionKind::IDHASH,
    SectionKind::NUMERIC,
    SectionKind::STOREDMETA,
    SectionKind::STATS,
];

const DELTA_PARITY_FIELDS: [FieldDescriptor; 6] = [
    FieldDescriptor {
        id: 0,
        name: "id",
        kind: FieldKind::Keyword,
        stored: true,
    },
    FieldDescriptor {
        id: 1,
        name: "content",
        kind: FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: true,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 2,
        name: "rank",
        kind: FieldKind::U64 {
            indexed: true,
            fast: true,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 3,
        name: "opaque",
        kind: FieldKind::StoredOnly,
        stored: true,
    },
    FieldDescriptor {
        id: 4,
        name: "title",
        kind: FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: true,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 5,
        name: "fast_rank",
        kind: FieldKind::U64 {
            indexed: false,
            fast: true,
        },
        stored: false,
    },
];

pub(super) const DELTA_PARITY_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "delta-seal-parity-v1",
    fields: &DELTA_PARITY_FIELDS,
};

pub(super) fn run_with_cx<F, Fut>(test: F)
where
    F: FnOnce(Cx) -> Fut,
    Fut: Future<Output = ()>,
{
    asupersync::test_utils::run_test_with_cx(test);
}

pub(super) fn run_with_blocking_cx<F, Fut>(test: F)
where
    F: FnOnce(Cx) -> Fut,
    Fut: Future<Output = ()>,
{
    let cx = Cx::for_testing();
    let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .blocking_threads(1, 4)
        .build()
        .expect("build test runtime with a blocking pool");
    runtime.block_on(test(cx));
}

pub(super) fn deterministic_config() -> QuillConfig {
    QuillConfig {
        deterministic_ingest: true,
        ..QuillConfig::default()
    }
}

pub(super) async fn create_root_generation(
    cx: &Cx,
    lexical_root: &Path,
    name: &str,
    document_id: &str,
    content: &str,
) -> PathBuf {
    let path = lexical_root.join(name);
    let index = QuillIndex::create(cx, &path, deterministic_config())
        .await
        .expect("create root-bound generation");
    LexicalWrite::index_document(&index, cx, &IndexableDocument::new(document_id, content))
        .await
        .expect("index root-bound fixture document");
    LexicalWrite::commit(&index, cx)
        .await
        .expect("commit root-bound fixture document");
    path
}

pub(super) fn root_pointer(name: &str) -> CurrentPointer {
    CurrentPointer::new(
        BlueGreenEngine::Quill,
        name,
        crate::segment::FSLX_FORMAT_VERSION,
    )
    .expect("valid root-bound Quill pointer")
}

pub(super) fn fuel_diagnostics(error: &QuillIndexError) -> (u64, u64, u64, u64, u64, u64) {
    let QuillIndexError::QueryFuelExhausted {
        budget,
        consumed,
        segments_touched,
        dictionary_blocks,
        posting_blocks,
        position_docs,
    } = error
    else {
        // ubs:ignore -- Test helper must reject the wrong query-fuel error variant.
        panic!("expected typed query fuel exhaustion, got {error:?}");
    };
    (
        *budget,
        *consumed,
        *segments_touched,
        *dictionary_blocks,
        *posting_blocks,
        *position_docs,
    )
}

const E3_9_PINNED_SEEDS: [u64; 4] = [
    0xe3_9000_0000_0001,
    0xe3_9000_0000_001d,
    0xe3_9000_0000_0101,
    0xe3_9000_0000_1009,
];

const E3_9_RANDOM_SEED_COUNT_ENV: &str = "QUILL_E3_9_RANDOM_SEEDS";

const E6_5_PINNED_SEEDS: [u64; 4] = [
    0xe6_5000_0000_0001,
    0xe6_5000_0000_001d,
    0xe6_5000_0000_0101,
    0xe6_5000_0000_1009,
];

const E6_5_RANDOM_SEED_COUNT_ENV: &str = "QUILL_E6_5_RANDOM_SEEDS";

fn lab_seed_corpus(pinned: &[u64], random_seed_count_env: &str, salt: u64) -> Vec<u64> {
    let random_seed_count = match std::env::var(random_seed_count_env) {
        Ok(value) => value.parse::<usize>().unwrap_or_else(|error| {
            panic!("{random_seed_count_env}={value:?} is not a seed count: {error}") // ubs:ignore -- Invalid test seeds must fail rather than skip cases.
        }),
        Err(std::env::VarError::NotPresent) => 0,
        // ubs:ignore -- Invalid test seed configuration must fail, not silently skip cases.
        Err(error) => panic!("failed to read {random_seed_count_env}: {error}"),
    };
    let mut seeds = pinned.to_vec();
    if random_seed_count == 0 {
        return seeds;
    }

    let epoch_nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("nightly seed clock predates the Unix epoch")
        .as_nanos();
    let mut state = u64::try_from(epoch_nanos & u128::from(u64::MAX))
        .expect("masked epoch nanoseconds fit u64")
        ^ salt;
    for _ in 0..random_seed_count {
        // SplitMix64 gives the scheduled campaign a fresh, well-spread
        // corpus while every assertion still prints the exact replay seed.
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut seed = state;
        seed = (seed ^ (seed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        seed = (seed ^ (seed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        seeds.push(seed ^ (seed >> 31));
    }
    seeds
}

pub(super) fn e3_9_seed_corpus() -> Vec<u64> {
    lab_seed_corpus(
        &E3_9_PINNED_SEEDS,
        E3_9_RANDOM_SEED_COUNT_ENV,
        0xe3_9000_9e37_79b9,
    )
}

pub(super) fn e6_5_seed_corpus() -> Vec<u64> {
    lab_seed_corpus(
        &E6_5_PINNED_SEEDS,
        E6_5_RANDOM_SEED_COUNT_ENV,
        0xe6_5000_9e37_79b9,
    )
}

pub(super) fn assert_e3_9_lab_report(
    scenario: &str,
    seed: u64,
    report: &asupersync::lab::LabRunReport,
) {
    let replay = format!(
        "{scenario}: seed={seed:#018x} fingerprint={:#018x}",
        report.trace_fingerprint
    );
    assert!(report.quiescent, "{replay}: LabRuntime did not quiesce");
    assert!(
        report.oracle_report.all_passed(),
        "{replay}: LabRuntime oracle failure: {:?}",
        report.oracle_report
    );
    assert!(
        report.invariant_violations.is_empty(),
        "{replay}: LabRuntime invariant failures: {:?}",
        report.invariant_violations
    );
}

pub(super) fn assert_pairwise_disjoint_manifest(segments: &[ManifestSegment]) {
    for pair in segments.windows(2) {
        assert!(
            pair[0].docid_hi <= pair[1].docid_lo,
            "manifest intervals overlap: {:?} then {:?}",
            pair[0],
            pair[1],
        );
    }
}

pub(super) fn shipping_content_hash(document_id: &str, content: &str) -> u64 {
    let document = IndexableDocument::new(document_id, content);
    let metadata = canonical_metadata(&document.metadata).expect("canonical fixture metadata");
    canonical_document_content_hash(&document, &metadata).expect("canonical fixture document hash")
}

pub(super) fn fixture_documents() -> Vec<IndexableDocument> {
    vec![
        IndexableDocument::new("rust-1", "rust ownership prevents data races")
            .with_title("Rust ownership")
            .with_metadata("cluster", "systems"),
        IndexableDocument::new("rust-2", "safe systems programming with rust")
            .with_title("Borrow checker")
            .with_metadata("cluster", "systems"),
        IndexableDocument::new("python-1", "python data science notebooks")
            .with_title("Python guide")
            .with_metadata("cluster", "data"),
    ]
}

pub(super) fn apply_alpha_delta(
    delta: &mut DeltaSegment,
    global_docid: u32,
    document_id: &str,
    content_frequency: u32,
) {
    apply_sealable_delta_document(delta, global_docid, document_id, "alpha", content_frequency);
}

pub(super) fn apply_sealable_delta_document(
    delta: &mut DeltaSegment,
    global_docid: u32,
    document_id: &str,
    content_term: &str,
    content_frequency: u32,
) {
    let positions = (0..content_frequency).collect::<Vec<_>>();
    let content = std::iter::repeat_n(
        content_term,
        usize::try_from(content_frequency).expect("fixture frequency fits usize"),
    )
    .collect::<Vec<_>>()
    .join(" ");
    let ordinal = u64::from(global_docid).to_le_bytes();
    let fieldnorms = [
        DeltaFieldNorm {
            field_ord: ID_FIELD,
            raw_length: 1,
            fieldnorm_id: fieldnorm_to_id(1),
        },
        DeltaFieldNorm {
            field_ord: CONTENT_FIELD,
            raw_length: content_frequency,
            fieldnorm_id: fieldnorm_to_id(content_frequency),
        },
        DeltaFieldNorm {
            field_ord: TITLE_FIELD,
            raw_length: 0,
            fieldnorm_id: fieldnorm_to_id(0),
        },
    ];
    let postings = [
        DeltaTermPosting {
            field_ord: ID_FIELD,
            term: document_id.as_bytes(),
            frequency: 1,
            positions: None,
        },
        DeltaTermPosting {
            field_ord: CONTENT_FIELD,
            term: content_term.as_bytes(),
            frequency: content_frequency,
            positions: Some(&positions),
        },
    ];
    let stored = [
        DeltaStoredValue::new(ID_FIELD, document_id.as_bytes()),
        DeltaStoredValue::new(CONTENT_FIELD, content.as_bytes()),
        DeltaStoredValue::new(TITLE_FIELD, b""),
        DeltaStoredValue::new(METADATA_FIELD, b"{}"),
        DeltaStoredValue::new(ORD_FIELD, &ordinal),
    ];
    delta
        .apply_document_with_values(
            global_docid,
            frankensearch_core::DocId::from(document_id),
            shipping_content_hash(document_id, &content),
            &fieldnorms,
            &postings,
            &[],
            &stored,
        )
        .expect("apply Delta fixture document");
}

pub(super) fn apply_tokenized_delta_document(
    delta: &mut DeltaSegment,
    global_docid: u32,
    document_id: &str,
    content: &str,
) {
    let content_has_positions = delta
        .schema()
        .fields
        .iter()
        .find(|field| field.id == CONTENT_FIELD)
        .is_some_and(|field| {
            matches!(
                field.kind,
                FieldKind::Text {
                    positions: true,
                    ..
                }
            )
        });
    let mut term_positions = BTreeMap::<&str, Vec<u32>>::new();
    for (position, term) in content.split_ascii_whitespace().enumerate() {
        term_positions.entry(term).or_default().push(
            u32::try_from(position).expect("fixture token position fits the Delta wire type"),
        );
    }
    let token_count = term_positions
        .values()
        .map(Vec::len)
        .try_fold(0_u32, |count, term_count| {
            count.checked_add(u32::try_from(term_count).expect("fixture term frequency fits u32"))
        })
        .expect("fixture token count fits u32");
    let fieldnorms = [
        DeltaFieldNorm {
            field_ord: ID_FIELD,
            raw_length: 1,
            fieldnorm_id: fieldnorm_to_id(1),
        },
        DeltaFieldNorm {
            field_ord: CONTENT_FIELD,
            raw_length: token_count,
            fieldnorm_id: fieldnorm_to_id(token_count),
        },
        DeltaFieldNorm {
            field_ord: TITLE_FIELD,
            raw_length: 0,
            fieldnorm_id: fieldnorm_to_id(0),
        },
    ];
    let mut postings = Vec::with_capacity(term_positions.len() + 1);
    postings.push(DeltaTermPosting {
        field_ord: ID_FIELD,
        term: document_id.as_bytes(),
        frequency: 1,
        positions: None,
    });
    for (term, positions) in &term_positions {
        postings.push(DeltaTermPosting {
            field_ord: CONTENT_FIELD,
            term: term.as_bytes(),
            frequency: u32::try_from(positions.len()).expect("fixture frequency fits u32"),
            positions: content_has_positions.then_some(positions.as_slice()),
        });
    }
    let ordinal = u64::from(global_docid).to_le_bytes();
    let stored = [
        DeltaStoredValue::new(ID_FIELD, document_id.as_bytes()),
        DeltaStoredValue::new(CONTENT_FIELD, content.as_bytes()),
        DeltaStoredValue::new(TITLE_FIELD, b""),
        DeltaStoredValue::new(METADATA_FIELD, b"{}"),
        DeltaStoredValue::new(ORD_FIELD, &ordinal),
    ];
    delta
        .apply_document_with_values(
            global_docid,
            DocId::from(document_id),
            shipping_content_hash(document_id, content),
            &fieldnorms,
            &postings,
            &[],
            &stored,
        )
        .expect("apply tokenized Delta fixture document");
}

pub(super) fn alpha_snapshot_tuple(snapshot: &QuillSearchSnapshot) -> (u64, u64, u64, u64, u64) {
    let stats = snapshot
        .bm25_field_stats(CONTENT_FIELD)
        .expect("content snapshot statistics");
    (
        snapshot.snapshot_epoch(),
        snapshot.live_doc_count(),
        stats.doc_count,
        stats.total_tokens,
        snapshot
            .bm25_doc_freq(CONTENT_FIELD, b"alpha")
            .expect("alpha document frequency"),
    )
}

pub(super) async fn concat_merge_fixture_index(cx: &Cx) -> QuillIndex {
    let documents = fixture_documents();
    let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
    for (batch_index, document) in documents.iter().enumerate() {
        index
            .index_documents(cx, std::slice::from_ref(document))
            .await
            .expect("accumulate concat-merge leaf");
        index.commit(cx).await.expect("commit concat-merge leaf");
        assert_eq!(
            index
                .snapshot()
                .expect("concat-merge fixture snapshot is authoritative")
                .segments()
                .len(),
            batch_index + 1,
            "each committed fixture batch must seal one leaf segment",
        );
    }
    index
}

pub(super) struct Q1Ob2aSeal {
    pub(super) encoded: EncodedSegment,
    pub(super) field_stats: BTreeMap<u16, (u64, u32)>,
}

pub(super) fn q1_ob2a_documents() -> Vec<IndexableDocument> {
    (0_u32..400)
        .map(|ordinal| {
            let cohort = if ordinal < 100 { "left" } else { "right" };
            IndexableDocument::new(
                format!("q1-ob2a-{ordinal:03}"),
                format!("shared {cohort} concat parity"),
            )
            .with_title(format!("{cohort} cohort"))
            .with_metadata("cohort", cohort)
        })
        .collect()
}

fn q1_ob2a_doc_ord(first_doc_ord: u32, offset: usize) -> u32 {
    first_doc_ord
        .checked_add(u32::try_from(offset).expect("Q1-OB2a offset fits u32"))
        .expect("Q1-OB2a document ordinal stays in one lease")
}

pub(super) fn seal_q1_ob2a_documents(
    documents: &[IndexableDocument],
    first_doc_ord: u32,
    segment_id: u64,
) -> Q1Ob2aSeal {
    let mut accumulator = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("Q1-OB2a accumulator");
    let mut content_hashes = Vec::with_capacity(documents.len());
    for (offset, document) in documents.iter().enumerate() {
        let doc_ord = q1_ob2a_doc_ord(first_doc_ord, offset);
        let metadata = canonical_metadata(&document.metadata).expect("canonical metadata");
        let ordinal = u64::from(doc_ord).to_le_bytes();
        let title = document.title.as_deref().unwrap_or("");
        let indexed = [
            IndexedFieldValue::new(ID_FIELD, &document.id),
            IndexedFieldValue::new(CONTENT_FIELD, &document.content),
            IndexedFieldValue::new(TITLE_FIELD, title),
        ];
        let stored = [
            StoredFieldValue::new(METADATA_FIELD, &metadata),
            StoredFieldValue::new(ORD_FIELD, &ordinal),
        ];
        accumulator
            .add_document_with_values(doc_ord, &indexed, &[], &stored)
            .expect("accumulate Q1-OB2a document");
        content_hashes.push(
            canonical_document_content_hash(document, &metadata)
                .expect("canonical Q1-OB2a content hash"),
        );
    }

    let doc_count = u32::try_from(documents.len()).expect("Q1-OB2a document count fits u32");
    let field_stats = accumulator
        .fields()
        .iter()
        .map(|field| (field.field_ord(), (field.total_tokens(), doc_count)))
        .collect::<BTreeMap<_, _>>();
    let identities = documents
        .iter()
        .zip(&content_hashes)
        .enumerate()
        .map(|(offset, (document, &content_hash))| {
            FlushDocumentInput::new(
                q1_ob2a_doc_ord(first_doc_ord, offset),
                &document.id,
                content_hash,
            )
        })
        .collect::<Vec<_>>();
    let encoded = flush_accumulator_with_mode(
        &accumulator,
        FlushSegmentInput {
            segment_id,
            lease_docid_base: 0,
            created_unix_s: 0,
            engine_version: CURRENT_ENGINE_VERSION,
            documents: &identities,
        },
        FlushMode::Scalar,
    )
    .expect("seal Q1-OB2a segment");
    Q1Ob2aSeal {
        encoded,
        field_stats,
    }
}

pub(super) fn q1_ob2a_owned_index(
    encoded_segments: Vec<EncodedSegment>,
    field_stats: Vec<ManifestFieldStats>,
) -> QuillIndex {
    let genesis = KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("Q1-OB2a genesis");
    let mut manifest = genesis.next_manifest().expect("Q1-OB2a manifest");
    manifest.docid_high_watermark = u64::from(DOC_ORDS_PER_LEASE);
    manifest.segments = encoded_segments
        .iter()
        .enumerate()
        .map(|(index, encoded)| {
            manifest_segment(
                encoded,
                u64::try_from(index + 1).expect("Q1-OB2a seal sequence fits u64"),
            )
        })
        .collect();
    manifest.field_stats = field_stats;
    let snapshot = genesis
        .publish_owned_segments(&manifest, encoded_segments)
        .expect("publish Q1-OB2a fixture");
    QuillIndex::from_backend(
        IndexBackend::Memory(snapshot),
        DEFAULT_SCHEMA,
        deterministic_config(),
    )
    .expect("bind Q1-OB2a fixture index")
}

pub(super) fn q1_ob2a_fixture_indexes() -> (QuillIndex, QuillIndex) {
    let documents = q1_ob2a_documents();
    let left = seal_q1_ob2a_documents(&documents[..100], 0, 0x0b2a_0001);
    let right = seal_q1_ob2a_documents(&documents[100..], 100, 0x0b2a_0002);
    let monolithic = seal_q1_ob2a_documents(&documents, 0, 0x0b2a_0003);

    let first_segment_stats =
        merge_field_stats(&[], &left.field_stats).expect("left Q1-OB2a stats");
    let combined_segment_stats =
        merge_field_stats(&first_segment_stats, &right.field_stats).expect("aggregate leaf stats");
    let monolithic_stats =
        merge_field_stats(&[], &monolithic.field_stats).expect("monolithic Q1-OB2a stats");
    assert_eq!(combined_segment_stats, monolithic_stats);

    let leaves = q1_ob2a_owned_index(vec![left.encoded, right.encoded], combined_segment_stats);
    let monolithic = q1_ob2a_owned_index(vec![monolithic.encoded], monolithic_stats);
    (leaves, monolithic)
}

pub(super) fn q1_ob2a_decoded_terms(index: &QuillIndex) -> Vec<(u16, Vec<u8>, u32, Vec<Posting>)> {
    let mut decoded = BTreeMap::<(u16, Vec<u8>), (u32, Vec<Posting>)>::new();
    for segment in index
        .snapshot()
        .expect("Q1-OB2a fixture snapshot is authoritative")
        .segments()
    {
        let dictionary = open_dictionary(segment, DEFAULT_SCHEMA).expect("Q1-OB2a TERMDICT");
        let limit = usize::try_from(dictionary.term_count()).expect("term count fits usize");
        let terms = dictionary
            .cursor()
            .expect("Q1-OB2a term cursor")
            .collect_bounded(limit)
            .expect("materialize Q1-OB2a terms");
        for term in terms {
            let postings = open_owned_cursor(
                segment,
                DEFAULT_SCHEMA,
                term.field_ord,
                &term.term,
                false,
                None,
            )
            .expect("decode Q1-OB2a postings")
            .postings;
            let entry = decoded.entry((term.field_ord, term.term)).or_default();
            entry.0 = entry
                .0
                .checked_add(term.metadata.doc_freq)
                .expect("Q1-OB2a aggregate df fits u32");
            entry.1.extend(postings);
        }
    }
    decoded
        .into_iter()
        .map(|((field_ord, term), (doc_freq, postings))| (field_ord, term, doc_freq, postings))
        .collect()
}

pub(super) fn q1_ob2a_decoded_stats(segment: &RecoveredSegment) -> StatsSection {
    let expected_fields = term_field_ords(DEFAULT_SCHEMA);
    StatsSection::parse(
        required_section(segment, SectionKind::STATS).expect("Q1-OB2a STATS bytes"),
        &expected_fields,
        segment.manifest().doc_count,
    )
    .expect("decode Q1-OB2a STATS")
}

pub(super) fn q1_ob2a_query_evidence(
    index: &QuillIndex,
    cx: &Cx,
) -> Vec<(QuillSearchResult, Vec<u32>)> {
    Q1_OB2A_QUERIES
        .iter()
        .map(|query| {
            let ranked = index
                .search_paginated(cx, query, 500, 0, true)
                .expect("Q1-OB2a ranked query");
            let docids = index
                .collect_docids(cx, query)
                .expect("Q1-OB2a scoreless query");
            (ranked, docids)
        })
        .collect()
}

pub(super) fn q1_ob4_tombstoned_index(document_count: usize, deleted: &[u32]) -> QuillIndex {
    let documents = q1_ob2a_documents();
    let documents = documents
        .get(..document_count)
        .expect("Q1-OB4 fixture document count");
    let sealed = seal_q1_ob2a_documents(documents, 0, 0x0b40_0001);
    let field_stats =
        merge_field_stats(&[], &sealed.field_stats).expect("Q1-OB4 segment statistics");
    let committed = q1_ob2a_owned_index(vec![sealed.encoded], field_stats);
    let mut manifest = committed
        .snapshot()
        .expect("Q1-OB4 committed fixture snapshot is authoritative")
        .next_manifest()
        .expect("Q1-OB4 tombstone MANIFEST");
    for &global_docid in deleted {
        assert!(
            committed
                .snapshot()
                .expect("Q1-OB4 committed fixture snapshot is authoritative")
                .delete_document(&mut manifest, &format!("q1-ob2a-{global_docid:03}"),)
                .expect("stage Q1-OB4 tombstone"),
            "Q1-OB4 document {global_docid} must be live before deletion",
        );
    }
    let tombstoned = committed
        .snapshot()
        .expect("Q1-OB4 committed fixture snapshot is authoritative")
        .publish_owned_segments(&manifest, Vec::new())
        .expect("publish Q1-OB4 tombstone generation");
    QuillIndex::from_backend(
        IndexBackend::Memory(tombstoned),
        DEFAULT_SCHEMA,
        deterministic_config(),
    )
    .expect("bind Q1-OB4 fixture index")
}

pub(super) type Q1Ob4QueryEvidence = (Vec<(String, u32)>, Option<u64>, u64, Vec<u32>);

pub(super) fn q1_ob4_query_evidence(index: &QuillIndex, cx: &Cx) -> Vec<Q1Ob4QueryEvidence> {
    Q1_OB4_QUERIES
        .iter()
        .map(|query| {
            let ranked = index
                .search_paginated(cx, query, 500, 0, true)
                .expect("Q1-OB4 ranked query");
            let hits = ranked
                .hits
                .iter()
                .map(|hit| (hit.document_id.clone(), hit.global_docid))
                .collect();
            let docids = index
                .collect_docids(cx, query)
                .expect("Q1-OB4 scoreless query");
            (hits, ranked.total_count, ranked.doc_count, docids)
        })
        .collect()
}

pub(super) fn committed_segment_ids(index: &QuillIndex) -> Vec<u64> {
    index
        .snapshot()
        .expect("concat-merge fixture snapshot is authoritative")
        .segments()
        .iter()
        .map(|segment| segment.manifest().segment_id)
        .collect()
}

pub(super) fn fresh_merge_segment_id(index: &QuillIndex, seed: u64) -> u64 {
    let mut candidate = seed;
    loop {
        if index
            .snapshot()
            .expect("concat-merge fixture snapshot is authoritative")
            .segments()
            .iter()
            .all(|segment| segment.manifest().segment_id != candidate)
        {
            return candidate;
        }
        candidate = candidate
            .checked_add(1)
            .expect("concat-merge test segment-id space exhausted");
    }
}

/// Names of the `seg-*.fslx` files and `*.retired` receipts in an index
/// directory, sorted, so a test can compare the on-disk population with
/// the manifest's live set.
#[cfg(unix)]
pub(super) fn segment_files_and_receipts(directory: &Path) -> (Vec<String>, Vec<String>) {
    let mut segments = Vec::new();
    let mut receipts = Vec::new();
    for entry in std::fs::read_dir(directory).expect("read index directory") {
        let name = entry
            .expect("index directory entry")
            .file_name()
            .to_string_lossy()
            .into_owned();
        let extension = Path::new(&name).extension().and_then(|ext| ext.to_str());
        if name.starts_with("seg-") && extension == Some("fslx") {
            segments.push(name);
        } else if extension == Some("retired") {
            receipts.push(name);
        }
    }
    segments.sort();
    receipts.sort();
    (segments, receipts)
}

#[cfg(unix)]
pub(super) fn live_segment_files(index: &QuillIndex) -> Vec<String> {
    let mut names = committed_segment_ids(index)
        .into_iter()
        .map(|segment_id| format!("seg-{segment_id:016x}.fslx"))
        .collect::<Vec<_>>();
    names.sort();
    names
}

/// Seeded multi-segment fixture with heavy cross-segment term overlap and
/// score ties so fan-out ordering mistakes would surface.
pub(super) async fn segment_fanout_fixture_index(
    cx: &Cx,
    segments: usize,
    docs_per_segment: usize,
    seed: u64,
) -> QuillIndex {
    const VOCABULARY: [&str; 10] = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "shared", "rare", "quill", "argus",
    ];
    let tier_fanout = segments
        .checked_add(1)
        .expect("fan-out fixture segment count leaves room for its tier policy");
    let index = QuillIndex::in_memory(QuillConfig {
        deterministic_ingest: true,
        tier_fanout,
        ..QuillConfig::default()
    })
    .expect("memory index");
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for segment in 0..segments {
        let mut batch = Vec::with_capacity(docs_per_segment);
        for ordinal in 0..docs_per_segment {
            let word_count = 3 + usize::try_from(next() % 10).expect("small word count");
            let mut text = String::new();
            for position in 0..word_count {
                if position != 0 {
                    text.push(' ');
                }
                let pick = usize::try_from(next() % 10).expect("vocabulary pick");
                text.push_str(VOCABULARY[pick]);
            }
            batch.push(IndexableDocument::new(
                format!("fanout-s{segment:02}-d{ordinal:04}"),
                text,
            ));
        }
        index
            .index_documents(cx, &batch)
            .await
            .expect("accumulate fan-out fixture batch");
        index
            .commit(cx)
            .await
            .expect("seal fan-out fixture segment");
    }
    assert_eq!(
        index
            .snapshot()
            .expect("fan-out fixture snapshot is authoritative")
            .segments()
            .len(),
        segments
    );
    index
}

pub(super) fn published_generation(index: &QuillIndex) -> u64 {
    index
        .segment_stats()
        .expect("segment stats")
        .published_generation
}

pub(super) fn parallel_budget_fixture_documents(count: usize) -> Vec<IndexableDocument> {
    (0..count)
        .map(|ordinal| {
            IndexableDocument::new(
                format!("parallel-budget-{ordinal:05}"),
                format!("shared budget token group-{}", ordinal % 17),
            )
            .with_title(format!("budget title {}", ordinal % 5))
            .with_metadata("ordinal", ordinal.to_string())
        })
        .collect()
}
