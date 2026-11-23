use super::{
    CassDocumentRef, CassFields, CassQueryFilters, CassTantivyIndex, Value as _, cass_build_schema,
    cass_build_tantivy_query, cass_fields_from_schema,
};

pub(super) fn fields() -> CassFields {
    let schema = cass_build_schema();
    cass_fields_from_schema(&schema).expect("cass fields")
}

pub(super) fn scored_cass_results(
    searcher: &tantivy::Searcher,
    fields: &CassFields,
    raw_query: &str,
    filters: &CassQueryFilters,
) -> std::collections::BTreeMap<u64, f32> {
    let query = cass_build_tantivy_query(raw_query, filters, fields);
    searcher
        .search(
            &*query,
            &tantivy::collector::TopDocs::with_limit(64).order_by_score(),
        )
        .expect("search CASS Boolean fixture")
        .into_iter()
        .map(|(score, address)| {
            let document: tantivy::TantivyDocument = searcher
                .doc(address)
                .expect("load CASS Boolean fixture document");
            let msg_idx = document
                .get_first(fields.msg_idx)
                .and_then(|value| value.as_u64())
                .expect("stored CASS fixture msg_idx");
            (msg_idx, score)
        })
        .collect()
}

pub(super) fn cass_result_ids(
    searcher: &tantivy::Searcher,
    fields: &CassFields,
    raw_query: &str,
    filters: &CassQueryFilters,
) -> Vec<u64> {
    scored_cass_results(searcher, fields, raw_query, filters)
        .into_keys()
        .collect()
}

pub(super) fn cass_result_fixture() -> (tempfile::TempDir, CassTantivyIndex) {
    let directory = tempfile::tempdir().expect("temporary CASS result index directory");
    let mut index = CassTantivyIndex::open_or_create_with_writer_parallelism(directory.path(), 1)
        .expect("create CASS result index");
    let documents = [
        CassDocumentRef {
            agent: "claude",
            workspace: Some("/alpha"),
            workspace_original: Some("/alpha"),
            source_path: "/fixture/0.jsonl",
            msg_idx: 0,
            created_at: Some(100),
            title: Some("Current Alpha"),
            content: "alpha active",
            source_id: "local-a",
            origin_kind: "local",
            origin_host: None,
            conversation_id: None,
        },
        CassDocumentRef {
            agent: "claude",
            workspace: Some("/alpha"),
            workspace_original: Some("/alpha"),
            source_path: "/fixture/1.jsonl",
            msg_idx: 1,
            created_at: Some(200),
            title: Some("Legacy Alpha"),
            content: "alpha deprecated",
            source_id: "local-b",
            origin_kind: "local",
            origin_host: None,
            conversation_id: None,
        },
        CassDocumentRef {
            agent: "codex",
            workspace: Some("/beta"),
            workspace_original: Some("/beta"),
            source_path: "/fixture/2.jsonl",
            msg_idx: 2,
            created_at: Some(300),
            title: Some("Current Beta"),
            content: "beta active",
            source_id: "remote-a",
            origin_kind: "ssh",
            origin_host: Some("worker-a"),
            conversation_id: None,
        },
        CassDocumentRef {
            agent: "claude",
            workspace: Some("/gamma"),
            workspace_original: Some("/gamma"),
            source_path: "/fixture/3.jsonl",
            msg_idx: 3,
            created_at: Some(400),
            title: Some("Current Gamma"),
            content: "gamma active",
            source_id: "remote-b",
            origin_kind: "ssh",
            origin_host: Some("worker-b"),
            conversation_id: None,
        },
        CassDocumentRef {
            agent: "codex",
            workspace: Some("/delta"),
            workspace_original: Some("/delta"),
            source_path: "/fixture/4.jsonl",
            msg_idx: 4,
            created_at: Some(500),
            title: Some("Currentish Omega"),
            content: "xalpha archived",
            source_id: "local-c",
            origin_kind: "local",
            origin_host: None,
            conversation_id: None,
        },
        CassDocumentRef {
            agent: "gemini",
            workspace: Some("/delta"),
            workspace_original: Some("/delta"),
            source_path: "/fixture/5.jsonl",
            msg_idx: 5,
            created_at: Some(600),
            title: Some("Middle Marker"),
            content: "alphabeta retained",
            source_id: "local-d",
            origin_kind: "local",
            origin_host: None,
            conversation_id: None,
        },
    ];
    index
        .add_cass_document_refs(&documents)
        .expect("index CASS result documents");
    index.commit().expect("commit CASS result documents");
    (directory, index)
}
