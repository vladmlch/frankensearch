use tantivy::schema::{FAST, Field, STORED, STRING, Schema, TextFieldIndexing, TextOptions};

/// Name for the custom tokenizer registered with the Tantivy index.
pub const TOKENIZER_NAME: &str = "frankensearch_default";

/// Named fields from the Tantivy schema for type-safe access.
#[derive(Debug, Clone, Copy)]
pub struct SchemaFields {
    pub id: Field,
    pub content: Field,
    pub title: Field,
    pub metadata_json: Field,
    /// Dense per-document insertion ordinal, stored as a `u64` FAST (columnar)
    /// field. `None` for indexes created before this field existed (resolved by
    /// name in `from_index`); such indexes materialize ids via the docstore.
    pub ord: Option<Field>,
}

/// Build the frankensearch Tantivy schema.
pub fn build_schema() -> (Schema, SchemaFields) {
    build_schema_with_positions(true)
}

pub fn build_schema_with_positions(positions: bool) -> (Schema, SchemaFields) {
    let mut builder = Schema::builder();

    // ID: exact-match only, stored for retrieval.
    let id = builder.add_text_field("id", STRING | STORED);

    // Content: full-text indexed with our custom tokenizer, stored for snippet use.
    let index_record_option = if positions {
        tantivy::schema::IndexRecordOption::WithFreqsAndPositions
    } else {
        tantivy::schema::IndexRecordOption::Basic
    };
    let content_options = TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer(TOKENIZER_NAME)
                .set_index_option(index_record_option),
        )
        .set_stored();
    let content = builder.add_text_field("content", content_options);

    // Title: full-text indexed with our custom tokenizer, stored.
    let title_options = TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer(TOKENIZER_NAME)
                .set_index_option(index_record_option),
        )
        .set_stored();
    let title = builder.add_text_field("title", title_options);

    // Metadata: stored as JSON string, not indexed.
    let metadata_json = builder.add_text_field("metadata_json", STORED);

    // Dense insertion ordinal: a flat bit-packed `u64` FAST column (no
    // dictionary) plus an external `ordinal -> doc_id` table lets id
    // materialization skip the stored-document decompress entirely. STORED too
    // so the table can be rebuilt from a reopened index if needed.
    let ord = builder.add_u64_field("ord", FAST | STORED);

    let schema = builder.build();
    let fields = SchemaFields {
        id,
        content,
        title,
        metadata_json,
        ord: Some(ord),
    };
    (schema, fields)
}
