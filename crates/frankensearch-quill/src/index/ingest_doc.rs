use crate::keeper::{ManifestFieldStats, ManifestSegment, TombstoneSet};
use crate::quiver::NumericValue;
use crate::scribe::{
    AccumulatorError, ColumnarAccumulator, DocumentAccumulation, IndexedFieldValue,
    IndexedNumericValue, StoredFieldValue,
};
use crate::segment::EncodedSegment;
use frankensearch_core::IndexableDocument;
use std::collections::{BTreeMap, HashMap};
use xxhash_rust::xxh3::Xxh3;

use super::error_results::{
    CONTENT_FIELD, CONTENT_HASH_DOMAIN, ID_FIELD, METADATA_FIELD, ORD_FIELD, QuillIndexError,
    TITLE_FIELD, invalid_state,
};

/// One document already projected onto an arbitrary compiled schema.
///
/// [`IndexableDocument`] is the five-field shipping shape and stays the input
/// type for [`super::QuillIndex::upsert_documents`]. This is its schema-general
/// sibling for indexes opened through [`super::QuillIndex::create_with_schema`]: every
/// value carries its own stable field ordinal, so a caller with a wider schema
/// — the CASS profile in [`crate::cass`], for instance — supplies each column
/// directly instead of squeezing it through `content`/`title`/`metadata`.
///
/// `id` is deliberately not a schema column here. It is the identity Quill uses
/// for dedup, tombstones, and IDMAP witnesses regardless of schema; supply it
/// in `indexed` as well when the schema also indexes it as a searchable field.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SchemaDocument {
    /// Unique document identifier (caller-defined).
    pub id: String,
    /// Indexed string columns as `(field_ord, text)`.
    pub indexed: Vec<(u16, String)>,
    /// Indexed or fast numeric columns as `(field_ord, value)`.
    pub numeric: Vec<(u16, NumericValue)>,
    /// Stored-only columns as `(field_ord, bytes)`.
    pub stored: Vec<(u16, Vec<u8>)>,
}

/// Domain separator for [`SchemaDocument`] content witnesses.
///
/// Distinct from [`CONTENT_HASH_DOMAIN`] so that a five-field document and a
/// schema document sharing an id can never collide into a false "unchanged"
/// verdict if both shapes are written to one index.
const SCHEMA_CONTENT_HASH_DOMAIN: &[u8] = b"frankensearch.quill.idmap-content.schema.v1\0";

/// One ingest batch, in either supported document shape.
///
/// `Copy` so it substitutes for the `&[IndexableDocument]` it replaced without
/// disturbing the ingest routes' borrow structure.
#[derive(Debug, Clone, Copy)]
pub(super) enum IngestBatch<'a> {
    Default(&'a [IndexableDocument]),
    Schema(&'a [SchemaDocument]),
}

/// One document from an [`IngestBatch`].
#[derive(Debug, Clone, Copy)]
pub(super) enum IngestDoc<'a> {
    Default(&'a IndexableDocument),
    Schema(&'a SchemaDocument),
}

impl<'a> IngestBatch<'a> {
    pub(super) fn len(self) -> usize {
        match self {
            Self::Default(documents) => documents.len(),
            Self::Schema(documents) => documents.len(),
        }
    }

    pub(super) fn is_empty(self) -> bool {
        self.len() == 0
    }

    pub(super) fn get(self, index: usize) -> Option<IngestDoc<'a>> {
        match self {
            Self::Default(documents) => documents.get(index).map(IngestDoc::Default),
            Self::Schema(documents) => documents.get(index).map(IngestDoc::Schema),
        }
    }

    /// Sub-batch over `range`, used to hand each shard its fan-out span.
    ///
    /// `None` when the range is outside the batch, so a bad shard assignment
    /// surfaces as a typed refusal rather than a slice panic.
    pub(super) fn try_slice(self, range: std::ops::Range<usize>) -> Option<Self> {
        match self {
            Self::Default(documents) => documents.get(range).map(Self::Default),
            Self::Schema(documents) => documents.get(range).map(Self::Schema),
        }
    }

    pub(super) fn iter(self) -> IngestBatchIter<'a> {
        IngestBatchIter {
            batch: self,
            next: 0,
        }
    }
}

/// An owned ingest batch.
///
/// The facade materializes one only when its input carries repeated ids and a
/// last-write-wins canonicalization has to be handed to the writer as a
/// contiguous batch.
pub(super) enum OwnedIngestBatch {
    Default(Vec<IndexableDocument>),
    Schema(Vec<SchemaDocument>),
}

impl OwnedIngestBatch {
    /// An empty batch of the same shape as `batch`, with `capacity` reserved.
    pub(super) fn with_capacity_like(
        batch: IngestBatch<'_>,
        capacity: usize,
    ) -> Result<Self, QuillIndexError> {
        let reserve_failed = || invalid_state("could not reserve canonical upsert batch");
        match batch {
            IngestBatch::Default(_) => {
                let mut documents = Vec::new();
                documents
                    .try_reserve_exact(capacity)
                    .map_err(|_| reserve_failed())?;
                Ok(Self::Default(documents))
            }
            IngestBatch::Schema(_) => {
                let mut documents = Vec::new();
                documents
                    .try_reserve_exact(capacity)
                    .map_err(|_| reserve_failed())?;
                Ok(Self::Schema(documents))
            }
        }
    }

    /// Append `document`, which must share this batch's shape.
    pub(super) fn push(&mut self, document: IngestDoc<'_>) -> Result<(), QuillIndexError> {
        match (self, document) {
            (Self::Default(batch), IngestDoc::Default(document)) => {
                batch.push(document.clone());
                Ok(())
            }
            (Self::Schema(batch), IngestDoc::Schema(document)) => {
                batch.push(document.clone());
                Ok(())
            }
            _ => Err(invalid_state(
                "canonical upsert batch received a document of another shape",
            )),
        }
    }

    pub(super) fn as_batch(&self) -> IngestBatch<'_> {
        match self {
            Self::Default(documents) => IngestBatch::Default(documents),
            Self::Schema(documents) => IngestBatch::Schema(documents),
        }
    }
}

impl<'a> IntoIterator for IngestBatch<'a> {
    type Item = IngestDoc<'a>;
    type IntoIter = IngestBatchIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub(super) struct IngestBatchIter<'a> {
    batch: IngestBatch<'a>,
    next: usize,
}

impl<'a> Iterator for IngestBatchIter<'a> {
    type Item = IngestDoc<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let document = self.batch.get(self.next)?;
        self.next += 1;
        Some(document)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.batch.len().saturating_sub(self.next);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for IngestBatchIter<'_> {}

impl<'a> IngestDoc<'a> {
    /// Schema-independent identity used for dedup, tombstones, and IDMAP.
    pub(super) fn id(self) -> &'a str {
        match self {
            Self::Default(document) => document.id.as_str(),
            Self::Schema(document) => document.id.as_str(),
        }
    }

    /// Canonicalize this document's metadata into the caller's reused buffer.
    ///
    /// Schema documents carry their stored bytes explicitly, so they leave the
    /// buffer empty; the buffer is cleared either way so a caller reusing one
    /// allocation across a run never observes a previous document's bytes.
    pub(super) fn write_canonical_metadata(
        self,
        out: &mut Vec<u8>,
    ) -> Result<(), serde_json::Error> {
        out.clear();
        match self {
            Self::Default(document) => write_canonical_metadata(out, &document.metadata),
            Self::Schema(_) => Ok(()),
        }
    }

    /// Project onto the index schema and hand the values to Scribe.
    pub(super) fn accumulate(
        self,
        accumulator: &mut ColumnarAccumulator,
        doc_ord: u32,
        global_docid: u64,
        canonical_metadata: &[u8],
    ) -> Result<DocumentAccumulation, AccumulatorError> {
        match self {
            Self::Default(document) => {
                let values = default_schema_values(document, global_docid, canonical_metadata);
                accumulator.add_document_with_values(
                    doc_ord,
                    &values.indexed,
                    &values.numeric,
                    &values.stored,
                )
            }
            Self::Schema(document) => {
                let indexed = document
                    .indexed
                    .iter()
                    .map(|(field_ord, text)| IndexedFieldValue::new(*field_ord, text))
                    .collect::<Vec<_>>();
                let numeric = document
                    .numeric
                    .iter()
                    .map(|(field_ord, value)| IndexedNumericValue {
                        field_ord: *field_ord,
                        value: *value,
                    })
                    .collect::<Vec<_>>();
                let stored = document
                    .stored
                    .iter()
                    .map(|(field_ord, bytes)| StoredFieldValue::new(*field_ord, bytes))
                    .collect::<Vec<_>>();
                accumulator.add_document_with_values(doc_ord, &indexed, &numeric, &stored)
            }
        }
    }

    /// The exact IDMAP content witness Quill persists for this document.
    pub(super) fn content_hash(self, canonical_metadata: &[u8]) -> Result<u64, QuillIndexError> {
        match self {
            Self::Default(document) => {
                canonical_document_content_hash(document, canonical_metadata)
            }
            Self::Schema(document) => {
                let mut hasher = Xxh3::new();
                hasher.update(SCHEMA_CONTENT_HASH_DOMAIN);
                schema_witness_field(&mut hasher, u16::MAX, document.id.as_bytes())?;
                for (field_ord, text) in &document.indexed {
                    schema_witness_field(&mut hasher, *field_ord, text.as_bytes())?;
                }
                for (field_ord, value) in &document.numeric {
                    let bytes = match value {
                        NumericValue::U64(value) => value.to_le_bytes(),
                        NumericValue::I64(value) => value.to_le_bytes(),
                    };
                    schema_witness_field(&mut hasher, *field_ord, &bytes)?;
                }
                for (field_ord, bytes) in &document.stored {
                    schema_witness_field(&mut hasher, *field_ord, bytes)?;
                }
                Ok(hasher.digest())
            }
        }
    }
}

/// Absorb one length- and ordinal-tagged field into a schema content witness.
fn schema_witness_field(
    hasher: &mut Xxh3,
    field_ord: u16,
    bytes: &[u8],
) -> Result<(), QuillIndexError> {
    let len = u64::try_from(bytes.len())
        .map_err(|_| invalid_state("schema document field length does not fit u64"))?;
    hasher.update(&field_ord.to_le_bytes());
    hasher.update(&len.to_le_bytes());
    hasher.update(bytes);
    Ok(())
}

/// One document's borrowed FSLX values under [`crate::schema::DEFAULT_SCHEMA`].
struct DefaultSchemaValues<'a> {
    pub(super) indexed: [IndexedFieldValue<'a>; 3],
    pub(super) numeric: [IndexedNumericValue; 1],
    pub(super) stored: [StoredFieldValue<'a>; 1],
}

/// Project one [`IndexableDocument`] onto [`crate::schema::DEFAULT_SCHEMA`].
///
/// Every ingest route — serial, sharded fan-out, and the parallel accumulator
/// helper — reaches Scribe through this one projection, so the routes cannot
/// drift apart in which fields they populate or how an absent title is spelled.
/// `metadata` is the already-canonicalized JSON; the caller owns that buffer so
/// the shard paths can reuse a single allocation per shard.
fn default_schema_values<'a>(
    document: &'a IndexableDocument,
    global_docid: u64,
    metadata: &'a [u8],
) -> DefaultSchemaValues<'a> {
    DefaultSchemaValues {
        indexed: [
            IndexedFieldValue::new(ID_FIELD, &document.id),
            IndexedFieldValue::new(CONTENT_FIELD, &document.content),
            IndexedFieldValue::new(TITLE_FIELD, document.title.as_deref().unwrap_or("")),
        ],
        numeric: [IndexedNumericValue::u64(ORD_FIELD, global_docid)],
        stored: [StoredFieldValue::new(METADATA_FIELD, metadata)],
    }
}

/// Append canonical metadata JSON to `out`.
///
/// `serde_json::to_vec` is exactly `to_writer` into a fresh `Vec`, so this
/// emits the same bytes the allocating form did; writing into a caller-owned
/// buffer is what lets the ingest path reuse one allocation per shard.
pub(super) fn write_canonical_metadata(
    out: &mut Vec<u8>,
    metadata: &std::collections::HashMap<String, String>,
) -> Result<(), serde_json::Error> {
    let ordered = metadata.iter().collect::<BTreeMap<_, _>>();
    serde_json::to_writer(out, &ordered)
}

pub(super) fn canonical_metadata(
    metadata: &std::collections::HashMap<String, String>,
) -> Result<Vec<u8>, serde_json::Error> {
    let mut out = Vec::new();
    write_canonical_metadata(&mut out, metadata)?;
    Ok(out)
}

pub(super) fn canonical_document_content_hash(
    document: &IndexableDocument,
    metadata: &[u8],
) -> Result<u64, QuillIndexError> {
    let mut hasher = Xxh3::new();
    hasher.update(CONTENT_HASH_DOMAIN);
    for field in [
        document.id.as_bytes(),
        document.content.as_bytes(),
        document.title.as_deref().unwrap_or("").as_bytes(),
        metadata,
    ] {
        let len = u64::try_from(field.len())
            .map_err(|_| invalid_state("canonical document field length does not fit u64"))?;
        hasher.update(&len.to_le_bytes());
        hasher.update(field);
    }
    Ok(hasher.digest())
}

/// Compute the exact IDMAP content witness Quill will persist for a document.
///
/// # Errors
///
/// Returns a JSON error if canonical metadata serialization fails.
pub fn indexable_document_content_hash(
    document: &IndexableDocument,
) -> Result<u64, QuillIndexError> {
    let metadata = canonical_metadata(&document.metadata)?;
    canonical_document_content_hash(document, &metadata)
}

pub(super) fn manifest_segment(encoded: &EncodedSegment, seal_seq: u64) -> ManifestSegment {
    let header = encoded.header();
    ManifestSegment {
        segment_id: header.segment_id,
        seal_seq,
        file_len: encoded.file_len(),
        file_xxh3: encoded.file_xxh3(),
        docid_lo: header.docid_lo,
        docid_hi: header.docid_hi,
        doc_count: header.doc_count,
        tombstones: TombstoneSet::new(),
    }
}

pub(super) fn merge_field_stats(
    committed: &[ManifestFieldStats],
    pending: &BTreeMap<u16, (u64, u32)>,
) -> Result<Vec<ManifestFieldStats>, QuillIndexError> {
    let mut merged = committed
        .iter()
        .map(|row| (row.field_ord, (row.total_tokens, row.doc_count)))
        .collect::<BTreeMap<_, _>>();
    for (&field_ord, &(tokens, documents)) in pending {
        let row = merged.entry(field_ord).or_insert((0, 0));
        row.0 = row
            .0
            .checked_add(tokens)
            .ok_or_else(|| invalid_state("manifest field token count overflow"))?;
        row.1 = row
            .1
            .checked_add(documents)
            .ok_or_else(|| invalid_state("manifest field document count overflow"))?;
    }
    Ok(merged
        .into_iter()
        .map(
            |(field_ord, (total_tokens, doc_count))| ManifestFieldStats {
                field_ord,
                total_tokens,
                doc_count,
            },
        )
        .collect())
}
