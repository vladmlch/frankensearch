use thiserror::Error;

use crate::contract::fieldnorm_to_id;
use crate::error::QuillError;
use crate::grimoire::MAX_TERM_BYTES;
use crate::quiver::NumericValue;
use crate::schema::{Analyzer as AnalyzerKind, FieldKind, SchemaDescriptor};

use super::analyzer::{TokenAnalyzer, analyze_admitted_borrowed};
use super::cass::CompiledAnalyzerFamily;
use super::interner::TermInterner;
use super::{DEFAULT_ARENA_CHUNK_BYTES, DOC_ORDS_PER_LEASE};

/// One indexed string value supplied for a document.
///
/// Values may name only [`FieldKind::Keyword`] or [`FieldKind::Text`] fields.
/// Omitted indexed string fields receive a zero document length and fieldnorm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexedFieldValue<'a> {
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Borrowed source text.
    pub text: &'a str,
}

impl<'a> IndexedFieldValue<'a> {
    /// Construct a borrowed indexed field value.
    #[must_use]
    pub const fn new(field_ord: u16, text: &'a str) -> Self {
        Self { field_ord, text }
    }
}

/// One opaque stored-field value supplied for a document.
///
/// Values are retained byte-for-byte for FSLX STOREDMETA. Indexed string
/// fields marked `stored` normally derive these bytes directly from their
/// [`IndexedFieldValue`]; this explicit form supplies stored-only and numeric
/// fields, or a stored indexed field that is intentionally not indexed for one
/// document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoredFieldValue<'a> {
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Borrowed opaque bytes.
    pub bytes: &'a [u8],
}

impl<'a> StoredFieldValue<'a> {
    /// Construct a borrowed stored-field value.
    #[must_use]
    pub const fn new(field_ord: u16, bytes: &'a [u8]) -> Self {
        Self { field_ord, bytes }
    }
}

/// One schema-typed indexed-or-fast numeric value supplied for a document.
///
/// Numeric values are retained in a dedicated typed column for NUMERIC. When
/// the descriptor also sets `stored=true`, Scribe writes the same canonical
/// eight little-endian bytes into STOREDMETA automatically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexedNumericValue {
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Signed or unsigned value matching the field descriptor.
    pub value: NumericValue,
}

impl IndexedNumericValue {
    /// Construct one signed indexed value.
    #[must_use]
    pub const fn i64(field_ord: u16, value: i64) -> Self {
        Self {
            field_ord,
            value: NumericValue::I64(value),
        }
    }

    /// Construct one unsigned indexed value.
    #[must_use]
    pub const fn u64(field_ord: u16, value: u64) -> Self {
        Self {
            field_ord,
            value: NumericValue::U64(value),
        }
    }
}

/// Typed validation failures from [`ColumnarAccumulator::add_document`].
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum AccumulatorError {
    /// A lease-relative ordinal cannot name a document in another lease.
    #[error(
        "document ordinal {doc_ord} is outside the lease-relative range 0..{DOC_ORDS_PER_LEASE}"
    )]
    DocumentOutsideLease {
        /// Rejected lease-relative ordinal.
        doc_ord: u32,
    },
    /// Document ordinals must ascend within one shard lease.
    #[error("document ordinal {current} is not greater than prior ordinal {previous}")]
    OutOfOrderDocument {
        /// Last completed document ordinal.
        previous: u32,
        /// Rejected ordinal.
        current: u32,
    },
    /// More documents were supplied than the public u32 counter can express.
    #[error("columnar accumulator document count exceeds u32")]
    TooManyDocuments,
    /// A value named no field in the validated dense descriptor.
    #[error("unknown schema field ordinal {field_ord}")]
    UnknownField {
        /// Rejected field ordinal.
        field_ord: u16,
    },
    /// A document supplied the same indexed field more than once.
    #[error("document {doc_ord} supplies field {field_ord} more than once")]
    DuplicateField {
        /// Current shard-relative document ordinal.
        doc_ord: u32,
        /// Repeated field ordinal.
        field_ord: u16,
    },
    /// A string value named a numeric or stored-only field.
    #[error("field {field_ord} ({field_name}) is not an indexed string field")]
    NonStringField {
        /// Rejected field ordinal.
        field_ord: u16,
        /// Stable schema field name.
        field_name: &'static str,
    },
    /// A typed numeric value named a string or stored-only field.
    #[error("field {field_ord} ({field_name}) is not numeric")]
    NonNumericField {
        /// Rejected field ordinal.
        field_ord: u16,
        /// Stable schema field name.
        field_name: &'static str,
    },
    /// A typed numeric value named a field with no NUMERIC column.
    #[error("numeric field {field_ord} ({field_name}) is neither indexed nor fast")]
    NonIndexedNumericField {
        /// Rejected field ordinal.
        field_ord: u16,
        /// Stable schema field name.
        field_name: &'static str,
    },
    /// An in-memory numeric tag disagreed with the schema descriptor.
    #[error("numeric field {field_ord} ({field_name}) expects {expected}, got {actual}")]
    NumericTypeMismatch {
        /// Rejected field ordinal.
        field_ord: u16,
        /// Stable schema field name.
        field_name: &'static str,
        /// Schema type name.
        expected: &'static str,
        /// Supplied type name.
        actual: &'static str,
    },
    /// Opaque bytes used as a numeric value must be exactly one scalar wide.
    #[error(
        "numeric field {field_ord} ({field_name}) has {actual} stored bytes, expected {expected}"
    )]
    InvalidNumericBytes {
        /// Rejected field ordinal.
        field_ord: u16,
        /// Stable schema field name.
        field_name: &'static str,
        /// Required byte width.
        expected: usize,
        /// Supplied byte width.
        actual: usize,
    },
    /// The schema requested an analyzer that has not landed in Scribe yet.
    #[error("field {field_ord} ({field_name}) requires unsupported analyzer {analyzer:?}")]
    UnsupportedAnalyzer {
        /// Rejected field ordinal.
        field_ord: u16,
        /// Stable schema field name.
        field_name: &'static str,
        /// Requested schema pipeline.
        analyzer: AnalyzerKind,
    },
    /// Source length exceeded the u32 position proof used by the accumulator.
    #[error("field {field_ord} source is {bytes} bytes and exceeds the u32 position domain")]
    SourceTooLarge {
        /// Rejected field ordinal.
        field_ord: u16,
        /// Source byte length.
        bytes: usize,
    },
    /// A value named a field whose descriptor does not store original bytes.
    #[error("field {field_ord} ({field_name}) is not stored")]
    NonStoredField {
        /// Rejected field ordinal.
        field_ord: u16,
        /// Stable schema field name.
        field_name: &'static str,
    },
    /// One stored value cannot be represented by the durable u32 offsets.
    #[error("stored field {field_ord} value is {bytes} bytes and exceeds the u32 offset domain")]
    StoredValueTooLarge {
        /// Rejected field ordinal.
        field_ord: u16,
        /// Rejected byte length.
        bytes: usize,
    },
    /// Appending a value would make one field blob exceed the durable u32 domain.
    #[error(
        "stored field {field_ord} blob cannot append {appended} bytes to its current {current} bytes"
    )]
    StoredBlobTooLarge {
        /// Rejected field ordinal.
        field_ord: u16,
        /// Current accumulated blob length.
        current: usize,
        /// Candidate value length.
        appended: usize,
    },
}

/// Completion report for one atomically accumulated document.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DocumentAccumulation {
    /// Admitted indexed token occurrences across all supplied fields.
    pub admitted_tokens: u64,
    /// Tokens dropped by the global 65,530-byte admission rule.
    pub oversized_tokens: u64,
    /// Complete retained accumulator footprint after the document.
    pub bytes_reserved: usize,
    /// Live logical accumulator footprint after the document.
    pub bytes_used: usize,
}

/// Per-field structure-of-arrays token and document-length columns.
///
/// `term_ids`, `doc_ords`, and (when present) `positions` are parallel. The
/// document-length and fieldnorm columns align with
/// [`ColumnarAccumulator::document_ords`] rather than with token rows.
#[derive(Debug)]
pub struct FieldTokenColumns {
    field_ord: u16,
    term_ids: Vec<u32>,
    doc_ords: Vec<u32>,
    positions: Option<Vec<u32>>,
    document_lengths: Vec<u32>,
    fieldnorm_ids: Vec<u8>,
    total_tokens: u64,
}

impl FieldTokenColumns {
    fn new(field_ord: u16, positions: bool) -> Self {
        Self {
            field_ord,
            term_ids: Vec::new(),
            doc_ords: Vec::new(),
            positions: positions.then(Vec::new),
            document_lengths: Vec::new(),
            fieldnorm_ids: Vec::new(),
            total_tokens: 0,
        }
    }

    /// Stable schema field ordinal represented by these columns.
    #[must_use]
    pub const fn field_ord(&self) -> u16 {
        self.field_ord
    }

    /// Local composite term IDs in document/token order.
    #[must_use]
    pub fn term_ids(&self) -> &[u32] {
        &self.term_ids
    }

    /// Shard-lease-relative document ordinals parallel to [`Self::term_ids`].
    #[must_use]
    pub fn doc_ords(&self) -> &[u32] {
        &self.doc_ords
    }

    /// Positions parallel to token rows, or `None` when the schema does not
    /// persist positions for this field.
    #[must_use]
    pub fn positions(&self) -> Option<&[u32]> {
        self.positions.as_deref()
    }

    /// Exact admitted token count for each completed document.
    #[must_use]
    pub fn document_lengths(&self) -> &[u32] {
        &self.document_lengths
    }

    /// Tantivy-compatible quantized fieldnorm for each completed document.
    #[must_use]
    pub fn fieldnorm_ids(&self) -> &[u8] {
        &self.fieldnorm_ids
    }

    /// Exact token total used by FSLX STATS/avgdl.
    #[must_use]
    pub const fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Number of admitted token triples for this field.
    #[must_use]
    pub fn len(&self) -> usize {
        self.term_ids.len()
    }

    /// Whether this field has no token triples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.term_ids.is_empty()
    }

    fn append_token(&mut self, term_id: u32, doc_ord: u32, position: u32) {
        self.term_ids.push(term_id);
        self.doc_ords.push(doc_ord);
        if let Some(positions) = &mut self.positions {
            positions.push(position);
        }
    }

    fn begin_document(&mut self) {
        self.document_lengths.push(0);
        self.fieldnorm_ids.push(fieldnorm_to_id(0));
    }

    fn finish_document_field(&mut self, length: u32) {
        let raw = self
            .document_lengths
            .last_mut()
            .expect("begin_document creates the document-length slot");
        *raw = length;
        let fieldnorm = self
            .fieldnorm_ids
            .last_mut()
            .expect("begin_document creates the fieldnorm slot");
        *fieldnorm = fieldnorm_to_id(length);
        self.total_tokens += u64::from(length);
    }

    pub(super) fn bytes_reserved(&self) -> usize {
        let u32_bytes = std::mem::size_of::<u32>();
        self.term_ids
            .capacity()
            .saturating_mul(u32_bytes)
            .saturating_add(self.doc_ords.capacity().saturating_mul(u32_bytes))
            .saturating_add(self.positions.as_ref().map_or(0, |positions| {
                positions.capacity().saturating_mul(u32_bytes)
            }))
            .saturating_add(self.document_lengths.capacity().saturating_mul(u32_bytes))
            .saturating_add(self.fieldnorm_ids.capacity())
    }

    fn bytes_used(&self) -> usize {
        let u32_bytes = std::mem::size_of::<u32>();
        self.term_ids
            .len()
            .saturating_mul(u32_bytes)
            .saturating_add(self.doc_ords.len().saturating_mul(u32_bytes))
            .saturating_add(
                self.positions
                    .as_ref()
                    .map_or(0, |positions| positions.len().saturating_mul(u32_bytes)),
            )
            .saturating_add(self.document_lengths.len().saturating_mul(u32_bytes))
            .saturating_add(self.fieldnorm_ids.len())
    }

    fn reset(&mut self) {
        self.term_ids.clear();
        self.doc_ords.clear();
        if let Some(positions) = &mut self.positions {
            positions.clear();
        }
        self.document_lengths.clear();
        self.fieldnorm_ids.clear();
        self.total_tokens = 0;
    }
}

/// One schema-ordered column of opaque stored values.
///
/// Offsets align with completed documents rather than lease ordinals. A
/// separate presence byte preserves the distinction between an absent value
/// and a present empty byte string; the FSLX writer packs those bytes into the
/// STOREDMETA presence bitmap.
#[derive(Debug)]
pub struct StoredFieldColumns {
    field_ord: u16,
    offsets: Vec<u32>,
    present: Vec<u8>,
    blob: Vec<u8>,
}

impl StoredFieldColumns {
    fn new(field_ord: u16) -> Self {
        Self {
            field_ord,
            offsets: vec![0],
            present: Vec::new(),
            blob: Vec::new(),
        }
    }

    /// Stable schema field ordinal.
    #[must_use]
    pub const fn field_ord(&self) -> u16 {
        self.field_ord
    }

    /// Number of completed documents represented by this column.
    #[must_use]
    pub fn document_count(&self) -> usize {
        self.present.len()
    }

    /// Monotone blob offsets, with exactly `document_count + 1` entries.
    #[must_use]
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// One canonical `0`/`1` presence byte per completed document.
    #[must_use]
    pub fn presence(&self) -> &[u8] {
        &self.present
    }

    /// Concatenated opaque value bytes.
    #[must_use]
    pub fn blob(&self) -> &[u8] {
        &self.blob
    }

    /// Borrow one value by completed-document index.
    ///
    /// `None` means the document index is out of range or the field was
    /// absent. `Some(&[])` is a present empty value.
    #[must_use]
    pub fn value(&self, document_index: usize) -> Option<&[u8]> {
        if self.present.get(document_index).copied()? == 0 {
            return None;
        }
        let start = usize::try_from(*self.offsets.get(document_index)?).ok()?;
        let end = usize::try_from(*self.offsets.get(document_index + 1)?).ok()?;
        self.blob.get(start..end)
    }

    fn can_append_len(&self, value_len: Option<usize>) -> bool {
        value_len.is_none_or(|bytes| {
            self.blob
                .len()
                .checked_add(bytes)
                .is_some_and(|len| u32::try_from(len).is_ok())
        })
    }

    fn append_document(&mut self, value: Option<&[u8]>) {
        self.present.push(u8::from(value.is_some()));
        if let Some(bytes) = value {
            self.blob.extend_from_slice(bytes);
        }
        self.offsets.push(
            u32::try_from(self.blob.len())
                .expect("stored blob append was validated before accumulator mutation"),
        );
    }

    pub(super) fn bytes_reserved(&self) -> usize {
        self.offsets
            .capacity()
            .saturating_mul(std::mem::size_of::<u32>())
            .saturating_add(self.present.capacity())
            .saturating_add(self.blob.capacity())
    }

    fn bytes_used(&self) -> usize {
        self.offsets
            .len()
            .saturating_mul(std::mem::size_of::<u32>())
            .saturating_add(self.present.len())
            .saturating_add(self.blob.len())
    }

    fn reset(&mut self) {
        self.offsets.clear();
        self.offsets.push(0);
        self.present.clear();
        self.blob.clear();
    }
}

/// One schema-ordered indexed-or-fast numeric column.
///
/// Values align with completed documents, not sparse lease ordinals. `None`
/// represents an absent optional value or a segment-range hole introduced when
/// Scribe seals sparse local ordinals.
#[derive(Debug)]
pub struct NumericFieldColumns {
    field_ord: u16,
    values: Vec<Option<NumericValue>>,
}

impl NumericFieldColumns {
    fn new(field_ord: u16) -> Self {
        Self {
            field_ord,
            values: Vec::new(),
        }
    }

    /// Stable schema field ordinal.
    #[must_use]
    pub const fn field_ord(&self) -> u16 {
        self.field_ord
    }

    /// One optional typed value per completed document.
    #[must_use]
    pub fn values(&self) -> &[Option<NumericValue>] {
        &self.values
    }

    fn append_document(&mut self, value: Option<NumericValue>) {
        self.values.push(value);
    }

    pub(super) fn bytes_reserved(&self) -> usize {
        self.values
            .capacity()
            .saturating_mul(std::mem::size_of::<Option<NumericValue>>())
    }

    fn bytes_used(&self) -> usize {
        self.values
            .len()
            .saturating_mul(std::mem::size_of::<Option<NumericValue>>())
    }

    fn reset(&mut self) {
        self.values.clear();
    }
}

/// Schema-driven shard-local columnar token accumulator.
///
/// A complete document is validated before any column is changed. Indexed
/// fields are then processed in schema order (not caller order), making local
/// term-ID assignment deterministic. The supplied `doc_ord` remains relative
/// to its Keeper lease until the flush/seal stage rebases it.
#[derive(Debug)]
pub struct ColumnarAccumulator<A = CompiledAnalyzerFamily> {
    schema: SchemaDescriptor,
    pub(super) terms: TermInterner,
    pub(super) fields: Vec<FieldTokenColumns>,
    pub(super) numeric_fields: Vec<NumericFieldColumns>,
    pub(super) stored_fields: Vec<StoredFieldColumns>,
    pub(super) document_ords: Vec<u32>,
    pub(super) seen_fields: Vec<bool>,
    last_doc_ord: Option<u32>,
    pub(super) analyzer: A,
}

impl ColumnarAccumulator<CompiledAnalyzerFamily> {
    /// Create an empty accumulator for a validated compile-time schema.
    ///
    /// # Errors
    ///
    /// Returns [`QuillError::InvalidDescriptor`] when `schema` violates the
    /// dense-ID and field-shape invariants, or [`QuillError::Resource`] when
    /// the default family does not implement every requested analyzer.
    pub fn new(schema: SchemaDescriptor) -> Result<Self, QuillError> {
        Self::with_analyzer(schema, CompiledAnalyzerFamily::default())
    }

    /// Create an empty accumulator whose term arena is sized for one bounded
    /// shared-nothing worker rather than the scalar one-mebibyte default.
    ///
    /// Logical accounting and flush boundaries are independent of arena
    /// capacity. This constructor lets the caller share scalar arena slack
    /// across a wide parallel generation, subject to `ByteArena`'s 4 KiB
    /// per-arena floor.
    pub(crate) fn with_arena_chunk_size(
        schema: SchemaDescriptor,
        arena_chunk_size: usize,
    ) -> Result<Self, QuillError> {
        Self::with_analyzer_and_arena_chunk_size(
            schema,
            CompiledAnalyzerFamily::default(),
            arena_chunk_size,
        )
    }
}

impl<A: TokenAnalyzer> ColumnarAccumulator<A> {
    /// Create an empty accumulator with an injected analyzer implementation.
    ///
    /// This is the stable scalar/SIMD swap seam. One family may advertise
    /// several [`AnalyzerKind`] values, and implementations of a given kind
    /// must emit the same token stream. A schema field requesting a missing
    /// kind is rejected atomically at construction.
    ///
    /// # Errors
    ///
    /// Returns [`QuillError::InvalidDescriptor`] when `schema` violates the
    /// dense-ID and field-shape invariants, or [`QuillError::Resource`] when
    /// the injected family does not implement every requested analyzer.
    pub fn with_analyzer(schema: SchemaDescriptor, analyzer: A) -> Result<Self, QuillError> {
        Self::with_analyzer_and_arena_chunk_size(schema, analyzer, DEFAULT_ARENA_CHUNK_BYTES)
    }

    fn with_analyzer_and_arena_chunk_size(
        schema: SchemaDescriptor,
        analyzer: A,
        arena_chunk_size: usize,
    ) -> Result<Self, QuillError> {
        schema.validate()?;
        if let Some((field, requested)) = schema.fields.iter().find_map(|field| {
            let FieldKind::Text {
                analyzer: requested,
                ..
            } = field.kind
            else {
                return None;
            };
            (!analyzer.supports(requested)).then_some((field, requested))
        }) {
            return Err(QuillError::Resource {
                resource: "analyzer pipeline",
                detail: format!(
                    "field {} ({}) requests unavailable analyzer {requested:?}",
                    field.id, field.name
                ),
            });
        }
        let fields = schema
            .fields
            .iter()
            .filter_map(|field| match field.kind {
                FieldKind::Keyword => Some(FieldTokenColumns::new(field.id, false)),
                FieldKind::Text { positions, .. } => {
                    Some(FieldTokenColumns::new(field.id, positions))
                }
                FieldKind::StoredOnly | FieldKind::I64 { .. } | FieldKind::U64 { .. } => None,
            })
            .collect();
        let numeric_fields = schema
            .fields
            .iter()
            .filter_map(|field| match field.kind {
                kind if kind.has_numeric_column() => Some(NumericFieldColumns::new(field.id)),
                _ => None,
            })
            .collect();
        let stored_fields = schema
            .fields
            .iter()
            .filter(|field| field.stored)
            .map(|field| StoredFieldColumns::new(field.id))
            .collect();
        Ok(Self {
            schema,
            terms: TermInterner::with_arena_chunk_size(arena_chunk_size),
            fields,
            numeric_fields,
            stored_fields,
            document_ords: Vec::new(),
            seen_fields: vec![false; schema.fields.len()],
            last_doc_ord: None,
            analyzer,
        })
    }

    /// Add one complete document.
    ///
    /// `values` may be in any order; Scribe processes them in schema order.
    /// Missing indexed string fields receive length/norm zero. A keyword uses
    /// raw exact bytes and emits one token even when empty. Analyzed text uses
    /// the descriptor's analyzer and the global term admission rule.
    ///
    /// # Errors
    ///
    /// Returns a typed validation error before mutating any column when the
    /// document ordinal, field set, field kind, analyzer, or source-size proof
    /// is invalid.
    pub fn add_document(
        &mut self,
        doc_ord: u32,
        values: &[IndexedFieldValue<'_>],
    ) -> Result<DocumentAccumulation, AccumulatorError> {
        self.add_document_with_values(doc_ord, values, &[], &[])
    }

    /// Add one complete document with explicit opaque stored values.
    ///
    /// Indexed string values whose descriptors set `stored=true` are retained
    /// automatically. `stored_values` supplies stored-only/numeric fields, or
    /// a stored indexed field omitted from `values`. Naming the same field in
    /// both slices is rejected rather than admitting divergent indexed and
    /// stored bytes.
    ///
    /// # Errors
    ///
    /// Returns a typed validation error before mutating any column when the
    /// document ordinal, either field set, field shape, analyzer, source-size
    /// proof, or durable stored-blob bound is invalid.
    pub fn add_document_with_stored(
        &mut self,
        doc_ord: u32,
        values: &[IndexedFieldValue<'_>],
        stored_values: &[StoredFieldValue<'_>],
    ) -> Result<DocumentAccumulation, AccumulatorError> {
        self.add_document_with_values(doc_ord, values, &[], stored_values)
    }

    /// Add one complete document with typed indexed-or-fast numeric values.
    ///
    /// Numeric fields marked `stored` automatically retain the exact canonical
    /// little-endian scalar bytes. Missing numeric fields remain absent.
    ///
    /// # Errors
    ///
    /// Returns a typed validation error before mutating any column when the
    /// document ordinal, field set, numeric type, analyzer, or durable bounds
    /// are invalid.
    pub fn add_document_with_numeric(
        &mut self,
        doc_ord: u32,
        values: &[IndexedFieldValue<'_>],
        numeric_values: &[IndexedNumericValue],
    ) -> Result<DocumentAccumulation, AccumulatorError> {
        self.add_document_with_values(doc_ord, values, numeric_values, &[])
    }

    /// Add one complete document with typed numeric and opaque stored values.
    ///
    /// A field may appear in only one input slice. For a stored
    /// indexed-or-fast numeric field, prefer `numeric_values`: Scribe derives
    /// its canonical stored bytes automatically. `stored_values` remains
    /// accepted for such a field when the bytes are exactly one schema-typed
    /// little-endian scalar.
    ///
    /// # Errors
    ///
    /// Returns a typed validation error before mutating any column when the
    /// document ordinal, any field set, numeric type/width, analyzer, source
    /// size, or durable stored-blob bound is invalid.
    pub fn add_document_with_values(
        &mut self,
        doc_ord: u32,
        values: &[IndexedFieldValue<'_>],
        numeric_values: &[IndexedNumericValue],
        stored_values: &[StoredFieldValue<'_>],
    ) -> Result<DocumentAccumulation, AccumulatorError> {
        let accumulate_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::SCRIBE_ACCUMULATE,
            phase = "accumulate",
            doc_count = self.document_ords.len(),
            doc_ord,
            result_count = tracing::field::Empty,
            admitted_tokens = tracing::field::Empty,
            oversized_tokens = tracing::field::Empty,
            token_count = tracing::field::Empty,
            arena_bytes_used_high_water = tracing::field::Empty,
            arena_bytes_reserved_high_water = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _accumulate_timer = crate::tracing_conventions::StageTimer::new(&accumulate_span);
        let _accumulate_entered = accumulate_span.enter();
        if doc_ord >= DOC_ORDS_PER_LEASE {
            return Err(AccumulatorError::DocumentOutsideLease { doc_ord });
        }
        if let Some(previous) = self.last_doc_ord
            && doc_ord <= previous
        {
            return Err(AccumulatorError::OutOfOrderDocument {
                previous,
                current: doc_ord,
            });
        }
        if self.document_ords.len() == u32::MAX as usize {
            return Err(AccumulatorError::TooManyDocuments);
        }

        self.seen_fields.fill(false);
        for value in values {
            let field_index = usize::from(value.field_ord);
            let Some(field) = self.schema.fields.get(field_index) else {
                return Err(AccumulatorError::UnknownField {
                    field_ord: value.field_ord,
                });
            };
            if field.id != value.field_ord {
                return Err(AccumulatorError::UnknownField {
                    field_ord: value.field_ord,
                });
            }
            if std::mem::replace(&mut self.seen_fields[field_index], true) {
                return Err(AccumulatorError::DuplicateField {
                    doc_ord,
                    field_ord: value.field_ord,
                });
            }
            match field.kind {
                FieldKind::Keyword => {}
                FieldKind::Text { analyzer, .. } if self.analyzer.supports(analyzer) => {}
                FieldKind::Text { analyzer, .. } => {
                    return Err(AccumulatorError::UnsupportedAnalyzer {
                        field_ord: field.id,
                        field_name: field.name,
                        analyzer,
                    });
                }
                FieldKind::StoredOnly | FieldKind::I64 { .. } | FieldKind::U64 { .. } => {
                    return Err(AccumulatorError::NonStringField {
                        field_ord: field.id,
                        field_name: field.name,
                    });
                }
            }
            if u32::try_from(value.text.len()).is_err() {
                return Err(AccumulatorError::SourceTooLarge {
                    field_ord: field.id,
                    bytes: value.text.len(),
                });
            }
        }

        for value in numeric_values {
            let field_index = usize::from(value.field_ord);
            let Some(field) = self.schema.fields.get(field_index) else {
                return Err(AccumulatorError::UnknownField {
                    field_ord: value.field_ord,
                });
            };
            if field.id != value.field_ord {
                return Err(AccumulatorError::UnknownField {
                    field_ord: value.field_ord,
                });
            }
            if std::mem::replace(&mut self.seen_fields[field_index], true) {
                return Err(AccumulatorError::DuplicateField {
                    doc_ord,
                    field_ord: value.field_ord,
                });
            }
            match (field.kind, value.value) {
                (FieldKind::I64 { indexed, fast }, NumericValue::I64(_)) if indexed || fast => {}
                (FieldKind::U64 { indexed, fast }, NumericValue::U64(_)) if indexed || fast => {}
                (
                    FieldKind::I64 {
                        indexed: false,
                        fast: false,
                    }
                    | FieldKind::U64 {
                        indexed: false,
                        fast: false,
                    },
                    _,
                ) => {
                    return Err(AccumulatorError::NonIndexedNumericField {
                        field_ord: field.id,
                        field_name: field.name,
                    });
                }
                (FieldKind::I64 { .. }, actual) => {
                    return Err(AccumulatorError::NumericTypeMismatch {
                        field_ord: field.id,
                        field_name: field.name,
                        expected: "i64",
                        actual: numeric_value_type_name(actual),
                    });
                }
                (FieldKind::U64 { .. }, actual) => {
                    return Err(AccumulatorError::NumericTypeMismatch {
                        field_ord: field.id,
                        field_name: field.name,
                        expected: "u64",
                        actual: numeric_value_type_name(actual),
                    });
                }
                _ => {
                    return Err(AccumulatorError::NonNumericField {
                        field_ord: field.id,
                        field_name: field.name,
                    });
                }
            }
        }

        for value in stored_values {
            let field_index = usize::from(value.field_ord);
            let Some(field) = self.schema.fields.get(field_index) else {
                return Err(AccumulatorError::UnknownField {
                    field_ord: value.field_ord,
                });
            };
            if field.id != value.field_ord {
                return Err(AccumulatorError::UnknownField {
                    field_ord: value.field_ord,
                });
            }
            if std::mem::replace(&mut self.seen_fields[field_index], true) {
                return Err(AccumulatorError::DuplicateField {
                    doc_ord,
                    field_ord: value.field_ord,
                });
            }
            if !field.stored {
                return Err(AccumulatorError::NonStoredField {
                    field_ord: field.id,
                    field_name: field.name,
                });
            }
            if matches!(field.kind, FieldKind::I64 { .. } | FieldKind::U64 { .. })
                && value.bytes.len() != std::mem::size_of::<u64>()
            {
                return Err(AccumulatorError::InvalidNumericBytes {
                    field_ord: field.id,
                    field_name: field.name,
                    expected: std::mem::size_of::<u64>(),
                    actual: value.bytes.len(),
                });
            }
            if u32::try_from(value.bytes.len()).is_err() {
                return Err(AccumulatorError::StoredValueTooLarge {
                    field_ord: field.id,
                    bytes: value.bytes.len(),
                });
            }
        }

        for field in &self.stored_fields {
            let value_len = stored_values
                .iter()
                .find(|value| value.field_ord == field.field_ord)
                .map(|value| value.bytes.len())
                .or_else(|| {
                    values
                        .iter()
                        .find(|value| value.field_ord == field.field_ord)
                        .map(|value| value.text.len())
                })
                .or_else(|| {
                    numeric_values
                        .iter()
                        .find(|value| value.field_ord == field.field_ord)
                        .map(|_| std::mem::size_of::<u64>())
                });
            if !field.can_append_len(value_len) {
                return Err(AccumulatorError::StoredBlobTooLarge {
                    field_ord: field.field_ord,
                    current: field.blob.len(),
                    appended: value_len.unwrap_or(0),
                });
            }
        }

        for field in &mut self.fields {
            field.begin_document();
        }

        let mut admitted_tokens = 0_u64;
        let mut oversized_tokens = 0_u64;
        for field_index in 0..self.fields.len() {
            let field_ord = self.fields[field_index].field_ord;
            let Some(value) = values.iter().find(|value| value.field_ord == field_ord) else {
                continue;
            };
            let descriptor = self.schema.fields[usize::from(field_ord)];
            let (length, dropped) = match descriptor.kind {
                FieldKind::Keyword => {
                    if value.text.len() > MAX_TERM_BYTES {
                        tracing::warn!(
                            field_ord,
                            token_bytes = value.text.len(),
                            max_token_bytes = MAX_TERM_BYTES,
                            "Quill dropped an oversized keyword token"
                        );
                        (0, 1)
                    } else {
                        let term_id = self.terms.intern(field_ord, value.text.as_bytes());
                        self.fields[field_index].append_token(term_id, doc_ord, 0);
                        (1, 0)
                    }
                }
                FieldKind::Text { analyzer, .. } => {
                    debug_assert!(self.analyzer.supports(analyzer));
                    let terms = &mut self.terms;
                    let column = &mut self.fields[field_index];
                    let report = {
                        let tokenize_span = tracing::info_span!(
                            target: crate::tracing_conventions::TARGET,
                            crate::tracing_conventions::SCRIBE_TOKENIZE,
                            phase = "tokenize",
                            field_ord,
                            source_bytes = value.text.len(),
                            result_count = tracing::field::Empty,
                            oversized_tokens = tracing::field::Empty,
                            analyzer_bytes_reserved = tracing::field::Empty,
                            duration_us = tracing::field::Empty,
                        );
                        let _tokenize_timer =
                            crate::tracing_conventions::StageTimer::new(&tokenize_span);
                        let _tokenize_entered = tokenize_span.enter();
                        let report = analyze_admitted_borrowed(
                            &mut self.analyzer,
                            analyzer,
                            value.text,
                            &mut |token| {
                                let term_id = terms.intern(field_ord, token.text.as_bytes());
                                column.append_token(term_id, doc_ord, token.position);
                            },
                        )
                        .expect("document validation checked analyzer-family support");
                        tokenize_span.record(
                            "result_count",
                            u64::try_from(report.admitted_tokens).unwrap_or(u64::MAX),
                        );
                        tokenize_span.record(
                            "oversized_tokens",
                            u64::try_from(report.oversized_tokens).unwrap_or(u64::MAX),
                        );
                        tokenize_span.record(
                            "analyzer_bytes_reserved",
                            u64::try_from(self.analyzer.bytes_reserved()).unwrap_or(u64::MAX),
                        );
                        report
                    };
                    (
                        u32::try_from(report.admitted_tokens)
                            .expect("validated source length bounds admitted token count to u32"),
                        u64::try_from(report.oversized_tokens)
                            .expect("token count always fits u64"),
                    )
                }
                FieldKind::StoredOnly | FieldKind::I64 { .. } | FieldKind::U64 { .. } => {
                    unreachable!("constructor excludes non-string fields from token columns")
                }
            };
            self.fields[field_index].finish_document_field(length);
            admitted_tokens += u64::from(length);
            oversized_tokens += dropped;
        }

        let schema_fields = self.schema.fields;
        for field in &mut self.numeric_fields {
            let value = numeric_values
                .iter()
                .find(|value| value.field_ord == field.field_ord)
                .map(|value| value.value)
                .or_else(|| {
                    stored_values
                        .iter()
                        .find(|value| value.field_ord == field.field_ord)
                        .and_then(|value| {
                            numeric_value_from_le_bytes(
                                schema_fields[usize::from(field.field_ord)].kind,
                                value.bytes,
                            )
                        })
                });
            field.append_document(value);
        }

        for field in &mut self.stored_fields {
            if let Some(value) = numeric_values
                .iter()
                .find(|value| value.field_ord == field.field_ord)
            {
                let bytes = value.value.to_le_bytes();
                field.append_document(Some(&bytes));
                continue;
            }
            let value = stored_values
                .iter()
                .find(|value| value.field_ord == field.field_ord)
                .map(|value| value.bytes)
                .or_else(|| {
                    values
                        .iter()
                        .find(|value| value.field_ord == field.field_ord)
                        .map(|value| value.text.as_bytes())
                });
            field.append_document(value);
        }

        self.document_ords.push(doc_ord);
        self.last_doc_ord = Some(doc_ord);
        let arena_bytes_used = self.bytes_used();
        let arena_bytes_reserved = self.bytes_reserved();
        accumulate_span.record(
            "doc_count",
            u64::try_from(self.document_ords.len()).unwrap_or(u64::MAX),
        );
        accumulate_span.record("result_count", 1_u64);
        accumulate_span.record("admitted_tokens", admitted_tokens);
        accumulate_span.record("oversized_tokens", oversized_tokens);
        accumulate_span.record(
            "token_count",
            u64::try_from(self.token_count()).unwrap_or(u64::MAX),
        );
        accumulate_span.record(
            "arena_bytes_used_high_water",
            u64::try_from(arena_bytes_used).unwrap_or(u64::MAX),
        );
        accumulate_span.record(
            "arena_bytes_reserved_high_water",
            u64::try_from(arena_bytes_reserved).unwrap_or(u64::MAX),
        );
        tracing::info!(
            target: crate::tracing_conventions::TARGET,
            phase = "accumulate.complete",
            doc_count = self.document_ords.len(),
            admitted_tokens,
            oversized_tokens,
            arena_bytes_used,
            arena_bytes_reserved,
            "scalar document accumulated"
        );
        Ok(DocumentAccumulation {
            admitted_tokens,
            oversized_tokens,
            bytes_reserved: arena_bytes_reserved,
            bytes_used: arena_bytes_used,
        })
    }

    /// Validated schema associated with the accumulator.
    #[must_use]
    pub const fn schema(&self) -> SchemaDescriptor {
        self.schema
    }

    /// Composite field-namespaced term interner.
    #[must_use]
    pub const fn terms(&self) -> &TermInterner {
        &self.terms
    }

    /// Indexed string field columns in stable schema order.
    #[must_use]
    pub fn fields(&self) -> &[FieldTokenColumns] {
        &self.fields
    }

    /// Look up one indexed string field's columns.
    #[must_use]
    pub fn field(&self, field_ord: u16) -> Option<&FieldTokenColumns> {
        self.fields
            .binary_search_by_key(&field_ord, FieldTokenColumns::field_ord)
            .ok()
            .map(|index| &self.fields[index])
    }

    /// Indexed numeric field columns in stable schema order.
    #[must_use]
    pub fn numeric_fields(&self) -> &[NumericFieldColumns] {
        &self.numeric_fields
    }

    /// Look up one indexed-or-fast numeric field's typed column.
    #[must_use]
    pub fn numeric_field(&self, field_ord: u16) -> Option<&NumericFieldColumns> {
        self.numeric_fields
            .binary_search_by_key(&field_ord, NumericFieldColumns::field_ord)
            .ok()
            .map(|index| &self.numeric_fields[index])
    }

    /// Stored-field columns in stable schema order.
    #[must_use]
    pub fn stored_fields(&self) -> &[StoredFieldColumns] {
        &self.stored_fields
    }

    /// Look up one stored-field column.
    #[must_use]
    pub fn stored_field(&self, field_ord: u16) -> Option<&StoredFieldColumns> {
        self.stored_fields
            .binary_search_by_key(&field_ord, StoredFieldColumns::field_ord)
            .ok()
            .map(|index| &self.stored_fields[index])
    }

    /// Completed shard-relative document ordinals. Per-field length/norm
    /// columns align with this slice.
    #[must_use]
    pub fn document_ords(&self) -> &[u32] {
        &self.document_ords
    }

    /// First completed local document ordinal, if any.
    #[must_use]
    pub fn first_doc_ord(&self) -> Option<u32> {
        self.document_ords.first().copied()
    }

    /// Last completed local document ordinal, if any.
    #[must_use]
    pub const fn last_doc_ord(&self) -> Option<u32> {
        self.last_doc_ord
    }

    /// Number of completed documents.
    #[must_use]
    pub fn document_count(&self) -> usize {
        self.document_ords.len()
    }

    /// Total admitted token triples across indexed string fields.
    #[must_use]
    pub fn token_count(&self) -> usize {
        self.fields.iter().map(FieldTokenColumns::len).sum()
    }

    /// Complete retained footprint for RSS and reuse diagnostics.
    #[must_use]
    pub fn bytes_reserved(&self) -> usize {
        self.terms
            .bytes_reserved()
            .saturating_add(
                self.fields
                    .capacity()
                    .saturating_mul(std::mem::size_of::<FieldTokenColumns>()),
            )
            .saturating_add(
                self.fields
                    .iter()
                    .map(FieldTokenColumns::bytes_reserved)
                    .fold(0, usize::saturating_add),
            )
            .saturating_add(
                self.numeric_fields
                    .capacity()
                    .saturating_mul(std::mem::size_of::<NumericFieldColumns>()),
            )
            .saturating_add(
                self.numeric_fields
                    .iter()
                    .map(NumericFieldColumns::bytes_reserved)
                    .fold(0, usize::saturating_add),
            )
            .saturating_add(
                self.stored_fields
                    .capacity()
                    .saturating_mul(std::mem::size_of::<StoredFieldColumns>()),
            )
            .saturating_add(
                self.stored_fields
                    .iter()
                    .map(StoredFieldColumns::bytes_reserved)
                    .fold(0, usize::saturating_add),
            )
            .saturating_add(
                self.document_ords
                    .capacity()
                    .saturating_mul(std::mem::size_of::<u32>()),
            )
            .saturating_add(self.seen_fields.capacity().div_ceil(8))
            .saturating_add(self.analyzer.bytes_reserved())
    }

    /// Live logical footprint used by the post-document flush trigger.
    ///
    /// Retained spare capacity is excluded so a reset shard can refill the
    /// same allocation instead of degenerating into one-document segments.
    #[must_use]
    pub fn bytes_used(&self) -> usize {
        self.terms
            .bytes_used()
            .saturating_add(
                self.fields
                    .len()
                    .saturating_mul(std::mem::size_of::<FieldTokenColumns>()),
            )
            .saturating_add(
                self.fields
                    .iter()
                    .map(FieldTokenColumns::bytes_used)
                    .fold(0, usize::saturating_add),
            )
            .saturating_add(
                self.numeric_fields
                    .len()
                    .saturating_mul(std::mem::size_of::<NumericFieldColumns>()),
            )
            .saturating_add(
                self.numeric_fields
                    .iter()
                    .map(NumericFieldColumns::bytes_used)
                    .fold(0, usize::saturating_add),
            )
            .saturating_add(
                self.stored_fields
                    .len()
                    .saturating_mul(std::mem::size_of::<StoredFieldColumns>()),
            )
            .saturating_add(
                self.stored_fields
                    .iter()
                    .map(StoredFieldColumns::bytes_used)
                    .fold(0, usize::saturating_add),
            )
            .saturating_add(
                self.document_ords
                    .len()
                    .saturating_mul(std::mem::size_of::<u32>()),
            )
            .saturating_add(self.seen_fields.len().div_ceil(8))
    }

    /// Whether a completed document has carried the retained footprint to or
    /// beyond the configured shard budget.
    #[must_use]
    pub fn should_flush(&self, budget_bytes: usize) -> bool {
        !self.document_ords.is_empty() && self.bytes_used() >= budget_bytes
    }

    /// Clear logical contents for the next flush cycle while retaining normal
    /// vector, arena, hash-map, and tokenizer scratch capacity.
    pub fn reset(&mut self) {
        self.terms.reset();
        for field in &mut self.fields {
            field.reset();
        }
        for field in &mut self.numeric_fields {
            field.reset();
        }
        for field in &mut self.stored_fields {
            field.reset();
        }
        self.document_ords.clear();
        self.seen_fields.fill(false);
        self.last_doc_ord = None;
        self.analyzer.reset();
    }
}

fn numeric_value_type_name(value: NumericValue) -> &'static str {
    match value {
        NumericValue::I64(_) => "i64",
        NumericValue::U64(_) => "u64",
    }
}

fn numeric_value_from_le_bytes(kind: FieldKind, bytes: &[u8]) -> Option<NumericValue> {
    let bytes: [u8; 8] = bytes.try_into().ok()?;
    match kind {
        FieldKind::I64 { .. } => Some(NumericValue::I64(i64::from_le_bytes(bytes))),
        FieldKind::U64 { .. } => Some(NumericValue::U64(u64::from_le_bytes(bytes))),
        FieldKind::Keyword | FieldKind::Text { .. } | FieldKind::StoredOnly => None,
    }
}
