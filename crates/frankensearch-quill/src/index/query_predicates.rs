use crate::argus::{
    ArgusError, CheckpointPostingCursor, DeltaPostingCursor, PostingCursor, QueryWorkCheckpoint,
    QueryWorkKind, ReferenceScorer, ScorerClause, TermRecordOption, ValidatedArrivalCursor,
};
use crate::delta::DeltaSnapshot;
use crate::grimoire::{TermDictionaryError, trailing_star_prefix};
use crate::query::QueryValue;
use crate::quiver::{
    EncodedNumericSection, NumericEntry, NumericField, NumericFieldInput, NumericSection,
    NumericValue,
};
use crate::schema::{FieldKind, SchemaDescriptor};
use crate::segment::SectionKind;
use asupersync::Cx;
use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};
use std::ops::Bound;

use super::checkpoint::{
    PreparedSnapshotDocFreqs, QueryCheckpointHandle, clone_query_checkpoint_for_argus,
    delta_ordered_view_blocks,
};
use super::error_results::{QuillIndexError, check_cancel, invalid_state};
use super::query_cursors::{
    leaf_glob_terms, lower_leaf_term, open_dictionary, open_sealed_term_cursor, required_section,
};
use super::query_lowering::{QueryLeaf, QueryLoweringMode, lower_boolean};
use super::snapshot::QuillSearchSnapshot;

#[allow(clippy::too_many_arguments)]
pub(super) fn lower_leaf_range<'a>(
    cx: &Cx,
    checkpoint: &QueryCheckpointHandle<'a>,
    leaf: QueryLeaf<'a>,
    snapshot: &QuillSearchSnapshot,
    prepared_doc_freqs: Option<&PreparedSnapshotDocFreqs>,
    schema: SchemaDescriptor,
    field_ord: u16,
    lower: &Bound<QueryValue>,
    upper: &Bound<QueryValue>,
    boost: f32,
    mode: QueryLoweringMode,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    match query_field_kind(schema, field_ord)? {
        FieldKind::I64 { indexed: true, .. }
        | FieldKind::I64 {
            indexed: false,
            fast: true,
        }
        | FieldKind::U64 { indexed: true, .. }
        | FieldKind::U64 {
            indexed: false,
            fast: true,
        } => lower_leaf_numeric_range(cx, leaf, schema, field_ord, lower, upper, boost, mode),
        FieldKind::Keyword | FieldKind::Text { .. } => {
            let (lower, upper) = string_query_bounds(field_ord, lower, upper)?;
            let terms =
                snapshot_string_range_terms(checkpoint, snapshot, schema, field_ord, lower, upper)?;
            lower_leaf_string_predicate(
                checkpoint,
                leaf,
                snapshot,
                prepared_doc_freqs,
                schema,
                field_ord,
                terms,
                boost,
                mode,
            )
        }
        FieldKind::I64 {
            indexed: false,
            fast: false,
        }
        | FieldKind::U64 {
            indexed: false,
            fast: false,
        } => Err(QuillIndexError::UnsupportedQuery {
            detail: format!("range names non-indexed numeric field {field_ord}"),
        }),
        FieldKind::StoredOnly => Err(QuillIndexError::UnsupportedQuery {
            detail: format!("range names stored-only field {field_ord}"),
        }),
    }
}

pub(super) fn lower_leaf_numeric_range<'a>(
    cx: &Cx,
    leaf: QueryLeaf<'a>,
    schema: SchemaDescriptor,
    field_ord: u16,
    lower: &Bound<QueryValue>,
    upper: &Bound<QueryValue>,
    boost: f32,
    mode: QueryLoweringMode,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    let (lower, upper) = numeric_query_bounds(schema, field_ord, lower, upper)?;
    let score = match mode {
        QueryLoweringMode::Scored => boost,
        QueryLoweringMode::Unscored => 1.0,
    };
    match leaf {
        QueryLeaf::Sealed(segment) => {
            let manifest = segment.manifest();
            let section = NumericSection::parse(
                required_section(segment, SectionKind::NUMERIC)?,
                schema,
                manifest.docid_lo,
                manifest.docid_hi,
            )
            .map_err(ArgusError::from)?;
            let field = section.field(field_ord).ok_or_else(|| {
                invalid_state(format!("NUMERIC has no indexed-or-fast field {field_ord}"))
            })?;
            // Two counts, deliberately: the persisted column holds one value
            // per at-seal row and can only be validated against that, while
            // the scorer must share the live domain its Boolean siblings use
            // (`QueryNode::All` and term leaves both take
            // `leaf.live_document_count()`). Passing the at-seal count for
            // both made a tombstoned segment fail a valid query (#49).
            materialize_live_numeric_range(
                cx,
                leaf,
                field,
                lower,
                upper,
                segment.at_seal_doc_count(),
                leaf.live_document_count()?,
                score,
            )
        }
        QueryLeaf::Delta(delta) => {
            let encoded = encode_live_delta_numeric(delta, schema)?;
            let (docid_lo, docid_hi) = leaf.docid_range();
            let section = NumericSection::parse(encoded.as_bytes(), schema, docid_lo, docid_hi)
                .map_err(ArgusError::from)?;
            let field = section.field(field_ord).ok_or_else(|| {
                invalid_state(format!(
                    "Delta NUMERIC has no indexed-or-fast field {field_ord}"
                ))
            })?;
            materialize_live_numeric_range(
                cx,
                leaf,
                field,
                lower,
                upper,
                // A Delta's encoded column is built from its live rows, so
                // both domains are the same count here.
                leaf.live_document_count()?,
                leaf.live_document_count()?,
                score,
            )
        }
    }
}

fn materialize_live_numeric_range<'a>(
    cx: &Cx,
    leaf: QueryLeaf<'a>,
    field: NumericField<'_>,
    lower: Bound<NumericValue>,
    upper: Bound<NumericValue>,
    column_num_docs: u32,
    live_num_docs: u32,
    score: f32,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    const CANCEL_CHECK_MASK: usize = 1_023;

    check_cancel(cx, "numeric range")?;
    let entries = field
        .range_entries(lower, upper)
        .map_err(ArgusError::from)?;
    let value_count = field.len();
    // Coverage is a property of the persisted column, so it is judged against
    // the column's own domain: a value for every at-seal row, all of them in
    // range. That implies coverage of the live rows too, since those are a
    // subset — judging it against the live count instead would silently drop
    // the optimisation on any segment holding a tombstone.
    let covers_every_document = value_count
        == usize::try_from(column_num_docs).unwrap_or(usize::MAX)
        && entries.len() == value_count;
    let mut docids = Vec::new();
    docids
        .try_reserve_exact(entries.len())
        .map_err(|_| invalid_state("could not allocate live numeric range docids"))?;
    for (index, entry) in entries.enumerate() {
        if index & CANCEL_CHECK_MASK == 0 {
            check_cancel(cx, "numeric range")?;
        }
        if leaf.is_live_document(entry.docid()) {
            docids.push(entry.docid());
        }
    }
    check_cancel(cx, "numeric range")?;
    docids.sort_unstable();
    docids.dedup();
    ReferenceScorer::materialized_numeric_range(
        field.field_ord(),
        docids,
        value_count,
        column_num_docs,
        live_num_docs,
        covers_every_document,
        score,
    )
    .map_err(QuillIndexError::from)
}

pub(super) fn numeric_query_bounds(
    schema: SchemaDescriptor,
    field_ord: u16,
    lower: &Bound<QueryValue>,
    upper: &Bound<QueryValue>,
) -> Result<(Bound<NumericValue>, Bound<NumericValue>), QuillIndexError> {
    let kind = query_field_kind(schema, field_ord)?;
    let expected = match kind {
        FieldKind::I64 { indexed: true, .. }
        | FieldKind::I64 {
            indexed: false,
            fast: true,
        } => NumericValueKind::I64,
        FieldKind::U64 { indexed: true, .. }
        | FieldKind::U64 {
            indexed: false,
            fast: true,
        } => NumericValueKind::U64,
        FieldKind::I64 {
            indexed: false,
            fast: false,
        }
        | FieldKind::U64 {
            indexed: false,
            fast: false,
        } => {
            return Err(QuillIndexError::UnsupportedQuery {
                detail: format!("numeric range names non-indexed field {field_ord}"),
            });
        }
        FieldKind::Keyword | FieldKind::Text { .. } | FieldKind::StoredOnly => {
            return Err(QuillIndexError::UnsupportedQuery {
                detail: format!("numeric range names non-numeric field {field_ord}"),
            });
        }
    };
    Ok((
        numeric_query_bound(field_ord, expected, lower)?,
        numeric_query_bound(field_ord, expected, upper)?,
    ))
}

pub(super) fn query_field_kind(
    schema: SchemaDescriptor,
    field_ord: u16,
) -> Result<FieldKind, QuillIndexError> {
    query_field_descriptor(schema, field_ord).map(|field| field.kind)
}

pub(super) fn term_record_option(
    schema: SchemaDescriptor,
    field_ord: u16,
) -> Result<TermRecordOption, QuillIndexError> {
    match query_field_kind(schema, field_ord)? {
        FieldKind::Keyword
        | FieldKind::Text {
            positions: false, ..
        } => Ok(TermRecordOption::Basic),
        FieldKind::Text {
            positions: true, ..
        } => Ok(TermRecordOption::WithFreqsAndPositions),
        FieldKind::StoredOnly | FieldKind::I64 { .. } | FieldKind::U64 { .. } => {
            Err(QuillIndexError::UnsupportedQuery {
                detail: format!("term scorer names non-string field {field_ord}"),
            })
        }
    }
}

fn query_field_descriptor(
    schema: SchemaDescriptor,
    field_ord: u16,
) -> Result<crate::schema::FieldDescriptor, QuillIndexError> {
    schema
        .fields
        .get(usize::from(field_ord))
        .filter(|field| field.id == field_ord)
        .copied()
        .ok_or_else(|| QuillIndexError::UnsupportedQuery {
            detail: format!("query names unknown field {field_ord}"),
        })
}

#[derive(Clone, Copy)]
enum NumericValueKind {
    I64,
    U64,
}

fn numeric_query_bound(
    field_ord: u16,
    expected: NumericValueKind,
    bound: &Bound<QueryValue>,
) -> Result<Bound<NumericValue>, QuillIndexError> {
    let convert = |value: &QueryValue| match (expected, value) {
        (NumericValueKind::I64, QueryValue::I64(value)) => Ok(NumericValue::I64(*value)),
        (NumericValueKind::U64, QueryValue::U64(value)) => Ok(NumericValue::U64(*value)),
        _ => Err(QuillIndexError::UnsupportedQuery {
            detail: format!("numeric range bound type does not match field {field_ord}"),
        }),
    };
    match bound {
        Bound::Included(value) => convert(value).map(Bound::Included),
        Bound::Excluded(value) => convert(value).map(Bound::Excluded),
        Bound::Unbounded => Ok(Bound::Unbounded),
    }
}

pub(super) type StringQueryBounds<'a> = (Bound<&'a [u8]>, Bound<&'a [u8]>);

pub(super) fn string_query_bounds<'a>(
    field_ord: u16,
    lower: &'a Bound<QueryValue>,
    upper: &'a Bound<QueryValue>,
) -> Result<StringQueryBounds<'a>, QuillIndexError> {
    Ok((
        string_query_bound(field_ord, lower)?,
        string_query_bound(field_ord, upper)?,
    ))
}

fn string_query_bound(
    field_ord: u16,
    bound: &Bound<QueryValue>,
) -> Result<Bound<&[u8]>, QuillIndexError> {
    match bound {
        Bound::Included(QueryValue::Str(value)) => Ok(Bound::Included(value.as_bytes())),
        Bound::Excluded(QueryValue::Str(value)) => Ok(Bound::Excluded(value.as_bytes())),
        Bound::Unbounded => Ok(Bound::Unbounded),
        Bound::Included(QueryValue::I64(_) | QueryValue::U64(_))
        | Bound::Excluded(QueryValue::I64(_) | QueryValue::U64(_)) => {
            Err(QuillIndexError::UnsupportedQuery {
                detail: format!("string range bound type does not match field {field_ord}"),
            })
        }
    }
}

pub(super) fn snapshot_string_range_terms(
    checkpoint: &QueryCheckpointHandle<'_>,
    snapshot: &QuillSearchSnapshot,
    schema: SchemaDescriptor,
    field_ord: u16,
    lower: Bound<&[u8]>,
    upper: Bound<&[u8]>,
) -> Result<Vec<Vec<u8>>, QuillIndexError> {
    validate_string_query_field(schema, field_ord, "range")?;
    if string_range_is_reversed(&lower, &upper) {
        return Ok(Vec::new());
    }
    let mut terms = BTreeSet::<Vec<u8>>::new();
    for segment in snapshot.keeper_snapshot().segments() {
        let dictionary = open_dictionary(segment, schema)?;
        let limit = usize::try_from(dictionary.term_count())
            .map_err(|_| invalid_state("dictionary term count does not fit usize"))?;
        let mut cursor = dictionary.range_cursor(field_ord, lower, upper)?;
        let mut admitted_block = None;
        while let Some(current) = cursor.current() {
            let block = cursor.current_block_index();
            if block != admitted_block {
                checkpoint.admit(QueryWorkKind::DictionaryBlock, 1)?;
                admitted_block = block;
            }
            if terms.len() >= limit {
                return Err(TermDictionaryError::MaterializationLimitExceeded { limit }.into());
            }
            terms.insert(current.term.to_vec());
            cursor.next()?;
        }
    }
    for delta in snapshot.delta_snapshots() {
        // `ordered_terms` materialises and radix-sorts the complete ordered
        // view on its first use before yielding its first term. Admit that
        // bounded work from the O(1) term count before the cache can be built;
        // otherwise a cancelled or under-budget range still pays the whole
        // cost before observing its first checkpoint.
        checkpoint.admit(
            QueryWorkKind::DictionaryBlock,
            delta_ordered_view_blocks(delta.segment().term_count()),
        )?;
        let field_terms = delta.ordered_terms_for_field(field_ord);
        if delta.segment().term_count() != 0 {
            // A nonempty Delta still needs to surface cancellation after its
            // cached view was obtained, even when this requested field is empty.
            checkpoint.admit(QueryWorkKind::DictionaryBlock, 0)?;
        }
        for (term_index, term) in field_terms.enumerate() {
            // Ordered-view blocks are paid above. Keep a zero-unit boundary
            // so cancellation during a long range walk is still observed.
            if term_index != 0 && term_index % 16 == 0 {
                checkpoint.admit(QueryWorkKind::DictionaryBlock, 0)?;
            }
            // Both tests are O(1) now that the live count is taken at freeze,
            // so this expansion costs the dictionary blocks it already
            // admits and nothing more.
            if term.live_doc_freq() != 0 && term_in_string_range(term.term(), &lower, &upper) {
                terms.insert(term.term().to_vec());
            }
        }
    }
    Ok(terms.into_iter().collect())
}

fn string_range_is_reversed(lower: &Bound<&[u8]>, upper: &Bound<&[u8]>) -> bool {
    match (lower, upper) {
        (
            Bound::Included(lower) | Bound::Excluded(lower),
            Bound::Included(upper) | Bound::Excluded(upper),
        ) => lower > upper,
        _ => false,
    }
}

fn term_in_string_range(term: &[u8], lower: &Bound<&[u8]>, upper: &Bound<&[u8]>) -> bool {
    let above_lower = match lower {
        Bound::Included(bound) => term >= *bound,
        Bound::Excluded(bound) => term > *bound,
        Bound::Unbounded => true,
    };
    let below_upper = match upper {
        Bound::Included(bound) => term <= *bound,
        Bound::Excluded(bound) => term < *bound,
        Bound::Unbounded => true,
    };
    above_lower && below_upper
}

pub(super) fn validate_string_query_field(
    schema: SchemaDescriptor,
    field_ord: u16,
    predicate: &'static str,
) -> Result<(), QuillIndexError> {
    match query_field_kind(schema, field_ord)? {
        FieldKind::Keyword | FieldKind::Text { .. } => Ok(()),
        FieldKind::StoredOnly | FieldKind::I64 { .. } | FieldKind::U64 { .. } => {
            Err(QuillIndexError::UnsupportedQuery {
                detail: format!("{predicate} names non-string field {field_ord}"),
            })
        }
    }
}

fn encode_live_delta_numeric(
    delta: &DeltaSnapshot,
    schema: SchemaDescriptor,
) -> Result<EncodedNumericSection, QuillIndexError> {
    let mut owned_fields = Vec::<(u16, Vec<NumericEntry>)>::new();
    for field in schema
        .fields
        .iter()
        .filter(|field| field.kind.has_numeric_column())
    {
        let mut entries = Vec::new();
        for (global_docid, _) in delta.live_documents() {
            let Some(value) = delta.numeric_value(field.id, global_docid) else {
                continue;
            };
            entries.push(match value {
                NumericValue::I64(value) => NumericEntry::i64(value, global_docid),
                NumericValue::U64(value) => NumericEntry::u64(value, global_docid),
            });
        }
        owned_fields.push((field.id, entries));
    }
    let inputs = owned_fields
        .iter()
        .map(|(field_ord, entries)| NumericFieldInput::new(*field_ord, entries))
        .collect::<Vec<_>>();
    let (docid_lo, docid_hi) = QueryLeaf::Delta(delta).docid_range();
    EncodedNumericSection::encode(schema, docid_lo, docid_hi, &inputs)
        .map_err(ArgusError::from)
        .map_err(QuillIndexError::from)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn lower_leaf_set<'a>(
    checkpoint: &QueryCheckpointHandle<'a>,
    leaf: QueryLeaf<'a>,
    snapshot: &QuillSearchSnapshot,
    prepared_doc_freqs: Option<&PreparedSnapshotDocFreqs>,
    schema: SchemaDescriptor,
    field_ord: u16,
    values: &[QueryValue],
    boost: f32,
    mode: QueryLoweringMode,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    match query_field_kind(schema, field_ord)? {
        FieldKind::Keyword | FieldKind::Text { .. } => {
            let mut terms = BTreeSet::<Vec<u8>>::new();
            for value in values {
                let QueryValue::Str(value) = value else {
                    return Err(QuillIndexError::UnsupportedQuery {
                        detail: format!("set value type does not match string field {field_ord}"),
                    });
                };
                terms.insert(value.as_bytes().to_vec());
            }
            lower_leaf_string_predicate(
                checkpoint,
                leaf,
                snapshot,
                prepared_doc_freqs,
                schema,
                field_ord,
                terms.into_iter().collect(),
                boost,
                mode,
            )
        }
        FieldKind::I64 { indexed: true, .. }
        | FieldKind::I64 {
            indexed: false,
            fast: true,
        }
        | FieldKind::U64 { indexed: true, .. }
        | FieldKind::U64 {
            indexed: false,
            fast: true,
        } => {
            let values = numeric_query_values(schema, field_ord, values)?;
            lower_leaf_numeric_set(leaf, schema, field_ord, &values, boost, mode)
        }
        FieldKind::I64 {
            indexed: false,
            fast: false,
        }
        | FieldKind::U64 {
            indexed: false,
            fast: false,
        } => Err(QuillIndexError::UnsupportedQuery {
            detail: format!("set names non-indexed and non-fast numeric field {field_ord}"),
        }),
        FieldKind::StoredOnly => Err(QuillIndexError::UnsupportedQuery {
            detail: format!("set names stored-only field {field_ord}"),
        }),
    }
}

pub(super) fn numeric_query_values(
    schema: SchemaDescriptor,
    field_ord: u16,
    values: &[QueryValue],
) -> Result<Vec<NumericValue>, QuillIndexError> {
    match query_field_kind(schema, field_ord)? {
        FieldKind::I64 { indexed: true, .. }
        | FieldKind::I64 {
            indexed: false,
            fast: true,
        } => {
            let mut unique = BTreeSet::new();
            for value in values {
                let QueryValue::I64(value) = value else {
                    return Err(QuillIndexError::UnsupportedQuery {
                        detail: format!("set value type does not match i64 field {field_ord}"),
                    });
                };
                unique.insert(*value);
            }
            Ok(unique.into_iter().map(NumericValue::I64).collect())
        }
        FieldKind::U64 { indexed: true, .. }
        | FieldKind::U64 {
            indexed: false,
            fast: true,
        } => {
            let mut unique = BTreeSet::new();
            for value in values {
                let QueryValue::U64(value) = value else {
                    return Err(QuillIndexError::UnsupportedQuery {
                        detail: format!("set value type does not match u64 field {field_ord}"),
                    });
                };
                unique.insert(*value);
            }
            Ok(unique.into_iter().map(NumericValue::U64).collect())
        }
        FieldKind::I64 {
            indexed: false,
            fast: false,
        }
        | FieldKind::U64 {
            indexed: false,
            fast: false,
        }
        | FieldKind::Keyword
        | FieldKind::Text { .. }
        | FieldKind::StoredOnly => Err(QuillIndexError::UnsupportedQuery {
            detail: format!("numeric set names incompatible field {field_ord}"),
        }),
    }
}

fn lower_leaf_numeric_set<'a>(
    leaf: QueryLeaf<'a>,
    schema: SchemaDescriptor,
    field_ord: u16,
    values: &[NumericValue],
    boost: f32,
    mode: QueryLoweringMode,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    match leaf {
        QueryLeaf::Sealed(segment) => {
            let manifest = segment.manifest();
            let section = NumericSection::parse(
                required_section(segment, SectionKind::NUMERIC)?,
                schema,
                manifest.docid_lo,
                manifest.docid_hi,
            )
            .map_err(ArgusError::from)?;
            let field = section.field(field_ord).ok_or_else(|| {
                invalid_state(format!("NUMERIC has no indexed field {field_ord}"))
            })?;
            // Same split as the range path: validate the persisted column
            // against its at-seal domain, score in the live one (#49).
            lower_numeric_field_set(
                field,
                values,
                segment.at_seal_doc_count(),
                leaf.live_document_count()?,
                boost,
                mode,
            )
        }
        QueryLeaf::Delta(delta) => {
            let encoded = encode_live_delta_numeric(delta, schema)?;
            let (docid_lo, docid_hi) = leaf.docid_range();
            let section = NumericSection::parse(encoded.as_bytes(), schema, docid_lo, docid_hi)
                .map_err(ArgusError::from)?;
            let field = section.field(field_ord).ok_or_else(|| {
                invalid_state(format!("Delta NUMERIC has no indexed field {field_ord}"))
            })?;
            // A Delta's encoded column is built from its live rows, so both
            // domains are the same count here.
            let live_num_docs = leaf.live_document_count()?;
            lower_numeric_field_set(field, values, live_num_docs, live_num_docs, boost, mode)
        }
    }
}

fn lower_numeric_field_set<'a>(
    field: NumericField<'_>,
    values: &[NumericValue],
    column_num_docs: u32,
    live_num_docs: u32,
    boost: f32,
    mode: QueryLoweringMode,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    let mut clauses = Vec::new();
    clauses
        .try_reserve_exact(values.len())
        .map_err(|_| invalid_state("could not allocate numeric set clauses"))?;
    for &value in values {
        clauses.push(ScorerClause::should(ReferenceScorer::numeric_range(
            field,
            Bound::Included(value),
            Bound::Included(value),
            column_num_docs,
            live_num_docs,
        )?));
    }
    let matching = lower_boolean(
        clauses,
        QueryLoweringMode::Unscored,
        false,
        #[cfg(feature = "pruning-conformance")]
        None,
    )?;
    match mode {
        QueryLoweringMode::Scored => {
            ReferenceScorer::constant_score(matching, boost).map_err(QuillIndexError::from)
        }
        QueryLoweringMode::Unscored => Ok(matching),
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_leaf_string_predicate<'a>(
    checkpoint: &QueryCheckpointHandle<'a>,
    leaf: QueryLeaf<'a>,
    snapshot: &QuillSearchSnapshot,
    prepared_doc_freqs: Option<&PreparedSnapshotDocFreqs>,
    schema: SchemaDescriptor,
    field_ord: u16,
    terms: Vec<Vec<u8>>,
    boost: f32,
    mode: QueryLoweringMode,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    let mut clauses = Vec::new();
    clauses
        .try_reserve_exact(terms.len())
        .map_err(|_| invalid_state("could not allocate string predicate clauses"))?;
    for term in terms {
        // A materialized range, set or glob term is lowered exactly like a
        // syntactic one, so it reads the same query-local statistics table.
        // Passing `None` here made every expanded term rescan every sealed
        // dictionary in every lowered segment, which is the quadratic charge
        // the table exists to retire (#41) -- and the resulting BM25 figure is
        // then discarded by the constant score below.
        clauses.push(ScorerClause::should(lower_leaf_term(
            leaf,
            snapshot,
            schema,
            field_ord,
            &term,
            prepared_doc_freqs,
            1.0,
            false,
            checkpoint,
        )?));
    }
    let matching = lower_boolean(
        clauses,
        QueryLoweringMode::Unscored,
        false,
        #[cfg(feature = "pruning-conformance")]
        Some(checkpoint),
    )?;
    match mode {
        QueryLoweringMode::Scored => {
            ReferenceScorer::constant_score(matching, boost).map_err(QuillIndexError::from)
        }
        QueryLoweringMode::Unscored => Ok(matching),
    }
}

/// One live cursor in a glob arm's doc-id merge. These are the same
/// validated, checkpoint-metered movement adapters the per-term scorers
/// drive ([`ValidatedArrivalCursor`] for sealed leaves,
/// [`CheckpointPostingCursor`] for Delta leaves), held by their concrete
/// types so the merge loop pays no per-document scorer dispatch on top of
/// the cursor's own movement.
enum GlobDocCursor<'a> {
    Sealed(Box<ValidatedArrivalCursor<'a>>),
    Delta(CheckpointPostingCursor<'a>),
}

impl GlobDocCursor<'_> {
    pub(super) fn doc(&self) -> Option<u32> {
        match self {
            Self::Sealed(cursor) => PostingCursor::doc(cursor),
            Self::Delta(cursor) => PostingCursor::doc(cursor),
        }
    }

    pub(super) fn segment_num_docs(&self) -> u32 {
        match self {
            Self::Sealed(cursor) => PostingCursor::segment_num_docs(cursor),
            Self::Delta(cursor) => PostingCursor::segment_num_docs(cursor),
        }
    }

    pub(super) fn size_hint(&self) -> u32 {
        match self {
            Self::Sealed(cursor) => PostingCursor::size_hint(cursor),
            Self::Delta(cursor) => PostingCursor::size_hint(cursor),
        }
    }

    pub(super) fn next(&mut self) -> Result<Option<u32>, QuillIndexError> {
        match self {
            Self::Sealed(cursor) => cursor.next().map_err(QuillIndexError::from),
            Self::Delta(cursor) => cursor.next().map_err(QuillIndexError::from),
        }
    }
}

/// Collect the exact unique document set of one glob arm's expansion on one
/// leaf and field: a single ascending merge over the expanded terms' posting
/// cursors, materialized straight into the document-id vector
/// [`ReferenceScorer::constant_score_from_docids`] consumes.
///
/// The replaced lowering drove one term scorer per expansion through an
/// unscored union and then drained that union through
/// [`ReferenceScorer::constant_score`] — every posting paid the union's heap
/// merge plus the drain's per-document dispatch, monotonicity check and
/// vector-growth probe, all to produce a set whose members the cursors had
/// already yielded. The merge here walks the same cursors with the same
/// per-move fuel admissions, so the glob arm's cost tracks its posting-block
/// decode work; a common-term arm costs the decode instead of a full
/// scorer-chain materialization (profiled on the owner archive: decode was
/// ~1% of the glob arm's lowered walk).
///
/// The typed [`TermDictionaryError::GlobExpansionLimitExceeded`] cap is
/// unchanged: it is enforced by [`leaf_glob_terms`] before this collector
/// runs, and a cancellation or fuel refusal during the walk propagates from
/// the same cursor admissions the term scorers paid.
///
/// # Errors
///
/// Returns a typed error for an invalid string field, cursor or allocation
/// failure, or a checkpoint refusal (cancellation or fuel exhaustion).
fn collect_leaf_glob_docids(
    checkpoint: &QueryCheckpointHandle<'_>,
    leaf: QueryLeaf<'_>,
    schema: SchemaDescriptor,
    field_ord: u16,
    terms: &[Vec<u8>],
) -> Result<(Vec<u32>, u32), QuillIndexError> {
    let mut cursors = Vec::new();
    cursors
        .try_reserve_exact(terms.len())
        .map_err(|_| invalid_state("could not allocate glob doc cursors"))?;
    for term in terms {
        match leaf {
            QueryLeaf::Sealed(segment) => {
                // The open mirrors `lower_leaf_term`'s sealed arm admission
                // for the dictionary lookup, so fuel and cancellation observe
                // the same units at the same points.
                checkpoint.admit(QueryWorkKind::DictionaryBlock, 1)?;
                let (cursor, _fieldnorms) =
                    open_sealed_term_cursor(segment, schema, field_ord, term, false, checkpoint)?;
                #[cfg(feature = "profile-internals")]
                checkpoint.record_term_dictionary_view();
                cursors.push(GlobDocCursor::Sealed(Box::new(
                    ValidatedArrivalCursor::new(
                        cursor,
                        clone_query_checkpoint_for_argus(checkpoint),
                    ),
                )));
            }
            QueryLeaf::Delta(delta) => {
                let cursor =
                    DeltaPostingCursor::new_admitted(delta, field_ord, term, checkpoint.as_ref())?;
                cursors.push(GlobDocCursor::Delta(CheckpointPostingCursor::new(
                    cursor,
                    clone_query_checkpoint_for_argus(checkpoint),
                )?));
            }
        }
    }
    if cursors.is_empty() {
        return Ok((Vec::new(), 0));
    }
    // All cursors of one leaf share the same segment document count, which is
    // the value the drained union reported through
    // `shared_segment_num_docs`.
    let segment_num_docs = cursors[0].segment_num_docs();
    let total_hint: u64 = cursors
        .iter()
        .map(GlobDocCursor::size_hint)
        .map(u64::from)
        .sum();
    let mut docids = Vec::new();
    docids
        .try_reserve(
            usize::try_from(total_hint.min(1 << 20))
                .map_err(|_| invalid_state("glob docid merge capacity does not fit usize"))?,
        )
        .map_err(|_| {
            QuillIndexError::from(ArgusError::Allocation {
                resource: "glob docid merge",
                count: usize::try_from(total_hint).unwrap_or(usize::MAX),
            })
        })?;
    // Ascending k-way merge over the smallest head. The head is read as an
    // owned pair (no heap borrow crosses the loop body), its cursor
    // advances, and on the common path the replacement doc id is written
    // straight into the heap root through `peek_mut`, whose sift-down then
    // early-exits whenever the same stream still holds the minimum — the
    // hot shape for a common-term arm, where one term dominates the
    // expansion and a pop-then-reinsert loop paid two full heap sifts per
    // posting of exactly that stream (profiled on the owner archive: the
    // reinsert push alone was ~65% of the arm's lowered walk). Duplicate
    // heads collapse into one doc id, so every member contributes the arm's
    // constant score exactly once — the same dedup the drained union
    // performed. The slot rides the heap as a `u32`, keeping each entry
    // eight bytes.
    let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
    for (slot, cursor) in cursors.iter().enumerate() {
        let Some(doc) = cursor.doc() else {
            continue;
        };
        let slot = u32::try_from(slot)
            .map_err(|_| invalid_state("glob doc merge cursor slot overflows u32"))?;
        heap.push(Reverse((doc, slot)));
    }
    let mut last: Option<u32> = None;
    let mut since_poll = 0_u64;
    while let Some((doc, slot)) = heap.peek().map(|&Reverse(pair)| pair) {
        let cursor_slot = slot as usize;
        if last != Some(doc) {
            if docids.len() == docids.capacity() {
                docids.try_reserve(1).map_err(|_| {
                    QuillIndexError::from(ArgusError::Allocation {
                        resource: "glob docid merge",
                        count: docids.len().saturating_add(1),
                    })
                })?;
            }
            docids.push(doc);
            last = Some(doc);
        }
        match cursors[cursor_slot].next()? {
            Some(next_doc) => {
                let mut top = heap
                    .peek_mut()
                    .ok_or_else(|| invalid_state("glob doc merge lost its heap root"))?;
                *top = Reverse((next_doc, slot));
            }
            None => {
                heap.pop();
            }
        }
        since_poll += 1;
        // The merge itself moves no cursor block, but a large expansion can
        // spend real time here; surface cancellation on the same cadence the
        // dictionary walks use.
        if since_poll % 16 == 0 {
            checkpoint.admit(QueryWorkKind::PostingBlock, 0)?;
        }
    }
    Ok((docids, segment_num_docs))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn lower_leaf_glob<'a>(
    checkpoint: &QueryCheckpointHandle<'a>,
    leaf: QueryLeaf<'a>,
    _snapshot: &QuillSearchSnapshot,
    _prepared_doc_freqs: Option<&PreparedSnapshotDocFreqs>,
    schema: SchemaDescriptor,
    field_ids: &[u16],
    pattern: &[u8],
    boost: f32,
    expansion_limit: usize,
    mode: QueryLoweringMode,
    exclude_exact_literal: bool,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    let mut fields = Vec::new();
    fields
        .try_reserve_exact(field_ids.len())
        .map_err(|_| invalid_state("could not allocate glob field clauses"))?;
    for &field_ord in field_ids {
        let mut terms = leaf_glob_terms(
            checkpoint,
            leaf,
            schema,
            field_ord,
            pattern,
            expansion_limit,
        )?;
        if exclude_exact_literal {
            // The accompanying exact arm (`two_arm_variants_glob` checked the
            // two-arm shape before routing here) walks this same literal over
            // the glob's one field, so every document the literal's posting
            // list names is already matched by the Boolean `Or`'s Term arm.
            // Dropping it skips the single largest posting decode of a
            // common-term expansion; the arm's constant boost no longer
            // lands on documents whose only globset membership was the
            // literal itself, whose match and BM25 come from the Term arm
            // unchanged. The expansion cap still fired on the unfiltered
            // expansion inside `leaf_glob_terms`, so the typed
            // `GlobExpansionLimitExceeded` contract is untouched.
            if let Some(literal) = trailing_star_prefix(pattern) {
                terms.retain(|term| term.as_slice() != literal);
            }
        }
        let (docids, segment_num_docs) =
            collect_leaf_glob_docids(checkpoint, leaf, schema, field_ord, &terms)?;
        let field_scorer =
            ReferenceScorer::constant_score_from_docids(docids, segment_num_docs, boost)
                .map_err(QuillIndexError::from)?;
        fields.push(ScorerClause::should(field_scorer));
    }
    lower_boolean(
        fields,
        mode,
        false,
        #[cfg(feature = "pruning-conformance")]
        Some(checkpoint),
    )
}
