use std::ops::Range;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSlice;

use crate::contract::fieldnorm_to_id;
use crate::delta::DeltaSnapshot;
use crate::grimoire::{ByteSpan, TermInput, TermMetadata};
use crate::quiver::{EncodedPositionList, EncodedPostingList, IdMapEntryInput, Posting};
use crate::schema::{FieldKind, SchemaDescriptor};

use super::accumulator::{ColumnarAccumulator, FieldTokenColumns};
use super::analyzer::TokenAnalyzer;
use super::flush::{FlushDocumentInput, FlushError, FlushMode};

const RADIX_DIGIT_BITS: u32 = 16;
const RADIX_DIGIT_BUCKETS: usize = 1 << RADIX_DIGIT_BITS;
pub(super) const PARALLEL_RADIX_ROWS_PER_CHUNK: usize = 4_096;
const MAX_PARALLEL_RADIX_RANGE_BYTES: usize = 64 * 1024 * 1024;

pub(super) type DocLenColumns = (Vec<u16>, Vec<Vec<Option<u32>>>);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct FlushTokenRow {
    pub(super) term_id: u32,
    pub(super) doc_ord: u32,
    pub(super) position: u32,
}

#[derive(Debug)]
pub(super) struct RadixPartition {
    pub(super) rows: Vec<FlushTokenRow>,
    pub(super) ranges: Vec<Range<usize>>,
}

pub(super) fn schema_has_positions(schema: SchemaDescriptor) -> bool {
    schema.fields.iter().any(|field| {
        matches!(
            field.kind,
            FieldKind::Text {
                positions: true,
                ..
            }
        )
    })
}

pub(super) fn validate_flush_documents(
    document_ords: &[u32],
    documents: &[FlushDocumentInput<'_>],
) -> Result<(), FlushError> {
    if documents.len() != document_ords.len() {
        return Err(FlushError::DocumentCountMismatch {
            expected: document_ords.len(),
            actual: documents.len(),
        });
    }
    for (index, (&expected, document)) in document_ords.iter().zip(documents).enumerate() {
        if document.doc_ord != expected {
            return Err(FlushError::DocumentOrdinalMismatch {
                index,
                expected,
                actual: document.doc_ord,
            });
        }
    }
    Ok(())
}

pub(super) fn rebase_doc_id(lease_docid_base: u64, doc_ord: u32) -> Result<u64, FlushError> {
    let docid =
        lease_docid_base
            .checked_add(u64::from(doc_ord))
            .ok_or(FlushError::DocumentIdOverflow {
                lease_docid_base,
                doc_ord,
            })?;
    if docid > u64::from(u32::MAX) {
        return Err(FlushError::DocumentIdOverflow {
            lease_docid_base,
            doc_ord,
        });
    }
    Ok(docid)
}

pub(super) fn durable_len(bytes: &[u8], field: &'static str) -> Result<u64, FlushError> {
    u64::try_from(bytes.len()).map_err(|_| FlushError::ArithmeticOverflow { field })
}

pub(super) fn build_doclen_columns<A: TokenAnalyzer>(
    accumulator: &ColumnarAccumulator<A>,
    first_doc_ord: u32,
    span: usize,
) -> Result<DocLenColumns, FlushError> {
    let document_ords = accumulator.document_ords();
    let mut field_ords = Vec::new();
    let mut columns = Vec::new();
    field_ords
        .try_reserve_exact(accumulator.fields().len())
        .map_err(|_| FlushError::Allocation {
            resource: "DOCLEN field ordinals",
            count: accumulator.fields().len(),
        })?;
    columns
        .try_reserve_exact(accumulator.fields().len())
        .map_err(|_| FlushError::Allocation {
            resource: "DOCLEN columns",
            count: accumulator.fields().len(),
        })?;

    for field in accumulator.fields() {
        if field.document_lengths().len() != document_ords.len()
            || field.fieldnorm_ids().len() != document_ords.len()
        {
            return Err(FlushError::InvalidTokenColumn {
                field_ord: field.field_ord(),
                detail: "document-length/fieldnorm row count drift",
            });
        }
        let mut column = filled_vec(span, None, "DOCLEN span")?;
        for (index, (&doc_ord, &length)) in document_ords
            .iter()
            .zip(field.document_lengths())
            .enumerate()
        {
            let relative = doc_ord.checked_sub(first_doc_ord).ok_or_else(|| {
                FlushError::InvalidTokenColumn {
                    field_ord: field.field_ord(),
                    detail: "document ordinal precedes segment lower bound",
                }
            })?;
            let relative =
                usize::try_from(relative).map_err(|_| FlushError::ArithmeticOverflow {
                    field: "DOCLEN relative document index",
                })?;
            let slot = column
                .get_mut(relative)
                .ok_or_else(|| FlushError::InvalidTokenColumn {
                    field_ord: field.field_ord(),
                    detail: "document ordinal exceeds segment span",
                })?;
            *slot = Some(length);
            if field.fieldnorm_ids()[index] != fieldnorm_to_id(length) {
                return Err(FlushError::InvalidTokenColumn {
                    field_ord: field.field_ord(),
                    detail: "fieldnorm disagrees with raw document length",
                });
            }
        }
        field_ords.push(field.field_ord());
        columns.push(column);
    }
    Ok((field_ords, columns))
}

pub(super) fn build_id_map_inputs<'a>(
    documents: &'a [FlushDocumentInput<'a>],
    first_doc_ord: u32,
    span: usize,
    document_ords: &[u32],
) -> Result<Vec<Option<IdMapEntryInput<'a>>>, FlushError> {
    let mut entries = filled_vec(span, None, "IDMAP span")?;
    for (&doc_ord, document) in document_ords.iter().zip(documents) {
        let relative =
            doc_ord
                .checked_sub(first_doc_ord)
                .ok_or(FlushError::ArithmeticOverflow {
                    field: "IDMAP relative document index",
                })?;
        let relative = usize::try_from(relative).map_err(|_| FlushError::ArithmeticOverflow {
            field: "IDMAP relative document host index",
        })?;
        let slot = entries
            .get_mut(relative)
            .ok_or(FlushError::ArithmeticOverflow {
                field: "IDMAP document span",
            })?;
        *slot = Some(IdMapEntryInput::new(
            document.document_id,
            document.content_hash,
        ));
    }
    Ok(entries)
}

pub(super) fn collect_flush_rows<A: TokenAnalyzer>(
    accumulator: &ColumnarAccumulator<A>,
) -> Result<Vec<FlushTokenRow>, FlushError> {
    let mut rows = Vec::new();
    rows.try_reserve_exact(accumulator.token_count())
        .map_err(|_| FlushError::Allocation {
            resource: "token rows",
            count: accumulator.token_count(),
        })?;
    for field in accumulator.fields() {
        if field.term_ids().len() != field.doc_ords().len()
            || field
                .positions()
                .is_some_and(|positions| positions.len() != field.term_ids().len())
        {
            return Err(FlushError::InvalidTokenColumn {
                field_ord: field.field_ord(),
                detail: "parallel token column length drift",
            });
        }
        let mut previous_doc = None;
        let mut completed_index = 0_usize;
        for index in 0..field.term_ids().len() {
            let term_id = field.term_ids()[index];
            let doc_ord = field.doc_ords()[index];
            if usize::try_from(term_id).map_or(true, |id| id >= accumulator.terms().len()) {
                return Err(FlushError::TermIdOutOfRange {
                    field_ord: field.field_ord(),
                    term_id,
                });
            }
            if accumulator.terms().field_and_term(term_id).0 != field.field_ord() {
                return Err(FlushError::InvalidTokenColumn {
                    field_ord: field.field_ord(),
                    detail: "term interner field namespace drift",
                });
            }
            if previous_doc.is_some_and(|previous| doc_ord < previous) {
                return Err(FlushError::InvalidTokenColumn {
                    field_ord: field.field_ord(),
                    detail: "token document ordinals descend",
                });
            }
            while accumulator
                .document_ords()
                .get(completed_index)
                .is_some_and(|completed| *completed < doc_ord)
            {
                completed_index += 1;
            }
            if accumulator.document_ords().get(completed_index) != Some(&doc_ord) {
                return Err(FlushError::InvalidTokenColumn {
                    field_ord: field.field_ord(),
                    detail: "token references an incomplete document",
                });
            }
            previous_doc = Some(doc_ord);
            rows.push(FlushTokenRow {
                term_id,
                doc_ord,
                position: field.positions().map_or(0, |positions| positions[index]),
            });
        }
    }
    Ok(rows)
}

pub(super) fn stable_radix_partition(
    rows: Vec<FlushTokenRow>,
    term_count: usize,
) -> Result<RadixPartition, FlushError> {
    let chunk_count = rayon::current_num_threads()
        .min(rows.len().div_ceil(PARALLEL_RADIX_ROWS_PER_CHUNK))
        .max(1);
    stable_radix_partition_with_chunks(rows, term_count, chunk_count)
}

pub(super) fn stable_radix_partition_for_mode(
    rows: Vec<FlushTokenRow>,
    term_count: usize,
    mode: FlushMode,
) -> Result<RadixPartition, FlushError> {
    match mode {
        FlushMode::Automatic => stable_radix_partition(rows, term_count),
        FlushMode::Scalar => stable_radix_partition_serial(rows, term_count),
    }
}

pub(super) fn stable_radix_partition_with_chunks(
    rows: Vec<FlushTokenRow>,
    term_count: usize,
    chunk_count: usize,
) -> Result<RadixPartition, FlushError> {
    let chunk_count = chunk_count.max(1).min(rows.len().max(1));
    let local_range_bytes = chunk_count
        .checked_mul(term_count)
        .and_then(|count| count.checked_mul(std::mem::size_of::<Range<usize>>()));
    if chunk_count == 1
        || rows.is_empty()
        || local_range_bytes.is_none_or(|bytes| bytes > MAX_PARALLEL_RADIX_RANGE_BYTES)
    {
        return stable_radix_partition_serial(rows, term_count);
    }
    let row_count = rows.len();
    let chunk_len = rows.len().div_ceil(chunk_count);
    let local_partitions: Result<Vec<_>, FlushError> = rows
        .par_chunks(chunk_len)
        .map(|chunk| {
            let mut owned = Vec::new();
            owned
                .try_reserve_exact(chunk.len())
                .map_err(|_| FlushError::Allocation {
                    resource: "thread-local radix rows",
                    count: chunk.len(),
                })?;
            owned.extend_from_slice(chunk);
            stable_radix_partition_serial(owned, term_count)
        })
        .collect();
    let local_partitions = local_partitions?;
    drop(rows);

    let mut counts = filled_vec(term_count, 0_usize, "global radix histogram")?;
    for partition in &local_partitions {
        for (count, range) in counts.iter_mut().zip(&partition.ranges) {
            *count = count
                .checked_add(range.len())
                .ok_or(FlushError::ArithmeticOverflow {
                    field: "global radix histogram count",
                })?;
        }
    }
    let mut ranges = Vec::new();
    ranges
        .try_reserve_exact(term_count)
        .map_err(|_| FlushError::Allocation {
            resource: "global radix prefix ranges",
            count: term_count,
        })?;
    let mut cursor = 0_usize;
    for count in counts {
        let end = cursor
            .checked_add(count)
            .ok_or(FlushError::ArithmeticOverflow {
                field: "global radix prefix sum",
            })?;
        ranges.push(cursor..end);
        cursor = end;
    }
    if cursor != row_count {
        return Err(FlushError::ArithmeticOverflow {
            field: "global radix prefix coverage",
        });
    }

    let mut output = filled_vec(row_count, FlushTokenRow::default(), "parallel radix output")?;
    // Split the destination only at term boundaries, then give each disjoint
    // slice to one Rayon callback. Every callback copies its term runs from
    // source chunks in original chunk order: O(rows + chunks * terms), stable,
    // and free of locks, atomics, or unsafe aliasing.
    let term_groups = local_partitions.len().min(term_count).max(1);
    let terms_per_group = term_count.div_ceil(term_groups);
    let mut groups = Vec::new();
    groups
        .try_reserve_exact(term_groups)
        .map_err(|_| FlushError::Allocation {
            resource: "parallel radix output groups",
            count: term_groups,
        })?;
    let mut output_tail = output.as_mut_slice();
    let mut previous_output_end = 0_usize;
    let mut term_start = 0_usize;
    while term_start < term_count {
        let term_end = term_start.saturating_add(terms_per_group).min(term_count);
        let output_end = ranges
            .get(term_end - 1)
            .ok_or(FlushError::ArithmeticOverflow {
                field: "parallel radix group term range",
            })?
            .end;
        let group_len =
            output_end
                .checked_sub(previous_output_end)
                .ok_or(FlushError::ArithmeticOverflow {
                    field: "parallel radix group output length",
                })?;
        let (group, tail) = output_tail.split_at_mut(group_len);
        groups.push((term_start, term_end, group));
        output_tail = tail;
        previous_output_end = output_end;
        term_start = term_end;
    }
    if !output_tail.is_empty() || previous_output_end != row_count {
        return Err(FlushError::ArithmeticOverflow {
            field: "parallel radix output group coverage",
        });
    }
    groups
        .into_par_iter()
        .try_for_each(|(term_start, term_end, destination)| {
            let mut destination_offset = 0_usize;
            for term_id in term_start..term_end {
                for partition in &local_partitions {
                    let source_range =
                        partition
                            .ranges
                            .get(term_id)
                            .ok_or(FlushError::ArithmeticOverflow {
                                field: "thread-local radix term range",
                            })?;
                    let source = partition.rows.get(source_range.clone()).ok_or(
                        FlushError::ArithmeticOverflow {
                            field: "thread-local radix source range",
                        },
                    )?;
                    let destination_end = destination_offset.checked_add(source.len()).ok_or(
                        FlushError::ArithmeticOverflow {
                            field: "parallel radix destination range",
                        },
                    )?;
                    let target = destination
                        .get_mut(destination_offset..destination_end)
                        .ok_or(FlushError::ArithmeticOverflow {
                            field: "parallel radix destination coverage",
                        })?;
                    target.copy_from_slice(source);
                    destination_offset = destination_end;
                }
            }
            if destination_offset != destination.len() {
                return Err(FlushError::ArithmeticOverflow {
                    field: "parallel radix group scatter coverage",
                });
            }
            Ok(())
        })?;
    Ok(RadixPartition {
        rows: output,
        ranges,
    })
}

fn stable_radix_partition_serial(
    rows: Vec<FlushTokenRow>,
    term_count: usize,
) -> Result<RadixPartition, FlushError> {
    if rows.is_empty() {
        return Ok(RadixPartition {
            rows,
            ranges: filled_vec(term_count, 0..0, "empty term ranges")?,
        });
    }
    for row in &rows {
        if usize::try_from(row.term_id).map_or(true, |term_id| term_id >= term_count) {
            return Err(FlushError::TermIdOutOfRange {
                field_ord: 0,
                term_id: row.term_id,
            });
        }
    }

    let sorted = if term_count <= RADIX_DIGIT_BUCKETS {
        stable_digit_scatter(&rows, 0, term_count)?.0
    } else {
        // MSD partition first, then one stable low-digit scatter inside each
        // disjoint high-digit range. This preserves source order among equal
        // ids and avoids a term-count-sized second histogram.
        let high_bucket_count = term_count.div_ceil(RADIX_DIGIT_BUCKETS);
        let (high_sorted, high_ranges) =
            stable_digit_scatter(&rows, RADIX_DIGIT_BITS, high_bucket_count)?;
        let mut output = filled_vec(rows.len(), FlushTokenRow::default(), "radix output")?;
        for (high_digit, range) in high_ranges.into_iter().enumerate() {
            if range.is_empty() {
                continue;
            }
            let remaining_terms = term_count.saturating_sub(high_digit * RADIX_DIGIT_BUCKETS);
            let low_bucket_count = remaining_terms.min(RADIX_DIGIT_BUCKETS);
            let (low_sorted, _) =
                stable_digit_scatter(&high_sorted[range.clone()], 0, low_bucket_count)?;
            output[range].copy_from_slice(&low_sorted);
        }
        output
    };

    let mut ranges = filled_vec(term_count, 0..0, "term ranges")?;
    let mut cursor = 0_usize;
    for (term_id, range) in ranges.iter_mut().enumerate() {
        let start = cursor;
        while sorted
            .get(cursor)
            .is_some_and(|row| row.term_id as usize == term_id)
        {
            cursor += 1;
        }
        *range = start..cursor;
    }
    if cursor != sorted.len() {
        return Err(FlushError::ArithmeticOverflow {
            field: "radix partition coverage",
        });
    }
    Ok(RadixPartition {
        rows: sorted,
        ranges,
    })
}

fn stable_digit_scatter(
    rows: &[FlushTokenRow],
    shift: u32,
    bucket_count: usize,
) -> Result<(Vec<FlushTokenRow>, Vec<Range<usize>>), FlushError> {
    let mut counts = filled_vec(bucket_count, 0_usize, "radix histogram")?;
    for row in rows {
        let digit = ((row.term_id >> shift) & 0xffff) as usize;
        let count = counts
            .get_mut(digit)
            .ok_or(FlushError::ArithmeticOverflow {
                field: "radix digit domain",
            })?;
        *count = count.checked_add(1).ok_or(FlushError::ArithmeticOverflow {
            field: "radix histogram count",
        })?;
    }

    let mut ranges = Vec::new();
    ranges
        .try_reserve_exact(bucket_count)
        .map_err(|_| FlushError::Allocation {
            resource: "radix prefix ranges",
            count: bucket_count,
        })?;
    let mut cursor = 0_usize;
    for count in counts {
        let end = cursor
            .checked_add(count)
            .ok_or(FlushError::ArithmeticOverflow {
                field: "radix prefix sum",
            })?;
        ranges.push(cursor..end);
        cursor = end;
    }
    if cursor != rows.len() {
        return Err(FlushError::ArithmeticOverflow {
            field: "radix prefix coverage",
        });
    }
    let mut write_offsets = Vec::new();
    write_offsets
        .try_reserve_exact(ranges.len())
        .map_err(|_| FlushError::Allocation {
            resource: "radix write offsets",
            count: ranges.len(),
        })?;
    write_offsets.extend(ranges.iter().map(|range| range.start));
    let mut output = filled_vec(rows.len(), FlushTokenRow::default(), "radix scatter")?;
    for row in rows {
        let digit = ((row.term_id >> shift) & 0xffff) as usize;
        let destination = write_offsets
            .get_mut(digit)
            .ok_or(FlushError::ArithmeticOverflow {
                field: "radix scatter digit",
            })?;
        output[*destination] = *row;
        *destination += 1;
    }
    Ok((output, ranges))
}

pub(super) type OrderedTermStreams<'a> = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<TermInput<'a>>);

pub(super) fn encode_ordered_term_streams<'a, A: TokenAnalyzer>(
    accumulator: &'a ColumnarAccumulator<A>,
    partition: &RadixPartition,
    lease_docid_base: u64,
    docid_lo: u64,
    doclen_columns: &[Vec<Option<u32>>],
) -> Result<OrderedTermStreams<'a>, FlushError> {
    let sorted_ids = accumulator.terms().sorted_ids();
    let mut postings_bytes = Vec::new();
    let mut positions_bytes = Vec::new();
    let mut blockmax_bytes = Vec::new();
    let mut inputs = Vec::new();
    inputs
        .try_reserve_exact(sorted_ids.len())
        .map_err(|_| FlushError::Allocation {
            resource: "TERMDICT inputs",
            count: sorted_ids.len(),
        })?;

    for term_id_u32 in sorted_ids {
        let term_id = usize::try_from(term_id_u32).map_err(|_| FlushError::ArithmeticOverflow {
            field: "local term host index",
        })?;
        let range = partition
            .ranges
            .get(term_id)
            .ok_or(FlushError::TermHasNoRows {
                term_id: term_id_u32,
            })?;
        if range.is_empty() {
            return Err(FlushError::TermHasNoRows {
                term_id: term_id_u32,
            });
        }
        let rows = &partition.rows[range.clone()];
        let field_ord = accumulator.terms().field_and_term(term_id_u32).0;
        let field_index = accumulator
            .fields()
            .binary_search_by_key(&field_ord, FieldTokenColumns::field_ord)
            .map_err(|_| FlushError::InvalidTokenColumn {
                field_ord,
                detail: "term references a non-indexed field",
            })?;
        let field = &accumulator.fields()[field_index];
        let (postings, positions) = build_term_rows(
            term_id_u32,
            rows,
            lease_docid_base,
            field.positions().is_some(),
        )?;
        let field_doclens =
            doclen_columns
                .get(field_index)
                .ok_or(FlushError::InvalidTokenColumn {
                    field_ord,
                    detail: "missing DOCLEN source column",
                })?;
        let (field_ord, term) = accumulator.terms().field_and_term(term_id_u32);
        append_canonical_term(
            &mut postings_bytes,
            &mut positions_bytes,
            &mut blockmax_bytes,
            &mut inputs,
            field_ord,
            term,
            &postings,
            positions.as_deref(),
            docid_lo,
            field_doclens,
        )?;
    }
    Ok((postings_bytes, positions_bytes, blockmax_bytes, inputs))
}

pub(super) fn encode_delta_term_streams<'a>(
    snapshot: &'a DeltaSnapshot,
    docid_lo: u64,
    expected_field_ords: &[u16],
    doclen_columns: &[Vec<Option<u32>>],
) -> Result<OrderedTermStreams<'a>, FlushError> {
    let term_count = snapshot.segment().term_count();
    let mut postings_bytes = Vec::new();
    let mut positions_bytes = Vec::new();
    let mut blockmax_bytes = Vec::new();
    let mut inputs = Vec::new();
    inputs
        .try_reserve_exact(term_count)
        .map_err(|_| FlushError::Allocation {
            resource: "Delta TERMDICT inputs",
            count: term_count,
        })?;

    for term in snapshot.ordered_terms() {
        let field_ord = term.field_ord();
        let field_index = expected_field_ords.binary_search(&field_ord).map_err(|_| {
            FlushError::InvalidTokenColumn {
                field_ord,
                detail: "Delta term references a non-indexed field",
            }
        })?;
        let field_doclens =
            doclen_columns
                .get(field_index)
                .ok_or(FlushError::InvalidTokenColumn {
                    field_ord,
                    detail: "missing Delta DOCLEN source column",
                })?;
        let stores_positions = snapshot
            .schema()
            .fields
            .get(usize::from(field_ord))
            .is_some_and(|field| {
                matches!(
                    field.kind,
                    FieldKind::Text {
                        positions: true,
                        ..
                    }
                )
            });
        let live_doc_freq = term.live_doc_freq();
        if live_doc_freq == 0 {
            continue;
        }
        let mut postings = Vec::new();
        postings
            .try_reserve_exact(live_doc_freq)
            .map_err(|_| FlushError::Allocation {
                resource: "Delta term postings",
                count: live_doc_freq,
            })?;
        let mut positions = stores_positions.then(Vec::new);
        for posting in term.postings() {
            if !term.is_live(posting) {
                continue;
            }
            postings.push(Posting::new(posting.global_docid, posting.frequency));
            if let Some(values) = &mut positions {
                let count = usize::try_from(posting.frequency).map_err(|_| {
                    FlushError::ArithmeticOverflow {
                        field: "Delta term position count",
                    }
                })?;
                values
                    .try_reserve(count)
                    .map_err(|_| FlushError::Allocation {
                        resource: "Delta term positions",
                        count,
                    })?;
                let resolved = term
                    .positions(posting)
                    .ok_or(FlushError::InvalidTokenColumn {
                        field_ord,
                        detail: "position-indexed live Delta posting omitted positions",
                    })?;
                values.extend(resolved);
            }
        }
        append_canonical_term(
            &mut postings_bytes,
            &mut positions_bytes,
            &mut blockmax_bytes,
            &mut inputs,
            field_ord,
            term.term(),
            &postings,
            positions.as_deref(),
            docid_lo,
            field_doclens,
        )?;
    }
    Ok((postings_bytes, positions_bytes, blockmax_bytes, inputs))
}

#[allow(clippy::too_many_arguments)]
fn append_canonical_term<'a>(
    postings_bytes: &mut Vec<u8>,
    positions_bytes: &mut Vec<u8>,
    blockmax_bytes: &mut Vec<u8>,
    inputs: &mut Vec<TermInput<'a>>,
    field_ord: u16,
    term: &'a [u8],
    postings: &[Posting],
    positions: Option<&[u32]>,
    docid_lo: u64,
    field_doclens: &[Option<u32>],
) -> Result<(), FlushError> {
    let (encoded_postings, encoded_blockmax) =
        EncodedPostingList::encode_with_block_max(postings, |global_docid| {
            let relative = u64::from(global_docid).checked_sub(docid_lo)?;
            let relative = usize::try_from(relative).ok()?;
            field_doclens
                .get(relative)
                .copied()
                .flatten()
                .map(fieldnorm_to_id)
        })?;
    let encoded_positions = positions
        .map(|values| EncodedPositionList::encode(postings, values))
        .transpose()?;
    let postings_span = append_span(postings_bytes, encoded_postings.as_bytes(), "POSTINGS span")?;
    let positions_span = encoded_positions
        .as_ref()
        .map(|encoded| append_span(positions_bytes, encoded.as_bytes(), "POSITIONS span"))
        .transpose()?;
    let blockmax_span = append_span(blockmax_bytes, encoded_blockmax.as_bytes(), "BLOCKMAX span")?;
    let doc_freq = encoded_postings.doc_freq();
    let metadata = positions_span.map_or_else(
        || TermMetadata::without_positions(doc_freq, postings_span, blockmax_span),
        |positions_span| {
            TermMetadata::with_positions(doc_freq, postings_span, positions_span, blockmax_span)
        },
    );
    inputs.push(TermInput::new(field_ord, term, metadata));
    Ok(())
}

fn build_term_rows(
    term_id: u32,
    rows: &[FlushTokenRow],
    lease_docid_base: u64,
    stores_positions: bool,
) -> Result<(Vec<Posting>, Option<Vec<u32>>), FlushError> {
    let posting_count = rows
        .windows(2)
        .filter(|pair| pair[0].doc_ord != pair[1].doc_ord)
        .count()
        .saturating_add(1);
    let mut postings = Vec::new();
    postings
        .try_reserve_exact(posting_count)
        .map_err(|_| FlushError::Allocation {
            resource: "term postings",
            count: posting_count,
        })?;
    let mut positions = if stores_positions {
        let mut values = Vec::new();
        values
            .try_reserve_exact(rows.len())
            .map_err(|_| FlushError::Allocation {
                resource: "term positions",
                count: rows.len(),
            })?;
        Some(values)
    } else {
        None
    };
    let first = rows.first().ok_or(FlushError::TermHasNoRows { term_id })?;
    let mut current_doc = first.doc_ord;
    let mut frequency = 0_u32;
    let mut previous_position = None;
    for (index, row) in rows.iter().enumerate() {
        if row.term_id != term_id {
            return Err(FlushError::TermIdOutOfRange {
                field_ord: 0,
                term_id: row.term_id,
            });
        }
        if row.doc_ord < current_doc {
            return Err(FlushError::NonAscendingTermDocuments {
                term_id,
                index,
                previous: current_doc,
                current: row.doc_ord,
            });
        }
        if row.doc_ord != current_doc {
            postings.push(Posting::new(
                u32::try_from(rebase_doc_id(lease_docid_base, current_doc)?).map_err(|_| {
                    FlushError::DocumentIdOverflow {
                        lease_docid_base,
                        doc_ord: current_doc,
                    }
                })?,
                frequency,
            ));
            current_doc = row.doc_ord;
            frequency = 0;
            previous_position = None;
        }
        if let Some(previous) = previous_position
            && row.position < previous
        {
            return Err(FlushError::NonAscendingPositions {
                term_id,
                doc_ord: row.doc_ord,
                previous,
                current: row.position,
            });
        }
        frequency = frequency
            .checked_add(1)
            .ok_or(FlushError::TermFrequencyOverflow {
                term_id,
                doc_ord: row.doc_ord,
            })?;
        if let Some(positions) = &mut positions {
            positions.push(row.position);
            previous_position = Some(row.position);
        }
    }
    postings.push(Posting::new(
        u32::try_from(rebase_doc_id(lease_docid_base, current_doc)?).map_err(|_| {
            FlushError::DocumentIdOverflow {
                lease_docid_base,
                doc_ord: current_doc,
            }
        })?,
        frequency,
    ));
    Ok((postings, positions))
}

fn append_span(
    output: &mut Vec<u8>,
    bytes: &[u8],
    field: &'static str,
) -> Result<ByteSpan, FlushError> {
    let offset = durable_len(output, field)?;
    output
        .try_reserve(bytes.len())
        .map_err(|_| FlushError::Allocation {
            resource: field,
            count: bytes.len(),
        })?;
    output.extend_from_slice(bytes);
    Ok(ByteSpan::new(offset, durable_len(bytes, field)?))
}

pub(super) fn filled_vec<T: Clone>(
    len: usize,
    value: T,
    resource: &'static str,
) -> Result<Vec<T>, FlushError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(len)
        .map_err(|_| FlushError::Allocation {
            resource,
            count: len,
        })?;
    output.resize(len, value);
    Ok(output)
}
