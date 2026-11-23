use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use asupersync::Cx;
use asupersync::runtime::spawn_blocking;
use asupersync::sync::OwnedMutexGuard;

use crate::grimoire::{
    ByteSpan, EncodedTermDictionary, TermInput, TermMetadata, TermSectionLengths,
};
use crate::quiver::{
    BlockMaxConcatList, DocLenSection, EncodedBlockMax, EncodedDocLenSection, EncodedIdHashSection,
    EncodedIdMapSection, EncodedNumericSection, EncodedPositionList, EncodedPostingList,
    EncodedStatsSection, EncodedStoredMetaSection, FieldStats, IdMapEntry, IdMapSection,
    NumericSection, PositionList, PostingList, StatsSection, StoredMetaSection,
    aggregate_field_stats,
};
use crate::schema::{FieldKind, SchemaDescriptor};
use crate::segment::{
    EncodedSegment, PendingSegmentFile, PlannedSection, SectionKind, SegmentAssembler,
    SegmentHeaderInput, SegmentReader,
};

use super::admission::WriterAdmissionInner;
use super::compaction::{ConcatMergeArtifact, ConcatSource};
use super::constants::CURRENT_ENGINE_VERSION;
use super::errors::{ConcatMergeError, KeeperError};
use super::gc::canonical_segment_name;
use super::manifest::{ManifestSegment, TombstoneSet};
use super::manifest_load::VerifiedProposalSegment;
use super::manifest_validation::{TombstoneContainers, array_value};
use super::publish::publish_lock_for_normalized_directory;
use super::recovered_segment::RecoveredSegment;
#[cfg(not(windows))]
use super::slot_io::sync_directory;
use super::snapshot::KeeperSnapshot;
use super::writer::WriterProtection;

#[derive(Debug, Eq, PartialEq)]
struct ConcatTermHeapItem {
    field_ord: u16,
    term: Vec<u8>,
    source_index: usize,
    term_index: usize,
}

impl Ord for ConcatTermHeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .field_ord
            .cmp(&self.field_ord)
            .then_with(|| other.term.cmp(&self.term))
            .then_with(|| other.source_index.cmp(&self.source_index))
            .then_with(|| other.term_index.cmp(&self.term_index))
    }
}

impl PartialOrd for ConcatTermHeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
pub(in crate::keeper) struct MergedTermRow {
    pub(in crate::keeper) field_ord: u16,
    pub(in crate::keeper) term: Vec<u8>,
    pub(in crate::keeper) metadata: TermMetadata,
}

type MergedTermSections = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<MergedTermRow>);

#[derive(Clone, Copy, Debug)]
struct IdentityRepresentative {
    lowest_global_docid: u32,
    live_global_docid: Option<u32>,
}

pub(in crate::keeper) fn build_concat_merge(
    snapshot: &KeeperSnapshot,
    source_segment_ids: &[u64],
    output_segment_id: u64,
    created_unix_s: i64,
) -> Result<ConcatMergeArtifact, ConcatMergeError> {
    if source_segment_ids.len() < 2 {
        return Err(ConcatMergeError::TooFewSources {
            count: source_segment_ids.len(),
        });
    }
    let segments = snapshot.segments();
    let first_id = source_segment_ids[0];
    let start = segments
        .iter()
        .position(|segment| segment.manifest.segment_id == first_id)
        .ok_or(ConcatMergeError::SourceNotFound {
            position: 0,
            segment_id: first_id,
        })?;
    for (position, &actual) in source_segment_ids.iter().enumerate() {
        let expected_index = start
            .checked_add(position)
            .ok_or(ConcatMergeError::SourceRunPastManifest { position, actual })?;
        let Some(expected) = segments.get(expected_index) else {
            if segments
                .iter()
                .any(|segment| segment.manifest.segment_id == actual)
            {
                return Err(ConcatMergeError::SourceRunPastManifest { position, actual });
            }
            return Err(ConcatMergeError::SourceNotFound {
                position,
                segment_id: actual,
            });
        };
        if expected.manifest.segment_id != actual {
            if segments
                .iter()
                .any(|segment| segment.manifest.segment_id == actual)
            {
                return Err(ConcatMergeError::NonConsecutiveSources {
                    position,
                    expected: expected.manifest.segment_id,
                    actual,
                });
            }
            return Err(ConcatMergeError::SourceNotFound {
                position,
                segment_id: actual,
            });
        }
    }
    if segments
        .iter()
        .any(|segment| segment.manifest.segment_id == output_segment_id)
    {
        return Err(ConcatMergeError::OutputSegmentCollision {
            segment_id: output_segment_id,
        });
    }
    let end = start.checked_add(source_segment_ids.len()).ok_or(
        ConcatMergeError::ArithmeticOverflow {
            field: "source manifest slice end",
        },
    )?;
    let selected =
        segments
            .get(start..end)
            .ok_or_else(|| ConcatMergeError::SourceRunPastManifest {
                position: source_segment_ids.len(),
                actual: *source_segment_ids.last().unwrap_or(&first_id),
            })?;
    for segment in selected {
        if let Some(entry) = segment.section_entries().iter().find(|entry| {
            !(SectionKind::TERMDICT.raw()..=SectionKind::STATS.raw()).contains(&entry.kind.raw())
        }) {
            return Err(ConcatMergeError::UnknownSection {
                segment_id: segment.manifest.segment_id,
                section_kind: entry.kind.raw(),
            });
        }
        segment
            .verify()
            .map_err(|source| ConcatMergeError::Segment { source })?;
    }

    let schema = snapshot.schema();
    let term_fields = concat_term_field_ords(schema)?;
    let stored_fields = concat_stored_field_ords(schema)?;
    let has_positions = concat_schema_has_positions(schema);
    let has_numeric = concat_schema_has_numeric(schema);
    let mut sources = Vec::new();
    sources
        .try_reserve_exact(selected.len())
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "source views",
            count: selected.len(),
        })?;
    for segment in selected {
        sources.push(open_concat_source(
            segment,
            schema,
            &term_fields,
            &stored_fields,
            has_positions,
            has_numeric,
        )?);
    }

    let docid_lo = selected
        .first()
        .map(|segment| segment.manifest.docid_lo)
        .ok_or(ConcatMergeError::TooFewSources { count: 0 })?;
    let docid_hi = selected
        .last()
        .map(|segment| segment.manifest.docid_hi)
        .ok_or(ConcatMergeError::TooFewSources { count: 0 })?;
    let doc_count_u64 = selected.iter().try_fold(0_u64, |count, segment| {
        count
            .checked_add(u64::from(segment.manifest.doc_count))
            .ok_or(ConcatMergeError::ArithmeticOverflow {
                field: "physical document count",
            })
    })?;
    let doc_count =
        u32::try_from(doc_count_u64).map_err(|_| ConcatMergeError::ArithmeticOverflow {
            field: "durable physical document count",
        })?;

    let (postings, positions, blockmax, merged_terms) =
        merge_concat_terms(&sources, has_positions)?;
    let mut term_inputs = Vec::new();
    term_inputs
        .try_reserve_exact(merged_terms.len())
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "TERMDICT inputs",
            count: merged_terms.len(),
        })?;
    for term in &merged_terms {
        term_inputs.push(TermInput::new(term.field_ord, &term.term, term.metadata));
    }
    let termdict = EncodedTermDictionary::encode_sorted(
        schema,
        TermSectionLengths {
            postings: durable_concat_len(&postings, "POSTINGS")?,
            positions: has_positions
                .then(|| durable_concat_len(&positions, "POSITIONS"))
                .transpose()?,
            blockmax: durable_concat_len(&blockmax, "BLOCKMAX")?,
        },
        &term_inputs,
    )
    .map_err(|error| concat_codec(SectionKind::TERMDICT, error))?;

    let mut doclen_sections = Vec::new();
    doclen_sections
        .try_reserve_exact(sources.len())
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "DOCLEN source views",
            count: sources.len(),
        })?;
    doclen_sections.extend(sources.iter().map(|source| source.doclen.clone()));
    let doclen = EncodedDocLenSection::concatenate(&doclen_sections, &term_fields)
        .map_err(|error| concat_codec(SectionKind::DOCLEN, error))?;
    let mut id_map_sections = Vec::new();
    id_map_sections
        .try_reserve_exact(sources.len())
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "IDMAP source views",
            count: sources.len(),
        })?;
    id_map_sections.extend(sources.iter().map(|source| source.id_map));
    let id_map_plan = EncodedIdMapSection::plan_concatenate(&id_map_sections)
        .map_err(|error| concat_codec(SectionKind::IDMAP, error))?;

    let tombstone_docids = collect_concat_tombstones(&sources)?;
    let tombstones = TombstoneSet::from_sorted_docids(&tombstone_docids).map_err(|error| {
        ConcatMergeError::InvalidManifest {
            detail: error.to_string(),
        }
    })?;
    let representative_ordinals = resolve_concat_representatives_from_sources(
        &id_map_sections,
        id_map_plan.docid_lo(),
        &tombstone_docids,
    )?;
    let id_hash = EncodedIdHashSection::encode_resolved_concat(
        &id_map_sections,
        &id_map_plan,
        &representative_ordinals,
    )
    .map_err(|error| concat_codec(SectionKind::IDHASH, error))?;

    let numeric = if has_numeric {
        let mut numeric_sections = Vec::new();
        numeric_sections
            .try_reserve_exact(sources.len())
            .map_err(|_| ConcatMergeError::Allocation {
                resource: "NUMERIC source views",
                count: sources.len(),
            })?;
        for source in &sources {
            numeric_sections.push(
                source
                    .numeric
                    .clone()
                    .ok_or_else(|| concat_missing_section(source.segment, SectionKind::NUMERIC))?,
            );
        }
        Some(
            EncodedNumericSection::merge_sorted(schema, &numeric_sections)
                .map_err(|error| concat_codec(SectionKind::NUMERIC, error))?,
        )
    } else {
        None
    };
    let mut stored_sections = Vec::new();
    let stored_meta_plan =
        if stored_fields.is_empty() {
            None
        } else {
            stored_sections
                .try_reserve_exact(sources.len())
                .map_err(|_| ConcatMergeError::Allocation {
                    resource: "STOREDMETA source views",
                    count: sources.len(),
                })?;
            for source in &sources {
                stored_sections.push(source.stored_meta.clone().ok_or_else(|| {
                    concat_missing_section(source.segment, SectionKind::STOREDMETA)
                })?);
            }
            Some(
                EncodedStoredMetaSection::plan_concatenate(&stored_sections, &stored_fields)
                    .map_err(|error| concat_codec(SectionKind::STOREDMETA, error))?,
            )
        };

    let aggregate = aggregate_field_stats(sources.iter().map(|source| &source.stats))
        .map_err(|error| concat_codec(SectionKind::STATS, error))?;
    let mut stats_rows = Vec::new();
    stats_rows
        .try_reserve_exact(aggregate.len())
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "STATS rows",
            count: aggregate.len(),
        })?;
    for row in aggregate {
        if row.doc_count != doc_count_u64 {
            return Err(ConcatMergeError::SectionCodec {
                section: SectionKind::STATS,
                detail: format!(
                    "field {} denominator {} differs from merged physical count {doc_count_u64}",
                    row.field_ord, row.doc_count
                ),
            });
        }
        stats_rows.push(FieldStats::new(row.field_ord, row.total_tokens, doc_count));
    }
    let stats = EncodedStatsSection::encode(&term_fields, &stats_rows, doc_count)
        .map_err(|error| concat_codec(SectionKind::STATS, error))?;

    let mut section_plan = Vec::new();
    section_plan
        .try_reserve_exact(10)
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "FSLX section plan",
            count: 10,
        })?;
    section_plan.push(PlannedSection::new(
        SectionKind::TERMDICT,
        termdict.as_bytes().len(),
    ));
    section_plan.push(PlannedSection::new(SectionKind::POSTINGS, postings.len()));
    if has_positions {
        section_plan.push(PlannedSection::new(SectionKind::POSITIONS, positions.len()));
    }
    section_plan.push(PlannedSection::new(SectionKind::BLOCKMAX, blockmax.len()));
    section_plan.push(PlannedSection::new(
        SectionKind::DOCLEN,
        doclen.as_bytes().len(),
    ));
    section_plan.push(PlannedSection::new(
        SectionKind::IDMAP,
        id_map_plan.encoded_len(),
    ));
    section_plan.push(PlannedSection::new(
        SectionKind::IDHASH,
        id_hash.as_bytes().len(),
    ));
    if let Some(numeric) = &numeric {
        section_plan.push(PlannedSection::new(
            SectionKind::NUMERIC,
            numeric.as_bytes().len(),
        ));
    }
    if let Some(stored_meta_plan) = &stored_meta_plan {
        section_plan.push(PlannedSection::new(
            SectionKind::STOREDMETA,
            stored_meta_plan.encoded_len(),
        ));
    }
    section_plan.push(PlannedSection::new(
        SectionKind::STATS,
        stats.as_bytes().len(),
    ));

    let mut assembler = SegmentAssembler::new(
        SegmentHeaderInput {
            segment_id: output_segment_id,
            schema,
            docid_lo,
            docid_hi,
            doc_count,
            created_unix_s,
            engine_version: CURRENT_ENGINE_VERSION,
        },
        &section_plan,
    )
    .map_err(|source| ConcatMergeError::Segment { source })?;
    assembler
        .copy_section(SectionKind::TERMDICT, termdict.as_bytes())
        .map_err(|source| ConcatMergeError::Segment { source })?;
    assembler
        .copy_section(SectionKind::POSTINGS, &postings)
        .map_err(|source| ConcatMergeError::Segment { source })?;
    if has_positions {
        assembler
            .copy_section(SectionKind::POSITIONS, &positions)
            .map_err(|source| ConcatMergeError::Segment { source })?;
    }
    assembler
        .copy_section(SectionKind::BLOCKMAX, &blockmax)
        .map_err(|source| ConcatMergeError::Segment { source })?;
    assembler
        .copy_section(SectionKind::DOCLEN, doclen.as_bytes())
        .map_err(|source| ConcatMergeError::Segment { source })?;
    assembler
        .write_section(SectionKind::IDMAP, |bytes| {
            id_map_plan.append_to(&id_map_sections, bytes);
        })
        .map_err(|source| ConcatMergeError::Segment { source })?;

    {
        let id_map_bytes = assembler
            .written_section(SectionKind::IDMAP)
            .map_err(|source| ConcatMergeError::Segment { source })?;
        let merged_id_map = IdMapSection::parse(id_map_bytes, docid_lo, docid_hi)
            .map_err(|error| concat_codec(SectionKind::IDMAP, error))?;
        id_hash
            .section(merged_id_map)
            .map_err(|error| concat_codec(SectionKind::IDHASH, error))?;
    }
    assembler
        .copy_section(SectionKind::IDHASH, id_hash.as_bytes())
        .map_err(|source| ConcatMergeError::Segment { source })?;
    if let Some(numeric) = &numeric {
        assembler
            .copy_section(SectionKind::NUMERIC, numeric.as_bytes())
            .map_err(|source| ConcatMergeError::Segment { source })?;
    }
    if let Some(stored_meta_plan) = &stored_meta_plan {
        assembler
            .write_section(SectionKind::STOREDMETA, |bytes| {
                stored_meta_plan.append_to(&stored_sections, &stored_fields, bytes);
            })
            .map_err(|source| ConcatMergeError::Segment { source })?;
    }
    assembler
        .copy_section(SectionKind::STATS, stats.as_bytes())
        .map_err(|source| ConcatMergeError::Segment { source })?;
    let encoded = assembler
        .finish()
        .map_err(|source| ConcatMergeError::Segment { source })?;
    let reader = SegmentReader::from_bytes(encoded.as_bytes(), schema)
        .map_err(|source| ConcatMergeError::Segment { source })?;
    reader
        .verify()
        .map_err(|source| ConcatMergeError::Segment { source })?;

    let next_seal_seq = snapshot
        .loaded
        .manifest
        .segments
        .iter()
        .map(|segment| segment.seal_seq)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or(ConcatMergeError::ArithmeticOverflow {
            field: "seal sequence",
        })?;
    let replacement = ManifestSegment {
        segment_id: output_segment_id,
        seal_seq: next_seal_seq,
        file_len: encoded.file_len(),
        file_xxh3: encoded.file_xxh3(),
        docid_lo,
        docid_hi,
        doc_count,
        tombstones,
    };
    let mut manifest = snapshot.loaded.manifest.clone();
    manifest.generation =
        manifest
            .generation
            .checked_add(1)
            .ok_or(ConcatMergeError::ArithmeticOverflow {
                field: "manifest generation",
            })?;
    manifest.last_publish_unix_s = 0;
    manifest.segments.splice(start..end, [replacement]);
    manifest
        .validate()
        .map_err(|error| ConcatMergeError::InvalidManifest {
            detail: error.to_string(),
        })?;
    Ok(ConcatMergeArtifact { encoded, manifest })
}

pub(in crate::keeper) fn open_concat_source<'a>(
    segment: &'a RecoveredSegment,
    schema: SchemaDescriptor,
    term_fields: &[u16],
    stored_fields: &[u16],
    has_positions: bool,
    has_numeric: bool,
) -> Result<ConcatSource<'a>, ConcatMergeError> {
    let postings = required_concat_section(segment, SectionKind::POSTINGS)?;
    let positions = if has_positions {
        Some(required_concat_section(segment, SectionKind::POSITIONS)?)
    } else {
        None
    };
    let blockmax = required_concat_section(segment, SectionKind::BLOCKMAX)?;
    let termdict = segment
        .term_dictionary(schema)
        .map_err(|error| concat_codec(SectionKind::TERMDICT, error))?;
    let term_count = usize::try_from(termdict.term_count()).map_err(|_| {
        ConcatMergeError::ArithmeticOverflow {
            field: "source term count",
        }
    })?;
    let terms = termdict
        .cursor()
        .and_then(|cursor| cursor.collect_bounded(term_count))
        .map_err(|error| concat_codec(SectionKind::TERMDICT, error))?;

    let manifest = segment.manifest();
    let doclen = DocLenSection::parse(
        required_concat_section(segment, SectionKind::DOCLEN)?,
        manifest.docid_lo,
        manifest.docid_hi,
        term_fields,
    )
    .map_err(|error| concat_codec(SectionKind::DOCLEN, error))?;
    let id_map = IdMapSection::parse(
        required_concat_section(segment, SectionKind::IDMAP)?,
        manifest.docid_lo,
        manifest.docid_hi,
    )
    .map_err(|error| concat_codec(SectionKind::IDMAP, error))?;
    let numeric = if has_numeric {
        Some(
            NumericSection::parse(
                required_concat_section(segment, SectionKind::NUMERIC)?,
                schema,
                manifest.docid_lo,
                manifest.docid_hi,
            )
            .map_err(|error| concat_codec(SectionKind::NUMERIC, error))?,
        )
    } else {
        None
    };
    let stored_meta = if stored_fields.is_empty() {
        None
    } else {
        Some(
            StoredMetaSection::parse(
                required_concat_section(segment, SectionKind::STOREDMETA)?,
                manifest.docid_lo,
                manifest.docid_hi,
                stored_fields,
            )
            .map_err(|error| concat_codec(SectionKind::STOREDMETA, error))?,
        )
    };
    let stats = StatsSection::parse(
        required_concat_section(segment, SectionKind::STATS)?,
        term_fields,
        manifest.doc_count,
    )
    .map_err(|error| concat_codec(SectionKind::STATS, error))?;
    Ok(ConcatSource {
        segment,
        terms,
        postings,
        positions,
        blockmax,
        doclen,
        id_map,
        numeric,
        stored_meta,
        stats,
    })
}

fn merge_concat_terms(
    sources: &[ConcatSource<'_>],
    has_positions: bool,
) -> Result<MergedTermSections, ConcatMergeError> {
    let mut heap = BinaryHeap::new();
    heap.try_reserve(sources.len())
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "term merge heap",
            count: sources.len(),
        })?;
    for (source_index, source) in sources.iter().enumerate() {
        push_concat_term(&mut heap, source, source_index, 0)?;
    }
    let mut postings_output = Vec::new();
    let mut positions_output = Vec::new();
    let mut blockmax_output = Vec::new();
    let total_terms = sources.iter().try_fold(0_usize, |count, source| {
        count
            .checked_add(source.terms.len())
            .ok_or(ConcatMergeError::ArithmeticOverflow {
                field: "source term count sum",
            })
    })?;
    let mut merged_terms = Vec::new();
    merged_terms
        .try_reserve(total_terms)
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "merged term rows",
            count: total_terms,
        })?;

    while let Some(first) = heap.pop() {
        let field_ord = first.field_ord;
        let term = first.term;
        let mut parts = Vec::new();
        parts
            .try_reserve_exact(sources.len())
            .map_err(|_| ConcatMergeError::Allocation {
                resource: "equal-term source rows",
                count: sources.len(),
            })?;
        parts.push((first.source_index, first.term_index));
        push_concat_term(
            &mut heap,
            &sources[first.source_index],
            first.source_index,
            first.term_index + 1,
        )?;
        while heap
            .peek()
            .is_some_and(|next| next.field_ord == field_ord && next.term == term)
        {
            let next = heap.pop().expect("peeked concat term remains present");
            parts.push((next.source_index, next.term_index));
            push_concat_term(
                &mut heap,
                &sources[next.source_index],
                next.source_index,
                next.term_index + 1,
            )?;
        }
        parts.sort_unstable_by_key(|(source_index, _)| *source_index);

        let mut posting_slices = Vec::new();
        posting_slices.try_reserve_exact(parts.len()).map_err(|_| {
            ConcatMergeError::Allocation {
                resource: "term POSTINGS slices",
                count: parts.len(),
            }
        })?;
        for &(source_index, term_index) in &parts {
            let source = &sources[source_index];
            let metadata = source.terms[term_index].metadata;
            posting_slices.push(concat_span(
                source.postings,
                metadata.postings,
                SectionKind::POSTINGS,
            )?);
        }
        let mut posting_lists = Vec::new();
        posting_lists
            .try_reserve_exact(parts.len())
            .map_err(|_| ConcatMergeError::Allocation {
                resource: "source POSTINGS views",
                count: parts.len(),
            })?;
        for (&bytes, &(source_index, term_index)) in posting_slices.iter().zip(&parts) {
            posting_lists.push(
                PostingList::parse(
                    bytes,
                    sources[source_index].terms[term_index].metadata.doc_freq,
                )
                .map_err(|error| concat_codec(SectionKind::POSTINGS, error))?,
            );
        }
        let mut posting_refs = Vec::new();
        posting_refs
            .try_reserve_exact(posting_lists.len())
            .map_err(|_| ConcatMergeError::Allocation {
                resource: "POSTINGS references",
                count: posting_lists.len(),
            })?;
        posting_refs.extend(posting_lists.iter());
        let merged_postings = EncodedPostingList::concatenate(&posting_refs)
            .map_err(|error| concat_codec(SectionKind::POSTINGS, error))?;
        let doc_freq = merged_postings.doc_freq();

        let mut blockmax_lists = Vec::new();
        blockmax_lists.try_reserve_exact(parts.len()).map_err(|_| {
            ConcatMergeError::Allocation {
                resource: "source BLOCKMAX views",
                count: parts.len(),
            }
        })?;
        for ((source_index, term_index), postings) in parts.iter().copied().zip(&posting_lists) {
            let source = &sources[source_index];
            let bytes = concat_span(
                source.blockmax,
                source.terms[term_index].metadata.blockmax,
                SectionKind::BLOCKMAX,
            )?;
            blockmax_lists.push(
                BlockMaxConcatList::parse(bytes, postings)
                    .map_err(|error| concat_codec(SectionKind::BLOCKMAX, error))?,
            );
        }
        let mut blockmax_refs = Vec::new();
        blockmax_refs
            .try_reserve_exact(blockmax_lists.len())
            .map_err(|_| ConcatMergeError::Allocation {
                resource: "BLOCKMAX references",
                count: blockmax_lists.len(),
            })?;
        blockmax_refs.extend(blockmax_lists.iter());
        let merged_blockmax = EncodedBlockMax::concatenate(&blockmax_refs)
            .map_err(|error| concat_codec(SectionKind::BLOCKMAX, error))?;

        let has_term_positions = sources[parts[0].0].terms[parts[0].1]
            .metadata
            .positions
            .is_some();
        let merged_positions = if has_term_positions {
            let mut position_lists = Vec::new();
            position_lists.try_reserve_exact(parts.len()).map_err(|_| {
                ConcatMergeError::Allocation {
                    resource: "source POSITIONS views",
                    count: parts.len(),
                }
            })?;
            for ((source_index, term_index), postings) in parts.iter().copied().zip(&posting_lists)
            {
                let source = &sources[source_index];
                let span = source.terms[term_index].metadata.positions.ok_or_else(|| {
                    ConcatMergeError::SectionCodec {
                        section: SectionKind::POSITIONS,
                        detail: format!(
                            "field {field_ord} term {:?} has inconsistent positional metadata",
                            String::from_utf8_lossy(&term)
                        ),
                    }
                })?;
                let bytes = concat_span(
                    source.positions.ok_or_else(|| {
                        concat_missing_section(source.segment, SectionKind::POSITIONS)
                    })?,
                    span,
                    SectionKind::POSITIONS,
                )?;
                position_lists.push(
                    PositionList::parse(bytes, postings)
                        .map_err(|error| concat_codec(SectionKind::POSITIONS, error))?,
                );
            }
            let mut refs = Vec::new();
            refs.try_reserve_exact(position_lists.len()).map_err(|_| {
                ConcatMergeError::Allocation {
                    resource: "POSITIONS references",
                    count: position_lists.len(),
                }
            })?;
            refs.extend(position_lists.iter());
            Some(
                EncodedPositionList::concatenate(&refs)
                    .map_err(|error| concat_codec(SectionKind::POSITIONS, error))?,
            )
        } else {
            for &(source_index, term_index) in &parts {
                if sources[source_index].terms[term_index]
                    .metadata
                    .positions
                    .is_some()
                {
                    return Err(ConcatMergeError::SectionCodec {
                        section: SectionKind::POSITIONS,
                        detail: "equal term has inconsistent positional metadata".to_owned(),
                    });
                }
            }
            None
        };
        if has_term_positions && !has_positions {
            return Err(ConcatMergeError::SectionCodec {
                section: SectionKind::POSITIONS,
                detail: "term carries positions but schema has no POSITIONS section".to_owned(),
            });
        }

        let postings_offset = durable_concat_len(&postings_output, "POSTINGS offset")?;
        append_concat_bytes(
            &mut postings_output,
            merged_postings.as_bytes(),
            "POSTINGS output",
        )?;
        let blockmax_offset = durable_concat_len(&blockmax_output, "BLOCKMAX offset")?;
        append_concat_bytes(
            &mut blockmax_output,
            merged_blockmax.as_bytes(),
            "BLOCKMAX output",
        )?;
        let positions_span = if let Some(merged_positions) = merged_positions {
            let offset = durable_concat_len(&positions_output, "POSITIONS offset")?;
            let len = durable_concat_len(merged_positions.as_bytes(), "term POSITIONS length")?;
            append_concat_bytes(
                &mut positions_output,
                merged_positions.as_bytes(),
                "POSITIONS output",
            )?;
            Some(ByteSpan::new(offset, len))
        } else {
            None
        };
        let postings_len = durable_concat_len(merged_postings.as_bytes(), "term POSTINGS length")?;
        let blockmax_len = durable_concat_len(merged_blockmax.as_bytes(), "term BLOCKMAX length")?;
        let metadata = positions_span.map_or_else(
            || {
                TermMetadata::without_positions(
                    doc_freq,
                    ByteSpan::new(postings_offset, postings_len),
                    ByteSpan::new(blockmax_offset, blockmax_len),
                )
            },
            |positions_span| {
                TermMetadata::with_positions(
                    doc_freq,
                    ByteSpan::new(postings_offset, postings_len),
                    positions_span,
                    ByteSpan::new(blockmax_offset, blockmax_len),
                )
            },
        );
        merged_terms.push(MergedTermRow {
            field_ord,
            term,
            metadata,
        });
    }
    Ok((
        postings_output,
        positions_output,
        blockmax_output,
        merged_terms,
    ))
}

fn push_concat_term(
    heap: &mut BinaryHeap<ConcatTermHeapItem>,
    source: &ConcatSource<'_>,
    source_index: usize,
    term_index: usize,
) -> Result<(), ConcatMergeError> {
    let Some(term) = source.terms.get(term_index) else {
        return Ok(());
    };
    let mut key = Vec::new();
    key.try_reserve_exact(term.term.len())
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "term heap key",
            count: term.term.len(),
        })?;
    key.extend_from_slice(&term.term);
    heap.push(ConcatTermHeapItem {
        field_ord: term.field_ord,
        term: key,
        source_index,
        term_index,
    });
    Ok(())
}

fn collect_concat_tombstones(sources: &[ConcatSource<'_>]) -> Result<Vec<u32>, ConcatMergeError> {
    let count = sources.iter().try_fold(0_usize, |count, source| {
        let source_count = usize::try_from(source.segment.manifest.tombstones.cardinality())
            .map_err(|_| ConcatMergeError::ArithmeticOverflow {
                field: "host tombstone cardinality",
            })?;
        count
            .checked_add(source_count)
            .ok_or(ConcatMergeError::ArithmeticOverflow {
                field: "merged tombstone cardinality",
            })
    })?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(count)
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "merged tombstone docids",
            count,
        })?;
    for source in sources {
        let mut containers = TombstoneContainers::new(
            source.segment.manifest.tombstones.as_bytes(),
        )
        .map_err(|error| ConcatMergeError::InvalidManifest {
            detail: error.to_string(),
        })?;
        while let Some(container) =
            containers
                .next_container()
                .map_err(|error| ConcatMergeError::InvalidManifest {
                    detail: error.to_string(),
                })?
        {
            match container.kind {
                0 => {
                    for index in 0..container.payload.len() / 2 {
                        output.push(
                            (u32::from(container.chunk_id) << 16)
                                | u32::from(array_value(container.payload, index)),
                        );
                    }
                }
                1 => {
                    for (byte_index, byte) in container.payload.iter().copied().enumerate() {
                        let mut bits = byte;
                        while bits != 0 {
                            let bit = bits.trailing_zeros() as usize;
                            let low = u16::try_from(byte_index * 8 + bit).map_err(|_| {
                                ConcatMergeError::ArithmeticOverflow {
                                    field: "bitmap tombstone low bits",
                                }
                            })?;
                            output.push((u32::from(container.chunk_id) << 16) | u32::from(low));
                            bits &= bits - 1;
                        }
                    }
                }
                _ => {
                    return Err(ConcatMergeError::InvalidManifest {
                        detail: "unknown validated tombstone container kind".to_owned(),
                    });
                }
            }
        }
    }
    if output.len() != count {
        return Err(ConcatMergeError::InvalidManifest {
            detail: format!(
                "decoded tombstone count {} differs from declared {count}",
                output.len()
            ),
        });
    }
    Ok(output)
}

#[cfg(test)]
pub(in crate::keeper) fn resolve_concat_representatives(
    id_map: IdMapSection<'_>,
    tombstone_docids: &[u32],
) -> Result<Vec<u32>, ConcatMergeError> {
    resolve_concat_representatives_from_entries(
        id_map.docid_lo(),
        id_map.entries(),
        tombstone_docids,
    )
}

fn resolve_concat_representatives_from_sources(
    id_maps: &[IdMapSection<'_>],
    docid_lo: u64,
    tombstone_docids: &[u32],
) -> Result<Vec<u32>, ConcatMergeError> {
    resolve_concat_representatives_from_entries(
        docid_lo,
        id_maps.iter().copied().flat_map(|id_map| id_map.entries()),
        tombstone_docids,
    )
}

fn resolve_concat_representatives_from_entries<'a>(
    docid_lo: u64,
    entries: impl IntoIterator<Item = (u64, IdMapEntry<'a>)>,
    tombstone_docids: &[u32],
) -> Result<Vec<u32>, ConcatMergeError> {
    let mut representatives = BTreeMap::<&str, IdentityRepresentative>::new();
    let mut tombstones = tombstone_docids.iter().copied().peekable();
    for (global_docid, entry) in entries {
        let global_docid =
            u32::try_from(global_docid).map_err(|_| ConcatMergeError::ArithmeticOverflow {
                field: "IDHASH representative global docid",
            })?;
        if let Some(&tombstone) = tombstones.peek()
            && tombstone < global_docid
        {
            return Err(ConcatMergeError::InvalidManifest {
                detail: format!(
                    "merged tombstone {tombstone} references an IDMAP hole before row {global_docid}"
                ),
            });
        }
        let is_live = if tombstones.peek().copied() == Some(global_docid) {
            tombstones.next();
            false
        } else {
            true
        };
        match representatives.get_mut(entry.document_id()) {
            Some(representative) => {
                representative.lowest_global_docid =
                    representative.lowest_global_docid.min(global_docid);
                if is_live {
                    if let Some(first_global_docid) = representative.live_global_docid {
                        let mut document_id = String::new();
                        document_id
                            .try_reserve_exact(entry.document_id().len())
                            .map_err(|_| ConcatMergeError::Allocation {
                                resource: "duplicate document id diagnosis",
                                count: entry.document_id().len(),
                            })?;
                        document_id.push_str(entry.document_id());
                        return Err(ConcatMergeError::MultipleLiveDocumentIds {
                            document_id,
                            first_global_docid,
                            duplicate_global_docid: global_docid,
                        });
                    }
                    representative.live_global_docid = Some(global_docid);
                }
            }
            None => {
                representatives.insert(
                    entry.document_id(),
                    IdentityRepresentative {
                        lowest_global_docid: global_docid,
                        live_global_docid: is_live.then_some(global_docid),
                    },
                );
            }
        }
    }
    if let Some(tombstone) = tombstones.next() {
        return Err(ConcatMergeError::InvalidManifest {
            detail: format!(
                "merged tombstone {tombstone} is not represented by a trailing IDMAP row"
            ),
        });
    }
    let mut ordinals = Vec::new();
    ordinals
        .try_reserve_exact(representatives.len())
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "IDHASH representative ordinals",
            count: representatives.len(),
        })?;
    for representative in representatives.values() {
        let global_docid = representative
            .live_global_docid
            .unwrap_or(representative.lowest_global_docid);
        let ordinal = u64::from(global_docid).checked_sub(docid_lo).ok_or(
            ConcatMergeError::ArithmeticOverflow {
                field: "IDHASH representative ordinal",
            },
        )?;
        ordinals.push(u32::try_from(ordinal).map_err(|_| {
            ConcatMergeError::ArithmeticOverflow {
                field: "durable IDHASH representative ordinal",
            }
        })?);
    }
    ordinals.sort_unstable();
    Ok(ordinals)
}

fn required_concat_section(
    segment: &RecoveredSegment,
    kind: SectionKind,
) -> Result<&[u8], ConcatMergeError> {
    segment
        .section(kind)
        .map_err(|source| ConcatMergeError::Segment { source })?
        .ok_or_else(|| concat_missing_section(segment, kind))
}

fn concat_missing_section(segment: &RecoveredSegment, kind: SectionKind) -> ConcatMergeError {
    ConcatMergeError::SectionCodec {
        section: kind,
        detail: format!(
            "source segment {:#018x} is missing the required section",
            segment.manifest.segment_id
        ),
    }
}

pub(in crate::keeper) fn concat_span(
    bytes: &[u8],
    span: ByteSpan,
    section: SectionKind,
) -> Result<&[u8], ConcatMergeError> {
    let start = usize::try_from(span.offset).map_err(|_| ConcatMergeError::ArithmeticOverflow {
        field: "section span offset",
    })?;
    let len = usize::try_from(span.len).map_err(|_| ConcatMergeError::ArithmeticOverflow {
        field: "section span length",
    })?;
    let end = start
        .checked_add(len)
        .ok_or(ConcatMergeError::ArithmeticOverflow {
            field: "section span end",
        })?;
    bytes
        .get(start..end)
        .ok_or_else(|| ConcatMergeError::SectionCodec {
            section,
            detail: format!(
                "referenced byte span [{start}, {end}) exceeds section length {}",
                bytes.len()
            ),
        })
}

fn durable_concat_len(bytes: &[u8], field: &'static str) -> Result<u64, ConcatMergeError> {
    u64::try_from(bytes.len()).map_err(|_| ConcatMergeError::ArithmeticOverflow { field })
}

fn append_concat_bytes(
    output: &mut Vec<u8>,
    bytes: &[u8],
    resource: &'static str,
) -> Result<(), ConcatMergeError> {
    output
        .try_reserve(bytes.len())
        .map_err(|_| ConcatMergeError::Allocation {
            resource,
            count: output.len().saturating_add(bytes.len()),
        })?;
    output.extend_from_slice(bytes);
    Ok(())
}

fn concat_codec(section: SectionKind, error: impl std::fmt::Display) -> ConcatMergeError {
    ConcatMergeError::SectionCodec {
        section,
        detail: error.to_string(),
    }
}

pub(in crate::keeper) fn concat_term_field_ords(
    schema: SchemaDescriptor,
) -> Result<Vec<u16>, ConcatMergeError> {
    let count = schema
        .fields
        .iter()
        .filter(|field| matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. }))
        .count();
    let mut output = Vec::new();
    output
        .try_reserve_exact(count)
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "schema term fields",
            count,
        })?;
    output.extend(schema.fields.iter().filter_map(|field| {
        matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. }).then_some(field.id)
    }));
    Ok(output)
}

pub(in crate::keeper) fn concat_stored_field_ords(
    schema: SchemaDescriptor,
) -> Result<Vec<u16>, ConcatMergeError> {
    let count = schema.fields.iter().filter(|field| field.stored).count();
    let mut output = Vec::new();
    output
        .try_reserve_exact(count)
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "schema stored fields",
            count,
        })?;
    output.extend(
        schema
            .fields
            .iter()
            .filter_map(|field| field.stored.then_some(field.id)),
    );
    Ok(output)
}

pub(in crate::keeper) fn concat_schema_has_positions(schema: SchemaDescriptor) -> bool {
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

pub(in crate::keeper) fn concat_schema_has_numeric(schema: SchemaDescriptor) -> bool {
    schema
        .fields
        .iter()
        .any(|field| field.kind.has_numeric_column())
}

pub(in crate::keeper) fn reconcile_published_segment(
    admission: &WriterAdmissionInner,
    protection: &WriterProtection,
    pending: &PendingSegmentFile,
) -> Result<Option<PathBuf>, KeeperError> {
    let published = admission
        .directory
        .join(canonical_segment_name(pending.segment_id()));
    let published_metadata = match std::fs::symlink_metadata(&published) {
        Ok(metadata) => metadata,
        Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "inspect retry segment destination",
                path: published,
                source,
            });
        }
    };
    let pending_metadata =
        std::fs::symlink_metadata(pending.path()).map_err(|source| KeeperError::Io {
            operation: "inspect retry segment temp",
            path: pending.path().to_path_buf(),
            source,
        })?;
    let expected_file_len = pending.file_len();
    if !published_metadata.file_type().is_file()
        || !pending_metadata.file_type().is_file()
        || published_metadata.len() != expected_file_len
        || pending_metadata.len() != expected_file_len
    {
        return Err(KeeperError::Io {
            operation: "reconcile published segment",
            path: published,
            source: io::Error::new(
                io::ErrorKind::AlreadyExists,
                "published segment and retry temp are not matching regular files",
            ),
        });
    }
    let expected = std::fs::read(pending.path()).map_err(|source| KeeperError::Io {
        operation: "read retry segment temp",
        path: pending.path().to_path_buf(),
        source,
    })?;
    let actual = std::fs::read(&published).map_err(|source| KeeperError::Io {
        operation: "read retry segment destination",
        path: published.clone(),
        source,
    })?;
    if actual != expected {
        return Err(KeeperError::Io {
            operation: "reconcile published segment",
            path: published,
            source: io::Error::new(
                io::ErrorKind::AlreadyExists,
                "published segment differs from the retry temp",
            ),
        });
    }
    File::open(&published)
        .and_then(|file| file.sync_all())
        .map_err(|source| KeeperError::Io {
            operation: "sync reconciled segment",
            path: published.clone(),
            source,
        })?;
    #[cfg(not(windows))]
    sync_directory(&admission.directory)?;
    let _ = protection;
    Ok(Some(published))
}

pub(in crate::keeper) fn reconcile_encoded_segment(
    admission: &WriterAdmissionInner,
    protection: &WriterProtection,
    encoded: &EncodedSegment,
) -> Result<Option<PathBuf>, KeeperError> {
    let published = admission
        .directory
        .join(canonical_segment_name(encoded.header().segment_id));
    let metadata = match std::fs::symlink_metadata(&published) {
        Ok(metadata) => metadata,
        Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "inspect retry segment destination",
                path: published,
                source,
            });
        }
    };
    if !metadata.file_type().is_file() || metadata.len() != encoded.file_len() {
        return Err(KeeperError::Io {
            operation: "reconcile encoded segment",
            path: published,
            source: io::Error::new(
                io::ErrorKind::AlreadyExists,
                "published segment is not the exact expected regular file",
            ),
        });
    }
    let actual = std::fs::read(&published).map_err(|source| KeeperError::Io {
        operation: "read retry segment destination",
        path: published.clone(),
        source,
    })?;
    if actual.as_slice() != encoded.as_bytes() {
        return Err(KeeperError::Io {
            operation: "reconcile encoded segment",
            path: published,
            source: io::Error::new(
                io::ErrorKind::AlreadyExists,
                "published segment differs from the retained canonical bytes",
            ),
        });
    }
    File::open(&published)
        .and_then(|file| file.sync_all())
        .map_err(|source| KeeperError::Io {
            operation: "sync reconciled segment",
            path: published.clone(),
            source,
        })?;
    #[cfg(not(windows))]
    sync_directory(&admission.directory)?;
    let _ = protection;
    Ok(Some(published))
}

pub(in crate::keeper) async fn writer_mutation_guard(
    cx: &Cx,
    directory: &Path,
) -> Result<OwnedMutexGuard<()>, KeeperError> {
    if cx.is_cancel_requested() {
        return Err(KeeperError::WriterAdmissionCancelled);
    }
    let guard = OwnedMutexGuard::lock(
        publish_lock_for_normalized_directory(directory.to_path_buf()),
        cx,
    )
    .await
    .map_err(|source| KeeperError::PublishLock { source })?;
    if cx.is_cancel_requested() {
        return Err(KeeperError::WriterAdmissionCancelled);
    }
    Ok(guard)
}

pub(in crate::keeper) fn recovery_retryable(error: &KeeperError) -> bool {
    matches!(
        error,
        KeeperError::SegmentOpen { .. } | KeeperError::SegmentMetadataMismatch { .. }
    )
}

pub(in crate::keeper) async fn open_snapshot_blocking(
    directory: PathBuf,
    schema: SchemaDescriptor,
) -> Result<KeeperSnapshot, KeeperError> {
    spawn_blocking(move || KeeperSnapshot::open(directory, schema)).await
}

/// The writer's snapshot after its own successful publication: as
/// [`open_snapshot_blocking`], but the segments the publication preflight just
/// verified are not verified again (see [`super::manifest_load::reuse_verified_segments`]).
pub(in crate::keeper) async fn open_published_snapshot_blocking(
    directory: PathBuf,
    schema: SchemaDescriptor,
    verified: Vec<VerifiedProposalSegment>,
) -> Result<KeeperSnapshot, KeeperError> {
    spawn_blocking(
        move || match KeeperSnapshot::open_once_reusing(&directory, schema, verified) {
            Err(error) if recovery_retryable(&error) => KeeperSnapshot::open(&directory, schema),
            result => result,
        },
    )
    .await
}
