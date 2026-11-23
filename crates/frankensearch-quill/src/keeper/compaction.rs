use std::collections::{BTreeMap, BTreeSet};

use crate::grimoire::{
    ByteSpan, EncodedTermDictionary, OwnedTerm, TermInput, TermMetadata, TermSectionLengths,
};
use crate::quiver::{
    DocLenFieldInput, DocLenSection, EncodedDocLenSection, EncodedIdHashSection,
    EncodedIdMapSection, EncodedNumericSection, EncodedPositionList, EncodedPostingList,
    EncodedStatsSection, EncodedStoredMetaSection, FieldStats, IdMapEntryInput, IdMapSection,
    NumericEntry, NumericFieldInput, NumericSection, PositionList, Posting, PostingList,
    StatsSection, StoredMetaFieldInput, StoredMetaSection,
};
use crate::segment::{
    EncodedSegment, PlannedSection, SectionKind, SegmentAssembler, SegmentHeaderInput,
    SegmentReader,
};

use super::concat_merge::{
    MergedTermRow, concat_schema_has_numeric, concat_schema_has_positions, concat_span,
    concat_stored_field_ords, concat_term_field_ords, open_concat_source,
};
use super::constants::CURRENT_ENGINE_VERSION;
use super::errors::{CompactionError, CompactionPolicy, CompactionReport, ConcatMergeError};
use super::manifest::{Manifest, ManifestFieldStats, ManifestSegment, TombstoneSet};
use super::recovered_segment::RecoveredSegment;
use super::snapshot::KeeperSnapshot;

pub(in crate::keeper) struct ConcatMergeArtifact {
    pub(in crate::keeper) encoded: EncodedSegment,
    pub(in crate::keeper) manifest: Manifest,
}

pub(in crate::keeper) struct CompactionArtifact {
    pub(in crate::keeper) encoded: Vec<EncodedSegment>,
    pub(in crate::keeper) manifest: Manifest,
    pub(in crate::keeper) report: CompactionReport,
}

struct CompactionReplacement {
    encoded: Option<EncodedSegment>,
    manifest: Option<ManifestSegment>,
    source_stats: Vec<FieldStats>,
    replacement_stats: Vec<FieldStats>,
}

struct CompactedTermSections {
    postings: Vec<u8>,
    positions: Vec<u8>,
    blockmax: Vec<u8>,
    terms: Vec<MergedTermRow>,
    total_tokens: BTreeMap<u16, u64>,
}

pub(in crate::keeper) struct ConcatSource<'a> {
    pub(in crate::keeper) segment: &'a RecoveredSegment,
    pub(in crate::keeper) terms: Vec<OwnedTerm>,
    pub(in crate::keeper) postings: &'a [u8],
    pub(in crate::keeper) positions: Option<&'a [u8]>,
    pub(in crate::keeper) blockmax: &'a [u8],
    pub(in crate::keeper) doclen: DocLenSection<'a>,
    pub(in crate::keeper) id_map: IdMapSection<'a>,
    pub(in crate::keeper) numeric: Option<NumericSection<'a>>,
    pub(in crate::keeper) stored_meta: Option<StoredMetaSection<'a>>,
    pub(in crate::keeper) stats: StatsSection,
}

pub(in crate::keeper) fn build_compaction(
    snapshot: &KeeperSnapshot,
    policy: CompactionPolicy,
    created_unix_s: i64,
) -> Result<CompactionArtifact, CompactionError> {
    validate_compaction_policy(policy)?;
    let generation_before = snapshot.loaded.manifest.generation;
    let mut report = CompactionReport {
        generation_before,
        generation_after: generation_before,
        examined_segments: snapshot.segments.len(),
        ..CompactionReport::default()
    };
    let mut eligible = Vec::new();
    eligible
        .try_reserve_exact(snapshot.segments.len())
        .map_err(|_| CompactionError::Allocation {
            resource: "eligible segment indexes",
            count: snapshot.segments.len(),
        })?;
    for (index, segment) in snapshot.segments.iter().enumerate() {
        let physical = segment.manifest.doc_count;
        let deleted = segment.manifest.tombstones.cardinality();
        if physical != 0
            && deleted != 0
            && deleted as f64 / f64::from(physical) > policy.tombstone_density
        {
            eligible.push(index);
        }
    }
    if eligible.is_empty() {
        return Ok(CompactionArtifact {
            encoded: Vec::new(),
            manifest: snapshot.loaded.manifest.clone(),
            report,
        });
    }

    let generation_after =
        generation_before
            .checked_add(1)
            .ok_or(CompactionError::ArithmeticOverflow {
                field: "manifest generation",
            })?;
    let mut next_seal_seq = snapshot
        .segments
        .iter()
        .map(|segment| segment.manifest.seal_seq)
        .max()
        .unwrap_or(0);
    let mut used_ids = BTreeSet::new();
    used_ids.extend(
        snapshot
            .segments
            .iter()
            .map(|segment| segment.manifest.segment_id),
    );
    let mut replacements = BTreeMap::new();
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(eligible.len())
        .map_err(|_| CompactionError::Allocation {
            resource: "replacement segments",
            count: eligible.len(),
        })?;
    let mut manifest = snapshot.loaded.manifest.clone();
    manifest.generation = generation_after;
    manifest.last_publish_unix_s = 0;

    for index in eligible {
        let source = &snapshot.segments[index];
        let deleted = source.manifest.tombstones.cardinality();
        let live = u64::from(source.manifest.live_doc_count());
        let output_id = if live == 0 {
            0
        } else {
            derive_compaction_segment_id(
                snapshot,
                source.manifest.segment_id,
                generation_after,
                created_unix_s,
                &mut used_ids,
            )?
        };
        if live != 0 {
            next_seal_seq =
                next_seal_seq
                    .checked_add(1)
                    .ok_or(CompactionError::ArithmeticOverflow {
                        field: "seal sequence",
                    })?;
        }
        let replacement =
            compact_segment(snapshot, source, output_id, next_seal_seq, created_unix_s)?;
        adjust_compaction_field_stats(
            &mut manifest.field_stats,
            &replacement.source_stats,
            &replacement.replacement_stats,
        )?;
        report.compacted_segments += 1;
        report.dropped_documents = report.dropped_documents.checked_add(deleted).ok_or(
            CompactionError::ArithmeticOverflow {
                field: "dropped document count",
            },
        )?;
        report.input_bytes = report
            .input_bytes
            .checked_add(source.manifest.file_len)
            .ok_or(CompactionError::ArithmeticOverflow {
                field: "input bytes",
            })?;
        if let Some(output) = replacement.encoded {
            report.output_bytes = report.output_bytes.checked_add(output.file_len()).ok_or(
                CompactionError::ArithmeticOverflow {
                    field: "output bytes",
                },
            )?;
            encoded.push(output);
        } else {
            report.removed_segments += 1;
        }
        replacements.insert(source.manifest.segment_id, replacement.manifest);
    }

    let mut segments = Vec::new();
    segments
        .try_reserve_exact(manifest.segments.len())
        .map_err(|_| CompactionError::Allocation {
            resource: "successor manifest segments",
            count: manifest.segments.len(),
        })?;
    for segment in &manifest.segments {
        match replacements.remove(&segment.segment_id) {
            Some(Some(replacement)) => segments.push(replacement),
            Some(None) => {}
            None => segments.push(segment.clone()),
        }
    }
    manifest.segments = segments;
    manifest
        .validate()
        .map_err(|error| CompactionError::InvalidManifest {
            detail: error.to_string(),
        })?;
    report.generation_after = generation_after;
    Ok(CompactionArtifact {
        encoded,
        manifest,
        report,
    })
}

fn validate_compaction_policy(policy: CompactionPolicy) -> Result<(), CompactionError> {
    if !policy.tombstone_density.is_finite()
        || policy.tombstone_density <= 0.0
        || policy.tombstone_density > 1.0
    {
        return Err(CompactionError::InvalidDensity {
            density: policy.tombstone_density,
        });
    }
    Ok(())
}

fn derive_compaction_segment_id(
    snapshot: &KeeperSnapshot,
    source_id: u64,
    generation: u64,
    created_unix_s: i64,
    used_ids: &mut BTreeSet<u64>,
) -> Result<u64, CompactionError> {
    for salt in 0_u64..=u64::from(u16::MAX) {
        let mut preimage = [0_u8; 44];
        preimage[..8].copy_from_slice(&snapshot.loaded.manifest.schema_id.to_le_bytes());
        preimage[8..16].copy_from_slice(&generation.to_le_bytes());
        preimage[16..24].copy_from_slice(&source_id.to_le_bytes());
        preimage[24..32].copy_from_slice(&created_unix_s.to_le_bytes());
        preimage[32..36].copy_from_slice(&CURRENT_ENGINE_VERSION.to_le_bytes());
        preimage[36..].copy_from_slice(&salt.to_le_bytes());
        let candidate = xxhash_rust::xxh3::xxh3_64(&preimage);
        if used_ids.insert(candidate) {
            return Ok(candidate);
        }
    }
    Err(CompactionError::OutputIdExhausted { source_id })
}

fn adjust_compaction_field_stats(
    manifest: &mut [ManifestFieldStats],
    source: &[FieldStats],
    replacement: &[FieldStats],
) -> Result<(), CompactionError> {
    if source.len() != replacement.len() || source.len() != manifest.len() {
        return Err(CompactionError::InvalidManifest {
            detail: "compaction field-stat sets differ".to_owned(),
        });
    }
    for ((manifest, source), replacement) in manifest.iter_mut().zip(source).zip(replacement) {
        if manifest.field_ord != source.field_ord || source.field_ord != replacement.field_ord {
            return Err(CompactionError::InvalidManifest {
                detail: "compaction field-stat ordinals differ".to_owned(),
            });
        }
        manifest.total_tokens = manifest
            .total_tokens
            .checked_sub(source.total_tokens)
            .and_then(|value| value.checked_add(replacement.total_tokens))
            .ok_or(CompactionError::ArithmeticOverflow {
                field: "manifest total tokens",
            })?;
        manifest.doc_count = manifest
            .doc_count
            .checked_sub(source.doc_count)
            .and_then(|value| value.checked_add(replacement.doc_count))
            .ok_or(CompactionError::ArithmeticOverflow {
                field: "manifest document count",
            })?;
    }
    Ok(())
}

fn compact_segment(
    snapshot: &KeeperSnapshot,
    segment: &RecoveredSegment,
    output_segment_id: u64,
    seal_seq: u64,
    created_unix_s: i64,
) -> Result<CompactionReplacement, CompactionError> {
    if let Some(entry) = segment.section_entries().iter().find(|entry| {
        !(SectionKind::TERMDICT.raw()..=SectionKind::STATS.raw()).contains(&entry.kind.raw())
    }) {
        return Err(CompactionError::UnknownSection {
            segment_id: segment.manifest.segment_id,
            section_kind: entry.kind.raw(),
        });
    }
    segment
        .verify()
        .map_err(|source| CompactionError::Segment { source })?;
    let schema = snapshot.schema();
    let term_fields =
        concat_term_field_ords(schema).map_err(|error| compaction_source_validation(&error))?;
    let stored_fields =
        concat_stored_field_ords(schema).map_err(|error| compaction_source_validation(&error))?;
    let has_positions = concat_schema_has_positions(schema);
    let has_numeric = concat_schema_has_numeric(schema);
    let source = open_concat_source(
        segment,
        schema,
        &term_fields,
        &stored_fields,
        has_positions,
        has_numeric,
    )
    .map_err(|error| compaction_source_validation(&error))?;
    let source_stats = source.stats.rows().to_vec();
    let doc_count = segment.manifest.live_doc_count();
    if doc_count == 0 {
        let replacement_stats = source_stats
            .iter()
            .map(|row| FieldStats::new(row.field_ord, 0, 0))
            .collect();
        return Ok(CompactionReplacement {
            encoded: None,
            manifest: None,
            source_stats,
            replacement_stats,
        });
    }

    let docid_lo = segment.manifest.docid_lo;
    let docid_hi = segment.manifest.docid_hi;
    let span =
        usize::try_from(docid_hi - docid_lo).map_err(|_| CompactionError::ArithmeticOverflow {
            field: "host document span",
        })?;
    let is_live = |global_docid: u64| {
        source.id_map.contains(global_docid)
            && u32::try_from(global_docid)
                .is_ok_and(|docid| !segment.manifest.tombstones.contains(docid))
    };

    let mut doclen_columns = Vec::new();
    doclen_columns
        .try_reserve_exact(term_fields.len())
        .map_err(|_| CompactionError::Allocation {
            resource: "DOCLEN columns",
            count: term_fields.len(),
        })?;
    for field in source.doclen.fields() {
        let mut lengths = Vec::new();
        lengths
            .try_reserve_exact(span)
            .map_err(|_| CompactionError::Allocation {
                resource: "DOCLEN values",
                count: span,
            })?;
        for ordinal in 0..span {
            let global_docid = docid_lo
                .checked_add(u64::try_from(ordinal).map_err(|_| {
                    CompactionError::ArithmeticOverflow {
                        field: "global document ordinal",
                    }
                })?)
                .ok_or(CompactionError::ArithmeticOverflow {
                    field: "global document id",
                })?;
            let length = if is_live(global_docid) {
                field.decoded_fieldnorm(global_docid)
            } else {
                None
            };
            lengths.push(length);
        }
        doclen_columns.push((field.field_ord(), lengths));
    }
    let doclen_inputs = doclen_columns
        .iter()
        .map(|(field_ord, values)| DocLenFieldInput::new(*field_ord, values))
        .collect::<Vec<_>>();
    let doclen = EncodedDocLenSection::encode(docid_lo, docid_hi, &term_fields, &doclen_inputs)
        .map_err(|error| compaction_codec(SectionKind::DOCLEN, error))?;

    let mut id_map_entries = Vec::new();
    id_map_entries
        .try_reserve_exact(span)
        .map_err(|_| CompactionError::Allocation {
            resource: "IDMAP rows",
            count: span,
        })?;
    for ordinal in 0..span {
        let global_docid = docid_lo
            .checked_add(u64::try_from(ordinal).map_err(|_| {
                CompactionError::ArithmeticOverflow {
                    field: "IDMAP ordinal",
                }
            })?)
            .ok_or(CompactionError::ArithmeticOverflow {
                field: "IDMAP global document id",
            })?;
        let entry = is_live(global_docid)
            .then(|| source.id_map.get(global_docid))
            .flatten()
            .map(|entry| IdMapEntryInput::new(entry.document_id(), entry.content_hash()));
        id_map_entries.push(entry);
    }
    let id_map = EncodedIdMapSection::encode(docid_lo, docid_hi, &id_map_entries)
        .map_err(|error| compaction_codec(SectionKind::IDMAP, error))?;
    let id_map_view = id_map
        .section()
        .map_err(|error| compaction_codec(SectionKind::IDMAP, error))?;
    let id_hash = EncodedIdHashSection::encode(id_map_view)
        .map_err(|error| compaction_codec(SectionKind::IDHASH, error))?;

    let CompactedTermSections {
        postings,
        positions,
        blockmax,
        terms: compacted_terms,
        total_tokens,
    } = compact_terms(&source, &segment.manifest.tombstones, has_positions)?;
    let mut replacement_stats = Vec::new();
    replacement_stats
        .try_reserve_exact(source_stats.len())
        .map_err(|_| CompactionError::Allocation {
            resource: "STATS rows",
            count: source_stats.len(),
        })?;
    for row in &source_stats {
        replacement_stats.push(FieldStats::new(
            row.field_ord,
            total_tokens.get(&row.field_ord).copied().unwrap_or(0),
            doc_count,
        ));
    }
    let mut term_inputs = Vec::new();
    term_inputs
        .try_reserve_exact(compacted_terms.len())
        .map_err(|_| CompactionError::Allocation {
            resource: "TERMDICT inputs",
            count: compacted_terms.len(),
        })?;
    for term in &compacted_terms {
        term_inputs.push(TermInput::new(term.field_ord, &term.term, term.metadata));
    }
    let termdict = EncodedTermDictionary::encode_sorted(
        schema,
        TermSectionLengths {
            postings: durable_compaction_len(&postings, "POSTINGS")?,
            positions: has_positions
                .then(|| durable_compaction_len(&positions, "POSITIONS"))
                .transpose()?,
            blockmax: durable_compaction_len(&blockmax, "BLOCKMAX")?,
        },
        &term_inputs,
    )
    .map_err(|error| compaction_codec(SectionKind::TERMDICT, error))?;

    let numeric = if let Some(source_numeric) = &source.numeric {
        let mut columns = Vec::new();
        columns
            .try_reserve_exact(source_numeric.field_count())
            .map_err(|_| CompactionError::Allocation {
                resource: "NUMERIC columns",
                count: source_numeric.field_count(),
            })?;
        for field in source_numeric.fields() {
            let mut entries = Vec::<NumericEntry>::new();
            entries
                .try_reserve_exact(field.len())
                .map_err(|_| CompactionError::Allocation {
                    resource: "NUMERIC entries",
                    count: field.len(),
                })?;
            entries.extend(
                field
                    .entries()
                    .filter(|entry| !segment.manifest.tombstones.contains(entry.docid())),
            );
            columns.push((field.field_ord(), entries));
        }
        let inputs = columns
            .iter()
            .map(|(field_ord, entries)| NumericFieldInput::new(*field_ord, entries))
            .collect::<Vec<_>>();
        Some(
            EncodedNumericSection::encode(schema, docid_lo, docid_hi, &inputs)
                .map_err(|error| compaction_codec(SectionKind::NUMERIC, error))?,
        )
    } else {
        None
    };

    let stored_meta = if let Some(source_stored) = &source.stored_meta {
        let mut columns = Vec::new();
        columns
            .try_reserve_exact(source_stored.field_count())
            .map_err(|_| CompactionError::Allocation {
                resource: "STOREDMETA columns",
                count: source_stored.field_count(),
            })?;
        for field in source_stored.fields() {
            let mut values = Vec::new();
            values
                .try_reserve_exact(span)
                .map_err(|_| CompactionError::Allocation {
                    resource: "STOREDMETA values",
                    count: span,
                })?;
            for ordinal in 0..span {
                let global_docid = docid_lo
                    .checked_add(u64::try_from(ordinal).map_err(|_| {
                        CompactionError::ArithmeticOverflow {
                            field: "STOREDMETA ordinal",
                        }
                    })?)
                    .ok_or(CompactionError::ArithmeticOverflow {
                        field: "STOREDMETA global document id",
                    })?;
                values.push(
                    is_live(global_docid)
                        .then(|| field.get(global_docid))
                        .flatten(),
                );
            }
            columns.push((field.field_ord(), values));
        }
        let inputs = columns
            .iter()
            .map(|(field_ord, values)| StoredMetaFieldInput::new(*field_ord, values))
            .collect::<Vec<_>>();
        Some(
            EncodedStoredMetaSection::encode(docid_lo, docid_hi, &stored_fields, &inputs)
                .map_err(|error| compaction_codec(SectionKind::STOREDMETA, error))?,
        )
    } else {
        None
    };
    let stats = EncodedStatsSection::encode(&term_fields, &replacement_stats, doc_count)
        .map_err(|error| compaction_codec(SectionKind::STATS, error))?;

    let mut section_plan = Vec::new();
    section_plan
        .try_reserve_exact(10)
        .map_err(|_| CompactionError::Allocation {
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
        id_map.as_bytes().len(),
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
    if let Some(stored_meta) = &stored_meta {
        section_plan.push(PlannedSection::new(
            SectionKind::STOREDMETA,
            stored_meta.as_bytes().len(),
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
    .map_err(|source| CompactionError::Segment { source })?;
    assembler
        .copy_section(SectionKind::TERMDICT, termdict.as_bytes())
        .map_err(|source| CompactionError::Segment { source })?;
    assembler
        .copy_section(SectionKind::POSTINGS, &postings)
        .map_err(|source| CompactionError::Segment { source })?;
    if has_positions {
        assembler
            .copy_section(SectionKind::POSITIONS, &positions)
            .map_err(|source| CompactionError::Segment { source })?;
    }
    assembler
        .copy_section(SectionKind::BLOCKMAX, &blockmax)
        .map_err(|source| CompactionError::Segment { source })?;
    assembler
        .copy_section(SectionKind::DOCLEN, doclen.as_bytes())
        .map_err(|source| CompactionError::Segment { source })?;
    assembler
        .copy_section(SectionKind::IDMAP, id_map.as_bytes())
        .map_err(|source| CompactionError::Segment { source })?;
    assembler
        .copy_section(SectionKind::IDHASH, id_hash.as_bytes())
        .map_err(|source| CompactionError::Segment { source })?;
    if let Some(numeric) = &numeric {
        assembler
            .copy_section(SectionKind::NUMERIC, numeric.as_bytes())
            .map_err(|source| CompactionError::Segment { source })?;
    }
    if let Some(stored_meta) = &stored_meta {
        assembler
            .copy_section(SectionKind::STOREDMETA, stored_meta.as_bytes())
            .map_err(|source| CompactionError::Segment { source })?;
    }
    assembler
        .copy_section(SectionKind::STATS, stats.as_bytes())
        .map_err(|source| CompactionError::Segment { source })?;
    let encoded = assembler
        .finish()
        .map_err(|source| CompactionError::Segment { source })?;
    let reader = SegmentReader::from_bytes(encoded.as_bytes(), schema)
        .map_err(|source| CompactionError::Segment { source })?;
    reader
        .verify()
        .map_err(|source| CompactionError::Segment { source })?;
    let manifest = ManifestSegment {
        segment_id: output_segment_id,
        seal_seq,
        file_len: encoded.file_len(),
        file_xxh3: encoded.file_xxh3(),
        docid_lo,
        docid_hi,
        doc_count,
        tombstones: TombstoneSet::new(),
    };
    Ok(CompactionReplacement {
        encoded: Some(encoded),
        manifest: Some(manifest),
        source_stats,
        replacement_stats,
    })
}

fn compact_terms(
    source: &ConcatSource<'_>,
    tombstones: &TombstoneSet,
    has_positions: bool,
) -> Result<CompactedTermSections, CompactionError> {
    let mut postings_output = Vec::new();
    let mut positions_output = Vec::new();
    let mut blockmax_output = Vec::new();
    let mut terms = Vec::new();
    let mut total_tokens = BTreeMap::<u16, u64>::new();
    terms
        .try_reserve_exact(source.terms.len())
        .map_err(|_| CompactionError::Allocation {
            resource: "retained term rows",
            count: source.terms.len(),
        })?;

    for term in &source.terms {
        let posting_bytes = concat_span(
            source.postings,
            term.metadata.postings,
            SectionKind::POSTINGS,
        )
        .map_err(|error| compaction_source_validation(&error))?;
        let posting_list = PostingList::parse(posting_bytes, term.metadata.doc_freq)
            .map_err(|error| compaction_codec(SectionKind::POSTINGS, error))?;
        let decoded = posting_list
            .decode_all_bounded(term.metadata.doc_freq as usize)
            .map_err(|error| compaction_codec(SectionKind::POSTINGS, error))?;
        let position_list = if let Some(span) = term.metadata.positions {
            let section = source
                .positions
                .ok_or_else(|| CompactionError::SectionCodec {
                    section: SectionKind::POSITIONS,
                    detail: "term references POSITIONS but the source section is absent".to_owned(),
                })?;
            let bytes = concat_span(section, span, SectionKind::POSITIONS)
                .map_err(|error| compaction_source_validation(&error))?;
            Some(
                PositionList::parse(bytes, &posting_list)
                    .map_err(|error| compaction_codec(SectionKind::POSITIONS, error))?,
            )
        } else {
            None
        };
        let mut retained = Vec::<Posting>::new();
        retained
            .try_reserve_exact(decoded.len())
            .map_err(|_| CompactionError::Allocation {
                resource: "retained postings",
                count: decoded.len(),
            })?;
        let mut retained_positions = Vec::new();
        for (ordinal, posting) in decoded.iter().copied().enumerate() {
            if tombstones.contains(posting.doc_id) {
                continue;
            }
            if let Some(position_list) = &position_list {
                let ordinal =
                    u32::try_from(ordinal).map_err(|_| CompactionError::ArithmeticOverflow {
                        field: "position posting ordinal",
                    })?;
                for position in position_list
                    .positions_for_ordinal(ordinal)
                    .map_err(|error| compaction_codec(SectionKind::POSITIONS, error))?
                {
                    retained_positions.push(
                        position
                            .map_err(|error| compaction_codec(SectionKind::POSITIONS, error))?,
                    );
                }
            }
            retained.push(posting);
        }
        if retained.is_empty() {
            continue;
        }
        let retained_token_count = retained.iter().try_fold(0_u64, |total, posting| {
            total
                .checked_add(u64::from(posting.freq))
                .ok_or(CompactionError::ArithmeticOverflow {
                    field: "retained posting frequency sum",
                })
        })?;
        let field_total = total_tokens.entry(term.field_ord).or_default();
        *field_total = field_total.checked_add(retained_token_count).ok_or(
            CompactionError::ArithmeticOverflow {
                field: "retained field token total",
            },
        )?;
        let fieldnorms =
            source
                .doclen
                .field(term.field_ord)
                .ok_or_else(|| CompactionError::SectionCodec {
                    section: SectionKind::DOCLEN,
                    detail: format!("missing fieldnorm column for field {}", term.field_ord),
                })?;
        let (encoded_postings, encoded_blockmax) =
            EncodedPostingList::encode_with_block_max(&retained, |docid| {
                fieldnorms.fieldnorm_id(u64::from(docid))
            })
            .map_err(|error| compaction_codec(SectionKind::BLOCKMAX, error))?;
        let encoded_positions = if position_list.is_some() {
            Some(
                EncodedPositionList::encode(&retained, &retained_positions)
                    .map_err(|error| compaction_codec(SectionKind::POSITIONS, error))?,
            )
        } else {
            None
        };
        if encoded_positions.is_some() && !has_positions {
            return Err(CompactionError::SectionCodec {
                section: SectionKind::POSITIONS,
                detail: "term carries positions but the schema has no POSITIONS section".to_owned(),
            });
        }

        let postings_offset = durable_compaction_len(&postings_output, "POSTINGS offset")?;
        append_compaction_bytes(
            &mut postings_output,
            encoded_postings.as_bytes(),
            "POSTINGS output",
        )?;
        let blockmax_offset = durable_compaction_len(&blockmax_output, "BLOCKMAX offset")?;
        append_compaction_bytes(
            &mut blockmax_output,
            encoded_blockmax.as_bytes(),
            "BLOCKMAX output",
        )?;
        let positions_span = if let Some(encoded_positions) = encoded_positions {
            let offset = durable_compaction_len(&positions_output, "POSITIONS offset")?;
            let len = durable_compaction_len(encoded_positions.as_bytes(), "POSITIONS length")?;
            append_compaction_bytes(
                &mut positions_output,
                encoded_positions.as_bytes(),
                "POSITIONS output",
            )?;
            Some(ByteSpan::new(offset, len))
        } else {
            None
        };
        let doc_freq = encoded_postings.doc_freq();
        let postings_len = durable_compaction_len(encoded_postings.as_bytes(), "POSTINGS length")?;
        let blockmax_len = durable_compaction_len(encoded_blockmax.as_bytes(), "BLOCKMAX length")?;
        let metadata = positions_span.map_or_else(
            || {
                TermMetadata::without_positions(
                    doc_freq,
                    ByteSpan::new(postings_offset, postings_len),
                    ByteSpan::new(blockmax_offset, blockmax_len),
                )
            },
            |span| {
                TermMetadata::with_positions(
                    doc_freq,
                    ByteSpan::new(postings_offset, postings_len),
                    span,
                    ByteSpan::new(blockmax_offset, blockmax_len),
                )
            },
        );
        terms.push(MergedTermRow {
            field_ord: term.field_ord,
            term: term.term.clone(),
            metadata,
        });
    }
    Ok(CompactedTermSections {
        postings: postings_output,
        positions: positions_output,
        blockmax: blockmax_output,
        terms,
        total_tokens,
    })
}

fn durable_compaction_len(bytes: &[u8], field: &'static str) -> Result<u64, CompactionError> {
    u64::try_from(bytes.len()).map_err(|_| CompactionError::ArithmeticOverflow { field })
}

fn append_compaction_bytes(
    output: &mut Vec<u8>,
    bytes: &[u8],
    resource: &'static str,
) -> Result<(), CompactionError> {
    output
        .try_reserve(bytes.len())
        .map_err(|_| CompactionError::Allocation {
            resource,
            count: output.len().saturating_add(bytes.len()),
        })?;
    output.extend_from_slice(bytes);
    Ok(())
}

fn compaction_codec(section: SectionKind, error: impl std::fmt::Display) -> CompactionError {
    CompactionError::SectionCodec {
        section,
        detail: error.to_string(),
    }
}

fn compaction_source_validation(error: &ConcatMergeError) -> CompactionError {
    CompactionError::SourceValidation {
        detail: error.to_string(),
    }
}
