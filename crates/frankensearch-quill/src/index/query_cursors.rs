use crate::argus::{
    ArgusError, Bm25FieldSnapshot, CheckpointPostingCursor, DeltaFieldNorms, DeltaPostingCursor,
    FieldNormReader, PositionsHandle, PositionsReader, PostingCursor, QueryWorkCheckpoint,
    QueryWorkKind, ReferenceScorer, SealedPostingCursor, TermRecordOption, TermScorer,
    ValidatedArrivalCursor,
};
#[cfg(test)]
use crate::delta::DeltaSnapshot;
use crate::grimoire::{
    ByteSpan, TermDictionary, TermDictionaryError, star_glob_matches, trailing_star_prefix,
};
use crate::keeper::{KeeperSnapshot, RecoveredSegment};
use crate::query::{Occur, Query, QueryNode, QueryValue};
use crate::quiver::{
    DocLenField, DocLenSection, PositionList, Posting, PostingCodecError, PostingList,
    SnapshotFieldStats,
};
use crate::schema::SchemaDescriptor;
use crate::segment::SectionKind;
use crate::snippet::SnippetTerm;
#[cfg(test)]
use std::cell::Cell;
use std::collections::BTreeSet;
use std::sync::Arc;

use super::checkpoint::{
    PreparedSnapshotDocFreqs, QueryCheckpointHandle, TANTIVY_PHRASE_PREFIX_MAX_EXPANSIONS,
    clone_query_checkpoint_for_argus, delta_ordered_view_blocks,
};
use super::error_results::{CONTENT_FIELD, QuillIndexError, invalid_state};
use super::query_lowering::QueryLeaf;
use super::query_predicates::{term_record_option, validate_string_query_field};
use super::snapshot::{QuillSearchSnapshot, SnapshotError};

#[cfg(test)]
#[derive(Debug)]
pub(super) struct OwnedFieldNorms {
    pub(super) field_ord: u16,
    pub(super) docid_lo: u64,
    pub(super) values: Vec<u8>,
}

#[cfg(test)]
impl FieldNormReader for OwnedFieldNorms {
    fn field_ord(&self) -> u16 {
        self.field_ord
    }

    fn fieldnorm_id(&self, global_docid: u32) -> Option<u8> {
        let offset = u64::from(global_docid).checked_sub(self.docid_lo)?;
        self.values.get(usize::try_from(offset).ok()?).copied()
    }
}

#[derive(Debug)]
pub(super) struct OwnedPostingCursor {
    pub(super) postings: Vec<Posting>,
    pub(super) positions: Option<Vec<Vec<u32>>>,
    pub(super) cursor: usize,
    pub(super) segment_num_docs: u32,
}

impl OwnedPostingCursor {
    pub(super) fn empty(segment_num_docs: u32, positioned: bool) -> Self {
        Self {
            postings: Vec::new(),
            positions: positioned.then(Vec::new),
            cursor: 0,
            segment_num_docs,
        }
    }

    pub(super) fn current(&self) -> Option<Posting> {
        self.postings.get(self.cursor).copied()
    }
}

impl PostingCursor for OwnedPostingCursor {
    fn doc(&self) -> Option<u32> {
        self.current().map(|posting| posting.doc_id)
    }

    fn freq(&self) -> Option<u32> {
        self.current().map(|posting| posting.freq)
    }

    fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
        self.current()?;
        self.positions.as_ref()?;
        u32::try_from(self.cursor)
            .ok()
            .map(|ordinal| PositionsHandle::new(self, ordinal))
    }

    fn size_hint(&self) -> u32 {
        u32::try_from(self.postings.len()).unwrap_or(u32::MAX)
    }

    fn segment_num_docs(&self) -> u32 {
        self.segment_num_docs
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        if self.cursor < self.postings.len() {
            self.cursor += 1;
        }
        Ok(self.doc())
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.doc().is_some_and(|doc| doc >= target) {
            return Ok(self.doc());
        }
        let relative = self.postings[self.cursor.min(self.postings.len())..]
            .partition_point(|posting| posting.doc_id < target);
        self.cursor = self
            .cursor
            .saturating_add(relative)
            .min(self.postings.len());
        Ok(self.doc())
    }
}

impl PositionsReader for OwnedPostingCursor {
    fn decode_positions(
        &self,
        posting_ordinal: u32,
        output: &mut Vec<u32>,
    ) -> Result<(), ArgusError> {
        let ordinal = usize::try_from(posting_ordinal)
            .map_err(|_| ArgusError::CursorInvariant("posting ordinal does not fit usize"))?;
        if ordinal != self.cursor {
            return Err(ArgusError::CursorInvariant(
                "positions handle no longer identifies the current posting",
            ));
        }
        let positions = self
            .positions
            .as_ref()
            .and_then(|rows| rows.get(ordinal))
            .ok_or(ArgusError::CursorInvariant(
                "positioned cursor has no current position run",
            ))?;
        output
            .try_reserve_exact(positions.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "owned position run",
                count: positions.len(),
            })?;
        output.extend_from_slice(positions);
        Ok(())
    }
}

pub(super) fn leaf_phrase_prefix_terms(
    checkpoint: &QueryCheckpointHandle<'_>,
    leaf: QueryLeaf<'_>,
    schema: SchemaDescriptor,
    field_ord: u16,
    prefix: &[u8],
) -> Result<Vec<Vec<u8>>, QuillIndexError> {
    validate_string_query_field(schema, field_ord, "phrase prefix")?;
    let mut terms = Vec::new();
    terms
        .try_reserve_exact(TANTIVY_PHRASE_PREFIX_MAX_EXPANSIONS)
        .map_err(|_| invalid_state("could not allocate leaf-local phrase-prefix expansions"))?;

    match leaf {
        QueryLeaf::Sealed(segment) => {
            let dictionary = open_dictionary(segment, schema)?;
            let mut cursor = dictionary.prefix_cursor(field_ord, prefix)?;
            let mut admitted_block = None;
            while terms.len() < TANTIVY_PHRASE_PREFIX_MAX_EXPANSIONS {
                let Some(current) = cursor.current() else {
                    break;
                };
                let block = cursor.current_block_index();
                if block != admitted_block {
                    checkpoint.admit(QueryWorkKind::DictionaryBlock, 1)?;
                    admitted_block = block;
                }
                let mut owned = Vec::new();
                owned
                    .try_reserve_exact(current.term.len())
                    .map_err(|_| invalid_state("could not copy sealed phrase-prefix expansion"))?;
                owned.extend_from_slice(current.term);
                terms.push(owned);
                cursor.next()?;
            }
        }
        QueryLeaf::Delta(delta) => {
            // The ordered view is materialized and radix-sorted as one unit on
            // first use, so admission must precede construction just as it does
            // for ordinary glob expansion. The 50-term semantic cap remains
            // leaf-local after that unavoidable generation-level work.
            checkpoint.admit(
                QueryWorkKind::DictionaryBlock,
                delta_ordered_view_blocks(delta.segment().term_count()),
            )?;
            let field_terms = delta.ordered_terms_for_field(field_ord);
            if delta.segment().term_count() != 0 {
                checkpoint.admit(QueryWorkKind::DictionaryBlock, 0)?;
            }
            for (term_index, term) in field_terms.enumerate() {
                if term_index != 0 && term_index % 16 == 0 {
                    checkpoint.admit(QueryWorkKind::DictionaryBlock, 0)?;
                }
                if term.term().starts_with(prefix) {
                    let mut owned = Vec::new();
                    owned.try_reserve_exact(term.term().len()).map_err(|_| {
                        invalid_state("could not copy Delta phrase-prefix expansion")
                    })?;
                    owned.extend_from_slice(term.term());
                    terms.push(owned);
                    if terms.len() == TANTIVY_PHRASE_PREFIX_MAX_EXPANSIONS {
                        break;
                    }
                }
            }
        }
    }
    Ok(terms)
}

fn snapshot_glob_terms(
    checkpoint: Option<&QueryCheckpointHandle<'_>>,
    snapshot: &QuillSearchSnapshot,
    schema: SchemaDescriptor,
    field_ord: u16,
    pattern: &[u8],
    expansion_limit: usize,
) -> Result<Vec<Vec<u8>>, QuillIndexError> {
    validate_string_query_field(schema, field_ord, "glob")?;
    let mut terms = BTreeSet::<Vec<u8>>::new();
    for segment in snapshot.keeper_snapshot().segments() {
        let dictionary = open_dictionary(segment, schema)?;
        if !pattern.contains(&b'*') {
            if let Some(checkpoint) = checkpoint {
                checkpoint.admit(QueryWorkKind::DictionaryBlock, 1)?;
            }
            if dictionary.lookup(field_ord, pattern)?.is_some() {
                insert_glob_term(
                    &mut terms,
                    field_ord,
                    pattern,
                    pattern.to_vec(),
                    expansion_limit,
                )?;
            }
            continue;
        }
        let trailing_prefix = trailing_star_prefix(pattern);
        let mut cursor = if let Some(prefix) = trailing_prefix {
            dictionary.prefix_cursor(field_ord, prefix)?
        } else {
            dictionary.field_cursor(field_ord)?
        };
        let mut admitted_block = None;
        while let Some(current) = cursor.current() {
            let block = cursor.current_block_index();
            if block != admitted_block {
                if let Some(checkpoint) = checkpoint {
                    checkpoint.admit(QueryWorkKind::DictionaryBlock, 1)?;
                }
                admitted_block = block;
            }
            if trailing_prefix.is_some() || star_glob_matches(pattern, current.term) {
                insert_glob_term(
                    &mut terms,
                    field_ord,
                    pattern,
                    current.term.to_vec(),
                    expansion_limit,
                )?;
            }
            cursor.next()?;
        }
    }
    for delta in snapshot.delta_snapshots() {
        // `ordered_terms` is not a cursor: its first use allocates one id per
        // term and MSD-radix-sorts the whole set (`TermInterner::sorted_ids`)
        // before the first term is examined. Admitting per 16-term block
        // DURING that initial walk would charge nothing for the one allocation
        // that scales with the generation, and a cancelled or zero-fuel query
        // would still pay for it in full. The ceiling comes from the O(1)
        // `term_count` and is admitted BEFORE the ordered view exists.
        //
        // The cache retains the wholesale order per frozen generation; the
        // first lookup remains conservatively pre-admitted.
        if let Some(checkpoint) = checkpoint {
            checkpoint.admit(
                QueryWorkKind::DictionaryBlock,
                delta_ordered_view_blocks(delta.segment().term_count()),
            )?;
        }
        let field_terms = delta.ordered_terms_for_field(field_ord);
        if delta.segment().term_count() != 0 {
            if let Some(checkpoint) = checkpoint {
                // A nonempty Delta still needs to surface cancellation after
                // its cached view was obtained, even when this field is empty.
                checkpoint.admit(QueryWorkKind::DictionaryBlock, 0)?;
            }
        }
        for (term_index, term) in field_terms.enumerate() {
            // The blocks are paid for above, so this admits ZERO units: it
            // charges no fuel and moves no counter, and exists only so that a
            // cancellation during a long walk is still surfaced promptly.
            if let Some(checkpoint) = checkpoint.filter(|_| term_index != 0 && term_index % 16 == 0)
            {
                checkpoint.admit(QueryWorkKind::DictionaryBlock, 0)?;
            }
            // The live count is taken at freeze, so this stays O(1) per term
            // even on the `checkpoint == None` path, which has nothing to
            // admit against and previously walked every chain unmetered.
            if term.live_doc_freq() != 0 && star_glob_matches(pattern, term.term()) {
                insert_glob_term(
                    &mut terms,
                    field_ord,
                    pattern,
                    term.term().to_vec(),
                    expansion_limit,
                )?;
            }
        }
    }
    Ok(terms.into_iter().collect())
}

/// Expand one glob against ONE query leaf: the sealed segment's own term
/// dictionary, or the Delta leaf's ordered term view.
///
/// `lower_leaf_glob` used to ask [`snapshot_glob_terms`] for the
/// snapshot-wide term union once per leaf, so a glob-bearing query over S
/// segments performed S full union walks — S^2 dictionary walks, ~81k of
/// them on a 285-segment corpus. The union's cross-leaf terms are absent
/// terms for this leaf: they lower to dictionary probes with no postings, so
/// expanding against the leaf's own dictionary yields the same match set
/// for the lowered disjunction while walking one dictionary per leaf.
/// The trailing-star fast path (`prefix_cursor`), the full-field scan for
/// complex globs, the indexed exact-lookup arm, and the per-block fuel
/// admission all mirror `snapshot_glob_terms` exactly; only the scope of
/// the walk changes.
///
/// The typed [`TermDictionaryError::GlobExpansionLimitExceeded`] cap applies
/// per leaf, mirroring the phrase-prefix path's leaf locality
/// ([`leaf_phrase_prefix_terms`]): each leaf refuses independently when its
/// own matching range crosses the configured limit, while the query fuel
/// budget remains the snapshot-wide backstop for the aggregate walk.
///
/// # Errors
///
/// Returns a typed error for an invalid string field, cursor or view
/// failure, fuel exhaustion, or the per-leaf glob expansion limit.
pub(super) fn leaf_glob_terms(
    checkpoint: &QueryCheckpointHandle<'_>,
    leaf: QueryLeaf<'_>,
    schema: SchemaDescriptor,
    field_ord: u16,
    pattern: &[u8],
    expansion_limit: usize,
) -> Result<Vec<Vec<u8>>, QuillIndexError> {
    validate_string_query_field(schema, field_ord, "glob")?;
    let mut terms = BTreeSet::<Vec<u8>>::new();
    match leaf {
        QueryLeaf::Sealed(segment) => {
            let dictionary = open_dictionary(segment, schema)?;
            if !pattern.contains(&b'*') {
                checkpoint.admit(QueryWorkKind::DictionaryBlock, 1)?;
                if dictionary.lookup(field_ord, pattern)?.is_some() {
                    insert_glob_term(
                        &mut terms,
                        field_ord,
                        pattern,
                        pattern.to_vec(),
                        expansion_limit,
                    )?;
                }
                return Ok(terms.into_iter().collect());
            }
            let trailing_prefix = trailing_star_prefix(pattern);
            let mut cursor = if let Some(prefix) = trailing_prefix {
                dictionary.prefix_cursor(field_ord, prefix)?
            } else {
                dictionary.field_cursor(field_ord)?
            };
            let mut admitted_block = None;
            while let Some(current) = cursor.current() {
                let block = cursor.current_block_index();
                if block != admitted_block {
                    checkpoint.admit(QueryWorkKind::DictionaryBlock, 1)?;
                    admitted_block = block;
                }
                if trailing_prefix.is_some() || star_glob_matches(pattern, current.term) {
                    insert_glob_term(
                        &mut terms,
                        field_ord,
                        pattern,
                        current.term.to_vec(),
                        expansion_limit,
                    )?;
                }
                cursor.next()?;
            }
        }
        QueryLeaf::Delta(delta) => {
            // The ordered view's one-generation materialization ceiling is
            // admitted before the view exists, exactly as in
            // `snapshot_glob_terms`: the sort allocates once per frozen
            // generation and a cancelled or zero-fuel query must still pay
            // for it in full before touching a single term.
            checkpoint.admit(
                QueryWorkKind::DictionaryBlock,
                delta_ordered_view_blocks(delta.segment().term_count()),
            )?;
            let field_terms = delta.ordered_terms_for_field(field_ord);
            if delta.segment().term_count() != 0 {
                // A nonempty Delta still needs to surface cancellation after
                // its cached view was obtained, even when this field is empty.
                checkpoint.admit(QueryWorkKind::DictionaryBlock, 0)?;
            }
            for (term_index, term) in field_terms.enumerate() {
                // The blocks are paid for above, so this admits ZERO units:
                // it exists only so that a cancellation during a long walk is
                // still surfaced promptly.
                if term_index != 0 && term_index % 16 == 0 {
                    checkpoint.admit(QueryWorkKind::DictionaryBlock, 0)?;
                }
                // The live count is taken at freeze, so this stays O(1) per
                // term and drops fully tombstoned terms the sealed cursor
                // arms never see at all.
                if term.live_doc_freq() != 0 && star_glob_matches(pattern, term.term()) {
                    insert_glob_term(
                        &mut terms,
                        field_ord,
                        pattern,
                        term.term().to_vec(),
                        expansion_limit,
                    )?;
                }
            }
        }
    }
    Ok(terms.into_iter().collect())
}

pub(super) fn compiled_snippet_terms(
    checkpoint: &QueryCheckpointHandle<'_>,
    query: &Query,
    snapshot: &QuillSearchSnapshot,
    schema: SchemaDescriptor,
    expansion_limit: usize,
) -> Result<Vec<SnippetTerm>, QuillIndexError> {
    let mut terms = BTreeSet::<Vec<u8>>::new();
    collect_snippet_term_bytes(
        checkpoint,
        query,
        snapshot,
        schema,
        expansion_limit,
        &mut terms,
    )?;
    terms
        .into_iter()
        .map(|term| {
            let document_frequency = checkpointed_snapshot_doc_freq(
                checkpoint,
                snapshot,
                CONTENT_FIELD,
                &term,
                SnapshotDocFreqDeltaAdmission::TailPerGeneration,
            )?;
            let text = String::from_utf8(term)
                .map_err(|_| invalid_state("content term dictionary contains non-UTF-8 bytes"))?;
            Ok(SnippetTerm::new(text, document_frequency))
        })
        .collect()
}

fn collect_snippet_term_bytes(
    checkpoint: &QueryCheckpointHandle<'_>,
    query: &Query,
    snapshot: &QuillSearchSnapshot,
    schema: SchemaDescriptor,
    expansion_limit: usize,
    terms: &mut BTreeSet<Vec<u8>>,
) -> Result<(), QuillIndexError> {
    let mut pending = Vec::new();
    pending
        .try_reserve_exact(query.node_count())
        .map_err(|_| ArgusError::Allocation {
            resource: "snippet query traversal",
            count: query.node_count(),
        })?;
    pending.push(query.root_id());
    while let Some(node_id) = pending.pop() {
        match query.node(node_id) {
            QueryNode::Term { fields, text }
                if fields.iter().any(|field| field.field_id == CONTENT_FIELD) =>
            {
                terms.insert(text.as_bytes().to_vec());
            }
            QueryNode::Phrase {
                fields,
                terms: phrase_terms,
                prefix,
                ..
            } if fields.iter().any(|field| field.field_id == CONTENT_FIELD) => {
                let prefix_index = (*prefix).then_some(phrase_terms.len().saturating_sub(1));
                for (index, term) in phrase_terms.iter().enumerate() {
                    if prefix_index == Some(index) {
                        let pattern = format!("{}*", term.text);
                        terms.extend(snapshot_glob_terms(
                            Some(checkpoint),
                            snapshot,
                            schema,
                            CONTENT_FIELD,
                            pattern.as_bytes(),
                            expansion_limit,
                        )?);
                    } else {
                        terms.insert(term.text.as_bytes().to_vec());
                    }
                }
            }
            QueryNode::Boolean { clauses, .. } => {
                pending.extend(
                    clauses
                        .iter()
                        .rev()
                        .filter(|clause| clause.occur != Occur::MustNot)
                        .map(|clause| clause.query),
                );
            }
            QueryNode::Set { field_id, values } if *field_id == CONTENT_FIELD => {
                terms.extend(values.iter().filter_map(|value| match value {
                    QueryValue::Str(text) => Some(text.as_bytes().to_vec()),
                    QueryValue::I64(_) | QueryValue::U64(_) => None,
                }));
            }
            QueryNode::Glob { field_ids, pattern } if field_ids.contains(&CONTENT_FIELD) => {
                terms.extend(snapshot_glob_terms(
                    Some(checkpoint),
                    snapshot,
                    schema,
                    CONTENT_FIELD,
                    pattern.as_bytes(),
                    expansion_limit,
                )?);
            }
            QueryNode::Boost { query: child, .. } => pending.push(*child),
            QueryNode::Empty
            | QueryNode::All
            | QueryNode::AllocationFailure { .. }
            | QueryNode::Term { .. }
            | QueryNode::Phrase { .. }
            | QueryNode::Range { .. }
            | QueryNode::Set { .. }
            | QueryNode::Glob { .. } => {}
        }
    }
    Ok(())
}

fn insert_glob_term(
    terms: &mut BTreeSet<Vec<u8>>,
    field_ord: u16,
    pattern: &[u8],
    term: Vec<u8>,
    expansion_limit: usize,
) -> Result<(), QuillIndexError> {
    if terms.insert(term) && terms.len() > expansion_limit {
        return Err(TermDictionaryError::GlobExpansionLimitExceeded {
            pattern: String::from_utf8_lossy(pattern).into_owned(),
            field_ord,
            limit: expansion_limit,
            actual: terms.len(),
        }
        .into());
    }
    Ok(())
}

pub(super) fn lower_leaf_term<'a>(
    leaf: QueryLeaf<'a>,
    snapshot: &QuillSearchSnapshot,
    schema: SchemaDescriptor,
    field_ord: u16,
    term: &[u8],
    prepared_doc_freqs: Option<&PreparedSnapshotDocFreqs>,
    boost: f32,
    rank_pruning: bool,
    checkpoint: &QueryCheckpointHandle<'a>,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    let stats = composite_snapshot_field(snapshot, field_ord)?;
    let doc_freq = match prepared_doc_freqs {
        Some(prepared) => prepared.get_or_compute(checkpoint, snapshot, field_ord, term)?,
        None => checkpointed_snapshot_doc_freq(
            checkpoint,
            snapshot,
            field_ord,
            term,
            SnapshotDocFreqDeltaAdmission::Ranking,
        )?,
    };
    let record_option = term_record_option(schema, field_ord)?;
    match leaf {
        QueryLeaf::Sealed(segment) => {
            checkpoint.admit(QueryWorkKind::DictionaryBlock, 1)?;
            let (cursor, fieldnorms) = open_sealed_term_cursor(
                segment,
                schema,
                field_ord,
                term,
                rank_pruning,
                checkpoint,
            )?;
            #[cfg(feature = "profile-internals")]
            checkpoint.record_term_dictionary_view();
            // The sealed leaf is the one path whose cursor type the crate knows
            // concretely, so it is the one that can carry arrival reads: the
            // scorer advances, scores and probes a single value it owns. Delta
            // and every custom cursor stay opaque below, which keeps their
            // pruned windows on the exact walk.
            let cursor =
                ValidatedArrivalCursor::new(cursor, clone_query_checkpoint_for_argus(checkpoint));
            build_validated_term_scorer(cursor, fieldnorms, stats, doc_freq, record_option, boost)
        }
        QueryLeaf::Delta(delta) => {
            // The construction pull can walk a tombstoned prefix, so it is
            // admitted as it walks rather than after the fact.
            let cursor =
                DeltaPostingCursor::new_admitted(delta, field_ord, term, checkpoint.as_ref())?;
            let cursor =
                CheckpointPostingCursor::new(cursor, clone_query_checkpoint_for_argus(checkpoint))?;
            build_term_scorer(
                cursor,
                DeltaFieldNorms::new(delta, field_ord),
                stats,
                doc_freq,
                record_option,
                boost,
            )
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum SnapshotDocFreqDeltaAdmission {
    /// Ranking keeps its existing no-Delta-admission behavior.
    Ranking,
    /// Snippet-tail doc-frequency probes pay one unit per Delta generation.
    TailPerGeneration,
}

pub(super) fn checkpointed_snapshot_doc_freq(
    checkpoint: &QueryCheckpointHandle<'_>,
    snapshot: &QuillSearchSnapshot,
    field_ord: u16,
    term: &[u8],
    delta_admission: SnapshotDocFreqDeltaAdmission,
) -> Result<u64, QuillIndexError> {
    #[cfg(feature = "profile-internals")]
    checkpoint.record_snapshot_doc_freq();
    if snapshot.bm25_field_stats(field_ord).is_none() {
        return Err(invalid_state(format!(
            "snapshot has no BM25 statistics for field {field_ord}"
        )));
    }
    let mut total = 0_u64;
    for segment in snapshot.keeper_snapshot().segments() {
        let dictionary = open_dictionary(segment, snapshot.keeper_snapshot().schema())?;
        #[cfg(feature = "profile-internals")]
        {
            checkpoint.record_global_doc_freq();
            checkpoint.record_term_dictionary_view();
        }
        checkpoint.admit(QueryWorkKind::DictionaryBlock, 1)?;
        if let Some(found) = dictionary.lookup(field_ord, term)? {
            total = total
                .checked_add(u64::from(found.metadata.doc_freq))
                .ok_or_else(|| invalid_state("snapshot document frequency overflow"))?;
        }
    }
    for delta in snapshot.delta_snapshots() {
        if matches!(
            delta_admission,
            SnapshotDocFreqDeltaAdmission::TailPerGeneration
        ) {
            // The frozen count is O(1), but the tail still crosses one
            // generation-specific lookup boundary. Admit immediately before
            // that lookup so cancellation or fuel exhaustion cannot cross it.
            checkpoint.admit(QueryWorkKind::DictionaryBlock, 1)?;
        }
        let delta_doc_freq = delta
            .find_term(field_ord, term)
            .map_or(0, |found| found.live_doc_freq());
        #[cfg(feature = "profile-internals")]
        checkpoint.record_global_doc_freq();
        total = total
            .checked_add(u64::try_from(delta_doc_freq).map_err(|_| {
                SnapshotError::CounterOverflow {
                    counter: "live Delta document frequency",
                }
            })?)
            .ok_or(SnapshotError::CounterOverflow {
                counter: "snapshot document frequency",
            })?;
    }
    Ok(total)
}

pub(super) fn open_sealed_term_cursor<'a>(
    segment: &'a RecoveredSegment,
    schema: SchemaDescriptor,
    field_ord: u16,
    term: &[u8],
    rank_pruning: bool,
    checkpoint: &QueryCheckpointHandle<'_>,
) -> Result<(SealedPostingCursor<'a>, DocLenField<'a>), QuillIndexError> {
    let fieldnorms = borrowed_fieldnorms(segment, schema, field_ord)?;
    let dictionary = open_dictionary(segment, schema)?;
    let found = dictionary.lookup(field_ord, term)?;
    // The lookup has to come first — it is what tells us whether a block will
    // be decoded at all — but the permit still precedes every decode below,
    // because construction is what reads the term's first posting block. An
    // absent term admits zero units rather than skipping the call, so even a
    // term that decodes nothing still observes cancellation here.
    CheckpointPostingCursor::admit_initial_block(
        checkpoint.as_ref(),
        found
            .as_ref()
            .is_some_and(|found| found.metadata.doc_freq > 0),
    )?;
    let Some(found) = found else {
        let postings = PostingList::parse(&[], 0)?.into_cursor()?;
        let cursor = SealedPostingCursor::from_owned(postings, 0, segment.doc_count());
        return Ok((cursor, fieldnorms));
    };

    let postings_section = required_section(segment, SectionKind::POSTINGS)?;
    let postings_bytes = span(postings_section, found.metadata.postings, "POSTINGS")?;
    if let Some(cached) = segment
        .cached_rank_pruning_metadata(found.term_ord, found.metadata)
        .map_err(invalid_state)?
    {
        let cursor = if rank_pruning {
            SealedPostingCursor::from_validated_pruning(
                postings_bytes,
                cached,
                segment.doc_count(),
            )?
        } else {
            let size_hint = cached.doc_freq();
            SealedPostingCursor::from_owned(
                cached.cursor(postings_bytes)?,
                size_hint,
                segment.doc_count(),
            )
        };
        return Ok((cursor, fieldnorms));
    }
    let postings = PostingList::parse(postings_bytes, found.metadata.doc_freq)?;
    if !rank_pruning {
        let size_hint = postings.doc_freq();
        let cursor = postings.into_cursor()?;
        return Ok((
            SealedPostingCursor::from_owned(cursor, size_hint, segment.doc_count()),
            fieldnorms,
        ));
    }
    let blockmax_section = required_section(segment, SectionKind::BLOCKMAX)?;
    let blockmax_bytes = span(blockmax_section, found.metadata.blockmax, "BLOCKMAX")?;
    let pruning = Arc::new(postings.into_pruning_metadata(blockmax_bytes, fieldnorms)?);
    let pruning = segment
        .cache_rank_pruning_metadata(found.term_ord, found.metadata, pruning)
        .map_err(invalid_state)?;
    Ok((
        SealedPostingCursor::from_validated_pruning(postings_bytes, pruning, segment.doc_count())?,
        fieldnorms,
    ))
}

pub(super) fn composite_snapshot_field(
    snapshot: &QuillSearchSnapshot,
    field_ord: u16,
) -> Result<SnapshotFieldStats, QuillIndexError> {
    snapshot
        .bm25_field_stats(field_ord)
        .ok_or_else(|| invalid_state(format!("snapshot has no field statistics for {field_ord}")))
}

#[cfg(test)]
pub(super) fn lower_term(
    segment: &RecoveredSegment,
    snapshot: &KeeperSnapshot,
    schema: SchemaDescriptor,
    field_ord: u16,
    term: &[u8],
    boost: f32,
) -> Result<ReferenceScorer<'static>, QuillIndexError> {
    let stats = snapshot_field(snapshot, field_ord)?;
    let doc_freq = snapshot_doc_freq(snapshot, schema, field_ord, term)?;
    let cursor = open_owned_cursor(segment, schema, field_ord, term, false, None)?;
    let norms = owned_fieldnorms(segment, schema, field_ord)?;
    build_term_scorer(
        cursor,
        norms,
        stats,
        doc_freq,
        term_record_option(schema, field_ord)?,
        boost,
    )
}

fn build_term_scorer<'a, C, F>(
    cursor: C,
    fieldnorms: F,
    stats: SnapshotFieldStats,
    snapshot_doc_freq: u64,
    record_option: TermRecordOption,
    boost: f32,
) -> Result<ReferenceScorer<'a>, QuillIndexError>
where
    C: PostingCursor + 'a,
    F: FieldNormReader + 'a,
{
    Ok(ReferenceScorer::term(TermScorer::new(
        cursor,
        fieldnorms,
        Bm25FieldSnapshot::new(stats)?,
        snapshot_doc_freq,
        record_option,
        boost,
    )?))
}

/// [`build_term_scorer`] for the one cursor the crate builds concretely.
///
/// This is not generic, and that is the point: the validated arm is reachable
/// only by handing over a [`ValidatedArrivalCursor`] by value, so no wrapper can
/// route itself here.
fn build_validated_term_scorer<'a, F>(
    cursor: ValidatedArrivalCursor<'a>,
    fieldnorms: F,
    stats: SnapshotFieldStats,
    snapshot_doc_freq: u64,
    record_option: TermRecordOption,
    boost: f32,
) -> Result<ReferenceScorer<'a>, QuillIndexError>
where
    F: FieldNormReader + 'a,
{
    Ok(ReferenceScorer::term(TermScorer::new_validated(
        cursor,
        fieldnorms,
        Bm25FieldSnapshot::new(stats)?,
        snapshot_doc_freq,
        record_option,
        boost,
    )?))
}

#[cfg(test)]
pub(super) fn lower_composite_sealed_term(
    segment: &RecoveredSegment,
    snapshot: &QuillSearchSnapshot,
    schema: SchemaDescriptor,
    field_ord: u16,
    term: &[u8],
    boost: f32,
) -> Result<ReferenceScorer<'static>, QuillIndexError> {
    let stats = snapshot.bm25_field_stats(field_ord).ok_or_else(|| {
        invalid_state(format!("snapshot has no field statistics for {field_ord}"))
    })?;
    let doc_freq = snapshot.bm25_doc_freq(field_ord, term)?;
    let cursor = open_owned_cursor(segment, schema, field_ord, term, false, None)?;
    let norms = owned_fieldnorms(segment, schema, field_ord)?;
    build_term_scorer(
        cursor,
        norms,
        stats,
        doc_freq,
        term_record_option(schema, field_ord)?,
        boost,
    )
}

#[cfg(test)]
pub(super) fn lower_delta_term<'a>(
    delta: &'a DeltaSnapshot,
    snapshot: &QuillSearchSnapshot,
    field_ord: u16,
    term: &[u8],
    boost: f32,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    let stats = snapshot.bm25_field_stats(field_ord).ok_or_else(|| {
        invalid_state(format!("snapshot has no field statistics for {field_ord}"))
    })?;
    let doc_freq = snapshot.bm25_doc_freq(field_ord, term)?;
    build_term_scorer(
        DeltaPostingCursor::new(delta, field_ord, term)?,
        DeltaFieldNorms::new(delta, field_ord),
        stats,
        doc_freq,
        term_record_option(delta.schema(), field_ord)?,
        boost,
    )
}

#[cfg(test)]
pub(super) fn snapshot_field(
    snapshot: &KeeperSnapshot,
    field_ord: u16,
) -> Result<SnapshotFieldStats, QuillIndexError> {
    snapshot
        .loaded_manifest()
        .manifest
        .field_stats
        .iter()
        .find(|row| row.field_ord == field_ord)
        .map(|row| SnapshotFieldStats {
            field_ord,
            total_tokens: row.total_tokens,
            doc_count: u64::from(row.doc_count),
        })
        .ok_or_else(|| invalid_state(format!("manifest has no field statistics for {field_ord}")))
}

pub(super) fn snapshot_doc_freq(
    snapshot: &KeeperSnapshot,
    schema: SchemaDescriptor,
    field_ord: u16,
    term: &[u8],
) -> Result<u64, QuillIndexError> {
    let mut total = 0_u64;
    for segment in snapshot.segments() {
        let dictionary = open_dictionary(segment, schema)?;
        if let Some(found) = dictionary.lookup(field_ord, term)? {
            total = total
                .checked_add(u64::from(found.metadata.doc_freq))
                .ok_or_else(|| invalid_state("snapshot document frequency overflow"))?;
        }
    }
    Ok(total)
}

pub(super) fn open_owned_cursor(
    segment: &RecoveredSegment,
    schema: SchemaDescriptor,
    field_ord: u16,
    term: &[u8],
    positioned: bool,
    checkpoint: Option<&QueryCheckpointHandle<'_>>,
) -> Result<OwnedPostingCursor, QuillIndexError> {
    let dictionary = open_dictionary(segment, schema)?;
    if let Some(checkpoint) = checkpoint {
        checkpoint.admit(QueryWorkKind::DictionaryBlock, 1)?;
    }
    let Some(found) = dictionary.lookup(field_ord, term)? else {
        return Ok(OwnedPostingCursor::empty(segment.doc_count(), positioned));
    };
    let postings_section = required_section(segment, SectionKind::POSTINGS)?;
    let postings_bytes = span(postings_section, found.metadata.postings, "POSTINGS")?;
    let postings = PostingList::parse(postings_bytes, found.metadata.doc_freq)?;
    let posting_count = usize::try_from(found.metadata.doc_freq)
        .map_err(|_| invalid_state("posting count does not fit usize"))?;
    let mut rows = Vec::new();
    rows.try_reserve_exact(posting_count).map_err(|_| {
        PostingCodecError::MaterializationAllocation {
            count: posting_count,
        }
    })?;
    let positions = if positioned {
        // Preserve the historical fuel/cancellation boundary: every validated
        // POSTINGS block is admitted before POSITIONS parsing begins. The
        // subsequent paired cursor traverses those same blocks without
        // charging them a second time.
        if let Some(checkpoint) = checkpoint {
            for _ in postings.blocks() {
                checkpoint.admit(QueryWorkKind::PostingBlock, 1)?;
            }
        }
        let position_span =
            found
                .metadata
                .positions
                .ok_or_else(|| QuillIndexError::UnsupportedQuery {
                    detail: format!("field {field_ord} has no positions"),
                })?;
        let position_section = required_section(segment, SectionKind::POSITIONS)?;
        let position_bytes = span(position_section, position_span, "POSITIONS")?;
        let positions = PositionList::parse(position_bytes, &postings)?;
        let mut decoded = Vec::new();
        decoded
            .try_reserve_exact(posting_count)
            .map_err(|_| invalid_state("could not allocate owned position rows"))?;

        // Phrase lowering needs owned position rows, but ordinal-at-a-time
        // lookup restarts at the beginning of each POSITIONS block. Traverse
        // the paired posting/position streams once so every run is decoded
        // exactly once before the cursor advances to the next posting.
        let mut position_cursor = positions.cursor()?;
        while let Some(posting) = position_cursor.current() {
            let mut row = Vec::new();
            if !position_cursor.decode_current_positions_into(&mut row)? {
                return Err(invalid_state(
                    "position cursor exhausted before its current posting",
                ));
            }
            let frequency = usize::try_from(posting.freq)
                .map_err(|_| invalid_state("posting frequency does not fit usize"))?;
            if row.len() != frequency {
                return Err(invalid_state(format!(
                    "position run length {} does not match posting frequency {frequency}",
                    row.len(),
                )));
            }
            rows.push(posting);
            decoded.push(row);
            position_cursor.next()?;
        }
        if rows.len() != posting_count {
            return Err(invalid_state(format!(
                "position cursor materialized {} of {posting_count} postings",
                rows.len(),
            )));
        }
        Some(decoded)
    } else {
        let mut posting_cursor = postings.cursor()?;
        let mut admitted_block = None;
        while let Some(posting) = posting_cursor.current() {
            let block = posting_cursor
                .block_index()
                .ok_or_else(|| invalid_state("posting cursor has no block index"))?;
            if admitted_block != Some(block) {
                if let Some(checkpoint) = checkpoint {
                    checkpoint.admit(QueryWorkKind::PostingBlock, 1)?;
                }
                admitted_block = Some(block);
            }
            rows.push(posting);
            posting_cursor.next()?;
        }
        if rows.len() != posting_count {
            return Err(invalid_state(format!(
                "posting cursor materialized {} of {posting_count} postings",
                rows.len(),
            )));
        }
        None
    };
    Ok(OwnedPostingCursor {
        postings: rows,
        positions,
        cursor: 0,
        segment_num_docs: segment.doc_count(),
    })
}

pub(super) fn borrowed_fieldnorms(
    segment: &RecoveredSegment,
    schema: SchemaDescriptor,
    field_ord: u16,
) -> Result<DocLenField<'_>, QuillIndexError> {
    let manifest = segment.manifest();
    let section = DocLenSection::parse_schema(
        required_section(segment, SectionKind::DOCLEN)?,
        manifest.docid_lo,
        manifest.docid_hi,
        schema,
    )?;
    section
        .field(field_ord)
        .ok_or_else(|| invalid_state(format!("DOCLEN has no field {field_ord}")))
}

#[cfg(test)]
fn owned_fieldnorms(
    segment: &RecoveredSegment,
    schema: SchemaDescriptor,
    field_ord: u16,
) -> Result<OwnedFieldNorms, QuillIndexError> {
    OWNED_FIELDNORM_BRIDGE_CALLS.with(|calls| calls.set(calls.get().saturating_add(1)));
    let manifest = segment.manifest();
    let field = borrowed_fieldnorms(segment, schema, field_ord)?;
    Ok(OwnedFieldNorms {
        field_ord,
        docid_lo: manifest.docid_lo,
        values: field.fieldnorm_ids().to_vec(),
    })
}

pub(super) fn open_dictionary(
    segment: &RecoveredSegment,
    schema: SchemaDescriptor,
) -> Result<TermDictionary<'_>, QuillIndexError> {
    Ok(segment.term_dictionary(schema)?)
}

pub(super) fn required_section(
    segment: &RecoveredSegment,
    kind: SectionKind,
) -> Result<&[u8], QuillIndexError> {
    segment
        .section(kind)?
        .ok_or_else(|| invalid_state(format!("segment is missing section {}", kind.raw())))
}

pub(super) fn span<'a>(
    section: &'a [u8],
    span: ByteSpan,
    name: &'static str,
) -> Result<&'a [u8], QuillIndexError> {
    let start = usize::try_from(span.offset)
        .map_err(|_| invalid_state(format!("{name} span start does not fit usize")))?;
    let len = usize::try_from(span.len)
        .map_err(|_| invalid_state(format!("{name} span length does not fit usize")))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| invalid_state(format!("{name} span overflows usize")))?;
    section
        .get(start..end)
        .ok_or_else(|| invalid_state(format!("{name} span lies outside its section")))
}

#[cfg(test)]
thread_local! {
    static TERM_FIELD_ORD_BRIDGE_CALLS: Cell<usize> = const { Cell::new(0) };
    static OWNED_FIELDNORM_BRIDGE_CALLS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(super) fn term_field_ords(schema: SchemaDescriptor) -> Vec<u16> {
    TERM_FIELD_ORD_BRIDGE_CALLS.with(|calls| calls.set(calls.get().saturating_add(1)));
    schema
        .fields
        .iter()
        .filter_map(|field| match field.kind {
            crate::schema::FieldKind::Keyword | crate::schema::FieldKind::Text { .. } => {
                Some(field.id)
            }
            crate::schema::FieldKind::StoredOnly
            | crate::schema::FieldKind::I64 { .. }
            | crate::schema::FieldKind::U64 { .. } => None,
        })
        .collect()
}

#[cfg(test)]
pub(super) fn reset_term_field_ord_bridge_calls() {
    TERM_FIELD_ORD_BRIDGE_CALLS.with(|calls| calls.set(0));
}

#[cfg(test)]
pub(super) fn term_field_ord_bridge_calls() -> usize {
    TERM_FIELD_ORD_BRIDGE_CALLS.with(Cell::get)
}

#[cfg(test)]
pub(super) fn reset_owned_fieldnorm_bridge_calls() {
    OWNED_FIELDNORM_BRIDGE_CALLS.with(|calls| calls.set(0));
}

#[cfg(test)]
pub(super) fn owned_fieldnorm_bridge_calls() -> usize {
    OWNED_FIELDNORM_BRIDGE_CALLS.with(Cell::get)
}
