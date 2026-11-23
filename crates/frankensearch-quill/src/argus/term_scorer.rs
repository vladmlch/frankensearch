use std::sync::Arc;

use crate::contract::{BM25_K1, compute_tf_cache, idf};
use crate::quiver::{DocLenField, SnapshotFieldStats};

use super::ArgusError;
use super::boolean_scorers::finite_score;
use super::cursor::{PostingCursor, TermCursor, ValidatedArrivalCursor};

/// Shared, validated BM25 denominator cache for one `(field, snapshot)` pair.
#[derive(Clone, Debug)]
pub struct Bm25FieldSnapshot {
    stats: SnapshotFieldStats,
    average_field_length: Option<f32>,
    pub(in crate::argus) tf_cache: Arc<[f32; 256]>,
}

impl Bm25FieldSnapshot {
    /// Validate snapshot counters and build the field's 256-entry tf cache.
    ///
    /// An empty or all-empty field has no scored postings and therefore keeps
    /// a zero cache. A term scorer rejects any posting that would try to use
    /// that cache.
    ///
    /// # Errors
    ///
    /// Returns a typed error when an empty snapshot reports tokens or the raw
    /// average cannot be represented as a finite f32.
    pub fn new(stats: SnapshotFieldStats) -> Result<Self, ArgusError> {
        if stats.doc_count == 0 && stats.total_tokens != 0 {
            return Err(ArgusError::InvalidSnapshot {
                field_ord: stats.field_ord,
                reason: "an empty snapshot cannot contain field tokens",
            });
        }
        let average_field_length = stats.average_field_length();
        if average_field_length.is_some_and(|average| !average.is_finite()) {
            return Err(ArgusError::InvalidSnapshot {
                field_ord: stats.field_ord,
                reason: "raw average field length is not finite",
            });
        }
        let tf_cache = average_field_length
            .filter(|average| *average > 0.0)
            .map_or_else(
                || Arc::new([0.0; 256]),
                |average| Arc::new(compute_tf_cache(average)),
            );
        Ok(Self {
            stats,
            average_field_length,
            tf_cache,
        })
    }

    /// Stable schema field ordinal.
    #[must_use]
    pub const fn field_ord(&self) -> u16 {
        self.stats.field_ord
    }

    /// Snapshot lifecycle population used as BM25 `N`.
    ///
    /// Composite views count Keeper's at-seal rows plus live Delta rows.
    #[must_use]
    pub const fn doc_count(&self) -> u64 {
        self.stats.doc_count
    }

    /// Snapshot token numerator under the same lifecycle as [`Self::doc_count`].
    ///
    /// Keeper tombstones remain until compaction; Delta tombstones are excluded
    /// immediately because those rows are never sealed.
    #[must_use]
    pub const fn total_tokens(&self) -> u64 {
        self.stats.total_tokens
    }

    /// Raw `total_tokens / N` average, or `None` for an empty snapshot.
    #[must_use]
    pub const fn average_field_length(&self) -> Option<f32> {
        self.average_field_length
    }
}

/// Backend-neutral fieldnorm access for sealed and delta segment views.
pub trait FieldNormReader {
    /// Stable schema field ordinal.
    fn field_ord(&self) -> u16;

    /// Stored Tantivy-compatible fieldnorm ID for one present global docid.
    fn fieldnorm_id(&self, global_docid: u32) -> Option<u8>;
}

impl FieldNormReader for DocLenField<'_> {
    fn field_ord(&self) -> u16 {
        DocLenField::field_ord(*self)
    }

    fn fieldnorm_id(&self, global_docid: u32) -> Option<u8> {
        DocLenField::fieldnorm_id(*self, u64::from(global_docid))
    }
}

/// Posting detail selected by the field schema for term scoring.
///
/// Snapshot field statistics retain the complete analyzed token count
/// independently of posting detail. Tantivy's `Basic` record option scores
/// every matching document with an effective term frequency of one, whether
/// the cursor exposes a canonical one or a raw sealed count. Frequency-bearing
/// options consume the cursor's recorded count unchanged.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TermRecordOption {
    /// Document presence only; BM25 uses an effective term frequency of one.
    Basic,
    /// Document presence plus term frequency.
    WithFreqs,
    /// Document presence, term frequency, and positions.
    WithFreqsAndPositions,
}

impl TermRecordOption {
    #[inline]
    const fn effective_frequency(self, raw_frequency: u32) -> u32 {
        match self {
            Self::Basic => 1,
            Self::WithFreqs | Self::WithFreqsAndPositions => raw_frequency,
        }
    }

    pub(in crate::argus) fn score_upper_bound(
        self,
        raw_max_frequency: u32,
        min_fieldnorm_id: u8,
        live_avgdl: f32,
        weight: f32,
    ) -> Option<f32> {
        crate::contract::block_max_score(
            crate::contract::block_max_frequency_to_code(
                self.effective_frequency(raw_max_frequency),
            ),
            min_fieldnorm_id,
            live_avgdl,
            weight,
        )
    }
}

/// One BM25 term leaf bound to a segment cursor and fieldnorm view.
pub struct TermScorer<'a> {
    pub(in crate::argus) cursor: TermCursor<'a>,
    fieldnorms: Box<dyn FieldNormReader + 'a>,
    snapshot: Bm25FieldSnapshot,
    pub(in crate::argus) record_option: TermRecordOption,
    weight: f32,
    pub(in crate::argus) term_score_upper_bound: Option<f32>,
    pub(in crate::argus) cost: u64,
    pub(in crate::argus) size_hint: u32,
    pub(in crate::argus) segment_num_docs: u32,
    validated_post_move_contract: bool,
}

impl<'a> TermScorer<'a> {
    /// Build a term scorer from snapshot-level statistics and segment-local
    /// cursor state.
    ///
    /// `snapshot_doc_freq` follows the same lifecycle population: Keeper
    /// postings include tombstones until compaction, while Delta postings count
    /// only live rows. Runtime cost, size hint, and segment `num_docs` come from
    /// the cursor so exact required-clause ordering cannot drift from its
    /// storage implementation.
    ///
    /// # Errors
    ///
    /// Rejects field drift, malformed snapshot statistics, inconsistent
    /// cursor accessors, invalid boosts, and `doc_freq > N` without invoking
    /// the vendored `idf` assertion.
    pub fn new<C, F>(
        cursor: C,
        fieldnorms: F,
        snapshot: Bm25FieldSnapshot,
        snapshot_doc_freq: u64,
        record_option: TermRecordOption,
        field_boost: f32,
    ) -> Result<Self, ArgusError>
    where
        C: PostingCursor + 'a,
        F: FieldNormReader + 'a,
    {
        Self::with_term_cursor(
            TermCursor::Opaque(Box::new(cursor)),
            fieldnorms,
            snapshot,
            snapshot_doc_freq,
            record_option,
            field_boost,
        )
    }

    /// Build a term scorer over the crate's own validated cursor.
    ///
    /// This is the only way into [`TermCursor::Validated`], and it takes the
    /// concrete type by value. A wrapper cannot reach this arm: it has nothing
    /// to pass, and handing over an embedded cursor would mean surrendering the
    /// instance it would otherwise drive — which is exactly the substitution the
    /// arm exists to prevent, since the scorer then advances, scores and probes
    /// that one value.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::new`].
    pub(crate) fn new_validated<F>(
        cursor: ValidatedArrivalCursor<'a>,
        fieldnorms: F,
        snapshot: Bm25FieldSnapshot,
        snapshot_doc_freq: u64,
        record_option: TermRecordOption,
        field_boost: f32,
    ) -> Result<Self, ArgusError>
    where
        F: FieldNormReader + 'a,
    {
        Self::with_term_cursor(
            TermCursor::Validated(Box::new(cursor)),
            fieldnorms,
            snapshot,
            snapshot_doc_freq,
            record_option,
            field_boost,
        )
    }

    fn with_term_cursor<F>(
        cursor: TermCursor<'a>,
        fieldnorms: F,
        snapshot: Bm25FieldSnapshot,
        snapshot_doc_freq: u64,
        record_option: TermRecordOption,
        field_boost: f32,
    ) -> Result<Self, ArgusError>
    where
        F: FieldNormReader + 'a,
    {
        if fieldnorms.field_ord() != snapshot.field_ord() {
            return Err(ArgusError::FieldMismatch {
                fieldnorm_field: fieldnorms.field_ord(),
                stats_field: snapshot.field_ord(),
            });
        }
        if snapshot_doc_freq > snapshot.doc_count() {
            return Err(ArgusError::InvalidDocFrequency {
                doc_freq: snapshot_doc_freq,
                doc_count: snapshot.doc_count(),
            });
        }
        let size_hint = cursor.size_hint();
        let cost = cursor.cost();
        let segment_num_docs = cursor.segment_num_docs();
        let validated_post_move_contract = cursor.validated_post_move_contract().is_some();
        if u64::from(segment_num_docs) > snapshot.doc_count() {
            return Err(ArgusError::InvalidSnapshot {
                field_ord: snapshot.field_ord(),
                reason: "segment num_docs exceeds snapshot BM25 document count",
            });
        }
        if !field_boost.is_finite() {
            return Err(ArgusError::InvalidBoost {
                field_ord: snapshot.field_ord(),
                boost_bits: field_boost.to_bits(),
            });
        }
        if snapshot_doc_freq != 0
            && snapshot
                .average_field_length()
                .is_none_or(|average| average <= 0.0)
        {
            return Err(ArgusError::InvalidSnapshot {
                field_ord: snapshot.field_ord(),
                reason: "a scored term requires a positive raw average field length",
            });
        }
        if cursor.doc().is_some() {
            if snapshot_doc_freq == 0 {
                return Err(ArgusError::InvalidSnapshot {
                    field_ord: snapshot.field_ord(),
                    reason: "a non-empty cursor cannot have zero snapshot doc_freq",
                });
            }
            if size_hint == 0 || cost == 0 {
                return Err(ArgusError::CursorInvariant(
                    "a non-empty cursor must have non-zero size and runtime cost",
                ));
            }
            validate_cursor_position(&cursor, &fieldnorms)?;
        } else {
            if cursor.freq().is_some() || cursor.positions_handle().is_some() {
                return Err(ArgusError::CursorInvariant(
                    "exhausted cursor retained frequency or positions state",
                ));
            }
            if size_hint != 0 || cost != 0 {
                return Err(ArgusError::CursorInvariant(
                    "an empty cursor must have zero size and runtime cost",
                ));
            }
        }

        let mut weight = idf(snapshot_doc_freq, snapshot.doc_count()) * (1.0 + BM25_K1);
        weight *= field_boost;
        if !weight.is_finite() {
            return Err(ArgusError::InvalidBoost {
                field_ord: snapshot.field_ord(),
                boost_bits: field_boost.to_bits(),
            });
        }
        // Query-time avgdl and idf/boost are immutable for this scorer. Scan a
        // sealed term's validated block table once here instead of once per
        // 4,096-doc competitive window.
        let term_score_upper_bound = snapshot
            .average_field_length()
            .filter(|average| average.is_finite() && *average > 0.0)
            .filter(|_| !weight.is_sign_negative())
            .and_then(|average| cursor.term_score_upper_bound(average, weight, record_option))
            .filter(|bound| bound.is_finite() && !bound.is_sign_negative());
        Ok(Self {
            cursor,
            fieldnorms: Box::new(fieldnorms),
            snapshot,
            record_option,
            weight,
            term_score_upper_bound,
            cost,
            size_hint,
            segment_num_docs,
            validated_post_move_contract,
        })
    }

    pub(in crate::argus) fn doc(&self) -> Option<u32> {
        self.cursor.doc()
    }

    pub(in crate::argus) fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        let previous = self.cursor.doc();
        let moved = self.cursor.next()?;
        let moved = if self.validated_post_move_contract {
            moved
        } else {
            validate_cursor_after_move(&self.cursor, self.fieldnorms.as_ref(), moved)?
        };
        match (previous, moved) {
            (None, Some(_)) => Err(ArgusError::CursorInvariant(
                "exhausted cursor resurrected during next",
            )),
            (Some(before), Some(after)) if after <= before => Err(ArgusError::CursorInvariant(
                "next did not move to a strictly greater document",
            )),
            _ => Ok(moved),
        }
    }

    pub(in crate::argus) fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        let previous = self.cursor.doc();
        let moved = self.cursor.advance(target)?;
        let moved = if self.validated_post_move_contract {
            moved
        } else {
            validate_cursor_after_move(&self.cursor, self.fieldnorms.as_ref(), moved)?
        };
        match (previous, moved) {
            (None, Some(_)) => Err(ArgusError::CursorInvariant(
                "exhausted cursor resurrected during advance",
            )),
            (Some(before), Some(after)) if before >= target && after != before => {
                Err(ArgusError::CursorInvariant(
                    "advance moved despite the current document satisfying its target",
                ))
            }
            (Some(before), None) if before >= target => Err(ArgusError::CursorInvariant(
                "advance exhausted despite the current document satisfying its target",
            )),
            (_, Some(after)) if after < target => Err(ArgusError::CursorInvariant(
                "advance landed below its requested target",
            )),
            _ => Ok(moved),
        }
    }

    pub(in crate::argus) fn score(&self) -> Result<f32, ArgusError> {
        let doc = self.cursor.doc().ok_or(ArgusError::CursorInvariant(
            "cannot score an exhausted cursor",
        ))?;
        let frequency = self.cursor.freq().ok_or(ArgusError::CursorInvariant(
            "current posting has no frequency",
        ))?;
        if frequency == 0 {
            return Err(ArgusError::CursorInvariant(
                "current posting has zero term frequency",
            ));
        }
        let frequency = self.record_option.effective_frequency(frequency);
        let fieldnorm_id =
            self.fieldnorms
                .fieldnorm_id(doc)
                .ok_or_else(|| ArgusError::MissingFieldnorm {
                    field_ord: self.fieldnorms.field_ord(),
                    global_docid: doc,
                })?;
        let frequency = frequency as f32;
        let norm = self.snapshot.tf_cache[usize::from(fieldnorm_id)];
        let tf_factor = frequency / (frequency + norm);
        finite_score(self.weight * tf_factor, doc)
    }

    pub(in crate::argus) fn competitive_cursor(&self) -> Option<CompetitiveTermCursor<'_>> {
        let live_avgdl = self.snapshot.average_field_length()?;
        let term_upper_bound = self.term_score_upper_bound?;
        Some(CompetitiveTermCursor {
            cursor: self.cursor.fork_for_pruning()?,
            live_avgdl,
            weight: self.weight,
            record_option: self.record_option,
            term_upper_bound,
        })
    }
}

pub(in crate::argus) struct CompetitiveTermCursor<'a> {
    cursor: Box<dyn PostingCursor + 'a>,
    live_avgdl: f32,
    weight: f32,
    record_option: TermRecordOption,
    term_upper_bound: f32,
}

impl CompetitiveTermCursor<'_> {
    pub(in crate::argus) fn doc(&self) -> Option<u32> {
        self.cursor.doc()
    }

    pub(in crate::argus) fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        self.cursor.next()
    }

    pub(in crate::argus) fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        self.cursor.advance(target)
    }

    pub(in crate::argus) const fn term_upper_bound(&self) -> f32 {
        self.term_upper_bound
    }

    pub(in crate::argus) fn current_block_upper_bound(&self) -> Option<f32> {
        let bound = self.cursor.current_block_score_upper_bound(
            self.live_avgdl,
            self.weight,
            self.record_option,
        )?;
        (bound.is_finite() && !bound.is_sign_negative()).then_some(bound)
    }

    pub(in crate::argus) fn current_block_last_doc(&self) -> Option<u32> {
        self.cursor.current_block_last_doc()
    }
}

fn validate_cursor_after_move(
    cursor: &dyn PostingCursor,
    fieldnorms: &dyn FieldNormReader,
    moved_doc: Option<u32>,
) -> Result<Option<u32>, ArgusError> {
    if cursor.doc() != moved_doc {
        return Err(ArgusError::CursorInvariant(
            "movement result disagrees with current document",
        ));
    }
    if moved_doc.is_none() {
        if cursor.freq().is_some() || cursor.positions_handle().is_some() {
            return Err(ArgusError::CursorInvariant(
                "exhausted cursor retained frequency or positions state",
            ));
        }
        return Ok(None);
    }
    validate_cursor_position(cursor, fieldnorms)?;
    Ok(moved_doc)
}

fn validate_cursor_position<C>(
    cursor: &C,
    fieldnorms: &dyn FieldNormReader,
) -> Result<(), ArgusError>
where
    C: PostingCursor + ?Sized,
{
    let doc = cursor.doc().ok_or(ArgusError::CursorInvariant(
        "positioned cursor has no document",
    ))?;
    if cursor.freq().is_none_or(|frequency| frequency == 0) {
        return Err(ArgusError::CursorInvariant(
            "positioned cursor has no positive frequency",
        ));
    }
    if fieldnorms.fieldnorm_id(doc).is_none() {
        return Err(ArgusError::MissingFieldnorm {
            field_ord: fieldnorms.field_ord(),
            global_docid: doc,
        });
    }
    Ok(())
}
