use std::marker::PhantomData;

use frankensearch_core::DocId;

use super::DeltaSegment;
use super::arena::{ChainIter, ChainRangeIter};
use super::types::{DeltaBlockMax, DeltaPosting, PostingRecord};

/// Owner-bound read view for one composite delta term.
#[derive(Debug, Clone, Copy)]
pub struct DeltaTerm<'a> {
    pub(in crate::delta) delta: &'a DeltaSegment,
    pub(in crate::delta) term_index: u32,
}

impl<'a> DeltaTerm<'a> {
    /// Schema field ordinal carried by this key.
    #[must_use]
    pub fn field_ord(self) -> u16 {
        self.delta.terms.field_and_term(self.term_index).0
    }

    /// Exact term bytes without the field prefix.
    #[must_use]
    pub fn term(self) -> &'a [u8] {
        self.delta.terms.field_and_term(self.term_index).1
    }

    /// Physical posting count, including tombstoned rows.
    #[must_use]
    pub fn physical_doc_freq(self) -> usize {
        self.delta.chains[self.term_index as usize].postings.len
    }

    /// Live posting count after applying delta tombstones.
    #[must_use]
    pub fn live_doc_freq(self) -> usize {
        // Maintained where liveness changes, so this is a field read rather
        // than a walk of the chain. See `TermChain::live_len`.
        self.delta.chains[self.term_index as usize].live_len as usize
    }

    /// Compute the live cardinality and whole-term pruning envelope in one
    /// physical-chain pass.
    #[must_use]
    pub(crate) fn live_doc_freq_and_block_max(self) -> (usize, Option<DeltaBlockMax>) {
        let live_doc_freq = self.live_doc_freq();
        let block_max = (live_doc_freq != 0).then(|| {
            let chain = &self.delta.chains[self.term_index as usize];
            DeltaBlockMax {
                max_frequency_code: chain.max_frequency_code,
                min_fieldnorm_id: chain.min_fieldnorm_id,
            }
        });
        (live_doc_freq, block_max)
    }

    /// Conservative whole-term impact envelope for future pruning paths.
    ///
    /// Returns `None` when every physical posting has been superseded or
    /// deleted. Otherwise the bound may include tombstoned historical maxima,
    /// which only makes it more conservative.
    #[must_use]
    pub fn block_max(self) -> Option<DeltaBlockMax> {
        self.live_doc_freq_and_block_max().1
    }

    /// Whether an owner-bound posting remains visible in this generation.
    pub(crate) fn is_live(self, posting: DeltaPosting<'a>) -> bool {
        posting.term_index == self.term_index
            && posting.owner_id == self.delta.owner_id
            && posting.generation == self.delta.generation
            && !self.delta.is_tombstoned(posting.global_docid)
    }

    /// Iterate physical postings in global-docid order.
    #[must_use]
    pub fn postings(self) -> DeltaPostings<'a> {
        let chain = &self.delta.chains[self.term_index as usize];
        DeltaPostings {
            inner: self.delta.posting_arena.iter(&chain.postings),
            term_index: self.term_index,
            owner_id: self.delta.owner_id,
            generation: self.delta.generation,
        }
    }

    /// Resolve positions for a posting yielded by this same term.
    #[must_use]
    pub fn positions(self, posting: DeltaPosting<'a>) -> Option<DeltaPositions<'a>> {
        if posting.term_index != self.term_index
            || posting.owner_id != self.delta.owner_id
            || posting.generation != self.delta.generation
            || !posting.has_positions
        {
            return None;
        }
        let chain = &self.delta.chains[self.term_index as usize];
        Some(DeltaPositions {
            inner: self.delta.position_arena.iter_range(
                &chain.positions,
                posting.position_start as usize,
                posting.frequency as usize,
            )?,
        })
    }
}

/// Physical posting iterator for one owner-bound term.
#[derive(Clone)]
pub struct DeltaPostings<'a> {
    inner: ChainIter<'a, PostingRecord>,
    term_index: u32,
    owner_id: u64,
    generation: u64,
}

impl<'a> Iterator for DeltaPostings<'a> {
    type Item = DeltaPosting<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let posting = *self.inner.next()?;
        Some(DeltaPosting {
            global_docid: posting.global_docid,
            frequency: posting.frequency,
            term_index: self.term_index,
            position_start: posting.position_start,
            has_positions: posting.has_positions,
            owner_id: self.owner_id,
            generation: self.generation,
            owner: PhantomData,
        })
    }
}

/// Exact position iterator for one posting.
pub struct DeltaPositions<'a> {
    inner: ChainRangeIter<'a, u32>,
}

impl Iterator for DeltaPositions<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().copied()
    }
}

/// Ordered live document iterator.
pub struct DeltaLiveDocuments<'a> {
    pub(in crate::delta) delta: &'a DeltaSegment,
    pub(in crate::delta) index: usize,
}

impl<'a> Iterator for DeltaLiveDocuments<'a> {
    type Item = (u32, &'a DocId);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.delta.document_docids.len() {
            let index = self.index;
            self.index += 1;
            let docid = self.delta.document_docids[index];
            if !self.delta.is_tombstoned(docid) {
                return Some((docid, &self.delta.document_ids[index]));
            }
        }
        None
    }
}
