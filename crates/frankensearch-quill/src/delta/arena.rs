use std::mem::size_of;

use super::ingest::reserve_vec;
use super::types::DeltaError;
use super::{DELTA_CHAIN_INITIAL_CAPACITY, DELTA_CHAIN_MAX_CAPACITY};

#[derive(Debug, Clone, Copy, Default)]
pub(in crate::delta) struct BlockMeta {
    start: usize,
    pub(in crate::delta) len: usize,
    limit: usize,
    pub(in crate::delta) reserved: usize,
    next: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(in crate::delta) struct ChainState {
    head: Option<usize>,
    pub(in crate::delta) tail: Option<usize>,
    pub(in crate::delta) len: usize,
}

/// Safe typed bump arena: initialized slots and integer block links only.
#[derive(Debug)]
pub(in crate::delta) struct TypedChainArena<T: Copy + Default> {
    resource: &'static str,
    pub(in crate::delta) blocks: Vec<BlockMeta>,
    pub(in crate::delta) slots: Vec<T>,
    pub(in crate::delta) active_blocks: usize,
}

impl<T: Copy + Default> TypedChainArena<T> {
    pub(in crate::delta) fn new(resource: &'static str) -> Self {
        Self {
            resource,
            blocks: Vec::new(),
            slots: Vec::new(),
            active_blocks: 0,
        }
    }

    fn allocate_reserved_block(&mut self, limit: usize) -> usize {
        let index = self.active_blocks;
        if index == self.blocks.len() {
            debug_assert!(self.blocks.len() < self.blocks.capacity());
            let start = self.slots.len();
            debug_assert!(start.saturating_add(limit) <= self.slots.capacity());
            self.slots.resize(start + limit, T::default());
            self.blocks.push(BlockMeta {
                start,
                len: 0,
                limit,
                reserved: limit,
                next: None,
            });
        } else {
            if self.blocks[index].reserved < limit {
                let start = self.slots.len();
                debug_assert!(start.saturating_add(limit) <= self.slots.capacity());
                self.slots.resize(start + limit, T::default());
                self.blocks[index].start = start;
                self.blocks[index].reserved = limit;
            }
            self.blocks[index].len = 0;
            self.blocks[index].limit = limit;
            self.blocks[index].next = None;
        }
        self.active_blocks += 1;
        index
    }

    pub(in crate::delta) fn append_reserved(&mut self, chain: &mut ChainState, value: T) -> usize {
        let mut added_bytes = 0_usize;
        let mut tail = match chain.tail {
            Some(tail) => tail,
            None => {
                let first = self.allocate_reserved_block(DELTA_CHAIN_INITIAL_CAPACITY);
                added_bytes = added_bytes.saturating_add(
                    size_of::<BlockMeta>()
                        + DELTA_CHAIN_INITIAL_CAPACITY.saturating_mul(size_of::<T>()),
                );
                chain.head = Some(first);
                chain.tail = Some(first);
                first
            }
        };
        if self.blocks[tail].len == self.blocks[tail].limit {
            let next_limit = self.blocks[tail]
                .limit
                .saturating_mul(2)
                .min(DELTA_CHAIN_MAX_CAPACITY);
            let next = self.allocate_reserved_block(next_limit);
            added_bytes = added_bytes
                .saturating_add(size_of::<BlockMeta>() + next_limit.saturating_mul(size_of::<T>()));
            self.blocks[tail].next = Some(next);
            chain.tail = Some(next);
            tail = next;
        }
        let block = &mut self.blocks[tail];
        self.slots[block.start + block.len] = value;
        block.len += 1;
        chain.len = chain.len.checked_add(1).expect("preflighted chain length");
        added_bytes
    }

    pub(in crate::delta) fn reserve_appends(
        &mut self,
        plans: &[(ChainState, usize)],
    ) -> Result<(), DeltaError> {
        let mut active_blocks = self.active_blocks;
        let mut additional_slots = 0_usize;

        for &(chain, additional) in plans {
            chain
                .len
                .checked_add(additional)
                .ok_or(DeltaError::CountOverflow {
                    resource: self.resource,
                    count: usize::MAX,
                })?;
            if additional == 0 {
                continue;
            }
            let mut remaining = additional;
            let (mut tail_len, mut tail_limit) = match chain.tail {
                Some(tail) => {
                    let block = &self.blocks[tail];
                    (block.len, block.limit)
                }
                None => {
                    self.simulate_block_allocation(
                        &mut active_blocks,
                        DELTA_CHAIN_INITIAL_CAPACITY,
                        &mut additional_slots,
                    )?;
                    (0, DELTA_CHAIN_INITIAL_CAPACITY)
                }
            };
            loop {
                let available = tail_limit.saturating_sub(tail_len);
                let admitted = remaining.min(available);
                remaining -= admitted;
                if remaining == 0 {
                    break;
                }
                let next_limit = tail_limit.saturating_mul(2).min(DELTA_CHAIN_MAX_CAPACITY);
                self.simulate_block_allocation(
                    &mut active_blocks,
                    next_limit,
                    &mut additional_slots,
                )?;
                tail_len = 0;
                tail_limit = next_limit;
            }
        }

        let additional_blocks = active_blocks.saturating_sub(self.blocks.len());
        reserve_vec(&mut self.blocks, additional_blocks, self.resource)?;
        reserve_vec(&mut self.slots, additional_slots, self.resource)
    }

    fn simulate_block_allocation(
        &self,
        active_blocks: &mut usize,
        limit: usize,
        additional_slots: &mut usize,
    ) -> Result<(), DeltaError> {
        let index = *active_blocks;
        if index >= self.blocks.len() || self.blocks[index].reserved < limit {
            *additional_slots =
                additional_slots
                    .checked_add(limit)
                    .ok_or(DeltaError::CountOverflow {
                        resource: self.resource,
                        count: usize::MAX,
                    })?;
        }
        *active_blocks = active_blocks
            .checked_add(1)
            .ok_or(DeltaError::CountOverflow {
                resource: self.resource,
                count: usize::MAX,
            })?;
        Ok(())
    }

    pub(in crate::delta) fn iter<'a>(&'a self, chain: &ChainState) -> ChainIter<'a, T> {
        ChainIter {
            arena: self,
            block: chain.head,
            offset: 0,
        }
    }

    pub(in crate::delta) fn iter_range<'a>(
        &'a self,
        chain: &ChainState,
        start: usize,
        len: usize,
    ) -> Option<ChainRangeIter<'a, T>> {
        if start.checked_add(len)? > chain.len {
            return None;
        }
        let mut block = chain.head;
        let mut offset = start;
        while let Some(index) = block {
            let meta = &self.blocks[index];
            if offset < meta.len {
                return Some(ChainRangeIter {
                    arena: self,
                    block: Some(index),
                    offset,
                    remaining: len,
                });
            }
            offset -= meta.len;
            block = meta.next;
        }
        (len == 0).then_some(ChainRangeIter {
            arena: self,
            block: None,
            offset: 0,
            remaining: 0,
        })
    }

    /// Copy only the active initialized slots into an immutable epoch arena.
    ///
    /// Mutable arenas retain reset-reusable blocks and relocated slot ranges.
    /// Carrying those inactive allocations into every published snapshot would
    /// keep prior generations' peak footprint alive after a seal. Block indices
    /// remain stable while starts are rebased into one compact owned slot slab.
    pub(in crate::delta) fn snapshot_copy(&self) -> Self {
        let mut blocks = Vec::with_capacity(self.active_blocks);
        let active_slots = self.blocks[..self.active_blocks]
            .iter()
            .map(|block| block.len)
            .sum();
        let mut slots = Vec::with_capacity(active_slots);
        for block in &self.blocks[..self.active_blocks] {
            let start = slots.len();
            slots.extend_from_slice(&self.slots[block.start..block.start + block.len]);
            blocks.push(BlockMeta {
                start,
                len: block.len,
                limit: block.len,
                reserved: block.len,
                next: block.next,
            });
        }
        Self {
            resource: self.resource,
            blocks,
            slots,
            active_blocks: self.active_blocks,
        }
    }

    pub(in crate::delta) fn reset(&mut self) {
        for block in &mut self.blocks[..self.active_blocks] {
            block.len = 0;
            block.next = None;
        }
        self.active_blocks = 0;
    }

    #[cfg(test)]
    pub(in crate::delta) fn bytes_used(&self) -> usize {
        self.blocks[..self.active_blocks]
            .iter()
            .map(|block| size_of::<BlockMeta>() + block.limit.saturating_mul(size_of::<T>()))
            .sum()
    }

    pub(in crate::delta) fn bytes_reserved(&self) -> usize {
        self.blocks
            .capacity()
            .saturating_mul(size_of::<BlockMeta>())
            .saturating_add(self.slots.capacity().saturating_mul(size_of::<T>()))
    }

    #[cfg(test)]
    pub(in crate::delta) fn block_limits(&self, chain: &ChainState) -> Vec<usize> {
        let mut limits = Vec::new();
        let mut block = chain.head;
        while let Some(index) = block {
            limits.push(self.blocks[index].limit);
            block = self.blocks[index].next;
        }
        limits
    }
}

#[derive(Clone)]
pub(in crate::delta) struct ChainIter<'a, T: Copy + Default> {
    arena: &'a TypedChainArena<T>,
    block: Option<usize>,
    offset: usize,
}

impl<'a, T: Copy + Default> Iterator for ChainIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let index = self.block?;
            let block = &self.arena.blocks[index];
            if self.offset < block.len {
                let item = &self.arena.slots[block.start + self.offset];
                self.offset += 1;
                return Some(item);
            }
            self.block = block.next;
            self.offset = 0;
        }
    }
}

pub(in crate::delta) struct ChainRangeIter<'a, T: Copy + Default> {
    arena: &'a TypedChainArena<T>,
    block: Option<usize>,
    offset: usize,
    remaining: usize,
}

impl<'a, T: Copy + Default> Iterator for ChainRangeIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        loop {
            let index = self.block?;
            let block = &self.arena.blocks[index];
            if self.offset < block.len {
                let item = &self.arena.slots[block.start + self.offset];
                self.offset += 1;
                self.remaining -= 1;
                return Some(item);
            }
            self.block = block.next;
            self.offset = 0;
        }
    }
}
