//! Native HNSW graph index (bd-kcek).
//!
//! frankensearch's own approximate-nearest-neighbour engine, owned in-tree
//! rather than adapted from a third-party crate. Two things motivate it:
//! we cannot publish to crates.io while the ANN engine is a git-only
//! dependency, and we should never be blocked on an external maintainer's
//! release timing for an algorithm this size.
//!
//! # Correctness is inherited, not re-derived
//!
//! bd-u3wt reverse-engineered the layer invariants by finding four coupled
//! defects in the previous engine. Those findings are the specification
//! this module implements, and each is enforced at the point it can be
//! violated:
//!
//! 1. **Descent must not attach above the sampled level.** The greedy
//!    descent that locates an entry point runs strictly above the new
//!    point's own level and never links there.
//! 2. **Connection must start at `min(new_level, previous_max_level)`.**
//!    Starting higher leaves a new point's upper layers empty; starting
//!    from the new level unconditionally links at layers no other point
//!    occupies yet.
//! 3. **Reverse edges belong to the layer being processed.** A reciprocal
//!    edge discovered while linking at layer `l` is stored at layer `l` —
//!    not at the new point's top layer. Storing it at the top layer both
//!    omits required lower-layer edges and can install an edge in a layer
//!    the endpoints do not occupy, which is what produced asymmetric
//!    underfill.
//! 4. **Layer participation is `level >= layer`, not `level == layer`.** A
//!    point participates in every layer at or below its sampled level, so
//!    an empty exact-level bucket does not mean the logical layer is empty.
//!    (Exposed by a `high, high, zero` sampled-level sequence.)
//!
//! # Safety
//!
//! This module contains no `unsafe`, and needs none: the graph stores ids
//! and distances, never reinterpreted bytes. So the class of defect the
//! previous engine's reload path carried — a misaligned pointer cast whose
//! element count came from a file header rather than the bytes actually
//! read — cannot occur here.
//!
//! That is a property of this module, not a blanket guarantee. The
//! workspace lints `unsafe_code` at `deny` rather than `forbid`
//! specifically so crates can opt in, and this crate does exactly that for
//! memory-mapping (`mapped_file.rs` and the `MmapMut` path in `lib.rs`).
//! Any future persistence layer for this graph inherits that risk the
//! moment it maps a file, so it should parse into owned values rather than
//! casting mapped bytes.

mod binding;
mod codec;
mod graph;
mod io_helpers;
mod receipt;
mod types;

#[cfg(test)]
mod tests;

pub use receipt::{NativeHnswGenerationReceiptV2, native_hnsw_generation_receipt_path};
pub use types::{
    HnswParams, MAX_LEVEL, NativeHnswParamsIdentityV1, ValidatedNativeHnsw, ValidatedNativeHnswHit,
};

const NATIVE_HNSW_MAGIC: [u8; 8] = *b"FSHNSW\0\0";
/// Current owned native-HNSW graph format.
pub const NATIVE_HNSW_FORMAT_VERSION: u32 = 1;
const NATIVE_HNSW_HEADER_LEN: usize = 96;
const NATIVE_HNSW_HEADER_LEN_U64: u64 = 96;
const NATIVE_HNSW_HEADER_CRC_OFFSET: usize = NATIVE_HNSW_HEADER_LEN - 4;
const NATIVE_HNSW_NO_ENTRY: u64 = u64::MAX;
const NATIVE_HNSW_RECEIPT_MAGIC: [u8; 8] = *b"FSHNRC\0\0";
/// Current schema for native-HNSW-to-FSVI generation receipts.
pub const NATIVE_HNSW_GENERATION_RECEIPT_SCHEMA_V2: u16 = 2;
/// Canonical sidecar suffix appended to the complete `.fshnsw` basename.
pub const NATIVE_HNSW_GENERATION_RECEIPT_SUFFIX: &str = ".receipt";
const NATIVE_HNSW_MAX_BASENAME_BYTES: usize = 255 - NATIVE_HNSW_GENERATION_RECEIPT_SUFFIX.len();
const SHA256_HEX_LEN: usize = 64;
const SHA256_BYTES: usize = 32;

#[cfg(test)]
impl NativeHnswGenerationReceiptV2 {
    /// Seal a synthetic receipt used by sibling-module unit tests.
    pub(crate) fn seal_for_test(&mut self) -> frankensearch_core::error::SearchResult<()> {
        self.seal()
    }
}
