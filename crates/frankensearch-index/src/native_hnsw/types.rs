use std::fmt;
use std::sync::Arc;

use crate::ValidatedFsviBytes;
use frankensearch_core::error::{SearchError, SearchResult};
use serde::{Deserialize, Serialize};

use super::binding::NativeHnswGenerationBindingV2;
use super::codec::NativeHnsw;
use super::io_helpers::{receipt_usize, usize_to_u64};

/// A violation of the graph's structural invariants.
///
/// Structured rather than stringly so an attestation gate can branch on the
/// defect class, and so the regression fixtures below assert on the variant
/// instead of matching message text. The graph itself indexes rows of a
/// caller-owned store and owns no file, so this is deliberately not a
/// [`SearchError`]: the wiring layer, which knows the sidecar path, maps a
/// defect into `SearchError::IndexCorrupted`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum GraphDefect {
    /// The graph and its caller-owned vector store describe different row
    /// counts, so graph ids cannot be interpreted against that store.
    StoreCardinalityMismatch {
        /// Rows indexed by the graph.
        graph_points: usize,
        /// Rows exposed by the vector store.
        store_rows: usize,
    },
    /// A graph with no points nonetheless names an entry point.
    EntryPointInEmptyGraph {
        /// The spurious entry point id.
        entry: u32,
    },
    /// A graph with no points nonetheless claims an occupied upper layer.
    MaxLevelInEmptyGraph {
        /// The spurious maximum level.
        max_level: usize,
    },
    /// A graph with points has no entry point, so nothing is searchable.
    MissingEntryPoint,
    /// The entry point is not a known row.
    EntryPointUnknown {
        /// The dangling entry point id.
        entry: u32,
    },
    /// The entry point does not sit at the graph's maximum level, so upper
    /// layers are unreachable.
    EntryPointBelowMaxLevel {
        /// The entry point id.
        entry: u32,
        /// The level it actually occupies.
        level: usize,
        /// The graph's maximum level.
        max_level: usize,
    },
    /// A point claims a level above the graph maximum.
    LevelAboveMax {
        /// The offending point.
        id: u32,
        /// Its claimed level.
        level: usize,
        /// The graph's maximum level.
        max_level: usize,
    },
    /// A neighbourhood exceeds its layer's degree budget.
    DegreeExceeded {
        /// The offending point.
        id: u32,
        /// The layer.
        layer: usize,
        /// Neighbours actually held.
        held: usize,
        /// The budget for this layer.
        budget: usize,
    },
    /// A neighbourhood lists the same neighbour twice.
    DuplicateNeighbour {
        /// The offending point.
        id: u32,
        /// The layer.
        layer: usize,
    },
    /// A point lists itself as its own neighbour.
    SelfEdge {
        /// The offending point.
        id: u32,
        /// The layer.
        layer: usize,
    },
    /// An edge names a neighbour that is not a known row.
    NeighbourUnknown {
        /// The offending point.
        id: u32,
        /// The layer.
        layer: usize,
        /// The dangling neighbour id.
        neighbour: u32,
    },
    /// An edge exists in a layer one endpoint does not occupy.
    ///
    /// This is the bd-u3wt asymmetric-underfill defect: reverse edges
    /// stored at the new point's top layer rather than the layer being
    /// processed installed edges in layers the target never reached.
    EdgeAboveNeighbourLevel {
        /// The point holding the edge.
        id: u32,
        /// The layer the edge sits in.
        layer: usize,
        /// The neighbour that cannot participate there.
        neighbour: u32,
        /// The level the neighbour actually reaches.
        neighbour_level: usize,
    },
    /// One endpoint names an edge whose peer does not name the reverse edge
    /// at the same logical layer.
    MissingReciprocalEdge {
        /// The point holding the one-way edge.
        id: u32,
        /// The layer containing the one-way edge.
        layer: usize,
        /// The peer missing the reverse edge.
        neighbour: u32,
    },
    /// A point's persisted layer count disagrees with the level sampled from
    /// the graph's seed and parameters.
    SampledLevelMismatch {
        /// The point whose level is not reproducible.
        id: u32,
        /// Level deterministically sampled for this point.
        expected: usize,
        /// Level stored in the graph.
        actual: usize,
    },
    /// Some points cannot be reached from the entry point at layer 0, so no
    /// query can find them however wide the beam.
    Unreachable {
        /// How many points were reached.
        reached: usize,
        /// How many points exist.
        total: usize,
        /// The lowest-numbered unreachable point.
        first_unreachable: u32,
    },
    /// A full-width search over every physical row did not produce the
    /// live-result cardinality promised by the admitted owner's census.
    ///
    /// This can only happen if graph reachability or the owner census has
    /// become inconsistent after admission. Returning a short result would
    /// make that corruption look like ordinary low recall, so search fails
    /// with this typed defect instead.
    LiveResultUnderfill {
        /// Result count requested by the caller.
        requested_k: usize,
        /// Live rows the owner census says must be returned.
        expected_live_hits: usize,
        /// Live rows actually found after searching every physical row.
        returned_live_hits: usize,
        /// Physical rows used as the hard search-width bound.
        physical_rows: usize,
    },
}

impl fmt::Display for GraphDefect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StoreCardinalityMismatch {
                graph_points,
                store_rows,
            } => write!(
                f,
                "graph indexes {graph_points} points but the vector store exposes {store_rows} rows"
            ),
            Self::EntryPointInEmptyGraph { entry } => {
                write!(f, "empty graph names entry point {entry}")
            }
            Self::MaxLevelInEmptyGraph { max_level } => {
                write!(f, "empty graph claims maximum level {max_level}")
            }
            Self::MissingEntryPoint => write!(f, "non-empty graph has no entry point"),
            Self::EntryPointUnknown { entry } => {
                write!(f, "entry point {entry} is not a known row")
            }
            Self::EntryPointBelowMaxLevel {
                entry,
                level,
                max_level,
            } => write!(
                f,
                "entry point {entry} is at level {level} but the graph maximum is {max_level}"
            ),
            Self::LevelAboveMax {
                id,
                level,
                max_level,
            } => write!(
                f,
                "point {id} claims level {level} above the graph maximum {max_level}"
            ),
            Self::DegreeExceeded {
                id,
                layer,
                held,
                budget,
            } => write!(
                f,
                "point {id} holds {held} neighbours at layer {layer}, above the {budget} budget"
            ),
            Self::DuplicateNeighbour { id, layer } => {
                write!(f, "point {id} has duplicate neighbours at layer {layer}")
            }
            Self::SelfEdge { id, layer } => {
                write!(f, "point {id} is its own neighbour at layer {layer}")
            }
            Self::NeighbourUnknown {
                id,
                layer,
                neighbour,
            } => write!(
                f,
                "point {id} names neighbour {neighbour} at layer {layer}, which is not a known row"
            ),
            Self::EdgeAboveNeighbourLevel {
                id,
                layer,
                neighbour,
                neighbour_level,
            } => write!(
                f,
                "point {id} has an edge to {neighbour} at layer {layer}, but {neighbour} only \
                 reaches level {neighbour_level}"
            ),
            Self::MissingReciprocalEdge {
                id,
                layer,
                neighbour,
            } => write!(
                f,
                "point {id} names neighbour {neighbour} at layer {layer}, but the reverse edge is \
                 missing"
            ),
            Self::SampledLevelMismatch {
                id,
                expected,
                actual,
            } => write!(
                f,
                "point {id} stores level {actual}, but seed and parameters sample level {expected}"
            ),
            Self::Unreachable {
                reached,
                total,
                first_unreachable,
            } => write!(
                f,
                "only {reached} of {total} points are reachable at layer 0 (first unreachable: \
                 {first_unreachable})"
            ),
            Self::LiveResultUnderfill {
                requested_k,
                expected_live_hits,
                returned_live_hits,
                physical_rows,
            } => write!(
                f,
                "full-width native HNSW search over {physical_rows} physical rows returned \
                 {returned_live_hits} live hits, but request k={requested_k} and the admitted \
                 owner census require {expected_live_hits}"
            ),
        }
    }
}

impl std::error::Error for GraphDefect {}

/// Double a search window without ever crossing its physical-row bound.
///
/// The `current + 1` floor guarantees progress even if a future caller starts
/// at zero; the early return makes that addition safe at `usize::MAX`.
pub(super) fn widen_search_width(current: usize, physical_rows: usize) -> usize {
    if current >= physical_rows {
        return physical_rows;
    }
    current
        .saturating_mul(2)
        .max(current + 1)
        .min(physical_rows)
}

/// Maximum number of graph layers.
///
/// Level assignment is geometric, so the probability of sampling a level at
/// all near this bound is negligible for any corpus that fits in memory;
/// the cap exists so layer tables are fixed-size and a corrupt or hostile
/// level can never drive an unbounded allocation.
pub const MAX_LEVEL: usize = 16;

/// Tuning parameters for graph construction and search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HnswParams {
    /// Neighbours kept per point per layer above layer 0.
    pub m: usize,
    /// Neighbours kept per point at layer 0.
    ///
    /// Layer 0 holds every point and carries the final hop of every query,
    /// so it is conventionally given twice the degree of upper layers.
    /// Must be at least two. Up to two slots preserve links to consecutive
    /// physical rows, keeping reciprocal pruning connected even for equal vectors.
    pub m0: usize,
    /// Beam width during construction.
    pub ef_construction: usize,
    /// Default beam width during search. Callers may override per query.
    pub ef_search: usize,
}

/// Integrity metadata read from or written with one owned graph artifact.
///
/// The CRC fields detect accidental corruption inside the graph format. They
/// are not a cryptographic generation identity; the production wiring layer
/// binds the complete artifact to its FSVI generation with a SHA-256 receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct NativeHnswFileMetadata {
    pub(super) format_version: u32,
    pub(super) byte_len: u64,
    pub(super) point_count: u64,
    pub(super) payload_crc32: u32,
    pub(super) header_crc32: u32,
}

impl NativeHnswFileMetadata {
    /// Binary format version.
    #[must_use]
    pub(super) const fn format_version(self) -> u32 {
        self.format_version
    }

    /// Complete file length.
    #[must_use]
    pub(super) const fn byte_len(self) -> u64 {
        self.byte_len
    }

    /// Number of graph points encoded in the artifact.
    #[must_use]
    pub(super) const fn point_count(self) -> u64 {
        self.point_count
    }

    /// CRC-32 of the canonical adjacency payload.
    #[must_use]
    pub(super) const fn payload_crc32(self) -> u32 {
        self.payload_crc32
    }

    /// CRC-32 of the fixed header before its checksum field.
    #[must_use]
    pub(super) const fn header_crc32(self) -> u32 {
        self.header_crc32
    }
}

/// Canonical persisted identity of native-HNSW construction parameters.
///
/// The receipt uses full-width integers so its encoding is independent of the
/// reader's pointer width. Conversion back to [`HnswParams`] remains checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeHnswParamsIdentityV1 {
    /// Neighbours retained per upper-layer point.
    pub m: u64,
    /// Neighbours retained per layer-zero point.
    pub m0: u64,
    /// Construction beam width.
    pub ef_construction: u64,
    /// Default search beam width.
    pub ef_search: u64,
}

impl NativeHnswParamsIdentityV1 {
    pub(super) fn from_params(params: HnswParams) -> SearchResult<Self> {
        Ok(Self {
            m: usize_to_u64(params.m, "receipt.params.m")?,
            m0: usize_to_u64(params.m0, "receipt.params.m0")?,
            ef_construction: usize_to_u64(
                params.ef_construction,
                "receipt.params.ef_construction",
            )?,
            ef_search: usize_to_u64(params.ef_search, "receipt.params.ef_search")?,
        })
    }

    pub(super) fn to_params(self) -> SearchResult<HnswParams> {
        let params = HnswParams {
            m: receipt_usize(self.m, "params.m")?,
            m0: receipt_usize(self.m0, "params.m0")?,
            ef_construction: receipt_usize(self.ef_construction, "params.ef_construction")?,
            ef_search: receipt_usize(self.ef_search, "params.ef_search")?,
        };
        params.validate()?;
        Ok(params)
    }
}

/// A receipted native HNSW graph that retains the exact admitted FSVI byte
/// owner used for construction or load.
///
/// The graph, receipt binding, and retained owner are private. Public callers
/// cannot extract a receipted graph and reattach it to another same-cardinality
/// vector store:
///
/// ```compile_fail
/// use frankensearch_index::native_hnsw::NativeHnsw;
///
/// fn bypass_receipts() {
///     let _ = std::mem::size_of::<NativeHnsw>();
/// }
/// ```
///
/// Even type inference cannot extract the private graph field:
///
/// ```compile_fail
/// use frankensearch_index::native_hnsw::ValidatedNativeHnsw;
///
/// fn detach(bound: ValidatedNativeHnsw) {
///     let _ = bound.graph;
/// }
/// ```
///
/// Search likewise accepts no caller-provided store:
///
/// ```compile_fail
/// use frankensearch_index::ValidatedFsviBytes;
/// use frankensearch_index::native_hnsw::ValidatedNativeHnsw;
///
/// fn substitute_store(
///     bound: &ValidatedNativeHnsw,
///     other: &ValidatedFsviBytes,
///     query: &[f32],
/// ) {
///     let _ = bound.search(query, 10, None, other);
/// }
/// ```
///
/// A hit borrows the retained owner through the graph handle and therefore
/// cannot escape after the handle is dropped:
///
/// ```compile_fail
/// use frankensearch_index::native_hnsw::ValidatedNativeHnsw;
///
/// fn detach_hit(bound: ValidatedNativeHnsw, query: &[f32]) {
///     let hit = bound.search(query, 1, None).unwrap().remove(0);
///     drop(bound);
///     let _ = hit.doc_id();
/// }
/// ```
#[derive(Debug)]
pub struct ValidatedNativeHnsw {
    pub(super) owner: Arc<ValidatedFsviBytes>,
    pub(super) binding: NativeHnswGenerationBindingV2,
    pub(super) graph: NativeHnsw,
}

/// One native-HNSW result resolved through the same admitted FSVI owner used
/// for graph construction or load.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValidatedNativeHnswHit<'owner> {
    pub(super) physical_row: u32,
    pub(super) distance: f32,
    pub(super) doc_id: &'owner str,
    pub(super) flags: crate::FsviRecordFlags,
}

impl ValidatedNativeHnswHit<'_> {
    /// Exact physical FSVI row indexed by the graph.
    #[must_use]
    pub const fn physical_row(&self) -> u32 {
        self.physical_row
    }

    /// Native HNSW distance; smaller values rank nearer.
    #[must_use]
    pub const fn distance(&self) -> f32 {
        self.distance
    }

    /// Document identifier resolved from the bound admitted owner.
    #[must_use]
    pub const fn doc_id(&self) -> &str {
        self.doc_id
    }

    /// Validated LIVE/TOMBSTONE state from the same physical owner row.
    #[must_use]
    pub const fn flags(&self) -> crate::FsviRecordFlags {
        self.flags
    }
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 200,
            ef_search: 100,
        }
    }
}

impl HnswParams {
    /// Validate the parameters.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when a parameter would make
    /// the graph degenerate: a zero degree disconnects the graph, and a
    /// beam narrower than one candidate cannot make progress.
    pub fn validate(&self) -> SearchResult<()> {
        let invalid = |field: &str, value: usize, reason: &str| SearchError::InvalidConfig {
            field: field.to_owned(),
            value: value.to_string(),
            reason: reason.to_owned(),
        };
        if self.m == 0 {
            return Err(invalid("m", self.m, "must be >= 1"));
        }
        if self.m0 < 2 {
            return Err(invalid(
                "m0",
                self.m0,
                "must be >= 2 to preserve connected layer-zero routing",
            ));
        }
        if self.ef_construction == 0 {
            return Err(invalid(
                "ef_construction",
                self.ef_construction,
                "must be >= 1",
            ));
        }
        if self.ef_search == 0 {
            return Err(invalid("ef_search", self.ef_search, "must be >= 1"));
        }
        Ok(())
    }

    /// Neighbour budget for `layer`.
    #[must_use]
    pub const fn degree_at(&self, layer: usize) -> usize {
        if layer == 0 { self.m0 } else { self.m }
    }
}
