use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use frankensearch_core::error::{SearchError, SearchResult};

use super::graph::{LevelSampler, VectorDistance, VisitedSet};
#[cfg(unix)]
use super::io_helpers::sync_parent_directory;
use super::io_helpers::{
    append_canonical_bytes, get_u32, get_u64, native_hnsw_corrupted,
    native_hnsw_receipt_config_error, open_regular_file, persisted_usize, put_u32, put_u64,
    reject_non_regular_destination, sha256_hex, usize_to_u64, write_payload_u32,
};
use super::types::{GraphDefect, HnswParams, MAX_LEVEL, NativeHnswFileMetadata};
use super::{
    NATIVE_HNSW_FORMAT_VERSION, NATIVE_HNSW_HEADER_CRC_OFFSET, NATIVE_HNSW_HEADER_LEN,
    NATIVE_HNSW_HEADER_LEN_U64, NATIVE_HNSW_MAGIC, NATIVE_HNSW_NO_ENTRY,
};

/// Per-point adjacency: one neighbour list per layer the point occupies.
#[derive(Debug, Clone, Default)]
pub(super) struct Adjacency {
    /// `layers[l]` holds the neighbours at layer `l`. Length is
    /// `level + 1`, so indexing beyond the point's level is impossible by
    /// construction rather than by check.
    pub(super) layers: Vec<Vec<u32>>,
}

impl Adjacency {
    pub(super) fn with_level(level: usize) -> Self {
        Self {
            layers: vec![Vec::new(); level + 1],
        }
    }

    /// The point's sampled level.
    pub(super) fn level(&self) -> usize {
        self.layers.len().saturating_sub(1)
    }

    /// Whether this point participates in `layer`.
    ///
    /// bd-u3wt defect (4): participation is `level >= layer`, never
    /// `level == layer`.
    pub(super) fn participates_in(&self, layer: usize) -> bool {
        layer < self.layers.len()
    }

    pub(super) fn neighbours(&self, layer: usize) -> &[u32] {
        self.layers.get(layer).map_or(&[], Vec::as_slice)
    }
}

/// Original neighbour lists touched by one insertion.
///
/// Insertion mutates only a bounded neighbourhood, so journaling those lists
/// avoids cloning the complete graph while still making every returned error
/// failure-atomic. Lists belonging to the newly appended point are omitted:
/// rollback removes that point wholesale.
#[derive(Debug)]
pub(super) struct MutationJournal {
    original_point_count: usize,
    original_lists: BTreeMap<(usize, usize), Vec<u32>>,
}

impl MutationJournal {
    pub(super) fn new(original_point_count: usize) -> Self {
        Self {
            original_point_count,
            original_lists: BTreeMap::new(),
        }
    }

    pub(super) fn record(&mut self, adjacency: &[Adjacency], id: u32, layer: usize) {
        let point_index = id as usize;
        if point_index >= self.original_point_count
            || self.original_lists.contains_key(&(point_index, layer))
        {
            return;
        }
        if let Some(original) = adjacency
            .get(point_index)
            .and_then(|point| point.layers.get(layer))
        {
            self.original_lists
                .insert((point_index, layer), original.clone());
        }
    }

    pub(super) fn rollback(self, adjacency: &mut [Adjacency]) {
        for ((point_index, layer), original) in self.original_lists {
            if let Some(neighbours) = adjacency
                .get_mut(point_index)
                .and_then(|point| point.layers.get_mut(layer))
            {
                *neighbours = original;
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct NativeHnswHeader {
    params: HnswParams,
    seed: u64,
    point_count: usize,
    entry: Option<u32>,
    max_level: usize,
    payload_len: u64,
    payload_crc32: u32,
    header_crc32: u32,
}

impl NativeHnswHeader {
    fn encode(self) -> SearchResult<[u8; NATIVE_HNSW_HEADER_LEN]> {
        let mut bytes = [0_u8; NATIVE_HNSW_HEADER_LEN];
        bytes[..NATIVE_HNSW_MAGIC.len()].copy_from_slice(&NATIVE_HNSW_MAGIC);
        put_u32(&mut bytes, 8, NATIVE_HNSW_FORMAT_VERSION);
        put_u32(
            &mut bytes,
            12,
            u32::try_from(NATIVE_HNSW_HEADER_LEN).map_err(|_| SearchError::InvalidConfig {
                field: "native_hnsw_header_len".to_owned(),
                value: NATIVE_HNSW_HEADER_LEN.to_string(),
                reason: "header length must fit in u32".to_owned(),
            })?,
        );
        put_u64(&mut bytes, 16, usize_to_u64(self.params.m, "m")?);
        put_u64(&mut bytes, 24, usize_to_u64(self.params.m0, "m0")?);
        put_u64(
            &mut bytes,
            32,
            usize_to_u64(self.params.ef_construction, "ef_construction")?,
        );
        put_u64(
            &mut bytes,
            40,
            usize_to_u64(self.params.ef_search, "ef_search")?,
        );
        put_u64(&mut bytes, 48, self.seed);
        put_u64(
            &mut bytes,
            56,
            usize_to_u64(self.point_count, "point_count")?,
        );
        put_u64(
            &mut bytes,
            64,
            self.entry.map_or(NATIVE_HNSW_NO_ENTRY, u64::from),
        );
        put_u64(&mut bytes, 72, usize_to_u64(self.max_level, "max_level")?);
        put_u64(&mut bytes, 80, self.payload_len);
        put_u32(&mut bytes, 88, self.payload_crc32);
        let header_crc32 = crc32fast::hash(&bytes[..NATIVE_HNSW_HEADER_CRC_OFFSET]);
        put_u32(&mut bytes, NATIVE_HNSW_HEADER_CRC_OFFSET, header_crc32);
        Ok(bytes)
    }

    // ubs:ignore — this parses the owned graph header, not a JWT or authentication token.
    fn decode(
        path: &Path,
        bytes: &[u8; NATIVE_HNSW_HEADER_LEN],
        file_len: u64,
    ) -> SearchResult<Self> {
        if bytes[..NATIVE_HNSW_MAGIC.len()] != NATIVE_HNSW_MAGIC {
            return Err(native_hnsw_corrupted(path, "bad native HNSW magic"));
        }
        let version = get_u32(bytes, 8);
        if version != NATIVE_HNSW_FORMAT_VERSION {
            return Err(native_hnsw_corrupted(
                path,
                format!(
                    "unsupported native HNSW format version {version}; expected \
                     {NATIVE_HNSW_FORMAT_VERSION}"
                ),
            ));
        }
        let header_len = usize::try_from(get_u32(bytes, 12)).map_err(|_| {
            native_hnsw_corrupted(path, "native HNSW header length does not fit usize")
        })?;
        if header_len != NATIVE_HNSW_HEADER_LEN {
            return Err(native_hnsw_corrupted(
                path,
                format!(
                    "native HNSW header length {header_len} does not equal \
                     {NATIVE_HNSW_HEADER_LEN}"
                ),
            ));
        }
        let header_crc32 = get_u32(bytes, NATIVE_HNSW_HEADER_CRC_OFFSET);
        let expected_header_crc32 = crc32fast::hash(&bytes[..NATIVE_HNSW_HEADER_CRC_OFFSET]);
        if header_crc32 != expected_header_crc32 {
            return Err(native_hnsw_corrupted(
                path,
                "native HNSW header CRC mismatch",
            ));
        }

        let params = HnswParams {
            m: persisted_usize(path, get_u64(bytes, 16), "m")?,
            m0: persisted_usize(path, get_u64(bytes, 24), "m0")?,
            ef_construction: persisted_usize(path, get_u64(bytes, 32), "ef_construction")?,
            ef_search: persisted_usize(path, get_u64(bytes, 40), "ef_search")?,
        };
        params.validate().map_err(|error| {
            native_hnsw_corrupted(path, format!("invalid persisted HNSW parameters: {error}"))
        })?;

        let point_count_wire = get_u64(bytes, 56);
        let u32_id_space = u64::from(u32::MAX) + 1;
        if point_count_wire > u32_id_space {
            return Err(native_hnsw_corrupted(
                path,
                "native HNSW point count exceeds u32 id space",
            ));
        }
        let point_count = persisted_usize(path, point_count_wire, "point_count")?;
        let entry_wire = get_u64(bytes, 64);
        let entry =
            if entry_wire == NATIVE_HNSW_NO_ENTRY {
                None
            } else {
                Some(u32::try_from(entry_wire).map_err(|_| {
                    native_hnsw_corrupted(path, "native HNSW entry point exceeds u32")
                })?)
            };
        let max_level = persisted_usize(path, get_u64(bytes, 72), "max_level")?;
        if max_level >= MAX_LEVEL {
            return Err(native_hnsw_corrupted(
                path,
                format!("native HNSW maximum level {max_level} exceeds format cap"),
            ));
        }
        let payload_len = get_u64(bytes, 80);
        let expected_file_len = NATIVE_HNSW_HEADER_LEN_U64
            .checked_add(payload_len)
            .ok_or_else(|| native_hnsw_corrupted(path, "native HNSW file length overflow"))?;
        if file_len != expected_file_len {
            return Err(native_hnsw_corrupted(
                path,
                format!(
                    "native HNSW file length {file_len} does not match header-declared \
                     length {expected_file_len}"
                ),
            ));
        }
        let minimum_payload = usize_to_u64(point_count, "point_count")?
            .checked_mul(8)
            .ok_or_else(|| native_hnsw_corrupted(path, "minimum payload length overflow"))?;
        if payload_len < minimum_payload {
            return Err(native_hnsw_corrupted(
                path,
                "native HNSW payload is too short for its point count",
            ));
        }

        Ok(Self {
            params,
            seed: get_u64(bytes, 48),
            point_count,
            entry,
            max_level,
            payload_len,
            payload_crc32: get_u32(bytes, 88),
            header_crc32,
        })
    }

    fn metadata(self) -> SearchResult<NativeHnswFileMetadata> {
        Ok(NativeHnswFileMetadata {
            format_version: NATIVE_HNSW_FORMAT_VERSION,
            byte_len: NATIVE_HNSW_HEADER_LEN_U64
                .checked_add(self.payload_len)
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "native_hnsw_file_len".to_owned(),
                    value: self.payload_len.to_string(),
                    reason: "file length overflow".to_owned(),
                })?,
            point_count: usize_to_u64(self.point_count, "point_count")?,
            payload_crc32: self.payload_crc32,
            header_crc32: self.header_crc32,
        })
    }
}

struct NativeHnswPayloadReader<'a> {
    file: &'a mut File,
    remaining: u64,
    hasher: crc32fast::Hasher,
}

impl<'a> NativeHnswPayloadReader<'a> {
    fn new(file: &'a mut File, payload_len: u64) -> Self {
        Self {
            file,
            remaining: payload_len,
            hasher: crc32fast::Hasher::new(),
        }
    }

    fn read_u32(&mut self, path: &Path, field: &str) -> SearchResult<u32> {
        let field_len = 4_u64;
        if self.remaining < field_len {
            return Err(native_hnsw_corrupted(
                path,
                format!("native HNSW payload ended while reading {field}"),
            ));
        }
        let mut bytes = [0_u8; 4];
        self.file.read_exact(&mut bytes).map_err(|error| {
            native_hnsw_corrupted(path, format!("could not read native HNSW {field}: {error}"))
        })?;
        self.hasher.update(&bytes);
        self.remaining -= field_len;
        Ok(u32::from_le_bytes(bytes))
    }

    fn finish(self, path: &Path, expected_crc32: u32) -> SearchResult<()> {
        if self.remaining != 0 {
            return Err(native_hnsw_corrupted(
                path,
                format!(
                    "native HNSW payload has {} unparsed trailing bytes",
                    self.remaining
                ),
            ));
        }
        let actual_crc32 = self.hasher.finalize();
        if actual_crc32 != expected_crc32 {
            return Err(native_hnsw_corrupted(
                path,
                "native HNSW payload CRC mismatch",
            ));
        }
        Ok(())
    }
}

/// A navigable small-world graph over rows of a vector store.
#[derive(Debug, Clone)]
pub(super) struct NativeHnsw {
    pub(super) params: HnswParams,
    pub(super) sampler: LevelSampler,
    pub(super) adjacency: Vec<Adjacency>,
    /// Entry point: the id of a point at the current maximum level.
    pub(super) entry: Option<u32>,
    pub(super) max_level: usize,
    /// Visited-set buffer reused across every insert, so construction does
    /// not re-zero an n-sized array per layer per point.
    pub(super) scratch: VisitedSet,
}

impl NativeHnsw {
    /// Create an empty graph.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for degenerate parameters.
    pub(super) fn new(params: HnswParams, seed: u64) -> SearchResult<Self> {
        params.validate()?;
        Ok(Self {
            sampler: LevelSampler::new(params.m, seed),
            params,
            adjacency: Vec::new(),
            entry: None,
            max_level: 0,
            scratch: VisitedSet::default(),
        })
    }

    /// Number of indexed points.
    #[must_use]
    pub(super) fn len(&self) -> usize {
        self.adjacency.len()
    }

    /// Whether the graph indexes no points.
    #[must_use]
    pub(super) fn is_empty(&self) -> bool {
        self.adjacency.is_empty()
    }

    /// The entry point id, if any point has been inserted.
    #[must_use]
    pub(super) const fn entry_point(&self) -> Option<u32> {
        self.entry
    }

    /// The current maximum level.
    #[must_use]
    pub(super) const fn max_level(&self) -> usize {
        self.max_level
    }

    /// Construction and search parameters persisted with this graph.
    #[must_use]
    pub(super) const fn params(&self) -> HnswParams {
        self.params
    }

    /// Deterministic level-sampling seed persisted with this graph.
    #[must_use]
    pub(super) const fn seed(&self) -> u64 {
        self.sampler.seed
    }

    /// SHA-256 of the graph's semantic topology and construction identity.
    ///
    /// File checksums and byte offsets are intentionally excluded. Point and
    /// neighbour order remain included because both are search-visible state
    /// and the owned builder is deterministic for a seed.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::IndexCorrupted`] when the graph is structurally
    /// invalid and [`SearchError::InvalidConfig`] when a count cannot be
    /// represented canonically.
    pub(super) fn topology_sha256(&self) -> SearchResult<String> {
        self.verify()
            .map_err(|defect| native_hnsw_corrupted(Path::new("<memory>"), defect.to_string()))?;
        let mut bytes = Vec::new();
        append_canonical_bytes(&mut bytes, b"frankensearch.native-hnsw.topology.v1")?;
        bytes.extend_from_slice(&NATIVE_HNSW_FORMAT_VERSION.to_be_bytes());
        for value in [
            usize_to_u64(self.params.m, "topology.params.m")?,
            usize_to_u64(self.params.m0, "topology.params.m0")?,
            usize_to_u64(
                self.params.ef_construction,
                "topology.params.ef_construction",
            )?,
            usize_to_u64(self.params.ef_search, "topology.params.ef_search")?,
            self.sampler.seed,
            usize_to_u64(self.adjacency.len(), "topology.point_count")?,
            self.entry.map_or(NATIVE_HNSW_NO_ENTRY, u64::from),
            usize_to_u64(self.max_level, "topology.max_level")?,
        ] {
            bytes.extend_from_slice(&value.to_be_bytes());
        }
        for point in &self.adjacency {
            bytes.extend_from_slice(
                &u32::try_from(point.layers.len())
                    .map_err(|_| {
                        native_hnsw_receipt_config_error(
                            "topology.layer_count",
                            &point.layers.len().to_string(),
                            "must fit the canonical u32 field",
                        )
                    })?
                    .to_be_bytes(),
            );
            for neighbours in &point.layers {
                bytes.extend_from_slice(
                    &u32::try_from(neighbours.len())
                        .map_err(|_| {
                            native_hnsw_receipt_config_error(
                                "topology.neighbour_count",
                                &neighbours.len().to_string(),
                                "must fit the canonical u32 field",
                            )
                        })?
                        .to_be_bytes(),
                );
                for neighbour in neighbours {
                    bytes.extend_from_slice(&neighbour.to_be_bytes());
                }
            }
        }
        Ok(sha256_hex(&bytes))
    }

    /// Persist this graph as an owned, versioned adjacency artifact.
    ///
    /// The graph is structurally attested before the destination is touched.
    /// Publication writes and syncs a temporary file, atomically renames it
    /// over `path`, then syncs the parent directory. Existing symlink and
    /// special-file targets are rejected rather than followed.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::IndexCorrupted`] when the in-memory graph fails
    /// structural attestation, [`SearchError::InvalidConfig`] when a value
    /// cannot be represented by the format, and [`SearchError::Io`] for
    /// filesystem failures.
    pub(super) fn save(&self, path: &Path) -> SearchResult<NativeHnswFileMetadata> {
        self.verify()
            .map_err(|defect| native_hnsw_corrupted(path, defect.to_string()))?;

        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent).map_err(SearchError::Io)?;
        reject_non_regular_destination(path)?;

        let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(SearchError::Io)?;
        temporary
            .as_file_mut()
            .write_all(&[0_u8; NATIVE_HNSW_HEADER_LEN])
            .map_err(SearchError::Io)?;

        let mut payload_hasher = crc32fast::Hasher::new();
        let mut payload_len = 0_u64;
        for point in &self.adjacency {
            write_payload_u32(
                temporary.as_file_mut(),
                &mut payload_hasher,
                &mut payload_len,
                u32::try_from(point.layers.len()).map_err(|_| SearchError::InvalidConfig {
                    field: "native_hnsw_layer_count".to_owned(),
                    value: point.layers.len().to_string(),
                    reason: "layer count must fit in u32".to_owned(),
                })?,
            )?;
            for neighbours in &point.layers {
                write_payload_u32(
                    temporary.as_file_mut(),
                    &mut payload_hasher,
                    &mut payload_len,
                    u32::try_from(neighbours.len()).map_err(|_| SearchError::InvalidConfig {
                        field: "native_hnsw_neighbour_count".to_owned(),
                        value: neighbours.len().to_string(),
                        reason: "neighbour count must fit in u32".to_owned(),
                    })?,
                )?;
                for &neighbour in neighbours {
                    write_payload_u32(
                        temporary.as_file_mut(),
                        &mut payload_hasher,
                        &mut payload_len,
                        neighbour,
                    )?;
                }
            }
        }

        let payload_crc32 = payload_hasher.finalize();
        let header = NativeHnswHeader {
            params: self.params,
            seed: self.sampler.seed,
            point_count: self.adjacency.len(),
            entry: self.entry,
            max_level: self.max_level,
            payload_len,
            payload_crc32,
            header_crc32: 0,
        };
        let encoded_header = header.encode()?;
        temporary
            .as_file_mut()
            .seek(SeekFrom::Start(0))
            .map_err(SearchError::Io)?;
        temporary
            .as_file_mut()
            .write_all(&encoded_header)
            .map_err(SearchError::Io)?;
        temporary.as_file().sync_all().map_err(SearchError::Io)?;

        let metadata = NativeHnswHeader {
            header_crc32: get_u32(&encoded_header, NATIVE_HNSW_HEADER_CRC_OFFSET),
            ..header
        }
        .metadata()?;
        temporary.persist(path).map_err(|error| {
            SearchError::Io(std::io::Error::new(
                error.error.kind(),
                format!(
                    "failed to atomically publish native HNSW graph '{}': {}",
                    path.display(),
                    error.error
                ),
            ))
        })?;
        #[cfg(unix)]
        sync_parent_directory(path)?;
        Ok(metadata)
    }

    /// Load and attest an owned graph artifact against its exact vector store.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::IndexNotFound`] when `path` is absent,
    /// [`SearchError::IndexCorrupted`] for malformed bytes or a graph/store
    /// mismatch, and [`SearchError::Io`] for other filesystem failures.
    #[cfg(test)]
    pub(super) fn load<D: VectorDistance>(path: &Path, store: &D) -> SearchResult<Self> {
        Self::load_with_metadata(path, store).map(|(graph, _)| graph)
    }

    /// Load a graph together with its checked format metadata.
    ///
    /// The parser reads bounded fields directly into owned adjacency lists;
    /// it never exposes persisted bytes as graph memory. The opened inode is
    /// compared with the path metadata to reject symlink and replacement
    /// races at the open boundary.
    ///
    /// # Errors
    ///
    /// Returns not-found, corruption, owner-cardinality, and filesystem
    /// errors from bounded parsing and structural attestation.
    pub(super) fn load_with_metadata<D: VectorDistance>(
        path: &Path,
        store: &D,
    ) -> SearchResult<(Self, NativeHnswFileMetadata)> {
        let mut file = open_regular_file(path)?;
        let file_len = file.metadata().map_err(SearchError::Io)?.len();
        if file_len < NATIVE_HNSW_HEADER_LEN_U64 {
            return Err(native_hnsw_corrupted(
                path,
                "native HNSW file is shorter than its fixed header",
            ));
        }

        let mut header_bytes = [0_u8; NATIVE_HNSW_HEADER_LEN];
        file.read_exact(&mut header_bytes).map_err(|error| {
            native_hnsw_corrupted(path, format!("could not read native HNSW header: {error}"))
        })?;
        // ubs:ignore — this decodes the owned graph header, not a JWT or authentication token.
        let header = NativeHnswHeader::decode(path, &header_bytes, file_len)?;
        let metadata = header.metadata()?;
        if header.point_count != store.len() {
            return Err(native_hnsw_corrupted(
                path,
                GraphDefect::StoreCardinalityMismatch {
                    graph_points: header.point_count,
                    store_rows: store.len(),
                }
                .to_string(),
            ));
        }

        let mut adjacency = Vec::new();
        adjacency
            .try_reserve_exact(header.point_count)
            .map_err(|error| {
                native_hnsw_corrupted(
                    path,
                    format!(
                        "could not allocate native HNSW point table for {} points: {error}",
                        header.point_count
                    ),
                )
            })?;
        let sampler = LevelSampler::new(header.params.m, header.seed);
        let mut payload = NativeHnswPayloadReader::new(&mut file, header.payload_len);
        for point_index in 0..header.point_count {
            let point_id = u32::try_from(point_index)
                .map_err(|_| native_hnsw_corrupted(path, "native HNSW point id exceeds u32"))?;
            let layer_count = persisted_usize(
                path,
                u64::from(payload.read_u32(path, "point layer count")?),
                "layer_count",
            )?;
            if !(1..=MAX_LEVEL).contains(&layer_count) {
                return Err(native_hnsw_corrupted(
                    path,
                    format!("point {point_id} has invalid native HNSW layer count {layer_count}"),
                ));
            }
            let expected_level = sampler.level_for(point_id);
            let actual_level = layer_count - 1;
            if actual_level != expected_level {
                return Err(native_hnsw_corrupted(
                    path,
                    GraphDefect::SampledLevelMismatch {
                        id: point_id,
                        expected: expected_level,
                        actual: actual_level,
                    }
                    .to_string(),
                ));
            }

            let mut layers = Vec::new();
            layers.try_reserve_exact(layer_count).map_err(|error| {
                native_hnsw_corrupted(
                    path,
                    format!(
                        "could not allocate {layer_count} native HNSW layers for point \
                         {point_id}: {error}"
                    ),
                )
            })?;
            for layer in 0..layer_count {
                let neighbour_count = persisted_usize(
                    path,
                    u64::from(payload.read_u32(path, "layer neighbour count")?),
                    "neighbour_count",
                )?;
                let budget = header.params.degree_at(layer);
                if neighbour_count > budget {
                    return Err(native_hnsw_corrupted(
                        path,
                        format!(
                            "point {point_id} layer {layer} holds {neighbour_count} neighbours, \
                             exceeding degree budget {budget}"
                        ),
                    ));
                }
                let neighbour_bytes = usize_to_u64(neighbour_count, "neighbour_count")?
                    .checked_mul(4)
                    .ok_or_else(|| {
                        native_hnsw_corrupted(path, "native HNSW neighbour-byte count overflow")
                    })?;
                if neighbour_bytes > payload.remaining {
                    return Err(native_hnsw_corrupted(
                        path,
                        format!(
                            "point {point_id} layer {layer} declares {neighbour_count} \
                             neighbours beyond the remaining payload"
                        ),
                    ));
                }
                let mut neighbours = Vec::new();
                neighbours
                    .try_reserve_exact(neighbour_count)
                    .map_err(|error| {
                        native_hnsw_corrupted(
                            path,
                            format!(
                                "could not allocate {neighbour_count} native HNSW neighbours for \
                                 point {point_id} layer {layer}: {error}"
                            ),
                        )
                    })?;
                for _ in 0..neighbour_count {
                    neighbours.push(payload.read_u32(path, "neighbour id")?);
                }
                layers.push(neighbours);
            }
            adjacency.push(Adjacency { layers });
        }
        payload.finish(path, header.payload_crc32)?;

        let graph = Self {
            params: header.params,
            sampler,
            adjacency,
            entry: header.entry,
            max_level: header.max_level,
            scratch: VisitedSet::default(),
        };
        graph
            .verify_for_store(store)
            .map_err(|defect| native_hnsw_corrupted(path, defect.to_string()))?;
        Ok((graph, metadata))
    }
}
