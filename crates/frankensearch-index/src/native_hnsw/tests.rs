mod graph_engine;
mod owner_binding;
mod receipt_validation;

use std::cell::Cell;
use std::path::Path;
use std::sync::Arc;

use crate::{FsviRecordFlags, FsviV2IdentityBinding, ValidatedFsviBytes, VectorIndex};
use ::tempfile as raw_tempfile;
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, QuantizationFormat,
};

use super::codec::NativeHnsw;
use super::graph::{LevelSampler, VectorDistance};
use super::io_helpers::{get_u32, get_u64, put_u32};
use super::receipt::NativeHnswGenerationReceiptV2;
use super::types::HnswParams;
use super::{NATIVE_HNSW_HEADER_CRC_OFFSET, NATIVE_HNSW_HEADER_LEN};

/// Test-only facade that preserves the production no-symlink-ancestor
/// contract on hosts whose default temporary path contains an alias.
///
/// macOS commonly returns `/var/folders/...` even though `/var` is a
/// symlink to `/private/var`. Canonicalising the fixture root keeps every
/// test artifact under the same directory while ensuring failures exercise
/// the artifact under test rather than that platform-provided alias.
pub(super) mod tempfile {
    use std::io;
    use std::path::{Path, PathBuf};

    use super::raw_tempfile;

    pub(super) struct CanonicalTempDir {
        _directory: raw_tempfile::TempDir,
        canonical_path: PathBuf,
    }

    impl CanonicalTempDir {
        pub(super) fn path(&self) -> &Path {
            &self.canonical_path
        }
    }

    pub(super) fn tempdir() -> io::Result<CanonicalTempDir> {
        let directory = raw_tempfile::tempdir()?;
        let canonical_path = std::fs::canonicalize(directory.path())?;
        Ok(CanonicalTempDir {
            _directory: directory,
            canonical_path,
        })
    }
}

/// An in-memory store of unit-normalised vectors using cosine distance
/// (`1 - dot`), which is the metric the two-tier index uses.
pub(super) struct TestStore {
    vectors: Vec<Vec<f32>>,
}

impl TestStore {
    pub(super) fn new(vectors: Vec<Vec<f32>>) -> Self {
        let vectors = vectors
            .into_iter()
            .map(|vector| {
                let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
                if norm <= f32::EPSILON {
                    vector
                } else {
                    vector.into_iter().map(|v| v / norm).collect()
                }
            })
            .collect();
        Self { vectors }
    }

    /// Deterministic pseudo-random corpus: seeded LCG, so a failure
    /// reproduces from the test name alone while the data still behaves
    /// like real embeddings.
    ///
    /// Keep this recall corpus varied: a modular lattice produces many
    /// equal distances, making a narrow beam's tied top-k ambiguous.
    /// Separate repeated-vector and lattice fixtures below enforce
    /// connectivity and exact full-width retrieval for those valid inputs.
    pub(super) fn synthetic(count: usize, dim: usize) -> Self {
        let vectors = (0..count)
            .map(|i| {
                let mut state = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0xdead_beef;
                (0..dim)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                    })
                    .collect()
            })
            .collect();
        Self::new(vectors)
    }

    /// A deterministic pseudo-random query vector.
    pub(super) fn query(seed: usize, dim: usize) -> Vec<f32> {
        let mut state = (seed as u64).wrapping_mul(0x2545_f491_4f6c_dd1d) ^ 0x00ab_cdef;
        (0..dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect()
    }

    pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Ground truth: exact nearest neighbours by full scan.
    pub(super) fn exact_top_k(&self, query: &[f32], k: usize) -> Vec<u32> {
        let mut scored: Vec<(f32, u32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let id = u32::try_from(i).expect("test corpus length fits u32");
                (1.0 - Self::dot(v, query), id)
            })
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        scored.into_iter().take(k).map(|(_, id)| id).collect()
    }
}

impl VectorDistance for TestStore {
    fn distance_to_query(&self, id: u32, query: &[f32]) -> SearchResult<f32> {
        let vector = self
            .vectors
            .get(id as usize)
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "row".to_owned(),
                value: id.to_string(),
                reason: "row index out of range".to_owned(),
            })?;
        Ok(1.0 - Self::dot(vector, query))
    }

    fn distance_between(&self, a: u32, b: u32) -> SearchResult<f32> {
        let left = self
            .vectors
            .get(a as usize)
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "row".to_owned(),
                value: a.to_string(),
                reason: "row index out of range".to_owned(),
            })?;
        self.distance_to_query(b, left)
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }
}

pub(super) struct FailingStore<'a> {
    inner: &'a TestStore,
    fail_between_at: Option<usize>,
    between_calls: Cell<usize>,
}

impl<'a> FailingStore<'a> {
    pub(super) fn new(inner: &'a TestStore, fail_between_at: Option<usize>) -> Self {
        Self {
            inner,
            fail_between_at,
            between_calls: Cell::new(0),
        }
    }

    pub(super) fn between_calls(&self) -> usize {
        self.between_calls.get()
    }
}

impl VectorDistance for FailingStore<'_> {
    fn distance_to_query(&self, id: u32, query: &[f32]) -> SearchResult<f32> {
        self.inner.distance_to_query(id, query)
    }

    fn distance_between(&self, a: u32, b: u32) -> SearchResult<f32> {
        let ordinal = self.between_calls.get() + 1;
        self.between_calls.set(ordinal);
        if self.fail_between_at == Some(ordinal) {
            return Err(SearchError::InvalidConfig {
                field: "fault_injection".to_owned(),
                value: ordinal.to_string(),
                reason: "injected distance_between failure".to_owned(),
            });
        }
        self.inner.distance_between(a, b)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

pub(super) fn topology_snapshot(
    graph: &NativeHnsw,
) -> (Vec<Vec<Vec<u32>>>, Option<u32>, usize, usize) {
    (
        graph
            .adjacency
            .iter()
            .map(|point| point.layers.clone())
            .collect(),
        graph.entry,
        graph.max_level,
        graph.len(),
    )
}

pub(super) fn reseal_persisted_checksums(bytes: &mut [u8]) {
    let payload_crc32 = crc32fast::hash(&bytes[NATIVE_HNSW_HEADER_LEN..]);
    put_u32(bytes, 88, payload_crc32);
    let header_crc32 = crc32fast::hash(&bytes[..NATIVE_HNSW_HEADER_CRC_OFFSET]);
    put_u32(bytes, NATIVE_HNSW_HEADER_CRC_OFFSET, header_crc32);
}

pub(super) fn first_persisted_neighbour(bytes: &[u8]) -> Option<(usize, u32)> {
    let point_count = usize::try_from(get_u64(bytes, 56)).expect("test point count");
    let mut offset = NATIVE_HNSW_HEADER_LEN;
    for point in 0..point_count {
        let layer_count =
            usize::try_from(get_u32(bytes, offset)).expect("test layer count fits usize");
        offset += 4;
        for _ in 0..layer_count {
            let neighbour_count =
                usize::try_from(get_u32(bytes, offset)).expect("test neighbour count fits usize");
            offset += 4;
            if neighbour_count > 0 {
                return Some((
                    offset,
                    u32::try_from(point).expect("test point id fits u32"),
                ));
            }
            offset += neighbour_count * 4;
        }
    }
    None
}

pub(super) fn params() -> HnswParams {
    HnswParams {
        m: 8,
        m0: 16,
        ef_construction: 64,
        ef_search: 64,
    }
}

pub(super) fn fsvi_v2_binding(
    sequence: u64,
    nonce_byte: u8,
    model_id: &str,
    dimension: u32,
) -> FsviV2IdentityBinding {
    fsvi_v2_binding_with_quantization(
        sequence,
        nonce_byte,
        model_id,
        dimension,
        QuantizationFormat::F16,
    )
}

pub(super) fn fsvi_v2_binding_with_quantization(
    sequence: u64,
    nonce_byte: u8,
    model_id: &str,
    dimension: u32,
    quantization: QuantizationFormat,
) -> FsviV2IdentityBinding {
    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(model_id, dimension);
    identity.storage.format = "fsvi-v2".to_owned();
    identity.storage.quantization = quantization;
    identity.storage.endianness = "little-endian".to_owned();
    let generation = ArtifactGenerationIdentityV1::new(sequence, [nonce_byte; 16])
        .expect("valid test generation");
    FsviV2IdentityBinding::new(
        generation,
        identity.freeze().expect("valid frozen FSVI v2 identity"),
    )
    .expect("valid FSVI v2 identity binding")
}

pub(super) fn fsvi_rows(count: usize, dimension: usize) -> Vec<(String, Vec<f32>)> {
    (0..count)
        .map(|row| {
            let mut vector = vec![0.0; dimension];
            vector[row % dimension] = 1.0;
            (format!("doc-{row:04}"), vector)
        })
        .collect()
}

pub(super) fn admit_owned_fsvi_fixture(
    path: &Path,
    binding: &FsviV2IdentityBinding,
) -> Arc<ValidatedFsviBytes> {
    let bytes = std::fs::read(path).expect("read completed FSVI v2 owner fixture");
    Arc::new(
        ValidatedFsviBytes::from_arc(Arc::<[u8]>::from(bytes), binding)
            .expect("admit owned FSVI v2 fixture bytes"),
    )
}

pub(super) fn admitted_fsvi_owner(
    directory: &Path,
    basename: &str,
    binding: &FsviV2IdentityBinding,
    rows: &[(String, Vec<f32>)],
) -> Arc<ValidatedFsviBytes> {
    let path = directory.join(basename);
    let mut writer =
        VectorIndex::create_v2(&path, binding.clone()).expect("create FSVI v2 owner fixture");
    for (doc_id, vector) in rows {
        writer
            .write_record(doc_id, vector)
            .expect("write FSVI v2 owner row");
    }
    // ubs:ignore -- finalizes a deterministic VectorIndex test fixture; no token or randomness.
    writer.finish().expect("finish FSVI v2 owner fixture");
    admit_owned_fsvi_fixture(&path, binding)
}

/// Offset of the bd-miio8 cross-tier publication nonce in an FSVI v2 header.
pub(super) const FSVI_V2_PUBLICATION_NONCE_OFFSET: usize = 14;

pub(super) fn fsvi_v2_header_size(bytes: &[u8]) -> usize {
    usize::try_from(u32::from_le_bytes(
        bytes[6..10].try_into().expect("v2 header-size field"),
    ))
    .expect("v2 header size fits usize")
}

pub(super) fn fsvi_v2_publication_nonce(bytes: &[u8]) -> u16 {
    u16::from_le_bytes(
        bytes[FSVI_V2_PUBLICATION_NONCE_OFFSET..FSVI_V2_PUBLICATION_NONCE_OFFSET + 2]
            .try_into()
            .expect("v2 publication-nonce field"),
    )
}

pub(super) fn admitted_fsvi_owner_with_publication_nonce(
    directory: &Path,
    basename: &str,
    binding: &FsviV2IdentityBinding,
    rows: &[(String, Vec<f32>)],
    publication_nonce: u16,
) -> Arc<ValidatedFsviBytes> {
    let path = directory.join(basename);
    let mut writer = VectorIndex::create_v2(&path, binding.clone())
        .expect("create FSVI v2 owner fixture")
        .with_publication_nonce(publication_nonce);
    for (doc_id, vector) in rows {
        writer
            .write_record(doc_id, vector)
            .expect("write FSVI v2 owner row");
    }
    // The publication nonce is public identity metadata, not a token.
    // ubs:ignore -- finalizes a deterministic VectorIndex test fixture; no randomness.
    writer.finish().expect("finish FSVI v2 owner fixture");
    admit_owned_fsvi_fixture(&path, binding)
}

pub(super) fn admitted_fsvi_owner_with_flags(
    directory: &Path,
    basename: &str,
    binding: &FsviV2IdentityBinding,
    rows: &[(String, Vec<f32>, FsviRecordFlags)],
) -> Arc<ValidatedFsviBytes> {
    let path = directory.join(basename);
    let mut writer =
        VectorIndex::create_v2(&path, binding.clone()).expect("create FSVI v2 owner fixture");
    for (doc_id, vector, flags) in rows {
        if *flags == FsviRecordFlags::TOMBSTONE {
            writer
                .write_tombstone_record(doc_id, vector)
                .expect("write FSVI v2 tombstone row");
        } else {
            assert_eq!(*flags, FsviRecordFlags::LIVE);
            writer
                .write_record(doc_id, vector)
                .expect("write FSVI v2 live row");
        }
    }
    writer.finish().expect("finish FSVI v2 owner fixture");
    admit_owned_fsvi_fixture(&path, binding)
}

pub(super) fn generation_owner(
    sequence: u64,
    nonce_byte: u8,
    model_id: &str,
    dimension: u32,
    physical_row_count: usize,
) -> Arc<ValidatedFsviBytes> {
    let directory = tempfile::tempdir().expect("temporary FSVI owner directory");
    let binding = fsvi_v2_binding(sequence, nonce_byte, model_id, dimension);
    let rows = fsvi_rows(
        physical_row_count,
        usize::try_from(dimension).expect("test dimension fits usize"),
    );
    admitted_fsvi_owner(directory.path(), "owner.fsvi", &binding, &rows)
}

pub(super) fn write_resealed_receipt(path: &Path, receipt: &mut NativeHnswGenerationReceiptV2) {
    receipt.seal().expect("seal mutated test receipt");
    std::fs::write(
        path,
        receipt
            .encode_unchecked()
            .expect("encode mutated test receipt"),
    )
    .expect("write mutated test receipt");
}

pub(super) fn sampled_entry_for_count(count: usize, seed: u64) -> Option<u32> {
    // ubs:ignore -- the public fixture cardinality is not secret material.
    if count == 0 {
        return None;
    }
    let sampler = LevelSampler::new(params().m, seed);
    let mut entry = 0_u32;
    let mut maximum = sampler.level_for(entry);
    for physical_index in 1..count {
        let id = u32::try_from(physical_index).expect("test owner count fits u32");
        let level = sampler.level_for(id);
        if level > maximum {
            entry = id;
            maximum = level;
        }
    }
    Some(entry)
}

pub(super) fn seed_with_tombstoned_entry(owner: &ValidatedFsviBytes) -> u64 {
    (0..10_000_u64)
        .find(|&seed| {
            let entry = sampled_entry_for_count(owner.record_count(), seed)
                .expect("nonempty test owner has an entry");
            owner
                .row(usize::try_from(entry).expect("entry fits usize"))
                .expect("entry row")
                .flags()
                .is_tombstone()
        })
        .expect("find deterministic seed whose entry point is tombstoned")
}
