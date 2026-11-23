use std::path::Path;
#[cfg(target_os = "linux")]
use std::path::PathBuf;
use std::sync::OnceLock;

use frankensearch_core::generation::EmbeddingSpaceIdentityV1;
use frankensearch_core::{SearchError, SearchResult};
use half::f16;
#[cfg(target_os = "linux")]
use sha2::{Digest, Sha256};

use crate::{ValidatedFsviBytes, VectorIndex};

#[cfg(all(test, target_os = "linux"))]
use super::EXACT_RESIDUAL_SIDECAR_BUILD_COUNT;
#[cfg(target_os = "linux")]
use super::residual_io::{
    acquire_exact_residual_sidecar_publication, exact_residual_sidecar_header_matches_source,
    exact_residual_sidecar_stream_is_self_consistent,
    exact_residual_sidecar_stream_matches_derived, open_exact_residual_sidecar_parent,
    publish_exact_residual_sidecar, publish_exact_residual_sidecar_with_capability,
};
use super::residual_io::{residual_sidecar_error, try_filled_sidecar_vec};
#[cfg(target_os = "linux")]
use super::residual_sidecar::ExactResidualPublication;
use super::residual_sidecar::{
    ExactResidualLayout, ExactResidualSidecar, ResidualSourceBinding, fallible_doc_id_copy,
    finite_f64_add, finite_f64_mul, outward_f32_norm,
};
use super::{EXACT_RESIDUAL_BLOCK, EXACT_RESIDUAL_LANES, InMemoryVectorIndex};
#[cfg(target_os = "linux")]
use super::{
    EXACT_RESIDUAL_CACHE_ATTEMPTS, EXACT_RESIDUAL_CACHE_COMPARISON_BYTE_BUDGET,
    EXACT_RESIDUAL_CACHE_COMPARISON_CANDIDATE_LIMIT, EXACT_RESIDUAL_CACHE_DIRECTORY_ENTRY_LIMIT,
    EXACT_RESIDUAL_CACHE_NONCE, EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES, EXACT_RESIDUAL_SIDECAR_MAGIC,
    EXACT_RESIDUAL_SIDECAR_VERSION,
};

/// Quantize an f16 slab to int8 using one corpus-wide max-abs scale.
///
/// Symmetric int8: `q = round(x / max_abs * 127)`, clamped to `[-127, 127]`. A
/// single global scale keeps `Σ q_a·q_b` monotonic with the true dot for a fixed
/// query, so pass-1 ranking is preserved; the exact f16 rescore restores values.
#[allow(clippy::cast_possible_truncation)] // round()+clamp() bounds the f32->i8 cast
pub(super) fn quantize_i8_slab(vectors_f16: &[f16]) -> Vec<i8> {
    // Runtime-dispatched (AVX2+F16C when available); see `simd` for the kernel.
    crate::simd::quantize_f16_slab_to_i8(vectors_f16)
}

/// Quantize an f32 query to int8 using its own max-abs scale (the scale is a
/// per-query constant and does not affect ranking).
#[allow(clippy::cast_possible_truncation)] // round()+clamp() bounds the f32->i8 cast
pub(super) fn quantize_i8_query(query: &[f32]) -> Vec<i8> {
    let max_abs = query.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    if max_abs <= 0.0 {
        return vec![0_i8; query.len()];
    }
    let scale = 127.0 / max_abs;
    query
        .iter()
        .map(|&x| (x * scale).round().clamp(-127.0, 127.0) as i8)
        .collect()
}

/// Quantize one component to a signed 4-bit nibble (`[-7, 7]`, 4-bit two's
/// complement in the low 4 bits) given a scale.
#[allow(clippy::cast_possible_truncation)] // round()+clamp() bounds the cast
fn nibble_of(value: f32, scale: f32) -> u8 {
    let q = (value * scale).round().clamp(-7.0, 7.0) as i8;
    q.cast_unsigned() & 0x0F
}

/// Pack an f32 query into signed 4-bit nibbles, 2 dims/byte (low = even dim, high =
/// odd dim), using the query's own max-abs scale (a per-query constant that does not
/// change the dot ranking). Matches `pack_4bit_slab`'s layout for `dot_packed_4bit`.
pub(super) fn pack_4bit_query(query: &[f32]) -> Vec<u8> {
    let max_abs = query.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    let scale = if max_abs > 1e-9 { 7.0 / max_abs } else { 0.0 };
    let mut packed = vec![0_u8; query.len().div_ceil(2)];
    for (d, &x) in query.iter().enumerate() {
        let nib = nibble_of(x, scale);
        if d % 2 == 0 {
            packed[d / 2] |= nib;
        } else {
            packed[d / 2] |= nib << 4;
        }
    }
    packed
}

/// Pack a contiguous f16 vector slab (`count·dim`) into signed 4-bit nibbles
/// (`dim.div_ceil(2)` bytes/vector) with one corpus-wide max-abs scale (a constant
/// factor, so the dot ranking is preserved).
pub(super) fn pack_4bit_slab(vectors_f16: &[f16], dim: usize) -> Vec<u8> {
    // Runtime-dispatched (AVX2+F16C when available); see `simd` for the kernel.
    crate::simd::pack_f16_slab_to_4bit(vectors_f16, dim)
}

impl InMemoryVectorIndex {
    /// Build from pre-computed f32 vectors, quantizing to f16.
    ///
    /// The resulting index carries no embedding-space identity
    /// ([`Self::space_fingerprint_hex`] returns `None`): nothing about a bare
    /// `Vec<f32>` proves which space produced it, and fabricating an identity
    /// here would defeat the space verifier. Callers that know the producing
    /// space should use [`Self::from_vectors_with_identity`].
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if any vector's length does not
    /// match `dimension`.
    pub fn from_vectors(
        doc_ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        dimension: usize,
    ) -> SearchResult<Self> {
        if doc_ids.len() != vectors.len() {
            return Err(SearchError::InvalidConfig {
                field: "vectors".to_owned(),
                value: format!("doc_ids={}, vectors={}", doc_ids.len(), vectors.len()),
                reason: "doc_ids and vectors must have the same length".to_owned(),
            });
        }
        let count = doc_ids.len();
        let flat_capacity =
            count
                .checked_mul(dimension)
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "vectors".to_owned(),
                    value: format!("count={count}, dimension={dimension}"),
                    reason: "vector slab length overflow".to_owned(),
                })?;
        let mut flat = Vec::new();
        flat.try_reserve_exact(flat_capacity)
            .map_err(|_| SearchError::InvalidConfig {
                field: "vectors".to_owned(),
                value: format!("count={count}, dimension={dimension}"),
                reason: "vector slab allocation failed".to_owned(),
            })?;
        for (i, vec) in vectors.into_iter().enumerate() {
            if vec.len() != dimension {
                return Err(SearchError::DimensionMismatch {
                    expected: dimension,
                    found: vec.len(),
                });
            }
            // Validate finite values
            for val in &vec {
                if !val.is_finite() {
                    return Err(SearchError::InvalidConfig {
                        field: "vectors".to_owned(),
                        value: format!("vector[{i}] contains non-finite value"),
                        reason: "all vector elements must be finite".to_owned(),
                    });
                }
            }
            crate::simd::encode_f32_to_f16_extend(&vec, &mut flat);
        }
        Ok(Self {
            doc_ids,
            vectors: flat,
            vectors_i8: OnceLock::new(),
            vectors_nibbles: OnceLock::new(),
            doc_id_hashes: OnceLock::new(),
            doc_id_index: OnceLock::new(),
            hash_to_pos: OnceLock::new(),
            space_fingerprint_hex: None,
            embedder_id: None,
            embedder_revision: None,
            space_identity_attested: false,
            dimension,
            residual_source_binding: None,
            exact_residual_sidecar: OnceLock::new(),
        })
    }

    /// Build from pre-computed f32 vectors that are known to have been
    /// produced in `space`, binding the index to that embedding space
    /// (bd-9xuj T2-C3).
    ///
    /// The space's fingerprint becomes this index's
    /// [`Self::space_fingerprint_hex`], the index-side join key for
    /// [`frankensearch_core::BoundQueryEmbedding::verify_space_identity`].
    /// The claim is checked before it is stored: `space` must itself
    /// validate, and its declared dimension must equal `dimension` — an
    /// index must never carry an identity that does not describe its
    /// vectors.
    ///
    /// # Errors
    ///
    /// Everything [`Self::from_vectors`] returns, plus
    /// `SearchError::InvalidConfig` when `space` fails validation or its
    /// dimension does not match `dimension`.
    pub fn from_vectors_with_identity(
        doc_ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        dimension: usize,
        space: &EmbeddingSpaceIdentityV1,
    ) -> SearchResult<Self> {
        space.validate()?;
        if usize::try_from(space.dimension).ok() != Some(dimension) {
            return Err(SearchError::InvalidConfig {
                field: "space_identity.dimension".to_owned(),
                value: space.dimension.to_string(),
                reason: format!(
                    "embedding-space dimension must equal the index dimension ({dimension}); \
                     refusing to bind an identity that does not describe this index's vectors"
                ),
            });
        }
        let mut index = Self::from_vectors(doc_ids, vectors, dimension)?;
        index.space_fingerprint_hex = Some(space.fingerprint());
        // The validated space also names its producing model; retain that as
        // the diagnostic embedder identity (bd-9xuj T2-C2). The join key
        // above stays the only compatibility authority.
        index.embedder_id = Some(space.logical_model_id.clone());
        index.embedder_revision = Some(space.immutable_revision.clone());
        // A caller-supplied space is a DECLARED identity, not one attested by
        // an artifact header (bd-9xuj T2-C4-write): the discriminator stays
        // false, and attested-only seams must not admit against it.
        index.space_identity_attested = false;
        Ok(index)
    }

    /// Load from an existing FSVI file, reading all data into memory.
    ///
    /// This reads the entire file-backed index into heap memory, eliminating
    /// page-fault latency on subsequent searches.
    ///
    /// # Errors
    ///
    /// Returns errors from [`VectorIndex::open_read_only`] or vector decoding failures.
    pub fn from_fsvi(path: &Path) -> SearchResult<Self> {
        let index = VectorIndex::open_read_only(path)?;
        Self::from_open_index(&index)
    }

    /// Build from a fully admitted immutable FSVI v2 artifact, preserving its
    /// validated embedding-space identity (bd-9xuj T2-C3).
    ///
    /// [`VectorIndex::open`] — and therefore [`Self::from_fsvi`] — reads
    /// legacy v1 files only; identity-complete v2 artifacts are opened
    /// exclusively through exact admission
    /// ([`VectorIndex::open_admitted_v2`]). This constructor is the in-memory
    /// load for that path: the resulting index always carries the artifact's
    /// validated space fingerprint in [`Self::space_fingerprint_hex`].
    ///
    /// # Errors
    ///
    /// Returns vector decoding failures from the admitted artifact.
    pub fn from_admitted_v2(source: &ValidatedFsviBytes) -> SearchResult<Self> {
        let mut index = Self::from_open_index(&source.index)?;
        index.residual_source_binding = Some(ResidualSourceBinding::from_witness(source.witness()));
        Ok(index)
    }

    /// Load an admitted FSVI v2 generation through the shipping residual-cache
    /// route. The cache directory may hold immutable sidecars for many source
    /// generations; only an exact generation-key match that re-derives bitwise
    /// from this admitted source is attached. Missing, stale, corrupt, or
    /// collision-held paths cause a new generation-keyed immutable candidate to
    /// be published without overwriting any prior artifact.
    ///
    /// This is the single-vector production construction seam. The two-tier
    /// shipping constructor below calls it for each admitted tier, so product
    /// users do not need test-only field mutation or disconnected I/O calls.
    ///
    /// # Errors
    ///
    /// Returns source-vector decoding errors. Optional sidecar failures are
    /// deliberately contained as flat-scan fallback.
    #[cfg(target_os = "linux")]
    pub fn from_admitted_v2_with_residual_sidecar_cache(
        source: &ValidatedFsviBytes,
        cache_dir: &Path,
    ) -> SearchResult<Self> {
        let index = Self::from_admitted_v2(source)?;
        let Ok(Some(candidates)) = index.exact_residual_sidecar_cache_candidates(cache_dir) else {
            // A cache directory that cannot be safely enumerated, or exhausts
            // its fixed discovery/comparison budget, is unavailable. Do not
            // build a large sidecar merely to discard it after an inevitable
            // publication failure; flat exact search is already the semantic
            // default. In this case publication is deliberately skipped.
            return Ok(index);
        };
        // Candidate verification is a read-only path: a valid immutable
        // sidecar must remain reusable from a read-only cache root. Its full
        // source-derived comparison is necessary before attachment, but must
        // not require a publication capability that reuse never consumes.
        #[cfg(target_os = "linux")]
        let Ok(candidate_layout) =
            ExactResidualLayout::for_shape(index.record_count(), index.dimension)
        else {
            return Ok(index);
        };
        let mut derived = None;
        for candidate in candidates {
            // Every read of `candidate` below is behind the Linux descriptor
            // API; off Linux the loop only re-derives, so name the deliberate
            // non-use rather than emit an unused-variable warning there.
            #[cfg(not(target_os = "linux"))]
            let _ = &candidate;
            // Fail-cheap admission. Discovery filters on the header alone, so a
            // corrupt body would otherwise reach the byte-exact comparison and
            // force a full source derivation solely to learn the candidate was
            // never reusable — work that is discarded whenever the optional
            // cache also cannot publish. One bounded streaming read over the
            // candidate's own self-covering digest decides that without
            // deriving anything, which is what lets the reservation-before-
            // derivation ordering below hold even when candidates exist.
            #[cfg(target_os = "linux")]
            match exact_residual_sidecar_stream_is_self_consistent(&candidate, &candidate_layout) {
                Ok(true) => {}
                Ok(false) | Err(_) => continue,
            }
            if derived.is_none() {
                let Ok(sidecar) = index.build_exact_residual_sidecar() else {
                    return Ok(index);
                };
                derived = Some(sidecar);
            }
            #[cfg(target_os = "linux")]
            let exactly_matches = {
                let Some(expected) = derived.as_ref() else {
                    return Ok(index);
                };
                exact_residual_sidecar_stream_matches_derived(&candidate, expected).unwrap_or(false)
            };
            #[cfg(not(target_os = "linux"))]
            let exactly_matches = false;
            if exactly_matches {
                let Some(sidecar) = derived.take() else {
                    return Ok(index);
                };
                let _ = index.exact_residual_sidecar.set(sidecar);
                return Ok(index);
            }
        }
        // No reusable candidate was attached. Reserve the create-only
        // publication primitive before a no-candidate derivation so an
        // unavailable optional cache cannot cause a discarded large build.
        // A corrupt header-matching candidate may already have required one
        // derivation for read-only authentication above; that work is never
        // repeated here.
        #[cfg(target_os = "linux")]
        let reserved_publication = {
            let Ok(candidate) = index.next_exact_residual_sidecar_cache_path(cache_dir) else {
                return Ok(index);
            };
            match acquire_exact_residual_sidecar_publication(&candidate) {
                Ok(capability) => capability,
                Err(_) => return Ok(index),
            }
        };
        let sidecar = match derived {
            Some(sidecar) => sidecar,
            None => match index.build_exact_residual_sidecar() {
                Ok(sidecar) => sidecar,
                Err(_) => return Ok(index),
            },
        };
        #[cfg(target_os = "linux")]
        match publish_exact_residual_sidecar_with_capability(reserved_publication, &sidecar) {
            Ok(ExactResidualPublication::Published) => {
                let _ = index.exact_residual_sidecar.set(sidecar);
                return Ok(index);
            }
            // A concurrent owner may have claimed the deterministic
            // generation-keyed name after capability acquisition. The
            // already-derived sidecar is safe to retry under a new name.
            Ok(ExactResidualPublication::DestinationExists) => {}
            Err(_) => return Ok(index),
        }
        #[cfg(target_os = "linux")]
        for _ in 0..EXACT_RESIDUAL_CACHE_ATTEMPTS {
            let Ok(candidate) = index.next_exact_residual_sidecar_cache_path(cache_dir) else {
                break;
            };
            match publish_exact_residual_sidecar(&candidate, &sidecar) {
                Ok(ExactResidualPublication::Published) => {
                    let _ = index.exact_residual_sidecar.set(sidecar);
                    break;
                }
                // A concurrent owner claimed this name; the next attempt draws
                // a fresh one. This is the final statement of the loop body, so
                // falling through is exactly one more attempt.
                Ok(ExactResidualPublication::DestinationExists) => {}
                Err(_) => break,
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = sidecar;
        Ok(index)
    }

    #[cfg(target_os = "linux")]
    pub(super) fn exact_residual_generation_cache_prefix(&self) -> SearchResult<String> {
        let source = self.residual_source_binding.as_ref().ok_or_else(|| {
            residual_sidecar_error(
                "source_generation",
                "generation-keyed caching requires an admitted FSVI v2 source",
            )
        })?;
        let mut digest = Sha256::new();
        digest.update(EXACT_RESIDUAL_SIDECAR_MAGIC);
        digest.update(EXACT_RESIDUAL_SIDECAR_VERSION.to_le_bytes());
        digest.update(source.generation_fingerprint);
        digest.update(source.vector_content_digest);
        digest.update(source.ordered_live_docset_digest);
        digest.update(source.space_fingerprint);
        digest.update(
            u64::try_from(self.record_count())
                .map_err(|_| residual_sidecar_error("record_count", "does not fit u64"))?
                .to_le_bytes(),
        );
        digest.update(
            u64::try_from(self.dimension)
                .map_err(|_| residual_sidecar_error("dimension", "does not fit u64"))?
                .to_le_bytes(),
        );
        let mut prefix = String::new();
        prefix
            .try_reserve_exact(9 + EXACT_RESIDUAL_SIDECAR_DIGEST_BYTES * 2)
            .map_err(|_| residual_sidecar_error("cache_path", "allocation failed"))?;
        prefix.push_str("fsrs-v2-");
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for byte in digest.finalize() {
            prefix.push(char::from(HEX[usize::from(byte >> 4)]));
            prefix.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        prefix.push('-');
        Ok(prefix)
    }

    #[cfg(target_os = "linux")]
    fn next_exact_residual_sidecar_cache_path(&self, cache_dir: &Path) -> SearchResult<PathBuf> {
        let prefix = self.exact_residual_generation_cache_prefix()?;
        let nonce = EXACT_RESIDUAL_CACHE_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(cache_dir.join(format!("{prefix}{}-{nonce}.fsrs", std::process::id())))
    }

    #[cfg(target_os = "linux")]
    fn exact_residual_sidecar_cache_candidates(
        &self,
        cache_dir: &Path,
    ) -> SearchResult<Option<Vec<PathBuf>>> {
        // Prove the optional cache root itself is a no-follow directory before
        // deriving a large payload. This catches a missing path, regular-file
        // obstacle, or symlinked cache root while the fallback is still free.
        if open_exact_residual_sidecar_parent(&cache_dir.join(".fsrs-cache-probe")).is_err() {
            return Ok(None);
        }
        let Some(source) = self.residual_source_binding.as_ref() else {
            return Ok(None);
        };
        let prefix = self.exact_residual_generation_cache_prefix()?;
        let Ok(layout) = ExactResidualLayout::for_shape(self.record_count(), self.dimension) else {
            return Ok(None);
        };
        let comparison_limit = EXACT_RESIDUAL_CACHE_COMPARISON_CANDIDATE_LIMIT
            .min(EXACT_RESIDUAL_CACHE_COMPARISON_BYTE_BUDGET / layout.encoded_bytes);
        if comparison_limit == 0 {
            return Ok(None);
        }
        let Ok(mut entries) = std::fs::read_dir(cache_dir) else {
            return Ok(None);
        };
        let mut paths = Vec::new();
        paths
            .try_reserve_exact(EXACT_RESIDUAL_CACHE_DIRECTORY_ENTRY_LIMIT)
            .map_err(|_| residual_sidecar_error("cache", "directory-entry allocation failed"))?;
        for _ in 0..=EXACT_RESIDUAL_CACHE_DIRECTORY_ENTRY_LIMIT {
            let entry = match entries.next() {
                Some(Ok(entry)) => entry,
                Some(Err(_)) => return Ok(None),
                None => break,
            };
            if paths.len() == EXACT_RESIDUAL_CACHE_DIRECTORY_ENTRY_LIMIT {
                return Ok(None);
            }
            paths.push(entry.path());
        }
        // `read_dir` order is filesystem-defined. Sorting the bounded set
        // makes both candidate selection and the work-budget verdict stable.
        paths.sort_unstable();
        let mut candidates = Vec::new();
        candidates
            .try_reserve_exact(comparison_limit)
            .map_err(|_| residual_sidecar_error("cache", "candidate allocation failed"))?;
        let mut comparison_bytes = 0_usize;
        for candidate in paths {
            let Some(name) = candidate.file_name() else {
                return Ok(None);
            };
            let Some(name) = name.to_str() else {
                continue;
            };
            // These names are generated by this module, so the suffix match is
            // deliberately exact: a differently-cased name is not one of ours
            // and must not be admitted. `strip_suffix` states that exactness
            // directly instead of a case-folding extension comparison.
            if !name.starts_with(&prefix) || name.strip_suffix(".fsrs").is_none() {
                continue;
            }
            let header_matches = exact_residual_sidecar_header_matches_source(
                &candidate,
                source,
                self.record_count(),
                self.dimension,
            )
            .unwrap_or(false);
            if header_matches {
                let Some(next_comparison_bytes) =
                    comparison_bytes.checked_add(layout.encoded_bytes)
                else {
                    return Ok(None);
                };
                if candidates.len() == comparison_limit
                    || next_comparison_bytes > EXACT_RESIDUAL_CACHE_COMPARISON_BYTE_BUDGET
                {
                    return Ok(None);
                }
                comparison_bytes = next_comparison_bytes;
                candidates.push(candidate);
            }
        }
        Ok(Some(candidates))
    }

    /// Load the admitted vectors without the Linux-only residual sidecar cache.
    ///
    /// # Errors
    ///
    /// Returns source-vector decoding errors.
    #[cfg(not(target_os = "linux"))]
    pub fn from_admitted_v2_with_residual_sidecar_cache(
        source: &ValidatedFsviBytes,
        _cache_dir: &Path,
    ) -> SearchResult<Self> {
        Self::from_admitted_v2(source)
    }

    /// Shared load path: read every live row (and any WAL tail) of an opened
    /// index into memory, preserving the embedding-space identity the source
    /// declares. An admitted v2 source carries a validated space fingerprint;
    /// a legacy v1 source carries none, and that absence is kept as the typed
    /// `None` state — never papered over with a fabricated identity.
    fn from_open_index(index: &VectorIndex) -> SearchResult<Self> {
        let count = index.record_count();
        let dimension = index.dimension();
        let capacity = count.checked_add(index.wal_entries.len()).ok_or_else(|| {
            SearchError::InvalidConfig {
                field: "vectors".to_owned(),
                value: format!("main={count}, wal={}", index.wal_entries.len()),
                reason: "document capacity overflow".to_owned(),
            }
        })?;
        let scalar_capacity =
            capacity
                .checked_mul(dimension)
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "vectors".to_owned(),
                    value: format!("count={capacity}, dimension={dimension}"),
                    reason: "vector slab length overflow".to_owned(),
                })?;
        let mut doc_ids = Vec::new();
        doc_ids
            .try_reserve_exact(capacity)
            .map_err(|_| SearchError::InvalidConfig {
                field: "vectors".to_owned(),
                value: capacity.to_string(),
                reason: "document-id allocation failed".to_owned(),
            })?;
        let mut flat = Vec::new();
        flat.try_reserve_exact(scalar_capacity)
            .map_err(|_| SearchError::InvalidConfig {
                field: "vectors".to_owned(),
                value: scalar_capacity.to_string(),
                reason: "vector slab allocation failed".to_owned(),
            })?;

        for i in 0..count {
            if index.is_deleted(i) {
                continue;
            }
            doc_ids.push(fallible_doc_id_copy(index.doc_id_at(i)?)?);
            index.extend_vector_at_f16(i, &mut flat)?;
        }

        for entry in &index.wal_entries {
            doc_ids.push(fallible_doc_id_copy(&entry.doc_id)?);
            flat.extend(entry.embedding.iter().copied().map(half::f16::from_f32));
        }

        // Capture the source's embedding-space identity before the backing
        // `VectorIndex` drops (bd-9xuj T2-C3). On the admitted-v2 path this
        // is the validated space fingerprint from the artifact's identity
        // header; on the legacy v1 `open` path `identity_v2()` is
        // structurally `None`, which is preserved as the typed absent state —
        // never substituted with a fabricated identity.
        let space_fingerprint_hex = index
            .identity_v2()
            .map(|identity| crate::fingerprint_hex(&identity.space_fingerprint));
        // The attested bit derives from WHERE the identity came from
        // (bd-9xuj T2-C4-write): `identity_v2()` is populated exclusively by
        // the v2 header parse inside exact admission (`VectorIndex::open` is
        // strictly v1), so its presence here means the fingerprint above was
        // read out of the artifact's own validated header bytes.
        let space_identity_attested = index.identity_v2().is_some();
        // Every FSVI header (v1 and v2) carries the embedder id/revision
        // strings; retain them verbatim instead of discarding them at load
        // (bd-9xuj T2-C2). A v1 header's empty revision stays `Some("")` —
        // that is what the header says, not a fabrication.
        let embedder_id = Some(index.embedder_id().to_owned());
        let embedder_revision = Some(index.embedder_revision().to_owned());

        Ok(Self {
            doc_ids,
            vectors: flat,
            vectors_i8: OnceLock::new(),
            vectors_nibbles: OnceLock::new(),
            doc_id_hashes: OnceLock::new(),
            doc_id_index: OnceLock::new(),
            hash_to_pos: OnceLock::new(),
            space_fingerprint_hex,
            embedder_id,
            embedder_revision,
            space_identity_attested,
            dimension,
            residual_source_binding: None,
            exact_residual_sidecar: OnceLock::new(),
        })
    }

    /// Number of vectors in the index.
    #[must_use]
    pub const fn record_count(&self) -> usize {
        self.doc_ids.len()
    }

    /// Vector dimensionality.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Lowercase hex SHA-256 fingerprint of the embedding space this index's
    /// vectors were produced in, when known (bd-9xuj T2-C3).
    ///
    /// This is the index-side join key for
    /// [`frankensearch_core::BoundQueryEmbedding::verify_space_identity`]:
    /// a bound query embedding is admissible against this index exactly when
    /// its space fingerprint equals this value. `None` means the source
    /// carried no identity (legacy v1 FSVI file, or the identity-less
    /// [`Self::from_vectors`] constructor); that absence is a legal, typed
    /// state which downstream seams route as legacy-unidentified — it must
    /// never be papered over with a fabricated fingerprint.
    #[must_use]
    pub fn space_fingerprint_hex(&self) -> Option<&str> {
        self.space_fingerprint_hex.as_deref()
    }

    /// Whether this index's space identity is FSVI-v2-HEADER-attested rather
    /// than builder/caller-declared (bd-9xuj T2-C4-write, guards 2+8).
    ///
    /// `true` exactly when the index was loaded through
    /// [`Self::from_admitted_v2`] from an artifact whose validated v2 header
    /// carried the identity — i.e. the fingerprint in
    /// [`Self::space_fingerprint_hex`] was read out of the artifact's own
    /// bytes. A `Some` fingerprint with `false` here is a DECLARED identity
    /// ([`Self::from_vectors_with_identity`]): usable for diagnostics and
    /// query-side joins, but never sufficient for an attested-only admission
    /// seam such as the refresh identity-bound merge.
    #[must_use]
    pub const fn space_identity_is_attested(&self) -> bool {
        self.space_identity_attested
    }

    /// Whether this index currently has a source-derived exact-residual
    /// sidecar attached. `true` means the cache passed descriptor-bound,
    /// generation-bound, and bitwise re-derivation checks; it says nothing
    /// about whether a particular query can prune, because non-finite or
    /// overflow-risk queries deliberately retain the exact flat fallback.
    #[must_use]
    pub fn has_exact_residual_sidecar(&self) -> bool {
        self.exact_residual_sidecar.get().is_some()
    }

    /// Embedder id recorded by this index's source, when the source carried
    /// one (bd-9xuj T2-C2): the FSVI header string on the file-backed load
    /// paths, or the declared space's `logical_model_id` on
    /// [`Self::from_vectors_with_identity`].
    ///
    /// Diagnostics only. Compatibility joins on
    /// [`Self::space_fingerprint_hex`]; an id string must never admit or
    /// reject a pairing. `None` means the source declared nothing
    /// ([`Self::from_vectors`]) — typed absence, never a default.
    #[must_use]
    pub fn embedder_id(&self) -> Option<&str> {
        self.embedder_id.as_deref()
    }

    /// Embedder revision recorded by this index's source, under the same
    /// retention rules as [`Self::embedder_id`] (bd-9xuj T2-C2). A v1
    /// header's empty revision is preserved as `Some("")`, distinct from the
    /// `None` of a source with no header at all.
    #[must_use]
    pub fn embedder_revision(&self) -> Option<&str> {
        self.embedder_revision.as_deref()
    }

    /// Build and atomically publish a versioned exact-residual sidecar for this
    /// admitted FSVI v2 generation.  Publication is descriptor-relative,
    /// no-symlink, `O_TMPFILE` + descriptor-bound `linkat` no-replace, and
    /// parent-directory durable; it never replaces an already published
    /// artifact. A failure before `linkat` has no visible temporary path. A
    /// failure syncing the parent after link is reported and intentionally does
    /// not unlink the visible artifact, because that path may already be
    /// durable. It never mutates the active FSVI image or a prior sidecar.
    ///
    /// Legacy and caller-built in-memory indexes have no source-generation
    /// witness, so they are deliberately rejected instead of fabricating one.
    ///
    /// # Errors
    ///
    /// Returns an error when the create-only publication capability cannot be
    /// acquired, when the sidecar cannot be derived from this generation
    /// (including a legacy or caller-built index carrying no source-generation
    /// witness), when the descriptor-bound link or parent-directory sync fails,
    /// or when the destination already exists, since a published sidecar is
    /// immutable and is never overwritten. On a non-Linux target it always
    /// returns an unsupported-platform error.
    pub fn write_exact_residual_sidecar(&self, path: &Path) -> SearchResult<()> {
        #[cfg(target_os = "linux")]
        {
            let capability = acquire_exact_residual_sidecar_publication(path)?;
            let sidecar = match self.build_exact_residual_sidecar() {
                Ok(sidecar) => sidecar,
                Err(error) => {
                    drop(capability);
                    return Err(error);
                }
            };
            match publish_exact_residual_sidecar_with_capability(capability, &sidecar)? {
                ExactResidualPublication::Published => Ok(()),
                ExactResidualPublication::DestinationExists => Err(residual_sidecar_error(
                    "publish",
                    "destination already exists; immutable sidecars are never overwritten",
                )),
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = path;
            Err(residual_sidecar_error(
                "platform",
                "exact residual sidecar publication requires Linux descriptor APIs",
            ))
        }
    }

    /// Attempt to admit an exact-residual sidecar.  A missing, malformed, stale,
    /// mismatched, or already-replaced sidecar returns `false` and leaves the
    /// established f16 flat scan selected.  The non-error result is intentional:
    /// sidecars are an optional optimization, never part of semantic admission.
    ///
    /// # Errors
    ///
    /// Returns an error only when re-deriving the comparison sidecar from the
    /// admitted source fails. A missing, malformed, stale, or mismatched
    /// sidecar is reported as `Ok(false)`, never as an error.
    pub fn try_open_exact_residual_sidecar(&self, path: &Path) -> SearchResult<bool> {
        let Some(source) = self.residual_source_binding.as_ref() else {
            return Ok(false);
        };
        // Both are consumed only by the Linux descriptor path below; off Linux
        // the header never matches, so name the deliberate non-use.
        #[cfg(not(target_os = "linux"))]
        let _ = (path, source);
        #[cfg(target_os = "linux")]
        let header_matches = exact_residual_sidecar_header_matches_source(
            path,
            source,
            self.record_count(),
            self.dimension,
        )
        .unwrap_or(false);
        #[cfg(not(target_os = "linux"))]
        let header_matches = false;
        if !header_matches {
            return Ok(false);
        }
        // The header preflight avoids derivation for absent/stale cache entries.
        // Once it matches, compare every serialized field and whole-file digest
        // directly from one descriptor with fixed stack buffers. This holds only
        // the trusted derived sidecar, never a second 512 MiB raw byte image.
        let expected = self.build_exact_residual_sidecar()?;
        #[cfg(target_os = "linux")]
        let exactly_matches =
            exact_residual_sidecar_stream_matches_derived(path, &expected).unwrap_or(false);
        #[cfg(not(target_os = "linux"))]
        let exactly_matches = false;
        if !exactly_matches {
            return Ok(false);
        }
        Ok(self.exact_residual_sidecar.set(expected).is_ok())
    }

    /// Sidecar content is treated as an untrusted derived cache, even after its
    /// file digest and source-generation header match.  Re-derive the complete
    /// transform from the admitted f16 source and require bitwise equality before
    /// it is allowed to prune a candidate.  This makes a finite, rehashed payload
    /// mutation fail closed rather than merely look structurally well-formed.
    #[cfg(test)]
    pub(super) fn admit_exact_residual_sidecar(
        &self,
        sidecar: &ExactResidualSidecar,
    ) -> SearchResult<bool> {
        let Some(source) = self.residual_source_binding.as_ref() else {
            return Ok(false);
        };
        if !sidecar.is_bound_to(source, self.record_count(), self.dimension) {
            return Ok(false);
        }
        let expected = self.build_exact_residual_sidecar()?;
        if !sidecar.exactly_matches_derived(&expected) {
            return Ok(false);
        }
        // The source-derived value, not caller-provided transformed data, is
        // what becomes live after a successful equality proof.
        Ok(self.exact_residual_sidecar.set(expected).is_ok())
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub(super) fn build_exact_residual_sidecar(&self) -> SearchResult<ExactResidualSidecar> {
        #[cfg(all(test, target_os = "linux"))]
        EXACT_RESIDUAL_SIDECAR_BUILD_COUNT.with(|count| {
            count.set(count.get().saturating_add(1));
        });
        let source = self.residual_source_binding.clone().ok_or_else(|| {
            residual_sidecar_error(
                "source_generation",
                "exact residual sidecars require an admitted FSVI v2 source generation",
            )
        })?;
        // Every platform requires source admission; persistence retains the
        // binding only where the descriptor-based codec is supported.
        #[cfg(not(any(test, target_os = "linux")))]
        let _ = source;
        let count = self.record_count();
        let dimension = self.dimension;
        let layout = ExactResidualLayout::for_shape(count, dimension)?;

        let mut energy = try_filled_sidecar_vec(dimension, 0.0_f64, "energy")?;
        for row in 0..count {
            for (dimension_index, value) in self.vector_slice(row).iter().enumerate() {
                let value = value.to_f32();
                if !value.is_finite() {
                    return Err(residual_sidecar_error(
                        "vectors",
                        "source contains a non-finite f16 value",
                    ));
                }
                energy[dimension_index] += f64::from(value) * f64::from(value);
            }
        }
        // Keep the final persisted permutation as the working sort buffer so
        // derivation never peaks at both a usize dimension list and a second
        // u32 payload list. The format cap is far below u32::MAX.
        let mut permutation = Vec::new();
        permutation
            .try_reserve_exact(dimension)
            .map_err(|_| residual_sidecar_error("permutation", "allocation failed"))?;
        for dimension_index in 0..dimension {
            permutation.push(u32::try_from(dimension_index).map_err(|_| {
                residual_sidecar_error("permutation", "dimension does not fit u32")
            })?);
        }
        permutation.sort_unstable_by(|left, right| {
            // `for_shape` caps the dimension at 65_536, so each stored
            // permutation value is at most 65_535 and fits every supported
            // `usize` without a fallible sorting closure.
            let left = *left as usize;
            let right = *right as usize;
            energy[right]
                .total_cmp(&energy[left])
                .then_with(|| left.cmp(&right))
        });
        drop(energy);

        let mut centroids = try_filled_sidecar_vec(layout.centroid_len, 0.0_f32, "centroids")?;
        let mut residuals = try_filled_sidecar_vec(layout.residual_len, 0.0_f32, "residuals")?;
        let mut suffix_norms = try_filled_sidecar_vec(layout.suffix_len, 0.0_f32, "suffix_norms")?;
        let mut correction_norms =
            try_filled_sidecar_vec(layout.lane_len, 0.0_f32, "correction_norms")?;
        for group in 0..layout.groups {
            let group_start = group * EXACT_RESIDUAL_LANES;
            let active_lanes = (count - group_start).min(EXACT_RESIDUAL_LANES);
            for (transformed_dimension, &source_dimension) in permutation.iter().enumerate() {
                let source_dimension = usize::try_from(source_dimension).map_err(|_| {
                    residual_sidecar_error("permutation", "dimension does not fit this platform")
                })?;
                let sum: f64 = (0..active_lanes)
                    .map(|lane| {
                        f64::from(self.vector_slice(group_start + lane)[source_dimension].to_f32())
                    })
                    .sum();
                centroids[group * dimension + transformed_dimension] =
                    (sum / active_lanes.max(1) as f64) as f32;
            }
            for lane in 0..active_lanes {
                let residual_base = group * dimension * EXACT_RESIDUAL_LANES + lane;
                let mut correction_sum = 0.0_f64;
                for (transformed_dimension, &source_dimension) in permutation.iter().enumerate() {
                    let source_dimension = usize::try_from(source_dimension).map_err(|_| {
                        residual_sidecar_error(
                            "permutation",
                            "dimension does not fit this platform",
                        )
                    })?;
                    let original = self.vector_slice(group_start + lane)[source_dimension].to_f32();
                    let centroid = centroids[group * dimension + transformed_dimension];
                    let residual = original - centroid;
                    residuals[residual_base + transformed_dimension * EXACT_RESIDUAL_LANES] =
                        residual;
                    // Keep the entire construction remainder in f64, rather
                    // than only the rounded f32 subtraction remainder. This
                    // contains both the residual-construction rounding and
                    // the later `(centroid + residual)` reconstruction
                    // rounding, so the Cauchy term bounds the actual f16 row.
                    let correction = finite_f64_add(
                        finite_f64_add(f64::from(original), -f64::from(centroid)).ok_or_else(
                            || {
                                residual_sidecar_error(
                                    "correction_norms",
                                    "construction subtraction overflowed",
                                )
                            },
                        )?,
                        -f64::from(residual),
                    )
                    .ok_or_else(|| {
                        residual_sidecar_error(
                            "correction_norms",
                            "construction subtraction overflowed",
                        )
                    })?;
                    let correction_squared =
                        finite_f64_mul(correction, correction).ok_or_else(|| {
                            residual_sidecar_error(
                                "correction_norms",
                                "construction square overflowed",
                            )
                        })?;
                    correction_sum = finite_f64_add(correction_sum, correction_squared)
                        .ok_or_else(|| {
                            residual_sidecar_error(
                                "correction_norms",
                                "construction norm overflowed",
                            )
                        })?;
                }
                correction_norms[group * EXACT_RESIDUAL_LANES + lane] =
                    outward_f32_norm(correction_sum, "correction_norms")?;
                let mut suffix_sum = 0.0_f64;
                for block_index in (0..layout.blocks).rev() {
                    let start = block_index * EXACT_RESIDUAL_BLOCK;
                    let end = (start + EXACT_RESIDUAL_BLOCK).min(dimension);
                    for transformed_dimension in start..end {
                        let residual =
                            residuals[residual_base + transformed_dimension * EXACT_RESIDUAL_LANES];
                        suffix_sum += f64::from(residual) * f64::from(residual);
                    }
                    let suffix_offset =
                        (group * (layout.blocks + 1) + block_index) * EXACT_RESIDUAL_LANES + lane;
                    suffix_norms[suffix_offset] = outward_f32_norm(suffix_sum, "suffix_norms")?;
                }
            }
        }
        Ok(ExactResidualSidecar {
            #[cfg(any(test, target_os = "linux"))]
            source,
            count,
            dimension,
            block: EXACT_RESIDUAL_BLOCK,
            lanes: EXACT_RESIDUAL_LANES,
            permutation,
            centroids,
            residuals,
            suffix_norms,
            correction_norms,
        })
    }
}
