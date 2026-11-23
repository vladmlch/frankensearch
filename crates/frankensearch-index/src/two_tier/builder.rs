use std::fs;
use std::path::{Path, PathBuf};

use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::{SearchError, SearchResult, TwoTierConfig};
use tracing::warn;

use crate::{Quantization, VectorIndex};

use super::{TwoTierIndex, VECTOR_INDEX_FAST_FILENAME, VECTOR_INDEX_QUALITY_FILENAME};

/// Tier that served a semantic vector
/// ([`TwoTierIndex::semantic_vector_with_tier_for_doc_id`]; bd-9xuj T2-C2).
///
/// The tier is the typed handle to the embedding space the vector lives in:
/// resolve it via [`TwoTierIndex::space_fingerprint_hex_for_tier`]. Vectors
/// from different tiers are NOT comparable — the silent quality→fast fallback
/// is exactly the seam this type makes observable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticVectorTier {
    /// Served from the quality tier (its embedding space is the quality
    /// tier's).
    Quality,
    /// Served from the fast tier: either no quality index is loaded or the
    /// quality tier has no vector for this document.
    Fast,
}

/// Builder for writing fast and optional quality FSVI indices.
#[derive(Debug)]
pub struct TwoTierIndexBuilder {
    dir: PathBuf,
    config: TwoTierConfig,
    fast_embedder_id: String,
    quality_embedder_id: String,
    fast_embedder_revision: String,
    quality_embedder_revision: String,
    fast_identity: Option<EmbeddingIdentityBundleV1>,
    quality_identity: Option<EmbeddingIdentityBundleV1>,
    fast_dimension: Option<usize>,
    quality_dimension: Option<usize>,
    fast_records: Vec<(String, Vec<f32>)>,
    quality_records: Vec<(String, Vec<f32>)>,
    fast_ids: std::collections::HashSet<String>,
    quality_ids: std::collections::HashSet<String>,
}

impl TwoTierIndexBuilder {
    pub(super) fn new(dir: PathBuf, config: TwoTierConfig) -> Self {
        Self {
            dir,
            config,
            fast_embedder_id: "fast-tier".to_owned(),
            quality_embedder_id: "quality-tier".to_owned(),
            fast_embedder_revision: String::new(),
            quality_embedder_revision: String::new(),
            fast_identity: None,
            quality_identity: None,
            fast_dimension: None,
            quality_dimension: None,
            fast_records: Vec::new(),
            quality_records: Vec::new(),
            fast_ids: std::collections::HashSet::new(),
            quality_ids: std::collections::HashSet::new(),
        }
    }

    /// Override the embedder id written to the fast-tier index header.
    pub fn set_fast_embedder_id(&mut self, embedder_id: impl Into<String>) -> &mut Self {
        self.fast_embedder_id = embedder_id.into();
        self
    }

    /// Override the embedder id written to the quality-tier index header.
    pub fn set_quality_embedder_id(&mut self, embedder_id: impl Into<String>) -> &mut Self {
        self.quality_embedder_id = embedder_id.into();
        self
    }

    /// Declare the complete identity of the embedder producing the fast
    /// tier's vectors (bd-9xuj T2-C2).
    ///
    /// The bundle is validated before it is accepted — a bound identity is a
    /// claim every downstream verifier trusts — and [`Self::finish`]
    /// additionally rejects it when its space dimension does not describe the
    /// vectors actually written. On success:
    ///
    /// - the v1 header's revision string stores the complete bundle fingerprint,
    ///   so reopening can reject another producer of the same model revision;
    /// - the finished [`TwoTierIndex`] retains the bundle
    ///   ([`TwoTierIndex::fast_declared_identity`]) and exposes the space
    ///   join key ([`TwoTierIndex::fast_space_fingerprint_hex`]).
    ///
    /// The operational id string ([`Self::set_fast_embedder_id`]) is left
    /// untouched: id strings are diagnostics, and this typed bundle — never
    /// a string — is the compatibility authority. Retention is process-local
    /// for the v1 artifacts this builder writes: a later reopen can compare
    /// the fingerprint but still retains no admitted v2 owner or typed bundle.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when the bundle fails its own
    /// validation.
    pub fn set_fast_identity(
        &mut self,
        identity: &EmbeddingIdentityBundleV1,
    ) -> SearchResult<&mut Self> {
        identity.validate()?;
        self.fast_embedder_revision = identity.fingerprint();
        self.fast_identity = Some(identity.clone());
        Ok(self)
    }

    /// Quality-tier counterpart of [`Self::set_fast_identity`].
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when the bundle fails its own
    /// validation.
    pub fn set_quality_identity(
        &mut self,
        identity: &EmbeddingIdentityBundleV1,
    ) -> SearchResult<&mut Self> {
        identity.validate()?;
        self.quality_embedder_revision = identity.fingerprint();
        self.quality_identity = Some(identity.clone());
        Ok(self)
    }

    /// Add a fast-tier vector record.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if this record dimension differs
    /// from previously added fast-tier vectors.
    pub fn add_fast_record(
        &mut self,
        doc_id: impl Into<String>,
        embedding: &[f32],
    ) -> SearchResult<()> {
        let dimension = embedding.len();
        let expected = self.fast_dimension.get_or_insert(dimension);
        if *expected != dimension {
            return Err(SearchError::DimensionMismatch {
                expected: *expected,
                found: dimension,
            });
        }
        let doc_id = doc_id.into();
        if !self.fast_ids.insert(doc_id.clone()) {
            return Err(SearchError::InvalidConfig {
                field: "doc_id".to_owned(),
                value: doc_id,
                reason: "duplicate doc_id in fast tier; each document must have a unique id"
                    .to_owned(),
            });
        }
        self.fast_records.push((doc_id, embedding.to_vec()));
        Ok(())
    }

    /// Bench-only comparator that adds an already-owned fast-tier vector record.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if this record dimension differs
    /// from previously added fast-tier vectors.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn add_fast_record_owned_for_benchmark(
        &mut self,
        doc_id: impl Into<String>,
        embedding: Vec<f32>,
    ) -> SearchResult<()> {
        let dimension = embedding.len();
        let expected = self.fast_dimension.get_or_insert(dimension);
        if *expected != dimension {
            return Err(SearchError::DimensionMismatch {
                expected: *expected,
                found: dimension,
            });
        }
        let doc_id = doc_id.into();
        if !self.fast_ids.insert(doc_id.clone()) {
            return Err(SearchError::InvalidConfig {
                field: "doc_id".to_owned(),
                value: doc_id,
                reason: "duplicate doc_id in fast tier; each document must have a unique id"
                    .to_owned(),
            });
        }
        self.fast_records.push((doc_id, embedding));
        Ok(())
    }

    /// Add a quality-tier vector record.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if this record dimension differs
    /// from previously added quality-tier vectors.
    pub fn add_quality_record(
        &mut self,
        doc_id: impl Into<String>,
        embedding: &[f32],
    ) -> SearchResult<()> {
        let dimension = embedding.len();
        let expected = self.quality_dimension.get_or_insert(dimension);
        if *expected != dimension {
            return Err(SearchError::DimensionMismatch {
                expected: *expected,
                found: dimension,
            });
        }
        let doc_id = doc_id.into();
        if !self.quality_ids.insert(doc_id.clone()) {
            return Err(SearchError::InvalidConfig {
                field: "doc_id".to_owned(),
                value: doc_id,
                reason: "duplicate doc_id in quality tier; each document must have a unique id"
                    .to_owned(),
            });
        }
        self.quality_records.push((doc_id, embedding.to_vec()));
        Ok(())
    }

    /// Add a fast record and optionally a matching quality record.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if either tier dimension is inconsistent.
    pub fn add_record(
        &mut self,
        doc_id: impl Into<String>,
        fast_embedding: &[f32],
        quality_embedding: Option<&[f32]>,
    ) -> SearchResult<()> {
        let doc_id = doc_id.into();
        self.add_fast_record(doc_id.clone(), fast_embedding)?;
        if let Some(quality_embedding) = quality_embedding {
            self.add_quality_record(doc_id, quality_embedding)?;
        }
        Ok(())
    }

    /// Write all buffered records and open the resulting `TwoTierIndex`.
    ///
    /// When a producing identity was declared
    /// ([`Self::set_fast_identity`] / [`Self::set_quality_identity`]), it is
    /// checked against the vectors actually written (its space dimension
    /// must equal the tier dimension — an index must never carry an identity
    /// that does not describe its vectors), written into the tier header's
    /// id/revision strings as far as the v1 format can carry it, and
    /// retained on the returned [`TwoTierIndex`] so consumers can verify
    /// per-tier space identity (bd-9xuj T2-C2). A declared quality identity
    /// is attached only when this build wrote a quality tier: it must never
    /// describe a stale quality artifact discovered on disk.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if no fast-tier records were
    /// added or a declared identity does not describe the written vectors,
    /// and propagates writer/open errors from `VectorIndex`.
    pub fn finish(self) -> SearchResult<TwoTierIndex> {
        let fast_dimension = self
            .fast_dimension
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "fast_records".to_owned(),
                value: "0".to_owned(),
                reason: "at least one fast-tier record is required".to_owned(),
            })?;
        if let Some(identity) = &self.fast_identity {
            ensure_identity_describes_tier(identity, fast_dimension, "fast_identity")?;
        }
        if let (Some(identity), Some(quality_dimension)) =
            (&self.quality_identity, self.quality_dimension)
        {
            ensure_identity_describes_tier(identity, quality_dimension, "quality_identity")?;
        }

        // Build each tier at a staging path and route through
        // install_replacement with a carried-forward generation: publishing
        // over a live directory with a plain in-place build left the
        // destination's WAL sidecar adoptable by the fresh generation
        // (bd-zhjv8 finding 1 — acknowledged appends from the superseded
        // generation resurrected inside the rebuild). Both tiers carry ONE
        // publication nonce, the successor of the destination pair's, so a
        // crash between the two installs is detectable at open time
        // (bd-miio8).
        let fast_path = self.dir.join(VECTOR_INDEX_FAST_FILENAME);
        // `create(path, id, dim)` is `create_with_revision(path, id, "", dim,
        // F16)`, so the written bytes are identical to the former `create`
        // call for builders that declared no identity (empty default
        // revision). Each tier is written at a sibling staging path and
        // installed over the canonical name through `publish_tier`, which
        // carries the successor compaction generation and the shared
        // publication nonce so `install_replacement`'s generation authority
        // atomically invalidates any surviving destination sidecar
        // (bd-zhjv8, bd-miio8; supersedes the r3 retire-before-publish
        // ordering — see `publish_tier`).
        let publication_nonce = crate::next_publication_nonce(if fast_path.exists() {
            VectorIndex::peek_publication_nonce(&fast_path)?
        } else {
            0
        });
        publish_tier(
            &fast_path,
            &self.fast_embedder_id,
            &self.fast_embedder_revision,
            fast_dimension,
            publication_nonce,
            &self.fast_records,
        )?;

        if let Some(quality_dimension) = self.quality_dimension {
            let quality_path = self.dir.join(VECTOR_INDEX_QUALITY_FILENAME);
            publish_tier(
                &quality_path,
                &self.quality_embedder_id,
                &self.quality_embedder_revision,
                quality_dimension,
                publication_nonce,
                &self.quality_records,
            )?;
        }

        let mut index = TwoTierIndex::open(&self.dir, self.config)?;

        // Retain the declared producing identities on the opened index
        // (bd-9xuj T2-C2). This is retention of a validated declaration by
        // the process that produced every vector it wrote — the same class
        // as `InMemoryVectorIndex::from_vectors_with_identity` — not
        // fabrication: an artifact-header identity, were one ever present,
        // wins, and a reopen from disk of these v1 artifacts stays typed
        // legacy-unidentified. The quality identity is dropped unless THIS
        // build wrote the quality tier, so it can never describe a stale
        // quality artifact `open` discovered on disk.
        if let Some(identity) = self.fast_identity {
            if index.fast_space_fingerprint_hex.is_none() {
                index.fast_space_fingerprint_hex = Some(identity.space.fingerprint());
            }
            index.fast_declared_identity = Some(identity);
        }
        if let Some(identity) = self.quality_identity
            && self.quality_dimension.is_some()
            && index.quality_source.is_some()
        {
            if index.quality_space_fingerprint_hex.is_none() {
                index.quality_space_fingerprint_hex = Some(identity.space.fingerprint());
            }
            index.quality_declared_identity = Some(identity);
        }
        Ok(index)
    }
}

/// Stage one tier's records and durably install them over `canonical_path`
/// with the generation discipline [`VectorIndex::install_replacement`]
/// requires (bd-zhjv8) and the shared cross-tier publication nonce
/// (bd-miio8).
///
/// Protocol:
/// 1. The replacement is FULLY written at a sibling staging path
///    (`temporary_output_path`), never at the canonical name, carrying
///    `next_generation(destination)` and the caller's publication nonce.
/// 2. [`VectorIndex::install_replacement`] validates the staged artifact
///    and atomically renames it over the canonical main. The generation
///    byte is the authority: the rename installs the new data AND
///    invalidates any surviving destination WAL (whose records bind to the
///    superseded generation), so sidecar retirement happens safely AFTER
///    publication.
///
/// This deliberately supersedes the bd-9xuj C4-write r3
/// retire-before-publish ordering (`write_tier_with_durable_wal_retirement`,
/// audits #8366/#8367): durably retiring the sidecar BEFORE the rename
/// opens a crash window in which the SURVIVING old generation loses its
/// acknowledged appends — the two-resource-commit finding that closed
/// bd-zhjv8 (authority, not ordering). A failed publication cleans up its
/// own staged artifact and leaves the old generation, main AND sidecar,
/// intact; a crash may leave a stray staging file, which lives under a
/// `.tmp.<pid>.<nanos>` name that no open or discovery path resolves.
fn publish_tier(
    canonical_path: &Path,
    embedder_id: &str,
    embedder_revision: &str,
    dimension: usize,
    publication_nonce: u16,
    records: &[(String, Vec<f32>)],
) -> SearchResult<()> {
    let generation = if canonical_path.exists() {
        crate::next_generation(VectorIndex::peek_compaction_gen(canonical_path)?)
    } else {
        1
    };
    let staging_path = crate::temporary_output_path(canonical_path);
    let result = (|| {
        let mut writer = VectorIndex::create_with_revision(
            &staging_path,
            embedder_id,
            embedder_revision,
            dimension,
            Quantization::F16,
        )?
        .with_generation(generation)
        .with_publication_nonce(publication_nonce);
        for (doc_id, embedding) in records {
            writer.write_record(doc_id, embedding)?;
        }
        writer.finish()?;
        VectorIndex::install_replacement(canonical_path, &staging_path).map(drop)
    })();
    if result.is_err()
        && staging_path.exists()
        && let Err(cleanup_err) = fs::remove_file(&staging_path)
    {
        // Best-effort cleanup of the staged artifact only — the staging file
        // is this function's own product under a name no open path resolves,
        // never the canonical main or its WAL (mirrors the error path of
        // `VectorIndexWriter::finish`).
        warn!(
            staging_path = %staging_path.display(),
            ?cleanup_err,
            "failed to clean up staged tier replacement after error"
        );
    }
    result
}

/// Reject a declared producing identity whose space dimension does not
/// describe the tier's written vectors (bd-9xuj T2-C2): the identity claim is
/// checked before it is retained, never trusted on assertion alone.
fn ensure_identity_describes_tier(
    identity: &EmbeddingIdentityBundleV1,
    tier_dimension: usize,
    tier_field: &str,
) -> SearchResult<()> {
    if usize::try_from(identity.space.dimension).ok() == Some(tier_dimension) {
        return Ok(());
    }
    Err(SearchError::InvalidConfig {
        field: format!("{tier_field}.space.dimension"),
        value: identity.space.dimension.to_string(),
        reason: format!(
            "declared embedding-space dimension must equal the written tier dimension \
             ({tier_dimension}); refusing to retain an identity that does not describe \
             this tier's vectors"
        ),
    })
}
