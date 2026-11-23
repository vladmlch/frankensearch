use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

use super::CommitRange;
use super::activation_manifest::GenerationComponentReceiptV1;
use super::authority_core::GenerationAuthorityErrorV1;
use super::codec::CanonicalEncoder;
use super::manifest::GenerationManifest;

// ---------------------------------------------------------------------------
// Exact generation component receipts (bd-exact-generation-component-receipts-rds6v)
// ---------------------------------------------------------------------------

/// Domain separator for the canonical, ENGINE-NEUTRAL ordered document-set
/// digest.
///
/// Deliberately distinct from the index crate's
/// `frankensearch.fsvi-v2.ordered-live-docset.v1`. That digest is computed
/// over FSVI physical rows and is meaningful only inside that format; this one
/// is what a vector index, a lexical index, an ANN sidecar, and a metadata
/// checkpoint can each compute from their own view and be required to agree
/// on. Sharing one domain between the two would let an FSVI-shaped digest
/// silently satisfy a cross-engine comparison it never actually made.
pub const CANONICAL_DOCSET_DIGEST_DOMAIN_V1: &[u8] =
    b"frankensearch.generation.canonical-ordered-docset.v1";

/// Which engine a component receipt speaks for.
///
/// Carried so a rejection can name the role that drifted: the acceptance
/// requires every mutation to be rejected "on the correct role", which a bare
/// boolean or a digest mismatch without attribution cannot express.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenerationComponentRole {
    /// The FSVI vector index.
    Vector,
    /// The lexical (Quill) index.
    Lexical,
    /// The ANN sidecar. Optional: it may be absent or disabled.
    Ann,
    /// The metadata/checkpoint store.
    Metadata,
}

impl GenerationComponentRole {
    /// Stable identifier used in errors and canonical encoding.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Vector => "vector",
            Self::Lexical => "lexical",
            Self::Ann => "ann",
            Self::Metadata => "metadata",
        }
    }

    pub(super) const fn wire(self) -> u8 {
        match self {
            Self::Vector => 1,
            Self::Lexical => 2,
            Self::Ann => 3,
            Self::Metadata => 4,
        }
    }
}

/// The canonical ordered document set a generation was built from.
///
/// Every engine derives this from its own live documents. Order is part of the
/// identity: two components holding the same documents in a different order
/// are not the same generation, because rank-ordered results and physical row
/// addressing both depend on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalDocsetV1 {
    documents: Vec<String>,
}

impl CanonicalDocsetV1 {
    /// Build a canonical document set from live document ids in their exact
    /// generation order.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an empty identifier or a duplicate. A
    /// duplicate is rejected rather than deduplicated: silently collapsing one
    /// would make a corrupted component agree with a healthy one.
    pub fn from_ordered_live_documents<I, S>(
        documents: I,
    ) -> Result<Self, GenerationAuthorityErrorV1>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let documents: Vec<String> = documents.into_iter().map(Into::into).collect();
        let mut seen = std::collections::BTreeSet::new();
        for document in &documents {
            if document.is_empty() {
                return Err(GenerationAuthorityErrorV1::InvalidField {
                    field: "component_receipt.docset.document_id",
                });
            }
            if !seen.insert(document.as_str()) {
                return Err(GenerationAuthorityErrorV1::InvalidField {
                    field: "component_receipt.docset.duplicate_document_id",
                });
            }
        }
        Ok(Self { documents })
    }

    /// Number of live documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Whether the generation has no live documents.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// The canonical digest every component must agree on.
    ///
    /// Length-prefixes both the count and each identifier, so neither a
    /// reordering, a hole, nor a boundary shift between adjacent identifiers
    /// can collide with a healthy set.
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        let mut encoder = CanonicalEncoder::new(CANONICAL_DOCSET_DIGEST_DOMAIN_V1);
        encoder.usize(self.documents.len());
        for document in &self.documents {
            encoder.text(document);
        }
        let mut hasher = Sha256::new();
        hasher.update(&encoder.bytes);
        hasher.finalize().into()
    }
}

/// One identity-complete component receipt.
///
/// Distinct from [`GenerationComponentReceiptV1`], which carries only
/// `byte_len` + `sha256` for the activation-authority lane. Exact bytes prove
/// an artifact was not rewritten; they cannot prove it was built from the same
/// documents and the same source checkpoint as its siblings. This carries
/// both, so a component that is internally perfect but belongs to a different
/// generation is still rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactComponentReceiptV1 {
    /// Which engine this receipt speaks for.
    pub role: GenerationComponentRole,
    /// Exact bytes of the component artifact.
    pub bytes: GenerationComponentReceiptV1,
    /// Canonical ordered document-set digest this component was built from.
    pub docset_digest: [u8; 32],
    /// Live document count, carried separately so a same-count substitution is
    /// distinguishable from a count drift in diagnostics.
    pub live_document_count: u64,
    /// The source checkpoint every component of one generation shares.
    pub source_checkpoint: [u8; 32],
}

impl ExactComponentReceiptV1 {
    /// Validate this receipt in isolation.
    ///
    /// # Errors
    ///
    /// Returns a typed error when the receipt cannot identify real bytes, or
    /// when a digest or checkpoint is the all-zero placeholder.
    pub fn validate(&self) -> Result<(), GenerationAuthorityErrorV1> {
        self.bytes.validate()?;
        if self.docset_digest == [0; 32] {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "component_receipt.docset_digest",
            });
        }
        if self.source_checkpoint == [0; 32] {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "component_receipt.source_checkpoint",
            });
        }
        Ok(())
    }

    pub(super) fn encode(&self, encoder: &mut CanonicalEncoder) {
        encoder.u8(self.role.wire());
        self.bytes.encode(encoder);
        encoder.bytes(&self.docset_digest);
        encoder.u64(self.live_document_count);
        encoder.bytes(&self.source_checkpoint);
    }
}

/// Why a composite generation was refused, and by which role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ComponentJoinErrorV1 {
    /// One component disagreed on the canonical document set.
    #[error("{role} component was built from a different document set")]
    DocsetDrift {
        /// The role whose digest disagreed with the vector anchor.
        role: &'static str,
    },
    /// One component disagreed on the live-document census.
    #[error("{role} component reports a different live-document count")]
    LiveDocumentCountDrift {
        /// The role whose count disagreed with the vector anchor.
        role: &'static str,
    },
    /// One component disagreed on the shared source checkpoint.
    #[error("{role} component was built from a different source checkpoint")]
    CheckpointDrift {
        /// The role whose checkpoint disagreed with the vector anchor.
        role: &'static str,
    },
    /// A receipt was filed under the wrong role.
    #[error("{expected} slot holds a {found} component receipt")]
    RoleMismatch {
        /// Slot the receipt was filed in.
        expected: &'static str,
        /// Role the receipt actually declares.
        found: &'static str,
    },
    /// A receipt failed its own validation.
    #[error("{role} component receipt is invalid: {source}")]
    InvalidComponent {
        /// The offending role.
        role: &'static str,
        /// Underlying field error.
        #[source]
        source: GenerationAuthorityErrorV1,
    },
}

/// Four component receipts proven to describe ONE generation.
///
/// The ANN slot is optional because ANN is an optional accelerator: a
/// generation with a stale or absent sidecar is still exact, provided the
/// mandatory roles agree. Vector, lexical, and metadata drift always fails
/// closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactGenerationComponentsV1 {
    vector: ExactComponentReceiptV1,
    lexical: ExactComponentReceiptV1,
    ann: Option<ExactComponentReceiptV1>,
    metadata: ExactComponentReceiptV1,
}

impl ExactGenerationComponentsV1 {
    /// Admit four component receipts as one generation.
    ///
    /// The vector receipt is the anchor: it is the only role that owns the
    /// identity-complete FSVI witness, so the others are compared against it
    /// rather than against each other. Comparing pairwise would let two
    /// equally-wrong components agree and outvote the anchor.
    ///
    /// # Errors
    ///
    /// Returns [`ComponentJoinErrorV1`] naming the role that drifted, its
    /// declared role when filed in the wrong slot, or its own field error.
    pub fn admit(
        vector: ExactComponentReceiptV1,
        lexical: ExactComponentReceiptV1,
        ann: Option<ExactComponentReceiptV1>,
        metadata: ExactComponentReceiptV1,
    ) -> Result<Self, ComponentJoinErrorV1> {
        for (slot, receipt) in [
            (GenerationComponentRole::Vector, &vector),
            (GenerationComponentRole::Lexical, &lexical),
            (GenerationComponentRole::Metadata, &metadata),
        ]
        .into_iter()
        .chain(
            ann.as_ref()
                .map(|receipt| (GenerationComponentRole::Ann, receipt)),
        ) {
            if receipt.role != slot {
                return Err(ComponentJoinErrorV1::RoleMismatch {
                    expected: slot.as_str(),
                    found: receipt.role.as_str(),
                });
            }
            receipt
                .validate()
                .map_err(|source| ComponentJoinErrorV1::InvalidComponent {
                    role: slot.as_str(),
                    source,
                })?;
        }

        let anchor_docset = vector.docset_digest;
        let anchor_live_document_count = vector.live_document_count;
        let anchor_checkpoint = vector.source_checkpoint;
        for receipt in [Some(&lexical), ann.as_ref(), Some(&metadata)]
            .into_iter()
            .flatten()
        {
            if receipt.docset_digest != anchor_docset {
                return Err(ComponentJoinErrorV1::DocsetDrift {
                    role: receipt.role.as_str(),
                });
            }
            if receipt.live_document_count != anchor_live_document_count {
                return Err(ComponentJoinErrorV1::LiveDocumentCountDrift {
                    role: receipt.role.as_str(),
                });
            }
            if receipt.source_checkpoint != anchor_checkpoint {
                return Err(ComponentJoinErrorV1::CheckpointDrift {
                    role: receipt.role.as_str(),
                });
            }
        }

        Ok(Self {
            vector,
            lexical,
            ann,
            metadata,
        })
    }

    /// The canonical document-set digest every admitted component agrees on.
    #[must_use]
    pub const fn docset_digest(&self) -> [u8; 32] {
        self.vector.docset_digest
    }

    /// The shared source checkpoint.
    #[must_use]
    pub const fn source_checkpoint(&self) -> [u8; 32] {
        self.vector.source_checkpoint
    }

    /// Whether an ANN component was admitted alongside the mandatory roles.
    #[must_use]
    pub const fn has_ann(&self) -> bool {
        self.ann.is_some()
    }

    /// Vector component receipt.
    #[must_use]
    pub const fn vector(&self) -> &ExactComponentReceiptV1 {
        &self.vector
    }

    /// Lexical component receipt.
    #[must_use]
    pub const fn lexical(&self) -> &ExactComponentReceiptV1 {
        &self.lexical
    }

    /// Metadata component receipt.
    #[must_use]
    pub const fn metadata(&self) -> &ExactComponentReceiptV1 {
        &self.metadata
    }

    /// ANN component receipt, when one was admitted.
    #[must_use]
    pub const fn ann(&self) -> Option<&ExactComponentReceiptV1> {
        self.ann.as_ref()
    }

    /// Digest binding the admitted composite, for a manifest to address.
    #[must_use]
    pub fn composite_digest(&self) -> [u8; 32] {
        let mut encoder = CanonicalEncoder::new(b"frankensearch.generation.exact-components.v1");
        self.vector.encode(&mut encoder);
        self.lexical.encode(&mut encoder);
        encoder.option(self.ann.as_ref(), |receipt, encoder| {
            receipt.encode(encoder);
        });
        self.metadata.encode(&mut encoder);
        let mut hasher = Sha256::new();
        hasher.update(&encoder.bytes);
        hasher.finalize().into()
    }
}

/// Domain separating the shared source checkpoint from every other digest.
///
/// Deliberately distinct from [`CANONICAL_DOCSET_DIGEST_DOMAIN_V1`] for the
/// same reason that one is distinct from the fsvi-v2 docset domain: a
/// checkpoint and a document-set digest are both 32 bytes derived from one
/// generation, and sharing a domain would let a component substitute one for
/// the other and still satisfy a comparison it never made.
pub const GENERATION_SOURCE_CHECKPOINT_DOMAIN_V1: &[u8] =
    b"frankensearch.generation.source-checkpoint.v1";

/// The shared source checkpoint every component of one generation carries.
///
/// [`ExactComponentReceiptV1::source_checkpoint`] is contractually "the same 32
/// bytes every role shares", but the keystone left the derivation open, and the
/// four producers then diverged: the metadata receipt derived its value while
/// the vector, ANN, and lexical producers accepted any `[u8; 32]` a caller
/// handed them (bd-z4zr3).
///
/// That was not merely untidy. `ExactGenerationComponentsV1::admit` compares
/// every role against the VECTOR anchor, so a caller passing an arbitrary
/// checkpoint to the three accepting producers made the one role that computed
/// its checkpoint correctly — metadata — the role reported as drifting. The
/// component that was right got the blame.
///
/// This type removes the failure at its source. It can only be constructed by
/// derivation from the commit range that produced the generation, so "an
/// arbitrary 32 bytes" is not a value any producer can be given. The wrong
/// checkpoint is unrepresentable rather than merely discouraged, which is the
/// difference between a contract and a comment.
///
/// # The guarantee is compile-time, so it is pinned at compile time
///
/// No runtime test can observe this property: a test can only exercise
/// constructors that exist, and the guarantee here is the ABSENCE of one. I
/// verified that directly — adding a `from_bytes` escape hatch to this type
/// reddens no test in the workspace, because nothing calls it; it merely makes
/// the hole available again for the next caller. The doctest below is therefore
/// the actual enforcement, and the error code is pinned so a snippet that stops
/// compiling for an unrelated reason fails it instead of passing.
///
/// ```compile_fail,E0423
/// use frankensearch_core::generation::SourceCheckpointV1;
///
/// // The field is private: bytes cannot become a checkpoint by assertion.
/// let forged = SourceCheckpointV1([0x5c; 32]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceCheckpointV1([u8; 32]);

impl SourceCheckpointV1 {
    /// Derive the checkpoint from the commit range that produced the generation.
    ///
    /// This is the ONLY constructor. There is deliberately no
    /// `from_bytes`/`from_raw`: every such escape hatch reintroduces exactly the
    /// split this type exists to close.
    #[must_use]
    pub fn derive(commit_range: &CommitRange) -> Self {
        let mut encoder = CanonicalEncoder::new(GENERATION_SOURCE_CHECKPOINT_DOMAIN_V1);
        encoder.u64(commit_range.low);
        encoder.u64(commit_range.high);
        let mut hasher = Sha256::new();
        hasher.update(&encoder.bytes);
        Self(hasher.finalize().into())
    }

    /// The 32 bytes a receipt carries on the wire.
    ///
    /// One-way on purpose: a receipt's field stays `[u8; 32]` so the serialized
    /// schema is unchanged, but bytes cannot travel back into a checkpoint.
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Derive the shared source checkpoint from a generation's commit range.
///
/// Prefer [`SourceCheckpointV1::derive`], which is the same derivation in a form
/// that cannot be confused with an arbitrary array. This returns raw bytes and
/// remains for callers comparing against a receipt's stored field.
#[must_use]
pub fn generation_source_checkpoint_v1(commit_range: &CommitRange) -> [u8; 32] {
    SourceCheckpointV1::derive(commit_range).to_bytes()
}

impl ExactComponentReceiptV1 {
    /// Build the metadata/checkpoint component receipt from a real manifest.
    ///
    /// The metadata role is the only component whose own artifact states the
    /// commit range, so this deliberately does NOT accept a caller-supplied
    /// checkpoint: a metadata receipt naming a checkpoint its manifest does not
    /// support is a contradiction rather than a drift to be discovered later at
    /// the join.
    ///
    /// `manifest_bytes` must be the exact immutable bytes that were or will be
    /// published for `manifest`. They are bound by length and SHA-256, so a
    /// manifest that is semantically equal but was rewritten produces a
    /// different receipt.
    ///
    /// `documents` is the caller's own ordered live document set, per the
    /// keystone contract: run [`CanonicalDocsetV1::from_ordered_live_documents`]
    /// over your ordered live ids rather than relabelling an engine digest.
    ///
    /// # Errors
    ///
    /// Returns a typed error when the manifest bytes are empty, or when the
    /// manifest's own `total_documents` disagrees with the canonical document
    /// set. The count cross-check exists because metadata is the only role
    /// carrying a document count independent of the docset it is claiming; a
    /// disagreement there is a defect in the receipt itself, and letting it
    /// form would push a diagnosable error into the join as an anonymous
    /// digest mismatch.
    pub fn for_metadata_manifest(
        manifest: &GenerationManifest,
        manifest_bytes: &[u8],
        documents: &CanonicalDocsetV1,
    ) -> Result<Self, GenerationAuthorityErrorV1> {
        if manifest_bytes.is_empty() {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "component_receipt.metadata.manifest_bytes",
            });
        }
        let live_document_count = u64::try_from(documents.len()).map_err(|_| {
            GenerationAuthorityErrorV1::InvalidField {
                field: "component_receipt.metadata.live_document_count",
            }
        })?;
        if manifest.total_documents != live_document_count {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "component_receipt.metadata.total_documents",
            });
        }

        let byte_len = u64::try_from(manifest_bytes.len()).map_err(|_| {
            GenerationAuthorityErrorV1::InvalidField {
                field: "component_receipt.metadata.byte_len",
            }
        })?;
        let sha256: [u8; 32] = Sha256::digest(manifest_bytes).into();

        Ok(Self {
            role: GenerationComponentRole::Metadata,
            bytes: GenerationComponentReceiptV1 { byte_len, sha256 },
            docset_digest: documents.digest(),
            live_document_count,
            source_checkpoint: generation_source_checkpoint_v1(&manifest.commit_range),
        })
    }

    /// Verify a stored metadata receipt against the manifest image a reader
    /// actually has, naming what drifted.
    ///
    /// This is the consumer half of [`Self::for_metadata_manifest`]. A receipt
    /// that cannot be re-checked against the artifact later only proves what
    /// was true at build time, which is exactly the window an artifact gets
    /// rewritten in.
    ///
    /// # What this can and cannot prove
    ///
    /// Everything checked here is derivable from the manifest image ALONE:
    /// the role, the exact bytes, the shared checkpoint (from the manifest's
    /// own `commit_range`), and the live document count (against its own
    /// `total_documents`).
    ///
    /// The canonical docset digest is deliberately NOT checked here, and that
    /// is not an oversight. `GenerationManifest` records document *counts*, not
    /// document *identifiers*, so the docset digest cannot be re-derived from
    /// the manifest at all. It is a claim the metadata role makes about a
    /// document set it does not itself enumerate, and the only thing that can
    /// falsify it is agreement with the vector anchor in
    /// [`ExactGenerationComponentsV1::admit`] -- which is precisely why that
    /// join exists. Use [`Self::verify_metadata_docset`] when the caller holds
    /// the ordered live ids, and the join for cross-engine proof. A verifier
    /// that silently "checked" the docset against a caller-supplied set would
    /// be asserting the caller's word back to itself.
    ///
    /// # Errors
    ///
    /// Returns [`MetadataReceiptDriftV1`] naming the first field that
    /// disagrees. Field order is deliberate: identity before content, so a
    /// receipt filed under the wrong role is reported as such rather than as a
    /// byte mismatch.
    pub fn verify_metadata_manifest(
        &self,
        manifest: &GenerationManifest,
        manifest_bytes: &[u8],
    ) -> Result<(), MetadataReceiptDriftV1> {
        if self.role != GenerationComponentRole::Metadata {
            return Err(MetadataReceiptDriftV1::NotAMetadataReceipt {
                found: self.role.as_str(),
            });
        }

        // The caller passes a parsed manifest AND its bytes. If those two
        // disagree the pair is incoherent, and every check below would be
        // measuring two different generations against one receipt. Compare the
        // decoded image rather than re-serializing: a manifest may legitimately
        // be published in an encoding this crate would not reproduce byte for
        // byte, so re-serializing would reject honest images.
        let decoded: GenerationManifest = serde_json::from_slice(manifest_bytes)
            .map_err(|_| MetadataReceiptDriftV1::UnreadableManifestImage)?;
        if decoded.commit_range != manifest.commit_range
            || decoded.total_documents != manifest.total_documents
            || decoded.generation_id != manifest.generation_id
        {
            return Err(MetadataReceiptDriftV1::ManifestImageMismatch);
        }

        let byte_len = u64::try_from(manifest_bytes.len())
            .map_err(|_| MetadataReceiptDriftV1::UnreadableManifestImage)?;
        if self.bytes.byte_len != byte_len {
            return Err(MetadataReceiptDriftV1::ByteLen {
                expected: self.bytes.byte_len,
                found: byte_len,
            });
        }
        if self.bytes.sha256 != <[u8; 32]>::from(Sha256::digest(manifest_bytes)) {
            return Err(MetadataReceiptDriftV1::ManifestBytes);
        }
        if self.source_checkpoint != generation_source_checkpoint_v1(&manifest.commit_range) {
            return Err(MetadataReceiptDriftV1::SourceCheckpoint);
        }
        if self.live_document_count != manifest.total_documents {
            return Err(MetadataReceiptDriftV1::LiveDocumentCount {
                expected: self.live_document_count,
                found: manifest.total_documents,
            });
        }
        Ok(())
    }

    /// Verify the receipt's docset claim against ordered live ids the caller
    /// holds independently of the manifest.
    ///
    /// Separate from [`Self::verify_metadata_manifest`] because it needs an
    /// input the artifact does not carry. Passing the same set that produced
    /// the receipt proves nothing; this is for a caller that obtained the ids
    /// from the corpus, not from the receipt.
    ///
    /// # Errors
    ///
    /// Returns [`MetadataReceiptDriftV1`] when the role is wrong, the digest
    /// disagrees, or the count disagrees.
    pub fn verify_metadata_docset(
        &self,
        documents: &CanonicalDocsetV1,
    ) -> Result<(), MetadataReceiptDriftV1> {
        if self.role != GenerationComponentRole::Metadata {
            return Err(MetadataReceiptDriftV1::NotAMetadataReceipt {
                found: self.role.as_str(),
            });
        }
        if self.docset_digest != documents.digest() {
            return Err(MetadataReceiptDriftV1::Docset);
        }
        let live = u64::try_from(documents.len())
            .map_err(|_| MetadataReceiptDriftV1::UnreadableManifestImage)?;
        if self.live_document_count != live {
            return Err(MetadataReceiptDriftV1::LiveDocumentCount {
                expected: self.live_document_count,
                found: live,
            });
        }
        Ok(())
    }
}

/// Why a stored metadata receipt does not describe the artifact presented.
///
/// Distinct from [`ComponentJoinErrorV1`], which answers "do these four
/// components describe one generation". This answers "does this receipt still
/// describe this artifact", which is a question a single reader can ask without
/// holding the other three roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum MetadataReceiptDriftV1 {
    /// The receipt speaks for another role entirely.
    #[error("expected a metadata component receipt, found {found}")]
    NotAMetadataReceipt {
        /// Role the receipt actually declares.
        found: &'static str,
    },
    /// The presented bytes are not a decodable manifest image.
    #[error("manifest image could not be decoded")]
    UnreadableManifestImage,
    /// The parsed manifest and the presented bytes describe different
    /// generations, so the pair cannot be verified against one receipt.
    #[error("parsed manifest and manifest image describe different generations")]
    ManifestImageMismatch,
    /// Exact byte length disagrees.
    #[error("manifest image is {found} bytes, receipt records {expected}")]
    ByteLen {
        /// Length the receipt records.
        expected: u64,
        /// Length actually presented.
        found: u64,
    },
    /// Exact content disagrees at the same length.
    #[error("manifest image content does not match the receipt's SHA-256")]
    ManifestBytes,
    /// The checkpoint the receipt records is not the one this manifest's
    /// commit range derives.
    #[error("receipt names a source checkpoint this manifest does not derive")]
    SourceCheckpoint,
    /// Live document count disagrees.
    #[error("receipt records {expected} live documents, found {found}")]
    LiveDocumentCount {
        /// Count the receipt records.
        expected: u64,
        /// Count actually presented.
        found: u64,
    },
    /// The canonical docset digest disagrees.
    #[error("receipt was built from a different canonical document set")]
    Docset,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
