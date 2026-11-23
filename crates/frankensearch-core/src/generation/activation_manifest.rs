use sha2::{Digest, Sha256};

use super::authority_core::{
    AuthorityRefV1, GENERATION_ACTIVATION_MANIFEST_MAX_BYTES_V1,
    GENERATION_ACTIVATION_MANIFEST_MAX_BYTES_V2, GENERATION_ACTIVATION_MANIFEST_SCHEMA_V2,
    GENERATION_AUTHORITY_SCHEMA_V1, GenerationAuthorityErrorV1,
};
use super::codec::{CanonicalDecoder, CanonicalEncoder};
use super::embedding_identity::ArtifactGenerationIdentityV1;
use super::receipts::GenerationComponentRole;

/// The transition that created an immutable activation manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenerationAuthorityActionV1 {
    /// Select a newly built generation.
    Activate,
    /// Select a previously published generation under a newer authority.
    Rollback,
    /// Publish a repaired replacement for a damaged generation.
    Repair,
    /// Publish a schema-preserving migrated generation.
    Migrate,
}

impl GenerationAuthorityActionV1 {
    const fn tag(self) -> u8 {
        match self {
            Self::Activate => 1,
            Self::Rollback => 2,
            Self::Repair => 3,
            Self::Migrate => 4,
        }
    }
}

/// Immutable receipt for one exact generation component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenerationComponentReceiptV1 {
    /// Exact component byte length.
    pub byte_len: u64,
    /// SHA-256 of the exact component bytes.
    pub sha256: [u8; 32],
}

impl GenerationComponentReceiptV1 {
    /// Validate the bounded immutable component receipt.
    ///
    /// # Errors
    ///
    /// Returns a typed error when the receipt cannot identify real bytes.
    pub fn validate(&self) -> Result<(), GenerationAuthorityErrorV1> {
        if self.byte_len == 0 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.component.byte_len",
            });
        }
        if self.sha256 == [0; 32] {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.component.sha256",
            });
        }
        Ok(())
    }

    pub(super) fn encode(self, encoder: &mut CanonicalEncoder) {
        encoder.u64(self.byte_len);
        encoder.bytes(&self.sha256);
    }
}

/// The four component receipts that make one generation selectable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenerationComponentReceiptsV1 {
    /// Vector-index component receipt.
    pub vector: GenerationComponentReceiptV1,
    /// Lexical-index component receipt.
    pub lexical: GenerationComponentReceiptV1,
    /// ANN component receipt.
    pub ann: GenerationComponentReceiptV1,
    /// Metadata component receipt.
    pub metadata: GenerationComponentReceiptV1,
}

impl GenerationComponentReceiptsV1 {
    fn validate(self) -> Result<(), GenerationAuthorityErrorV1> {
        self.vector.validate()?;
        self.lexical.validate()?;
        self.ann.validate()?;
        self.metadata.validate()
    }

    fn encode(self, encoder: &mut CanonicalEncoder) {
        self.vector.encode(encoder);
        self.lexical.encode(encoder);
        self.ann.encode(encoder);
        self.metadata.encode(encoder);
    }
}

/// Immutable payload addressed by [`AuthorityRefV1`].
///
/// It is pure data: publishing the payload, selecting its object ID, and any
/// filesystem action belong to the generation-root publisher lane. The
/// self-seal detects an in-memory or decoded field substitution before a
/// caller uses the externally addressed manifest digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationManifestV1 {
    /// [`GENERATION_AUTHORITY_SCHEMA_V1`].
    pub schema_version: u16,
    /// Authority sequence this manifest is eligible to serve.
    pub authority_sequence: u64,
    /// Exact preceding authority reference. Genesis has none.
    pub predecessor: Option<AuthorityRefV1>,
    /// Transition that produced this generation selection.
    pub action: GenerationAuthorityActionV1,
    /// Artifact-generation identity selected by this authority transition.
    pub generation: ArtifactGenerationIdentityV1,
    /// Immutable writer-fence witness.
    pub writer_fence_sha256: [u8; 32],
    /// Immutable source-checkpoint witness.
    pub source_checkpoint_sha256: [u8; 32],
    /// Canonical document-set witness.
    pub document_set_sha256: [u8; 32],
    /// Exact vector, lexical, ANN, and metadata component receipts.
    pub components: GenerationComponentReceiptsV1,
    /// SHA-256 self-seal over every preceding field.
    pub self_seal_sha256: [u8; 32],
}

impl ActivationManifestV1 {
    /// Construct and self-seal a canonical immutable activation manifest.
    ///
    /// # Errors
    ///
    /// Returns a typed error when a field or predecessor relation is invalid.
    pub fn new(
        authority_sequence: u64,
        predecessor: Option<AuthorityRefV1>,
        action: GenerationAuthorityActionV1,
        generation: ArtifactGenerationIdentityV1,
        writer_fence_sha256: [u8; 32],
        source_checkpoint_sha256: [u8; 32],
        document_set_sha256: [u8; 32],
        components: GenerationComponentReceiptsV1,
    ) -> Result<Self, GenerationAuthorityErrorV1> {
        let mut manifest = Self {
            schema_version: GENERATION_AUTHORITY_SCHEMA_V1,
            authority_sequence,
            predecessor,
            action,
            generation,
            writer_fence_sha256,
            source_checkpoint_sha256,
            document_set_sha256,
            components,
            self_seal_sha256: [0; 32],
        };
        manifest.validate_unsealed()?;
        manifest.self_seal_sha256 = manifest.computed_self_seal();
        Ok(manifest)
    }

    /// Validate every field and the self-seal without performing I/O.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an invalid field, broken predecessor relation,
    /// or mismatched self-seal.
    pub fn validate(&self) -> Result<(), GenerationAuthorityErrorV1> {
        self.validate_unsealed()?;
        // ubs:ignore — the self-seal is public immutable-integrity evidence, not a secret.
        if self.self_seal_sha256 != self.computed_self_seal() {
            return Err(GenerationAuthorityErrorV1::ManifestSelfSealMismatch);
        }
        Ok(())
    }

    /// Canonical bytes excluding the self-seal field.
    #[must_use]
    pub fn canonical_unsealed_bytes(&self) -> Vec<u8> {
        let mut encoder = CanonicalEncoder::new(b"frankensearch.activation-manifest.v1");
        encoder.u16(self.schema_version);
        encoder.u64(self.authority_sequence);
        encoder.u8(self.action.tag());
        encoder.option(self.predecessor.as_ref(), |predecessor, encoder| {
            encoder.u16(predecessor.schema_version);
            encoder.u64(predecessor.sequence);
            encoder.bytes(&predecessor.object_id);
            encoder.u64(predecessor.manifest_len);
            encoder.bytes(&predecessor.manifest_sha256);
            encoder.option(predecessor.predecessor.as_ref(), |ancestor, encoder| {
                encoder.bytes(ancestor);
            });
        });
        encoder.u16(self.generation.schema_version);
        encoder.u64(self.generation.sequence);
        encoder.bytes(&self.generation.nonce);
        encoder.bytes(&self.writer_fence_sha256);
        encoder.bytes(&self.source_checkpoint_sha256);
        encoder.bytes(&self.document_set_sha256);
        self.components.encode(&mut encoder);
        encoder.finish()
    }

    /// Canonical bytes including the validated self-seal.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = self.canonical_unsealed_bytes();
        bytes.extend_from_slice(&self.self_seal_sha256);
        bytes
    }

    /// Parse, validate, and re-encode one bounded canonical activation
    /// manifest. The input must match [`Self::canonical_bytes`] byte-for-byte.
    ///
    /// # Errors
    ///
    /// Returns a typed error for truncated, extended, future, malformed, or
    /// non-canonical manifest bytes without allocating based on input lengths.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, GenerationAuthorityErrorV1> {
        if bytes.len() < 32 || bytes.len() > GENERATION_ACTIVATION_MANIFEST_MAX_BYTES_V1 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.canonical_bytes",
            });
        }
        let unsealed_len = bytes.len() - 32;
        let (unsealed, seal) = bytes.split_at(unsealed_len);
        let mut decoder = CanonicalDecoder::new(unsealed);
        if decoder.bytes("activation_manifest.domain", 64)?
            != b"frankensearch.activation-manifest.v1"
        {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.domain",
            });
        }
        let schema_version = decoder.u16("activation_manifest.schema_version")?;
        let authority_sequence = decoder.u64("activation_manifest.authority_sequence")?;
        let action = match decoder.u8("activation_manifest.action")? {
            1 => GenerationAuthorityActionV1::Activate,
            2 => GenerationAuthorityActionV1::Rollback,
            3 => GenerationAuthorityActionV1::Repair,
            4 => GenerationAuthorityActionV1::Migrate,
            _ => {
                return Err(GenerationAuthorityErrorV1::InvalidField {
                    field: "activation_manifest.action",
                });
            }
        };
        let predecessor = match decoder.u8("activation_manifest.predecessor.present")? {
            0 => None,
            1 => Some(AuthorityRefV1 {
                schema_version: decoder.u16("activation_manifest.predecessor.schema_version")?,
                sequence: decoder.u64("activation_manifest.predecessor.sequence")?,
                object_id: decoder.fixed_bytes("activation_manifest.predecessor.object_id")?,
                manifest_len: decoder.u64("activation_manifest.predecessor.manifest_len")?,
                manifest_sha256: decoder
                    .fixed_bytes("activation_manifest.predecessor.manifest_sha256")?,
                predecessor: match decoder.u8("activation_manifest.predecessor.ancestor.present")? {
                    0 => None,
                    1 => Some(decoder.fixed_bytes("activation_manifest.predecessor.ancestor")?),
                    _ => {
                        return Err(GenerationAuthorityErrorV1::InvalidField {
                            field: "activation_manifest.predecessor.ancestor.present",
                        });
                    }
                },
            }),
            _ => {
                return Err(GenerationAuthorityErrorV1::InvalidField {
                    field: "activation_manifest.predecessor.present",
                });
            }
        };
        let generation = ArtifactGenerationIdentityV1 {
            schema_version: decoder.u16("activation_manifest.generation.schema_version")?,
            sequence: decoder.u64("activation_manifest.generation.sequence")?,
            nonce: decoder.fixed_bytes("activation_manifest.generation.nonce")?,
        };
        let writer_fence_sha256 = decoder.fixed_bytes("activation_manifest.writer_fence_sha256")?;
        let source_checkpoint_sha256 =
            decoder.fixed_bytes("activation_manifest.source_checkpoint_sha256")?;
        let document_set_sha256 = decoder.fixed_bytes("activation_manifest.document_set_sha256")?;
        let components = GenerationComponentReceiptsV1 {
            vector: decoder.component("activation_manifest.components.vector")?,
            lexical: decoder.component("activation_manifest.components.lexical")?,
            ann: decoder.component("activation_manifest.components.ann")?,
            metadata: decoder.component("activation_manifest.components.metadata")?,
        };
        decoder.finish()?;
        let mut self_seal_sha256 = [0_u8; 32];
        self_seal_sha256.copy_from_slice(seal);
        let manifest = Self {
            schema_version,
            authority_sequence,
            predecessor,
            action,
            generation,
            writer_fence_sha256,
            source_checkpoint_sha256,
            document_set_sha256,
            components,
            self_seal_sha256,
        };
        manifest.validate()?;
        if !manifest.canonical_bytes().eq(bytes) {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.canonical_bytes",
            });
        }
        Ok(manifest)
    }

    /// Exact length and SHA-256 used by the external [`AuthorityRefV1`].
    #[must_use]
    pub fn object_receipt(&self) -> (u64, [u8; 32]) {
        let bytes = self.canonical_bytes();
        (
            u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            Sha256::digest(bytes).into(),
        )
    }

    fn validate_unsealed(&self) -> Result<(), GenerationAuthorityErrorV1> {
        if self.schema_version != GENERATION_AUTHORITY_SCHEMA_V1 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.schema_version",
            });
        }
        if self.authority_sequence == 0 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.authority_sequence",
            });
        }
        match (self.authority_sequence, self.predecessor) {
            (1, None) => {}
            (1, Some(_)) => {
                return Err(GenerationAuthorityErrorV1::InvalidField {
                    field: "activation_manifest.predecessor",
                });
            }
            (_, None) => {
                return Err(GenerationAuthorityErrorV1::InvalidField {
                    field: "activation_manifest.predecessor",
                });
            }
            (sequence, Some(predecessor)) => {
                predecessor.validate()?;
                if predecessor.sequence.checked_add(1) != Some(sequence) {
                    return Err(GenerationAuthorityErrorV1::BrokenPredecessorLink);
                }
            }
        }
        self.generation
            .validate()
            .map_err(|_| GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.generation",
            })?;
        for (field, digest) in [
            (
                "activation_manifest.writer_fence_sha256",
                self.writer_fence_sha256,
            ),
            (
                "activation_manifest.source_checkpoint_sha256",
                self.source_checkpoint_sha256,
            ),
            (
                "activation_manifest.document_set_sha256",
                self.document_set_sha256,
            ),
        ] {
            // ubs:ignore — component digests are public immutable-integrity identities.
            if digest == [0; 32] {
                return Err(GenerationAuthorityErrorV1::InvalidField { field });
            }
        }
        self.components.validate()
    }

    pub(super) fn computed_self_seal(&self) -> [u8; 32] {
        Sha256::digest(self.canonical_unsealed_bytes()).into()
    }
}

/// Verify that an immutable activation manifest is the exact object named by
/// one authority reference.
///
/// The caller supplies bytes/object storage outside this pure helper. This
/// check binds the authority sequence, predecessor fingerprint, exact object
/// length, and exact object digest before a consumer can select the manifest.
///
/// # Errors
///
/// Returns a typed error for malformed inputs or any reference/manifest
/// mismatch. It never falls back to a different manifest.
pub fn verify_authority_manifest_reference_v1(
    authority: &AuthorityRefV1,
    manifest: &ActivationManifestV1,
) -> Result<(), GenerationAuthorityErrorV1> {
    authority.validate()?;
    manifest.validate()?;
    // ubs:ignore — authority sequence/predecessor fingerprints are public integrity identities.
    if authority.sequence != manifest.authority_sequence
        || authority.predecessor
            != manifest
                .predecessor
                .map(|predecessor| predecessor.fingerprint())
    {
        return Err(GenerationAuthorityErrorV1::ManifestReferenceMismatch);
    }
    let (manifest_len, manifest_sha256) = manifest.object_receipt();
    // ubs:ignore — manifest lengths and digests are public immutable object identities.
    if authority.manifest_len != manifest_len || authority.manifest_sha256 != manifest_sha256 {
        return Err(GenerationAuthorityErrorV1::ManifestReferenceMismatch);
    }
    Ok(())
}

/// Mandatory exact components and an explicitly optional ANN accelerator.
///
/// Absence is authenticated by the schema-2 manifest. A present receipt is
/// always validated and admitted; an invalid accelerator is never silently
/// converted to absence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenerationComponentReceiptsV2 {
    /// Vector-index component receipt.
    pub vector: GenerationComponentReceiptV1,
    /// Lexical-index component receipt.
    pub lexical: GenerationComponentReceiptV1,
    /// ANN receipt, or an explicit exact-search generation with no accelerator.
    pub ann: Option<GenerationComponentReceiptV1>,
    /// Metadata component receipt.
    pub metadata: GenerationComponentReceiptV1,
}

impl GenerationComponentReceiptsV2 {
    /// Declared receipt for one role. Only ANN can be absent.
    #[must_use]
    pub const fn get(self, role: GenerationComponentRole) -> Option<GenerationComponentReceiptV1> {
        match role {
            GenerationComponentRole::Vector => Some(self.vector),
            GenerationComponentRole::Lexical => Some(self.lexical),
            GenerationComponentRole::Ann => self.ann,
            GenerationComponentRole::Metadata => Some(self.metadata),
        }
    }

    /// Declared roles in canonical order, with no invented receipt for absence.
    pub fn iter(
        self,
    ) -> impl Iterator<Item = (GenerationComponentRole, GenerationComponentReceiptV1)> {
        [
            (GenerationComponentRole::Vector, Some(self.vector)),
            (GenerationComponentRole::Lexical, Some(self.lexical)),
            (GenerationComponentRole::Ann, self.ann),
            (GenerationComponentRole::Metadata, Some(self.metadata)),
        ]
        .into_iter()
        .filter_map(|(role, receipt)| receipt.map(|receipt| (role, receipt)))
    }

    fn validate(self) -> Result<(), GenerationAuthorityErrorV1> {
        for (_, receipt) in self.iter() {
            receipt.validate()?;
        }
        Ok(())
    }

    fn encode(self, encoder: &mut CanonicalEncoder) {
        self.vector.encode(encoder);
        self.lexical.encode(encoder);
        encoder.option(self.ann.as_ref(), |receipt, encoder| {
            receipt.encode(encoder);
        });
        self.metadata.encode(encoder);
    }
}

/// Schema-2 immutable activation payload, with authenticated ANN presence.
///
/// Authority references, slot frames, writer fences and anti-rollback floors
/// retain their existing schemas. This version changes only the addressed
/// manifest domain, schema and component encoding. Schema-1 objects remain
/// immutable and are decoded through [`ActivationManifest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationManifestV2 {
    /// [`GENERATION_ACTIVATION_MANIFEST_SCHEMA_V2`].
    pub schema_version: u16,
    /// Authority sequence this manifest is eligible to serve.
    pub authority_sequence: u64,
    /// Exact preceding authority reference. Genesis has none.
    pub predecessor: Option<AuthorityRefV1>,
    /// Transition that produced this generation selection.
    pub action: GenerationAuthorityActionV1,
    /// Artifact generation selected by this authority transition.
    pub generation: ArtifactGenerationIdentityV1,
    /// Immutable writer-fence witness.
    pub writer_fence_sha256: [u8; 32],
    /// Immutable source-checkpoint witness.
    pub source_checkpoint_sha256: [u8; 32],
    /// Canonical document-set witness.
    pub document_set_sha256: [u8; 32],
    /// All mandatory roles and the explicit ANN selection.
    pub components: GenerationComponentReceiptsV2,
    /// SHA-256 self-seal over every preceding field, including ANN presence.
    pub self_seal_sha256: [u8; 32],
}

impl ActivationManifestV2 {
    /// Construct and self-seal a manifest with an explicit ANN selection.
    ///
    /// # Errors
    ///
    /// Rejects invalid component receipts, identities or predecessor relations.
    pub fn new(
        authority_sequence: u64,
        predecessor: Option<AuthorityRefV1>,
        action: GenerationAuthorityActionV1,
        generation: ArtifactGenerationIdentityV1,
        writer_fence_sha256: [u8; 32],
        source_checkpoint_sha256: [u8; 32],
        document_set_sha256: [u8; 32],
        components: GenerationComponentReceiptsV2,
    ) -> Result<Self, GenerationAuthorityErrorV1> {
        let mut manifest = Self {
            schema_version: GENERATION_ACTIVATION_MANIFEST_SCHEMA_V2,
            authority_sequence,
            predecessor,
            action,
            generation,
            writer_fence_sha256,
            source_checkpoint_sha256,
            document_set_sha256,
            components,
            self_seal_sha256: [0; 32],
        };
        manifest.validate_unsealed()?;
        manifest.self_seal_sha256 = Sha256::digest(manifest.canonical_unsealed_bytes()).into();
        Ok(manifest)
    }

    /// Validate every field and the authenticated ANN presence without I/O.
    ///
    /// # Errors
    ///
    /// Rejects invalid fields, predecessor relations and self-seal substitutions.
    pub fn validate(&self) -> Result<(), GenerationAuthorityErrorV1> {
        self.validate_unsealed()?;
        let computed: [u8; 32] = Sha256::digest(self.canonical_unsealed_bytes()).into();
        // ubs:ignore — this self-seal is public immutable-integrity evidence.
        if self.self_seal_sha256 != computed {
            return Err(GenerationAuthorityErrorV1::ManifestSelfSealMismatch);
        }
        Ok(())
    }

    /// Canonical schema-2 bytes excluding the self-seal field.
    #[must_use]
    pub fn canonical_unsealed_bytes(&self) -> Vec<u8> {
        let mut encoder = CanonicalEncoder::new(b"frankensearch.activation-manifest.v2");
        encoder.u16(self.schema_version);
        encoder.u64(self.authority_sequence);
        encoder.u8(self.action.tag());
        encoder.option(self.predecessor.as_ref(), |predecessor, encoder| {
            encoder.u16(predecessor.schema_version);
            encoder.u64(predecessor.sequence);
            encoder.bytes(&predecessor.object_id);
            encoder.u64(predecessor.manifest_len);
            encoder.bytes(&predecessor.manifest_sha256);
            encoder.option(predecessor.predecessor.as_ref(), |ancestor, encoder| {
                encoder.bytes(ancestor);
            });
        });
        encoder.u16(self.generation.schema_version);
        encoder.u64(self.generation.sequence);
        encoder.bytes(&self.generation.nonce);
        encoder.bytes(&self.writer_fence_sha256);
        encoder.bytes(&self.source_checkpoint_sha256);
        encoder.bytes(&self.document_set_sha256);
        self.components.encode(&mut encoder);
        encoder.finish()
    }

    /// Canonical schema-2 bytes including the self-seal.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = self.canonical_unsealed_bytes();
        bytes.extend_from_slice(&self.self_seal_sha256);
        bytes
    }

    /// Parse one bounded canonical schema-2 manifest without length-driven allocation.
    ///
    /// # Errors
    ///
    /// Rejects malformed presence tags, missing declared receipts, extra
    /// receipts after absence, schema substitutions and noncanonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, GenerationAuthorityErrorV1> {
        if bytes.len() < 32 || bytes.len() > GENERATION_ACTIVATION_MANIFEST_MAX_BYTES_V2 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.canonical_bytes",
            });
        }
        let (unsealed, seal) = bytes.split_at(bytes.len() - 32);
        let mut decoder = CanonicalDecoder::new(unsealed);
        if decoder.bytes("activation_manifest.domain", 64)?
            != b"frankensearch.activation-manifest.v2"
        {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.domain",
            });
        }
        let schema_version = decoder.u16("activation_manifest.schema_version")?;
        let authority_sequence = decoder.u64("activation_manifest.authority_sequence")?;
        let action = match decoder.u8("activation_manifest.action")? {
            1 => GenerationAuthorityActionV1::Activate,
            2 => GenerationAuthorityActionV1::Rollback,
            3 => GenerationAuthorityActionV1::Repair,
            4 => GenerationAuthorityActionV1::Migrate,
            _ => {
                return Err(GenerationAuthorityErrorV1::InvalidField {
                    field: "activation_manifest.action",
                });
            }
        };
        let predecessor = match decoder.u8("activation_manifest.predecessor.present")? {
            0 => None,
            1 => Some(AuthorityRefV1 {
                schema_version: decoder.u16("activation_manifest.predecessor.schema_version")?,
                sequence: decoder.u64("activation_manifest.predecessor.sequence")?,
                object_id: decoder.fixed_bytes("activation_manifest.predecessor.object_id")?,
                manifest_len: decoder.u64("activation_manifest.predecessor.manifest_len")?,
                manifest_sha256: decoder
                    .fixed_bytes("activation_manifest.predecessor.manifest_sha256")?,
                predecessor: match decoder.u8("activation_manifest.predecessor.ancestor.present")? {
                    0 => None,
                    1 => Some(decoder.fixed_bytes("activation_manifest.predecessor.ancestor")?),
                    _ => {
                        return Err(GenerationAuthorityErrorV1::InvalidField {
                            field: "activation_manifest.predecessor.ancestor.present",
                        });
                    }
                },
            }),
            _ => {
                return Err(GenerationAuthorityErrorV1::InvalidField {
                    field: "activation_manifest.predecessor.present",
                });
            }
        };
        let generation = ArtifactGenerationIdentityV1 {
            schema_version: decoder.u16("activation_manifest.generation.schema_version")?,
            sequence: decoder.u64("activation_manifest.generation.sequence")?,
            nonce: decoder.fixed_bytes("activation_manifest.generation.nonce")?,
        };
        let writer_fence_sha256 = decoder.fixed_bytes("activation_manifest.writer_fence_sha256")?;
        let source_checkpoint_sha256 =
            decoder.fixed_bytes("activation_manifest.source_checkpoint_sha256")?;
        let document_set_sha256 = decoder.fixed_bytes("activation_manifest.document_set_sha256")?;
        let components = GenerationComponentReceiptsV2 {
            vector: decoder.component("activation_manifest.components.vector")?,
            lexical: decoder.component("activation_manifest.components.lexical")?,
            ann: match decoder.u8("activation_manifest.components.ann.present")? {
                0 => None,
                1 => Some(decoder.component("activation_manifest.components.ann")?),
                _ => {
                    return Err(GenerationAuthorityErrorV1::InvalidField {
                        field: "activation_manifest.components.ann.present",
                    });
                }
            },
            metadata: decoder.component("activation_manifest.components.metadata")?,
        };
        decoder.finish()?;
        let mut self_seal_sha256 = [0; 32];
        self_seal_sha256.copy_from_slice(seal);
        let manifest = Self {
            schema_version,
            authority_sequence,
            predecessor,
            action,
            generation,
            writer_fence_sha256,
            source_checkpoint_sha256,
            document_set_sha256,
            components,
            self_seal_sha256,
        };
        manifest.validate()?;
        if !manifest.canonical_bytes().eq(bytes) {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.canonical_bytes",
            });
        }
        Ok(manifest)
    }

    /// Exact length and SHA-256 addressed by the existing authority reference.
    #[must_use]
    pub fn object_receipt(&self) -> (u64, [u8; 32]) {
        let bytes = self.canonical_bytes();
        (
            u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            Sha256::digest(bytes).into(),
        )
    }

    fn validate_unsealed(&self) -> Result<(), GenerationAuthorityErrorV1> {
        if self.schema_version != GENERATION_ACTIVATION_MANIFEST_SCHEMA_V2 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.schema_version",
            });
        }
        if self.authority_sequence == 0 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.authority_sequence",
            });
        }
        match (self.authority_sequence, self.predecessor) {
            (1, None) => {}
            (1, Some(_)) | (_, None) => {
                return Err(GenerationAuthorityErrorV1::InvalidField {
                    field: "activation_manifest.predecessor",
                });
            }
            (sequence, Some(predecessor)) => {
                predecessor.validate()?;
                if predecessor.sequence.checked_add(1) != Some(sequence) {
                    return Err(GenerationAuthorityErrorV1::BrokenPredecessorLink);
                }
            }
        }
        self.generation
            .validate()
            .map_err(|_| GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.generation",
            })?;
        for (field, digest) in [
            (
                "activation_manifest.writer_fence_sha256",
                self.writer_fence_sha256,
            ),
            (
                "activation_manifest.source_checkpoint_sha256",
                self.source_checkpoint_sha256,
            ),
            (
                "activation_manifest.document_set_sha256",
                self.document_set_sha256,
            ),
        ] {
            // ubs:ignore — these are public immutable-integrity identities.
            if digest == [0; 32] {
                return Err(GenerationAuthorityErrorV1::InvalidField { field });
            }
        }
        self.components.validate()
    }
}

/// An addressed activation manifest in its original authenticated wire version.
///
/// Decoding never upgrades or reseals an older object. Its canonical bytes and
/// object receipt remain the exact bytes selected by the authority reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationManifest {
    /// Schema 1, requiring all four components.
    V1(ActivationManifestV1),
    /// Schema 2, explicitly declaring whether ANN participates.
    V2(ActivationManifestV2),
}

impl ActivationManifest {
    /// Parse and authenticate either supported immutable manifest format.
    ///
    /// # Errors
    ///
    /// Rejects unknown domains or versions, malformed fields and invalid seals.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, GenerationAuthorityErrorV1> {
        if bytes.len() < 32 || bytes.len() > GENERATION_ACTIVATION_MANIFEST_MAX_BYTES_V2 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.canonical_bytes",
            });
        }
        let mut decoder = CanonicalDecoder::new(bytes);
        match decoder.bytes("activation_manifest.domain", 64)? {
            b"frankensearch.activation-manifest.v1" => {
                ActivationManifestV1::from_canonical_bytes(bytes).map(Self::V1)
            }
            b"frankensearch.activation-manifest.v2" => {
                ActivationManifestV2::from_canonical_bytes(bytes).map(Self::V2)
            }
            _ => Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.domain",
            }),
        }
    }

    /// Validate the original version's fields and self-seal.
    ///
    /// # Errors
    ///
    /// Returns that version's typed validation failure.
    pub fn validate(&self) -> Result<(), GenerationAuthorityErrorV1> {
        match self {
            Self::V1(manifest) => manifest.validate(),
            Self::V2(manifest) => manifest.validate(),
        }
    }

    /// Original canonical bytes, without upgrading or resealing.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        match self {
            Self::V1(manifest) => manifest.canonical_bytes(),
            Self::V2(manifest) => manifest.canonical_bytes(),
        }
    }

    /// Exact original object receipt.
    #[must_use]
    pub fn object_receipt(&self) -> (u64, [u8; 32]) {
        match self {
            Self::V1(manifest) => manifest.object_receipt(),
            Self::V2(manifest) => manifest.object_receipt(),
        }
    }

    /// Authority sequence authenticated by this manifest.
    #[must_use]
    pub const fn authority_sequence(&self) -> u64 {
        match self {
            Self::V1(manifest) => manifest.authority_sequence,
            Self::V2(manifest) => manifest.authority_sequence,
        }
    }

    /// Exact authenticated predecessor, or genesis.
    #[must_use]
    pub const fn predecessor(&self) -> Option<AuthorityRefV1> {
        match self {
            Self::V1(manifest) => manifest.predecessor,
            Self::V2(manifest) => manifest.predecessor,
        }
    }

    /// Selected artifact generation; distinct from the authority sequence.
    #[must_use]
    pub const fn generation(&self) -> ArtifactGenerationIdentityV1 {
        match self {
            Self::V1(manifest) => manifest.generation,
            Self::V2(manifest) => manifest.generation,
        }
    }

    /// The common source checkpoint authenticated by the manifest.
    #[must_use]
    pub const fn source_checkpoint_sha256(&self) -> [u8; 32] {
        match self {
            Self::V1(manifest) => manifest.source_checkpoint_sha256,
            Self::V2(manifest) => manifest.source_checkpoint_sha256,
        }
    }

    /// The common canonical document-set digest authenticated by the manifest.
    #[must_use]
    pub const fn document_set_sha256(&self) -> [u8; 32] {
        match self {
            Self::V1(manifest) => manifest.document_set_sha256,
            Self::V2(manifest) => manifest.document_set_sha256,
        }
    }

    /// Declared component roles. Schema 1 always returns a present ANN receipt.
    #[must_use]
    pub const fn components(&self) -> GenerationComponentReceiptsV2 {
        match self {
            Self::V1(manifest) => GenerationComponentReceiptsV2 {
                vector: manifest.components.vector,
                lexical: manifest.components.lexical,
                ann: Some(manifest.components.ann),
                metadata: manifest.components.metadata,
            },
            Self::V2(manifest) => manifest.components,
        }
    }
}

/// Verify the exact addressed manifest while preserving its original wire format.
///
/// # Errors
///
/// Rejects malformed references, invalid manifests, and any sequence,
/// predecessor, length or digest mismatch. No alternate object is selected.
pub fn verify_authority_manifest_reference(
    authority: &AuthorityRefV1,
    manifest: &ActivationManifest,
) -> Result<(), GenerationAuthorityErrorV1> {
    authority.validate()?;
    manifest.validate()?;
    // ubs:ignore — sequence and fingerprints are public integrity identities.
    if authority.sequence != manifest.authority_sequence()
        || authority.predecessor
            != manifest
                .predecessor()
                .map(|previous| previous.fingerprint())
    {
        return Err(GenerationAuthorityErrorV1::ManifestReferenceMismatch);
    }
    let (manifest_len, manifest_sha256) = manifest.object_receipt();
    // ubs:ignore — lengths and digests are public immutable object identities.
    if authority.manifest_len != manifest_len || authority.manifest_sha256 != manifest_sha256 {
        return Err(GenerationAuthorityErrorV1::ManifestReferenceMismatch);
    }
    Ok(())
}
