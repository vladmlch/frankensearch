use std::path::Path;

use frankensearch_core::SearchResult;
use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, FrozenEmbeddingIdentityBundleV1, QuantizationFormat,
};

use crate::Quantization;

use super::{
    IDENTITY_BOUND_WAL_CRC_OFFSET, IDENTITY_BOUND_WAL_HEADER_SIZE,
    IDENTITY_BOUND_WAL_HEADER_SIZE_U16, IDENTITY_BOUND_WAL_VERSION,
    IDENTITY_BUNDLE_FINGERPRINT_OFFSET, INPUT_FINGERPRINT_OFFSET, PRODUCER_FINGERPRINT_OFFSET,
    SHA256_FINGERPRINT_BYTES, SPACE_FINGERPRINT_OFFSET, STORAGE_FINGERPRINT_OFFSET, WAL_MAGIC,
    copy_array, crc32_of, decode_sha256_fingerprint, quantization_to_wire, wal_corrupted,
    wal_identity_config_error,
};

/// Parsed, validated identity header for a version-2 WAL.
///
/// The header duplicates the storage dimension and quantization so a reader
/// can reject incompatible payload bytes before parsing batches. Component
/// fingerprints are retained separately from the complete bundle fingerprint
/// to make an exact mismatch diagnosable without trusting display labels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalIdentityHeaderV2 {
    generation: ArtifactGenerationIdentityV1,
    dimension: u32,
    quantization: Quantization,
    identity_bundle_fingerprint: [u8; SHA256_FINGERPRINT_BYTES],
    space_fingerprint: [u8; SHA256_FINGERPRINT_BYTES],
    producer_fingerprint: [u8; SHA256_FINGERPRINT_BYTES],
    input_fingerprint: [u8; SHA256_FINGERPRINT_BYTES],
    storage_fingerprint: [u8; SHA256_FINGERPRINT_BYTES],
}

impl WalIdentityHeaderV2 {
    /// Decode and structurally validate the fixed header prefix.
    ///
    /// Additional WAL batch bytes after the fixed header are intentionally
    /// ignored. Exact compatibility with an expected artifact is established
    /// by [`WalIdentityBindingV2::validate_header`].
    ///
    /// # Errors
    ///
    /// Returns [`frankensearch_core::SearchError::IndexCorrupted`] for a truncated header, unknown
    /// version, non-zero reserved byte, invalid generation, unsupported
    /// quantization, zero fingerprint, or CRC mismatch.
    pub fn decode(data: &[u8], path: &Path) -> SearchResult<Self> {
        if data.len() < IDENTITY_BOUND_WAL_HEADER_SIZE {
            return Err(wal_corrupted(
                path,
                format!(
                    "identity-bound WAL header truncated: expected at least \
                     {IDENTITY_BOUND_WAL_HEADER_SIZE} bytes, got {}",
                    data.len()
                ),
            ));
        }
        if data[..4] != WAL_MAGIC {
            return Err(wal_corrupted(
                path,
                "identity-bound WAL has bad magic bytes",
            ));
        }

        let version = u16::from_le_bytes([data[4], data[5]]);
        if version != IDENTITY_BOUND_WAL_VERSION {
            return Err(wal_corrupted(
                path,
                format!(
                    "identity-bound WAL version mismatch: expected \
                     {IDENTITY_BOUND_WAL_VERSION}, got {version}"
                ),
            ));
        }
        let header_size = u16::from_le_bytes([data[6], data[7]]);
        if header_size != IDENTITY_BOUND_WAL_HEADER_SIZE_U16 {
            return Err(wal_corrupted(
                path,
                format!(
                    "identity-bound WAL header size mismatch: expected \
                     {IDENTITY_BOUND_WAL_HEADER_SIZE_U16}, got {header_size}"
                ),
            ));
        }
        if data[13..16] != [0; 3] || data[18..20] != [0; 2] {
            return Err(wal_corrupted(
                path,
                "identity-bound WAL reserved bytes must be zero",
            ));
        }

        let stored_crc = u32::from_le_bytes([
            data[IDENTITY_BOUND_WAL_CRC_OFFSET],
            data[IDENTITY_BOUND_WAL_CRC_OFFSET + 1],
            data[IDENTITY_BOUND_WAL_CRC_OFFSET + 2],
            data[IDENTITY_BOUND_WAL_CRC_OFFSET + 3],
        ]);
        let computed_crc = crc32_of(&data[..IDENTITY_BOUND_WAL_CRC_OFFSET]);
        if stored_crc != computed_crc {
            return Err(wal_corrupted(
                path,
                "identity-bound WAL header CRC mismatch",
            ));
        }

        let dimension = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        if dimension == 0 {
            return Err(wal_corrupted(
                path,
                "identity-bound WAL dimension must be greater than zero",
            ));
        }
        let quantization = Quantization::from_wire(data[12], path)?;
        let generation = ArtifactGenerationIdentityV1 {
            schema_version: u16::from_le_bytes([data[16], data[17]]),
            sequence: u64::from_le_bytes([
                data[20], data[21], data[22], data[23], data[24], data[25], data[26], data[27],
            ]),
            nonce: copy_array::<16>(&data[28..44], path, "artifact generation nonce")?,
        };
        if let Err(error) = generation.validate() {
            return Err(wal_corrupted(
                path,
                format!("identity-bound WAL artifact generation is invalid: {error}"),
            ));
        }

        let header = Self {
            generation,
            dimension,
            quantization,
            identity_bundle_fingerprint: copy_array(
                &data[IDENTITY_BUNDLE_FINGERPRINT_OFFSET..SPACE_FINGERPRINT_OFFSET],
                path,
                "embedding identity bundle fingerprint",
            )?,
            space_fingerprint: copy_array(
                &data[SPACE_FINGERPRINT_OFFSET..PRODUCER_FINGERPRINT_OFFSET],
                path,
                "embedding space fingerprint",
            )?,
            producer_fingerprint: copy_array(
                &data[PRODUCER_FINGERPRINT_OFFSET..INPUT_FINGERPRINT_OFFSET],
                path,
                "embedding producer fingerprint",
            )?,
            input_fingerprint: copy_array(
                &data[INPUT_FINGERPRINT_OFFSET..STORAGE_FINGERPRINT_OFFSET],
                path,
                "embedding input fingerprint",
            )?,
            storage_fingerprint: copy_array(
                &data[STORAGE_FINGERPRINT_OFFSET..IDENTITY_BOUND_WAL_CRC_OFFSET],
                path,
                "vector storage fingerprint",
            )?,
        };
        header.validate_fingerprints(path)?;
        Ok(header)
    }

    /// Encode this validated header to its fixed-width representation.
    #[must_use]
    pub fn encode(&self) -> [u8; IDENTITY_BOUND_WAL_HEADER_SIZE] {
        let mut bytes = [0; IDENTITY_BOUND_WAL_HEADER_SIZE];
        bytes[..4].copy_from_slice(&WAL_MAGIC);
        bytes[4..6].copy_from_slice(&IDENTITY_BOUND_WAL_VERSION.to_le_bytes());
        bytes[6..8].copy_from_slice(&IDENTITY_BOUND_WAL_HEADER_SIZE_U16.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.dimension.to_le_bytes());
        bytes[12] = quantization_to_wire(self.quantization);
        bytes[16..18].copy_from_slice(&self.generation.schema_version.to_le_bytes());
        bytes[20..28].copy_from_slice(&self.generation.sequence.to_le_bytes());
        bytes[28..44].copy_from_slice(&self.generation.nonce);
        bytes[IDENTITY_BUNDLE_FINGERPRINT_OFFSET..SPACE_FINGERPRINT_OFFSET]
            .copy_from_slice(&self.identity_bundle_fingerprint);
        bytes[SPACE_FINGERPRINT_OFFSET..PRODUCER_FINGERPRINT_OFFSET]
            .copy_from_slice(&self.space_fingerprint);
        bytes[PRODUCER_FINGERPRINT_OFFSET..INPUT_FINGERPRINT_OFFSET]
            .copy_from_slice(&self.producer_fingerprint);
        bytes[INPUT_FINGERPRINT_OFFSET..STORAGE_FINGERPRINT_OFFSET]
            .copy_from_slice(&self.input_fingerprint);
        bytes[STORAGE_FINGERPRINT_OFFSET..IDENTITY_BOUND_WAL_CRC_OFFSET]
            .copy_from_slice(&self.storage_fingerprint);
        let crc = crc32_of(&bytes[..IDENTITY_BOUND_WAL_CRC_OFFSET]);
        bytes[IDENTITY_BOUND_WAL_CRC_OFFSET..].copy_from_slice(&crc.to_le_bytes());
        bytes
    }

    /// Exact immutable generation bound to this WAL.
    #[must_use]
    pub const fn generation(&self) -> ArtifactGenerationIdentityV1 {
        self.generation
    }

    /// Stored vector dimension.
    #[must_use]
    pub const fn dimension(&self) -> u32 {
        self.dimension
    }

    /// Stored vector quantization.
    #[must_use]
    pub const fn quantization(&self) -> Quantization {
        self.quantization
    }

    /// SHA-256 of the complete frozen embedding identity bundle.
    #[must_use]
    pub const fn identity_bundle_fingerprint(&self) -> &[u8; SHA256_FINGERPRINT_BYTES] {
        &self.identity_bundle_fingerprint
    }

    /// SHA-256 of the mathematical embedding-space identity.
    #[must_use]
    pub const fn space_fingerprint(&self) -> &[u8; SHA256_FINGERPRINT_BYTES] {
        &self.space_fingerprint
    }

    /// SHA-256 of the embedding-producer attestation.
    #[must_use]
    pub const fn producer_fingerprint(&self) -> &[u8; SHA256_FINGERPRINT_BYTES] {
        &self.producer_fingerprint
    }

    /// SHA-256 of the outer embedding-input contract.
    #[must_use]
    pub const fn input_fingerprint(&self) -> &[u8; SHA256_FINGERPRINT_BYTES] {
        &self.input_fingerprint
    }

    /// SHA-256 of the physical vector-storage identity.
    #[must_use]
    pub const fn storage_fingerprint(&self) -> &[u8; SHA256_FINGERPRINT_BYTES] {
        &self.storage_fingerprint
    }

    fn validate_fingerprints(&self, path: &Path) -> SearchResult<()> {
        for (field, fingerprint) in [
            (
                "embedding identity bundle fingerprint",
                &self.identity_bundle_fingerprint,
            ),
            ("embedding space fingerprint", &self.space_fingerprint),
            ("embedding producer fingerprint", &self.producer_fingerprint),
            ("embedding input fingerprint", &self.input_fingerprint),
            ("vector storage fingerprint", &self.storage_fingerprint),
        ] {
            if *fingerprint == [0; SHA256_FINGERPRINT_BYTES] {
                return Err(wal_corrupted(
                    path,
                    format!("identity-bound WAL {field} must not be all zero"),
                ));
            }
        }
        Ok(())
    }
}

/// Exact generation and embedding-identity contract expected by an FSVI v2
/// WAL reader or writer.
///
/// Construction validates the full frozen identity bundle and requires the
/// physical storage contract to name `fsvi-v2`. A binding can emit the
/// canonical fixed header and reject any persisted component drift.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalIdentityBindingV2 {
    expected: WalIdentityHeaderV2,
}

impl WalIdentityBindingV2 {
    /// Build an exact identity binding for one FSVI v2 artifact generation.
    ///
    /// # Errors
    ///
    /// Returns [`frankensearch_core::SearchError::InvalidConfig`] when the generation or frozen
    /// identity is invalid, the storage format is not exactly `fsvi-v2`, or
    /// the quantization cannot be represented by the FSVI slab format.
    pub fn new(
        generation: ArtifactGenerationIdentityV1,
        identity: &FrozenEmbeddingIdentityBundleV1,
    ) -> SearchResult<Self> {
        generation.validate()?;
        identity.validate()?;
        if identity.identity.storage.format != "fsvi-v2" {
            return Err(wal_identity_config_error(
                "storage.format",
                "must be exactly fsvi-v2 for an identity-bound WAL",
            ));
        }
        let quantization = match identity.identity.storage.quantization {
            QuantizationFormat::F32 => Quantization::F32,
            QuantizationFormat::F16 => Quantization::F16,
            QuantizationFormat::Int8 | QuantizationFormat::Int4 => {
                return Err(wal_identity_config_error(
                    "storage.quantization",
                    "is not supported by the FSVI v2 slab",
                ));
            }
        };
        let expected = WalIdentityHeaderV2 {
            generation,
            dimension: identity.identity.storage.dimension,
            quantization,
            identity_bundle_fingerprint: decode_sha256_fingerprint(
                "frozen_bundle.fingerprint",
                &identity.fingerprint,
            )?,
            space_fingerprint: decode_sha256_fingerprint(
                "space.fingerprint",
                &identity.identity.space.fingerprint(),
            )?,
            producer_fingerprint: decode_sha256_fingerprint(
                "producer.fingerprint",
                &identity.identity.producer.fingerprint(),
            )?,
            input_fingerprint: decode_sha256_fingerprint(
                "input.fingerprint",
                &identity.identity.input.fingerprint(),
            )?,
            storage_fingerprint: decode_sha256_fingerprint(
                "storage.fingerprint",
                &identity.identity.storage.fingerprint(),
            )?,
        };
        Ok(Self { expected })
    }

    /// Expected parsed header.
    #[must_use]
    pub const fn expected_header(&self) -> &WalIdentityHeaderV2 {
        &self.expected
    }

    /// Canonical fixed-width WAL header for this exact binding.
    #[must_use]
    pub fn encode_header(&self) -> [u8; IDENTITY_BOUND_WAL_HEADER_SIZE] {
        self.expected.encode()
    }

    /// Validate a persisted WAL header against every expected identity field.
    ///
    /// # Errors
    ///
    /// Returns [`frankensearch_core::SearchError::IndexCorrupted`] when the header itself is
    /// malformed or any generation, dimension, quantization, or component
    /// fingerprint differs.
    pub fn validate_header(&self, data: &[u8], path: &Path) -> SearchResult<()> {
        let actual = WalIdentityHeaderV2::decode(data, path)?;
        if actual.generation != self.expected.generation {
            return Err(wal_corrupted(
                path,
                "identity-bound WAL artifact generation mismatch",
            ));
        }
        if actual.dimension != self.expected.dimension {
            return Err(wal_corrupted(
                path,
                "identity-bound WAL storage dimension mismatch",
            ));
        }
        if actual.quantization != self.expected.quantization {
            return Err(wal_corrupted(
                path,
                "identity-bound WAL storage quantization mismatch",
            ));
        }
        for (field, actual, expected) in [
            (
                "embedding identity bundle fingerprint",
                actual.identity_bundle_fingerprint,
                self.expected.identity_bundle_fingerprint,
            ),
            (
                "embedding space fingerprint",
                actual.space_fingerprint,
                self.expected.space_fingerprint,
            ),
            (
                "embedding producer fingerprint",
                actual.producer_fingerprint,
                self.expected.producer_fingerprint,
            ),
            (
                "embedding input fingerprint",
                actual.input_fingerprint,
                self.expected.input_fingerprint,
            ),
            (
                "vector storage fingerprint",
                actual.storage_fingerprint,
                self.expected.storage_fingerprint,
            ),
        ] {
            if actual != expected {
                return Err(wal_corrupted(
                    path,
                    format!("identity-bound WAL {field} mismatch"),
                ));
            }
        }
        Ok(())
    }
}
