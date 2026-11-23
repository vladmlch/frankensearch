use std::path::{Path, PathBuf};

use frankensearch_core::{error::SearchResult, generation::ArtifactGenerationIdentityV1};
use serde::{Deserialize, Serialize};

use super::io_helpers::{
    canonical_graph_basename, decode_sha256_hex, encode_lower_hex,
    native_hnsw_receipt_config_error, native_hnsw_receipt_corrupted, sha256_hex, usize_to_u64,
    validate_graph_basename, validate_native_hnsw_graph_path, validate_sha256_hex,
};
use super::types::{MAX_LEVEL, NativeHnswParamsIdentityV1};
use super::{
    NATIVE_HNSW_FORMAT_VERSION, NATIVE_HNSW_GENERATION_RECEIPT_SCHEMA_V2,
    NATIVE_HNSW_GENERATION_RECEIPT_SUFFIX, NATIVE_HNSW_HEADER_LEN_U64,
    NATIVE_HNSW_MAX_BASENAME_BYTES, NATIVE_HNSW_NO_ENTRY, NATIVE_HNSW_RECEIPT_MAGIC, SHA256_BYTES,
};

/// Cryptographic binding between one owned FSHNSW artifact and one FSVI v2
/// generation.
///
/// The public serde representation rejects unknown fields. The on-disk
/// sidecar is a separate canonical binary encoding with a SHA-256 body seal;
/// readers reject alternate encodings and trailing bytes rather than silently
/// accepting a second representation of the same receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeHnswGenerationReceiptV2 {
    /// [`NATIVE_HNSW_GENERATION_RECEIPT_SCHEMA_V2`].
    pub schema_version: u16,
    /// Exact immutable FSVI artifact generation.
    pub artifact_generation: ArtifactGenerationIdentityV1,
    /// SHA-256 of the generation's canonical identity bytes.
    pub artifact_generation_fingerprint: String,
    /// SHA-256 of the complete frozen embedding identity bundle.
    pub embedding_identity_fingerprint: String,
    /// SHA-256 of the mathematical embedding-space identity.
    pub embedding_space_fingerprint: String,
    /// SHA-256 of the embedding-producer attestation.
    pub embedding_producer_fingerprint: String,
    /// SHA-256 of the outer embedding-input contract.
    pub embedding_input_fingerprint: String,
    /// SHA-256 of the physical FSVI v2 storage identity.
    pub vector_storage_fingerprint: String,
    /// SHA-256 of the exact persisted FSVI vector slab and shape.
    pub vector_content_digest: String,
    /// SHA-256 of the ordered searchable FSVI document identifiers.
    pub ordered_live_docset_digest: String,
    /// SHA-256 of every byte in the admitted immutable FSVI image.
    pub fsvi_whole_image_sha256: String,
    /// Exact number of physical FSVI rows, including tombstones.
    pub fsvi_physical_row_count: u64,
    /// Canonical UTF-8 basename of the adjacent FSHNSW artifact.
    pub graph_basename: String,
    /// Complete FSHNSW file length.
    pub graph_byte_len: u64,
    /// SHA-256 of the complete FSHNSW file bytes.
    pub graph_sha256: String,
    /// Owned FSHNSW binary format version.
    pub native_format_version: u32,
    /// Construction and default-search parameters.
    pub params: NativeHnswParamsIdentityV1,
    /// Deterministic level-sampling seed.
    pub seed: u64,
    /// Number of graph rows, which must equal the vector-store cardinality.
    pub point_count: u64,
    /// Search entry point, absent only for an empty graph.
    pub entry_point: Option<u32>,
    /// Highest occupied graph layer.
    pub max_level: u64,
    /// CRC-32 of the canonical adjacency payload.
    pub payload_crc32: u32,
    /// CRC-32 of the fixed FSHNSW header.
    pub header_crc32: u32,
    /// SHA-256 of the semantic topology, independent of file checksums.
    pub topology_sha256: String,
    /// SHA-256 of every preceding field in canonical binary order.
    pub receipt_sha256: String,
}

impl NativeHnswGenerationReceiptV2 {
    /// Validate every internal schema, digest, cardinality, and topology field.
    ///
    /// This does not establish compatibility with caller-held expectations;
    /// use [`super::ValidatedNativeHnsw::load`] for admission.
    ///
    /// # Errors
    ///
    /// Returns [`frankensearch_core::error::SearchError::InvalidConfig`] for malformed or internally
    /// inconsistent receipt material.
    pub fn validate(&self) -> SearchResult<()> {
        if self.schema_version != NATIVE_HNSW_GENERATION_RECEIPT_SCHEMA_V2 {
            return Err(native_hnsw_receipt_config_error(
                "schema_version",
                &self.schema_version.to_string(),
                "unsupported native HNSW receipt schema",
            ));
        }
        self.artifact_generation.validate()?;
        validate_sha256_hex(
            "artifact_generation_fingerprint",
            &self.artifact_generation_fingerprint,
        )?;
        // ubs:ignore — the generation SHA-256 is public integrity metadata.
        if self.artifact_generation_fingerprint != self.artifact_generation.fingerprint() {
            return Err(native_hnsw_receipt_config_error(
                "artifact_generation_fingerprint",
                "redacted-digest-mismatch",
                "does not bind the canonical artifact generation",
            ));
        }
        for (field, fingerprint) in [
            (
                "embedding_identity_fingerprint",
                &self.embedding_identity_fingerprint,
            ),
            (
                "embedding_space_fingerprint",
                &self.embedding_space_fingerprint,
            ),
            (
                "embedding_producer_fingerprint",
                &self.embedding_producer_fingerprint,
            ),
            (
                "embedding_input_fingerprint",
                &self.embedding_input_fingerprint,
            ),
            (
                "vector_storage_fingerprint",
                &self.vector_storage_fingerprint,
            ),
            ("vector_content_digest", &self.vector_content_digest),
            (
                "ordered_live_docset_digest",
                &self.ordered_live_docset_digest,
            ),
            ("fsvi_whole_image_sha256", &self.fsvi_whole_image_sha256),
            ("graph_sha256", &self.graph_sha256),
            ("topology_sha256", &self.topology_sha256),
            ("receipt_sha256", &self.receipt_sha256),
        ] {
            validate_sha256_hex(field, fingerprint)?;
        }
        validate_graph_basename(&self.graph_basename)?;
        if self.graph_byte_len < NATIVE_HNSW_HEADER_LEN_U64 {
            return Err(native_hnsw_receipt_config_error(
                "graph_byte_len",
                &self.graph_byte_len.to_string(),
                "must include the complete fixed FSHNSW header",
            ));
        }
        if self.native_format_version != NATIVE_HNSW_FORMAT_VERSION {
            return Err(native_hnsw_receipt_config_error(
                "native_format_version",
                &self.native_format_version.to_string(),
                "unsupported owned FSHNSW format version",
            ));
        }
        let _ = self.params.to_params()?;
        if self.point_count > u64::from(u32::MAX) {
            return Err(native_hnsw_receipt_config_error(
                "point_count",
                &self.point_count.to_string(),
                "must fit the native u32 point-id space",
            ));
        }
        // ubs:ignore — these are public physical graph-row cardinalities, not credentials.
        if self.fsvi_physical_row_count != self.point_count {
            return Err(native_hnsw_receipt_config_error(
                "fsvi_physical_row_count",
                &self.fsvi_physical_row_count.to_string(),
                "must equal the native HNSW point count",
            ));
        }
        if self.max_level >= usize_to_u64(MAX_LEVEL, "max_level_limit")? {
            return Err(native_hnsw_receipt_config_error(
                "max_level",
                &self.max_level.to_string(),
                "must be below the native HNSW layer bound",
            ));
        }
        match (self.point_count, self.entry_point, self.max_level) {
            (0, None, 0) => {}
            (0, Some(_), _) => {
                return Err(native_hnsw_receipt_config_error(
                    "entry_point",
                    "redacted",
                    "must be absent for an empty graph",
                ));
            }
            (0, None, _) => {
                return Err(native_hnsw_receipt_config_error(
                    "max_level",
                    &self.max_level.to_string(),
                    "must be zero for an empty graph",
                ));
            }
            (_, None, _) => {
                return Err(native_hnsw_receipt_config_error(
                    "entry_point",
                    "redacted",
                    "must be present for a non-empty graph",
                ));
            }
            (count, Some(entry), _) if u64::from(entry) >= count => {
                return Err(native_hnsw_receipt_config_error(
                    "entry_point",
                    &entry.to_string(),
                    "must name a row within point_count",
                ));
            }
            _ => {}
        }
        let expected_receipt_sha256 = sha256_hex(&self.canonical_body_bytes()?);
        // ubs:ignore — the receipt SHA-256 is an unkeyed public integrity seal.
        if self.receipt_sha256 != expected_receipt_sha256 {
            return Err(native_hnsw_receipt_config_error(
                "receipt_sha256",
                "redacted-digest-mismatch",
                "does not match the canonical receipt body",
            ));
        }
        Ok(())
    }

    /// Encode this validated receipt to its one canonical binary form.
    ///
    /// # Errors
    ///
    /// Returns [`frankensearch_core::error::SearchError::InvalidConfig`] when the receipt is invalid.
    pub fn to_bytes(&self) -> SearchResult<Vec<u8>> {
        self.validate()?;
        self.encode_unchecked()
    }

    /// Decode and validate one canonical binary receipt.
    ///
    /// # Errors
    ///
    /// Returns [`frankensearch_core::error::SearchError::IndexCorrupted`] for truncation, an unknown
    /// schema, invalid fields, a body-seal mismatch, or noncanonical/trailing
    /// bytes.
    pub fn from_bytes(bytes: &[u8], receipt_path: &Path) -> SearchResult<Self> {
        let receipt = Self::decode_unchecked(bytes, receipt_path)?;
        receipt.validate().map_err(|error| {
            native_hnsw_receipt_corrupted(
                receipt_path,
                format!("native HNSW generation receipt is invalid: {error}"),
            )
        })?;
        let canonical = receipt.encode_unchecked().map_err(|error| {
            native_hnsw_receipt_corrupted(
                receipt_path,
                format!("native HNSW generation receipt cannot be canonicalized: {error}"),
            )
        })?;
        if canonical != bytes {
            return Err(native_hnsw_receipt_corrupted(
                receipt_path,
                "native HNSW generation receipt is not canonically encoded",
            ));
        }
        Ok(receipt)
    }

    pub(super) fn seal(&mut self) -> SearchResult<()> {
        self.receipt_sha256 = sha256_hex(&self.canonical_body_bytes()?);
        Ok(())
    }

    fn canonical_body_bytes(&self) -> SearchResult<Vec<u8>> {
        let basename = self.graph_basename.as_bytes();
        let basename_len = u16::try_from(basename.len()).map_err(|_| {
            native_hnsw_receipt_config_error(
                "graph_basename",
                "redacted-oversized",
                "basename length must fit the canonical u16 field",
            )
        })?;
        let mut bytes = Vec::with_capacity(488 + basename.len());
        bytes.extend_from_slice(&NATIVE_HNSW_RECEIPT_MAGIC);
        bytes.extend_from_slice(&self.schema_version.to_be_bytes());
        bytes.extend_from_slice(&self.artifact_generation.schema_version.to_be_bytes());
        bytes.extend_from_slice(&self.artifact_generation.sequence.to_be_bytes());
        bytes.extend_from_slice(&self.artifact_generation.nonce);
        for fingerprint in [
            &self.artifact_generation_fingerprint,
            &self.embedding_identity_fingerprint,
            &self.embedding_space_fingerprint,
            &self.embedding_producer_fingerprint,
            &self.embedding_input_fingerprint,
            &self.vector_storage_fingerprint,
        ] {
            bytes.extend_from_slice(&decode_sha256_hex("receipt fingerprint", fingerprint)?);
        }
        bytes.extend_from_slice(&decode_sha256_hex(
            "vector_content_digest",
            &self.vector_content_digest,
        )?);
        bytes.extend_from_slice(&decode_sha256_hex(
            "ordered_live_docset_digest",
            &self.ordered_live_docset_digest,
        )?);
        bytes.extend_from_slice(&decode_sha256_hex(
            "fsvi_whole_image_sha256",
            &self.fsvi_whole_image_sha256,
        )?);
        bytes.extend_from_slice(&self.fsvi_physical_row_count.to_be_bytes());
        bytes.extend_from_slice(&basename_len.to_be_bytes());
        bytes.extend_from_slice(basename);
        bytes.extend_from_slice(&self.graph_byte_len.to_be_bytes());
        bytes.extend_from_slice(&decode_sha256_hex("graph_sha256", &self.graph_sha256)?);
        bytes.extend_from_slice(&self.native_format_version.to_be_bytes());
        for value in [
            self.params.m,
            self.params.m0,
            self.params.ef_construction,
            self.params.ef_search,
            self.seed,
            self.point_count,
            self.entry_point.map_or(NATIVE_HNSW_NO_ENTRY, u64::from),
            self.max_level,
        ] {
            bytes.extend_from_slice(&value.to_be_bytes());
        }
        bytes.extend_from_slice(&self.payload_crc32.to_be_bytes());
        bytes.extend_from_slice(&self.header_crc32.to_be_bytes());
        bytes.extend_from_slice(&decode_sha256_hex(
            "topology_sha256",
            &self.topology_sha256,
        )?);
        Ok(bytes)
    }

    pub(super) fn encode_unchecked(&self) -> SearchResult<Vec<u8>> {
        let mut bytes = self.canonical_body_bytes()?;
        bytes.extend_from_slice(&decode_sha256_hex("receipt_sha256", &self.receipt_sha256)?);
        Ok(bytes)
    }

    fn decode_unchecked(bytes: &[u8], receipt_path: &Path) -> SearchResult<Self> {
        let mut reader = NativeHnswReceiptReader::new(bytes, receipt_path);
        let magic = reader.take_array::<8>("receipt magic")?;
        if magic != NATIVE_HNSW_RECEIPT_MAGIC {
            return Err(native_hnsw_receipt_corrupted(
                receipt_path,
                "bad native HNSW generation receipt magic",
            ));
        }
        let schema_version = reader.read_u16("schema version")?;
        if schema_version != NATIVE_HNSW_GENERATION_RECEIPT_SCHEMA_V2 {
            return Err(native_hnsw_receipt_corrupted(
                receipt_path,
                format!(
                    "unsupported native HNSW generation receipt schema {schema_version}; expected \
                     {NATIVE_HNSW_GENERATION_RECEIPT_SCHEMA_V2}"
                ),
            ));
        }
        let artifact_generation = ArtifactGenerationIdentityV1 {
            schema_version: reader.read_u16("artifact generation schema")?,
            sequence: reader.read_u64("artifact generation sequence")?,
            nonce: reader.take_array::<16>("artifact generation nonce")?,
        };
        let artifact_generation_fingerprint =
            encode_lower_hex(reader.take_array::<SHA256_BYTES>("artifact generation fingerprint")?);
        let embedding_identity_fingerprint =
            encode_lower_hex(reader.take_array::<SHA256_BYTES>("embedding identity fingerprint")?);
        let embedding_space_fingerprint =
            encode_lower_hex(reader.take_array::<SHA256_BYTES>("embedding space fingerprint")?);
        let embedding_producer_fingerprint =
            encode_lower_hex(reader.take_array::<SHA256_BYTES>("embedding producer fingerprint")?);
        let embedding_input_fingerprint =
            encode_lower_hex(reader.take_array::<SHA256_BYTES>("embedding input fingerprint")?);
        let vector_storage_fingerprint =
            encode_lower_hex(reader.take_array::<SHA256_BYTES>("vector storage fingerprint")?);
        let vector_content_digest =
            encode_lower_hex(reader.take_array::<SHA256_BYTES>("vector content digest")?);
        let ordered_live_docset_digest =
            encode_lower_hex(reader.take_array::<SHA256_BYTES>("ordered live-docset digest")?);
        let fsvi_whole_image_sha256 =
            encode_lower_hex(reader.take_array::<SHA256_BYTES>("FSVI whole-image SHA-256")?);
        let fsvi_physical_row_count = reader.read_u64("FSVI physical row count")?;
        let basename_len = usize::from(reader.read_u16("graph basename length")?);
        if basename_len == 0 || basename_len > NATIVE_HNSW_MAX_BASENAME_BYTES {
            return Err(native_hnsw_receipt_corrupted(
                receipt_path,
                "native HNSW receipt graph basename length is invalid",
            ));
        }
        let graph_basename = std::str::from_utf8(reader.take(basename_len, "graph basename")?)
            .map_err(|_| {
                native_hnsw_receipt_corrupted(
                    receipt_path,
                    "native HNSW receipt graph basename is not UTF-8",
                )
            })?
            .to_owned();
        let graph_byte_len = reader.read_u64("graph byte length")?;
        let graph_sha256 = encode_lower_hex(reader.take_array::<SHA256_BYTES>("graph SHA-256")?);
        let native_format_version = reader.read_u32("native format version")?;
        let params = NativeHnswParamsIdentityV1 {
            m: reader.read_u64("m")?,
            m0: reader.read_u64("m0")?,
            ef_construction: reader.read_u64("ef_construction")?,
            ef_search: reader.read_u64("ef_search")?,
        };
        let seed = reader.read_u64("seed")?;
        let point_count = reader.read_u64("point count")?;
        let entry_wire = reader.read_u64("entry point")?;
        let entry_point = if entry_wire == NATIVE_HNSW_NO_ENTRY {
            None
        } else {
            Some(u32::try_from(entry_wire).map_err(|_| {
                native_hnsw_receipt_corrupted(
                    receipt_path,
                    "native HNSW receipt entry point exceeds u32",
                )
            })?)
        };
        let max_level = reader.read_u64("maximum level")?;
        let payload_crc32 = reader.read_u32("payload CRC")?;
        let header_crc32 = reader.read_u32("header CRC")?;
        let topology_sha256 =
            encode_lower_hex(reader.take_array::<SHA256_BYTES>("topology SHA-256")?);
        let receipt_sha256 =
            encode_lower_hex(reader.take_array::<SHA256_BYTES>("receipt SHA-256")?);
        reader.finish()?;

        Ok(Self {
            schema_version,
            artifact_generation,
            artifact_generation_fingerprint,
            embedding_identity_fingerprint,
            embedding_space_fingerprint,
            embedding_producer_fingerprint,
            embedding_input_fingerprint,
            vector_storage_fingerprint,
            vector_content_digest,
            ordered_live_docset_digest,
            fsvi_whole_image_sha256,
            fsvi_physical_row_count,
            graph_basename,
            graph_byte_len,
            graph_sha256,
            native_format_version,
            params,
            seed,
            point_count,
            entry_point,
            max_level,
            payload_crc32,
            header_crc32,
            topology_sha256,
            receipt_sha256,
        })
    }
}

/// Canonical adjacent receipt path for `graph_path`.
///
/// # Errors
///
/// Returns [`frankensearch_core::error::SearchError::InvalidConfig`] for a non-UTF-8, non-`.fshnsw`,
/// control-bearing, dot-relative, or otherwise noncanonical graph path.
pub fn native_hnsw_generation_receipt_path(graph_path: &Path) -> SearchResult<PathBuf> {
    validate_native_hnsw_graph_path(graph_path)?;
    let basename = canonical_graph_basename(graph_path)?;
    let receipt_basename = format!("{basename}{NATIVE_HNSW_GENERATION_RECEIPT_SUFFIX}");
    Ok(graph_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(receipt_basename))
}

struct NativeHnswReceiptReader<'a> {
    bytes: &'a [u8],
    offset: usize,
    path: &'a Path,
}

impl<'a> NativeHnswReceiptReader<'a> {
    const fn new(bytes: &'a [u8], path: &'a Path) -> Self {
        Self {
            bytes,
            offset: 0,
            path,
        }
    }

    fn take(&mut self, len: usize, field: &str) -> SearchResult<&'a [u8]> {
        let end = self.offset.checked_add(len).ok_or_else(|| {
            native_hnsw_receipt_corrupted(
                self.path,
                format!("native HNSW receipt offset overflow while reading {field}"),
            )
        })?;
        let value = self.bytes.get(self.offset..end).ok_or_else(|| {
            native_hnsw_receipt_corrupted(
                self.path,
                format!("native HNSW receipt ended while reading {field}"),
            )
        })?;
        self.offset = end;
        Ok(value)
    }

    fn take_array<const N: usize>(&mut self, field: &str) -> SearchResult<[u8; N]> {
        let mut value = [0_u8; N];
        value.copy_from_slice(self.take(N, field)?);
        Ok(value)
    }

    fn read_u16(&mut self, field: &str) -> SearchResult<u16> {
        Ok(u16::from_be_bytes(self.take_array(field)?))
    }

    fn read_u32(&mut self, field: &str) -> SearchResult<u32> {
        Ok(u32::from_be_bytes(self.take_array(field)?))
    }

    fn read_u64(&mut self, field: &str) -> SearchResult<u64> {
        Ok(u64::from_be_bytes(self.take_array(field)?))
    }

    fn finish(self) -> SearchResult<()> {
        if self.offset != self.bytes.len() {
            return Err(native_hnsw_receipt_corrupted(
                self.path,
                format!(
                    "native HNSW generation receipt has {} trailing bytes",
                    self.bytes.len() - self.offset
                ),
            ));
        }
        Ok(())
    }
}
