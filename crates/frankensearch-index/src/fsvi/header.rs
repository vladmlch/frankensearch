use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, EMBEDDING_SPACE_IDENTITY_SCHEMA_V1, EmbeddingArtifactIdentityV1,
    EmbeddingProjectionV1, EmbeddingSpaceIdentityV1, EmbeddingSpaceKindV1, HashControlProfileV1,
    QuantizationFormat, VECTOR_STORAGE_IDENTITY_SCHEMA_V1, VectorStorageIdentityV1,
};
use frankensearch_core::{SearchError, SearchResult};

use super::codec::{
    crc32, decode_sha256_fingerprint_from_index, index_corrupted, read_array, read_slice,
    sha256_bytes,
};
use super::types::{
    EMBEDDING_BUNDLE_CANONICAL_DOMAIN, EMBEDDING_SPACE_CANONICAL_DOMAIN, FSVI_MAGIC,
    FSVI_V1_MAX_HEADER_BYTES, FSVI_V2_FIXED_PREFIX_BYTES, FSVI_V2_IDENTITY_BINDING_SCHEMA,
    FSVI_V2_MAX_CANONICAL_IDENTITY_BYTES, FSVI_V2_MAX_HEADER_BYTES, FSVI_V2_MIN_HEADER_BYTES,
    FSVI_V2_VERSION, FSVI_VERSION, FsviV2IdentityMetadata, Quantization, SHA256_BYTES,
    VECTOR_STORAGE_CANONICAL_DOMAIN, VectorMetadata,
};

pub fn parse_header(path: &Path, data: &[u8]) -> SearchResult<(VectorMetadata, usize)> {
    let mut cursor = 0usize;
    let magic = read_array::<4>(path, data, &mut cursor, "magic")?;
    if magic != FSVI_MAGIC {
        return Err(index_corrupted(
            path,
            format!("bad magic bytes: expected {FSVI_MAGIC:?}, found {magic:?}"),
        ));
    }

    let version = u16::from_le_bytes(read_array::<2>(path, data, &mut cursor, "version")?);
    if version != FSVI_VERSION {
        return Err(SearchError::IndexVersionMismatch {
            expected: FSVI_VERSION,
            found: version,
        });
    }

    let embedder_id_len = usize::from(u16::from_le_bytes(read_array::<2>(
        path,
        data,
        &mut cursor,
        "embedder_id_len",
    )?));
    let embedder_id_bytes = read_slice(path, data, &mut cursor, embedder_id_len, "embedder_id")?;
    let embedder_id = std::str::from_utf8(embedder_id_bytes)
        .map_err(|error| index_corrupted(path, format!("invalid UTF-8 in embedder_id: {error}")))?
        .to_owned();

    let embedder_revision_len = usize::from(u16::from_le_bytes(read_array::<2>(
        path,
        data,
        &mut cursor,
        "embedder_revision_len",
    )?));
    let embedder_revision_bytes = read_slice(
        path,
        data,
        &mut cursor,
        embedder_revision_len,
        "embedder_revision",
    )?;
    let embedder_revision = std::str::from_utf8(embedder_revision_bytes)
        .map_err(|error| {
            index_corrupted(path, format!("invalid UTF-8 in embedder_revision: {error}"))
        })?
        .to_owned();

    let dimension_u32 = u32::from_le_bytes(read_array::<4>(path, data, &mut cursor, "dimension")?);
    let dimension = usize::try_from(dimension_u32)
        .map_err(|_| index_corrupted(path, "dimension does not fit in usize"))?;
    if dimension == 0 {
        return Err(index_corrupted(path, "dimension must be greater than zero"));
    }

    let quantization_byte = read_array::<1>(path, data, &mut cursor, "quantization")?[0];
    let quantization = Quantization::from_wire(quantization_byte, path)?;

    // Reserved bytes: [0] compaction generation, [1..3] publication nonce
    // (little-endian u16; zero on pre-nonce files).
    let reserved = read_array::<3>(path, data, &mut cursor, "reserved")?;
    let compaction_gen = reserved[0];
    let publication_nonce = u16::from_le_bytes([reserved[1], reserved[2]]);

    let record_count_u64 =
        u64::from_le_bytes(read_array::<8>(path, data, &mut cursor, "record_count")?);
    let record_count = usize::try_from(record_count_u64)
        .map_err(|_| index_corrupted(path, "record_count does not fit in usize"))?;
    let vectors_offset =
        u64::from_le_bytes(read_array::<8>(path, data, &mut cursor, "vectors_offset")?);
    let expected_crc =
        u32::from_le_bytes(read_array::<4>(path, data, &mut cursor, "header_crc32")?);
    let actual_crc = crc32(&data[..cursor - 4]);
    if actual_crc != expected_crc {
        return Err(index_corrupted(
            path,
            format!("header CRC mismatch: expected {expected_crc:#010x}, got {actual_crc:#010x}"),
        ));
    }

    Ok((
        VectorMetadata {
            fsvi_version: FSVI_VERSION,
            embedder_id,
            embedder_revision,
            dimension,
            quantization,
            compaction_gen,
            publication_nonce,
            record_count,
            vectors_offset,
            identity_v2: None,
        },
        cursor,
    ))
}

pub(super) fn read_header_for_inspection(path: &Path) -> SearchResult<(u16, Vec<u8>)> {
    if !path.exists() {
        return Err(SearchError::IndexNotFound {
            path: path.to_path_buf(),
        });
    }
    let mut file = File::open(path).map_err(SearchError::Io)?;
    let file_len = file.metadata().map_err(SearchError::Io)?.len();
    let mut prefix = [0_u8; 6];
    read_exact_index_bytes(path, &mut file, &mut prefix, "magic and version")?;
    if prefix[..4] != FSVI_MAGIC {
        return Err(index_corrupted(
            path,
            format!(
                "bad magic bytes: expected {FSVI_MAGIC:?}, found {:?}",
                &prefix[..4]
            ),
        ));
    }
    let version = u16::from_le_bytes([prefix[4], prefix[5]]);
    match version {
        FSVI_VERSION => {
            let bounded_len = usize::try_from(file_len)
                .unwrap_or(usize::MAX)
                .min(FSVI_V1_MAX_HEADER_BYTES);
            file.seek(SeekFrom::Start(0)).map_err(SearchError::Io)?;
            let mut header = Vec::with_capacity(bounded_len);
            file.take(u64::try_from(bounded_len).unwrap_or(u64::MAX))
                .read_to_end(&mut header)
                .map_err(SearchError::Io)?;
            Ok((version, header))
        }
        FSVI_V2_VERSION => {
            let mut encoded_size = [0_u8; 4];
            read_exact_index_bytes(path, &mut file, &mut encoded_size, "v2 header_size")?;
            let header_size = usize::try_from(u32::from_le_bytes(encoded_size))
                .map_err(|_| index_corrupted(path, "v2 header_size does not fit in usize"))?;
            validate_v2_header_size(path, header_size)?;
            if u64::try_from(header_size).is_ok_and(|size| size > file_len) {
                return Err(index_corrupted(
                    path,
                    format!(
                        "v2 header is truncated: declared {header_size} bytes, file has {file_len}"
                    ),
                ));
            }
            file.seek(SeekFrom::Start(0)).map_err(SearchError::Io)?;
            let mut header = vec![0_u8; header_size];
            read_exact_index_bytes(path, &mut file, &mut header, "v2 header")?;
            Ok((version, header))
        }
        _ => Ok((version, prefix.to_vec())),
    }
}

pub fn read_exact_index_bytes(
    path: &Path,
    reader: &mut impl Read,
    bytes: &mut [u8],
    field: &str,
) -> SearchResult<()> {
    reader.read_exact(bytes).map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            index_corrupted(path, format!("{field} is truncated"))
        } else {
            SearchError::Io(error)
        }
    })
}

pub fn validate_v2_header_size(path: &Path, header_size: usize) -> SearchResult<()> {
    if !(FSVI_V2_MIN_HEADER_BYTES..=FSVI_V2_MAX_HEADER_BYTES).contains(&header_size) {
        return Err(index_corrupted(
            path,
            format!(
                "v2 header_size {header_size} is outside [{FSVI_V2_MIN_HEADER_BYTES}, {FSVI_V2_MAX_HEADER_BYTES}]"
            ),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
pub fn parse_v2_header(path: &Path, data: &[u8]) -> SearchResult<(VectorMetadata, usize)> {
    let mut cursor = 0usize;
    let magic = read_array::<4>(path, data, &mut cursor, "magic")?;
    if magic != FSVI_MAGIC {
        return Err(index_corrupted(
            path,
            format!("bad magic bytes: expected {FSVI_MAGIC:?}, found {magic:?}"),
        ));
    }
    let version = u16::from_le_bytes(read_array::<2>(path, data, &mut cursor, "version")?);
    if version != FSVI_V2_VERSION {
        return Err(SearchError::IndexVersionMismatch {
            expected: FSVI_V2_VERSION,
            found: version,
        });
    }
    let header_size = usize::try_from(u32::from_le_bytes(read_array::<4>(
        path,
        data,
        &mut cursor,
        "header_size",
    )?))
    .map_err(|_| index_corrupted(path, "v2 header_size does not fit in usize"))?;
    validate_v2_header_size(path, header_size)?;
    if header_size > data.len() {
        return Err(index_corrupted(
            path,
            format!(
                "v2 header is truncated: declared {header_size} bytes, have {}",
                data.len()
            ),
        ));
    }
    let header = &data[..header_size];

    let binding_schema = u16::from_le_bytes(read_array::<2>(
        path,
        header,
        &mut cursor,
        "identity_binding_schema",
    )?);
    if binding_schema != FSVI_V2_IDENTITY_BINDING_SCHEMA {
        return Err(index_corrupted(
            path,
            format!(
                "unsupported v2 identity binding schema {binding_schema}; expected {FSVI_V2_IDENTITY_BINDING_SCHEMA}"
            ),
        ));
    }
    let quantization = Quantization::from_wire(
        read_array::<1>(path, header, &mut cursor, "quantization")?[0],
        path,
    )?;
    let flags = read_array::<1>(path, header, &mut cursor, "header_flags")?[0];
    // Formerly a reserved-zero u16; now the cross-tier publication nonce
    // (bd-miio8), zero on single-tier and pre-nonce v2 files.
    let publication_nonce = u16::from_le_bytes(read_array::<2>(
        path,
        header,
        &mut cursor,
        "publication_nonce",
    )?);
    if flags != 0 {
        return Err(index_corrupted(path, "v2 header flags must be zero"));
    }
    let dimension_u32 =
        u32::from_le_bytes(read_array::<4>(path, header, &mut cursor, "dimension")?);
    let dimension = usize::try_from(dimension_u32)
        .map_err(|_| index_corrupted(path, "dimension does not fit in usize"))?;
    if dimension == 0 {
        return Err(index_corrupted(path, "dimension must be greater than zero"));
    }
    let record_count_u64 =
        u64::from_le_bytes(read_array::<8>(path, header, &mut cursor, "record_count")?);
    let record_count = usize::try_from(record_count_u64)
        .map_err(|_| index_corrupted(path, "record_count does not fit in usize"))?;
    let vectors_offset = u64::from_le_bytes(read_array::<8>(
        path,
        header,
        &mut cursor,
        "vectors_offset",
    )?);

    let generation_schema = u16::from_le_bytes(read_array::<2>(
        path,
        header,
        &mut cursor,
        "generation_schema",
    )?);
    let generation_reserved = u16::from_le_bytes(read_array::<2>(
        path,
        header,
        &mut cursor,
        "generation_reserved",
    )?);
    if generation_reserved != 0 {
        return Err(index_corrupted(
            path,
            "v2 generation reserved field must be zero",
        ));
    }
    let generation_sequence = u64::from_le_bytes(read_array::<8>(
        path,
        header,
        &mut cursor,
        "generation_sequence",
    )?);
    let generation_nonce = read_array::<16>(path, header, &mut cursor, "generation_nonce")?;
    let generation = ArtifactGenerationIdentityV1 {
        schema_version: generation_schema,
        sequence: generation_sequence,
        nonce: generation_nonce,
    };
    generation.validate().map_err(|error| {
        index_corrupted(path, format!("v2 artifact generation is invalid: {error}"))
    })?;

    let bundle_len = usize::try_from(u32::from_le_bytes(read_array::<4>(
        path,
        header,
        &mut cursor,
        "identity_bundle_len",
    )?))
    .map_err(|_| index_corrupted(path, "identity_bundle_len does not fit in usize"))?;
    let space_len = usize::try_from(u32::from_le_bytes(read_array::<4>(
        path,
        header,
        &mut cursor,
        "space_identity_len",
    )?))
    .map_err(|_| index_corrupted(path, "space_identity_len does not fit in usize"))?;
    let storage_len = usize::try_from(u32::from_le_bytes(read_array::<4>(
        path,
        header,
        &mut cursor,
        "storage_identity_len",
    )?))
    .map_err(|_| index_corrupted(path, "storage_identity_len does not fit in usize"))?;
    for (field, len) in [
        ("identity_bundle_len", bundle_len),
        ("space_identity_len", space_len),
        ("storage_identity_len", storage_len),
    ] {
        if len == 0 || len > FSVI_V2_MAX_CANONICAL_IDENTITY_BYTES {
            return Err(index_corrupted(
                path,
                format!("{field} must be non-zero and no larger than 1 MiB"),
            ));
        }
    }

    let identity_bundle_fingerprint =
        read_array::<SHA256_BYTES>(path, header, &mut cursor, "identity_bundle_fingerprint")?;
    let space_fingerprint =
        read_array::<SHA256_BYTES>(path, header, &mut cursor, "space_fingerprint")?;
    let producer_fingerprint =
        read_array::<SHA256_BYTES>(path, header, &mut cursor, "producer_fingerprint")?;
    let input_fingerprint =
        read_array::<SHA256_BYTES>(path, header, &mut cursor, "input_fingerprint")?;
    let storage_fingerprint =
        read_array::<SHA256_BYTES>(path, header, &mut cursor, "storage_fingerprint")?;
    let generation_fingerprint =
        read_array::<SHA256_BYTES>(path, header, &mut cursor, "generation_fingerprint")?;
    let ordered_live_docset_digest =
        read_array::<SHA256_BYTES>(path, header, &mut cursor, "ordered_live_docset_digest")?;
    let vector_content_digest =
        read_array::<SHA256_BYTES>(path, header, &mut cursor, "vector_content_digest")?;
    for (field, fingerprint) in [
        ("identity_bundle_fingerprint", identity_bundle_fingerprint),
        ("space_fingerprint", space_fingerprint),
        ("producer_fingerprint", producer_fingerprint),
        ("input_fingerprint", input_fingerprint),
        ("storage_fingerprint", storage_fingerprint),
        ("generation_fingerprint", generation_fingerprint),
        ("ordered_live_docset_digest", ordered_live_docset_digest),
        ("vector_content_digest", vector_content_digest),
    ] {
        if fingerprint == [0; SHA256_BYTES] {
            return Err(index_corrupted(
                path,
                format!("v2 {field} must not be all zero"),
            ));
        }
    }
    if cursor != FSVI_V2_FIXED_PREFIX_BYTES {
        return Err(index_corrupted(
            path,
            "internal v2 fixed header layout disagreement",
        ));
    }

    let identity_bundle_canonical_bytes =
        read_slice(path, header, &mut cursor, bundle_len, "identity_bundle")?.to_vec();
    let space_identity_canonical_bytes =
        read_slice(path, header, &mut cursor, space_len, "space_identity")?.to_vec();
    let storage_identity_canonical_bytes =
        read_slice(path, header, &mut cursor, storage_len, "storage_identity")?.to_vec();
    let crc_offset = header_size
        .checked_sub(4)
        .ok_or_else(|| index_corrupted(path, "v2 CRC offset underflow"))?;
    if cursor != crc_offset {
        return Err(index_corrupted(
            path,
            format!(
                "v2 canonical identity lengths end at byte {cursor}, expected CRC at {crc_offset}"
            ),
        ));
    }
    let expected_crc =
        u32::from_le_bytes(read_array::<4>(path, header, &mut cursor, "header_crc32")?);
    let actual_crc = crc32(&header[..crc_offset]);
    if actual_crc != expected_crc {
        return Err(index_corrupted(
            path,
            format!(
                "v2 header CRC mismatch: expected {expected_crc:#010x}, got {actual_crc:#010x}"
            ),
        ));
    }

    validate_canonical_fingerprint(
        path,
        "identity bundle",
        &identity_bundle_canonical_bytes,
        identity_bundle_fingerprint,
    )?;
    validate_canonical_fingerprint(
        path,
        "embedding space",
        &space_identity_canonical_bytes,
        space_fingerprint,
    )?;
    validate_canonical_fingerprint(
        path,
        "vector storage",
        &storage_identity_canonical_bytes,
        storage_fingerprint,
    )?;
    validate_canonical_fingerprint(
        path,
        "artifact generation",
        &generation.canonical_bytes(),
        generation_fingerprint,
    )?;
    let bundle_components =
        parse_bundle_component_fingerprints(path, &identity_bundle_canonical_bytes)?;
    if bundle_components
        != [
            space_fingerprint,
            producer_fingerprint,
            input_fingerprint,
            storage_fingerprint,
        ]
    {
        return Err(index_corrupted(
            path,
            "v2 complete identity bundle does not bind the stored component fingerprints",
        ));
    }
    let (embedder_id, embedder_revision, space_normalization) = parse_complete_space_identity(
        path,
        &space_identity_canonical_bytes,
        dimension_u32,
        input_fingerprint,
    )?;
    let storage_normalization = validate_storage_identity_bytes(
        path,
        &storage_identity_canonical_bytes,
        dimension_u32,
        quantization,
    )?;
    if storage_normalization != space_normalization {
        return Err(index_corrupted(
            path,
            "v2 storage normalization disagrees with the embedding-space output normalization",
        ));
    }

    Ok((
        VectorMetadata {
            fsvi_version: FSVI_V2_VERSION,
            embedder_id,
            embedder_revision,
            dimension,
            quantization,
            compaction_gen: 0,
            // Cross-tier publication nonce from the v2 header's nonce slot
            // (bd-miio8): binds the fast/quality tiers of one staged v2
            // publication; zero on single-tier and pre-nonce files.
            publication_nonce,
            record_count,
            vectors_offset,
            identity_v2: Some(FsviV2IdentityMetadata {
                generation,
                identity_bundle_canonical_bytes,
                space_identity_canonical_bytes,
                storage_identity_canonical_bytes,
                identity_bundle_fingerprint,
                space_fingerprint,
                producer_fingerprint,
                input_fingerprint,
                storage_fingerprint,
                generation_fingerprint,
                ordered_live_docset_digest,
                vector_content_digest,
                header_size,
            }),
        },
        header_size,
    ))
}

fn validate_canonical_fingerprint(
    path: &Path,
    field: &str,
    canonical_bytes: &[u8],
    expected: [u8; SHA256_BYTES],
) -> SearchResult<()> {
    let actual = sha256_bytes(canonical_bytes);
    if actual != expected {
        return Err(index_corrupted(
            path,
            format!("v2 {field} canonical bytes disagree with their SHA-256 fingerprint"),
        ));
    }
    Ok(())
}

fn parse_bundle_component_fingerprints(
    path: &Path,
    bytes: &[u8],
) -> SearchResult<[[u8; SHA256_BYTES]; 4]> {
    let mut cursor = 0usize;
    let domain = read_canonical_bytes(path, bytes, &mut cursor, "bundle.domain")?;
    if domain != EMBEDDING_BUNDLE_CANONICAL_DOMAIN {
        return Err(index_corrupted(
            path,
            "v2 identity bundle canonical domain is invalid",
        ));
    }
    let mut fingerprints = [[0_u8; SHA256_BYTES]; 4];
    for (index, field) in ["space", "producer", "input", "storage"]
        .into_iter()
        .enumerate()
    {
        let value = read_canonical_text(
            path,
            bytes,
            &mut cursor,
            &format!("bundle.{field}_fingerprint"),
        )?;
        fingerprints[index] = decode_sha256_fingerprint_from_index(path, field, value)?;
    }
    if cursor != bytes.len() {
        return Err(index_corrupted(
            path,
            "v2 identity bundle canonical bytes contain a trailing suffix",
        ));
    }
    Ok(fingerprints)
}

#[allow(clippy::too_many_lines)]
pub fn parse_complete_space_identity(
    path: &Path,
    bytes: &[u8],
    expected_dimension: u32,
    expected_input_fingerprint: [u8; SHA256_BYTES],
) -> SearchResult<(String, String, String)> {
    let mut cursor = 0usize;
    let domain = read_canonical_bytes(path, bytes, &mut cursor, "space.domain")?;
    if domain != EMBEDDING_SPACE_CANONICAL_DOMAIN {
        return Err(index_corrupted(
            path,
            "v2 embedding-space canonical domain is invalid",
        ));
    }
    let schema = read_canonical_u16(path, bytes, &mut cursor, "space.schema")?;
    if schema != EMBEDDING_SPACE_IDENTITY_SCHEMA_V1 {
        return Err(index_corrupted(
            path,
            format!(
                "v2 embedding-space schema {schema} is unsupported; expected {EMBEDDING_SPACE_IDENTITY_SCHEMA_V1}"
            ),
        ));
    }
    let model_id =
        read_canonical_text(path, bytes, &mut cursor, "space.logical_model_id")?.to_owned();
    let revision =
        read_canonical_text(path, bytes, &mut cursor, "space.immutable_revision")?.to_owned();
    let kind = match read_canonical_u8(path, bytes, &mut cursor, "space.kind")? {
        1 => EmbeddingSpaceKindV1::Semantic,
        2 => EmbeddingSpaceKindV1::HashControl,
        found => {
            return Err(index_corrupted(
                path,
                format!("v2 embedding-space kind tag {found} is invalid"),
            ));
        }
    };
    let artifact_manifest_fingerprint = read_canonical_text(
        path,
        bytes,
        &mut cursor,
        "space.artifact_manifest_fingerprint",
    )?
    .to_owned();
    let artifact_count_u64 = read_canonical_u64(path, bytes, &mut cursor, "space.artifact_count")?;
    let artifact_count = usize::try_from(artifact_count_u64)
        .map_err(|_| index_corrupted(path, "v2 artifact count does not fit in usize"))?;
    let minimum_artifact_bytes = 24usize;
    let maximum_artifact_count = bytes
        .len()
        .saturating_sub(cursor)
        .checked_div(minimum_artifact_bytes)
        .unwrap_or(0);
    if artifact_count > maximum_artifact_count {
        return Err(index_corrupted(
            path,
            format!(
                "v2 artifact count {artifact_count} cannot fit in the remaining canonical bytes"
            ),
        ));
    }
    let mut artifacts = Vec::new();
    for index in 0..artifact_count {
        artifacts.push(EmbeddingArtifactIdentityV1 {
            role: read_canonical_text(
                path,
                bytes,
                &mut cursor,
                &format!("space.artifacts[{index}].role"),
            )?
            .to_owned(),
            sha256: read_canonical_text(
                path,
                bytes,
                &mut cursor,
                &format!("space.artifacts[{index}].sha256"),
            )?
            .to_owned(),
            size: read_canonical_u64(
                path,
                bytes,
                &mut cursor,
                &format!("space.artifacts[{index}].size"),
            )?,
        });
    }

    let tokenizer_fingerprint =
        read_canonical_text(path, bytes, &mut cursor, "space.tokenizer_fingerprint")?.to_owned();
    let vocabulary_fingerprint =
        read_canonical_text(path, bytes, &mut cursor, "space.vocabulary_fingerprint")?.to_owned();
    let model_config_fingerprint =
        read_canonical_text(path, bytes, &mut cursor, "space.model_config_fingerprint")?.to_owned();
    let model_preprocessing =
        read_canonical_text(path, bytes, &mut cursor, "space.model_preprocessing")?.to_owned();
    let sequence_policy =
        read_canonical_text(path, bytes, &mut cursor, "space.sequence_policy")?.to_owned();
    let query_instruction =
        read_canonical_text(path, bytes, &mut cursor, "space.query_instruction")?.to_owned();
    let document_instruction =
        read_canonical_text(path, bytes, &mut cursor, "space.document_instruction")?.to_owned();
    let pooling = read_canonical_text(path, bytes, &mut cursor, "space.pooling")?.to_owned();
    let output_normalization =
        read_canonical_text(path, bytes, &mut cursor, "space.output_normalization")?.to_owned();
    let dimension = read_canonical_u32(path, bytes, &mut cursor, "space.dimension")?;
    let input_contract_fingerprint =
        read_canonical_text(path, bytes, &mut cursor, "space.input_contract_fingerprint")?
            .to_owned();

    let hash_control =
        match read_canonical_u8(path, bytes, &mut cursor, "space.hash_control.option")? {
            0 => None,
            1 => Some(HashControlProfileV1 {
                algorithm: read_canonical_text(
                    path,
                    bytes,
                    &mut cursor,
                    "space.hash_control.algorithm",
                )?
                .to_owned(),
                algorithm_revision: read_canonical_text(
                    path,
                    bytes,
                    &mut cursor,
                    "space.hash_control.algorithm_revision",
                )?
                .to_owned(),
                seed: read_canonical_u64(path, bytes, &mut cursor, "space.hash_control.seed")?,
                feature_rules: read_canonical_text(
                    path,
                    bytes,
                    &mut cursor,
                    "space.hash_control.feature_rules",
                )?
                .to_owned(),
                tokenization_rules: read_canonical_text(
                    path,
                    bytes,
                    &mut cursor,
                    "space.hash_control.tokenization_rules",
                )?
                .to_owned(),
                signing_rules: read_canonical_text(
                    path,
                    bytes,
                    &mut cursor,
                    "space.hash_control.signing_rules",
                )?
                .to_owned(),
                normalization_rules: read_canonical_text(
                    path,
                    bytes,
                    &mut cursor,
                    "space.hash_control.normalization_rules",
                )?
                .to_owned(),
            }),
            found => {
                return Err(index_corrupted(
                    path,
                    format!("v2 hash-control option tag {found} is invalid"),
                ));
            }
        };
    let projection = match read_canonical_u8(path, bytes, &mut cursor, "space.projection.option")? {
        0 => None,
        1 => Some(EmbeddingProjectionV1 {
            parent_space_fingerprint: read_canonical_text(
                path,
                bytes,
                &mut cursor,
                "space.projection.parent_space_fingerprint",
            )?
            .to_owned(),
            source_dimension: read_canonical_u32(
                path,
                bytes,
                &mut cursor,
                "space.projection.source_dimension",
            )?,
            output_dimension: read_canonical_u32(
                path,
                bytes,
                &mut cursor,
                "space.projection.output_dimension",
            )?,
            projection_rule: read_canonical_text(
                path,
                bytes,
                &mut cursor,
                "space.projection.projection_rule",
            )?
            .to_owned(),
            renormalization_rule: read_canonical_text(
                path,
                bytes,
                &mut cursor,
                "space.projection.renormalization_rule",
            )?
            .to_owned(),
        }),
        found => {
            return Err(index_corrupted(
                path,
                format!("v2 projection option tag {found} is invalid"),
            ));
        }
    };
    if cursor != bytes.len() {
        return Err(index_corrupted(
            path,
            "v2 embedding-space canonical bytes contain a trailing suffix",
        ));
    }

    let identity = EmbeddingSpaceIdentityV1 {
        schema_version: schema,
        logical_model_id: model_id,
        immutable_revision: revision,
        kind,
        artifact_manifest_fingerprint,
        artifacts,
        tokenizer_fingerprint,
        vocabulary_fingerprint,
        model_config_fingerprint,
        model_preprocessing,
        sequence_policy,
        query_instruction,
        document_instruction,
        pooling,
        output_normalization,
        dimension,
        input_contract_fingerprint,
        hash_control,
        projection,
    };
    identity.validate().map_err(|error| {
        index_corrupted(
            path,
            format!("v2 embedding-space identity is incomplete or invalid: {error}"),
        )
    })?;
    if identity.dimension != expected_dimension {
        return Err(index_corrupted(
            path,
            "v2 embedding-space dimension disagrees with the fixed header",
        ));
    }
    let persisted_input_fingerprint = decode_sha256_fingerprint_from_index(
        path,
        "space.input_contract",
        &identity.input_contract_fingerprint,
    )?;
    if persisted_input_fingerprint != expected_input_fingerprint {
        return Err(index_corrupted(
            path,
            "v2 embedding-space input contract disagrees with the complete identity bundle",
        ));
    }
    if identity.canonical_bytes() != bytes {
        return Err(index_corrupted(
            path,
            "v2 embedding-space bytes are not the exact canonical encoding",
        ));
    }
    let model_id = identity.logical_model_id;
    let revision = identity.immutable_revision;
    let output_normalization = identity.output_normalization;
    Ok((model_id, revision, output_normalization))
}

fn validate_storage_identity_bytes(
    path: &Path,
    bytes: &[u8],
    expected_dimension: u32,
    expected_quantization: Quantization,
) -> SearchResult<String> {
    let mut cursor = 0usize;
    let domain = read_canonical_bytes(path, bytes, &mut cursor, "storage.domain")?;
    if domain != VECTOR_STORAGE_CANONICAL_DOMAIN {
        return Err(index_corrupted(
            path,
            "v2 vector-storage canonical domain is invalid",
        ));
    }
    let schema = read_canonical_u16(path, bytes, &mut cursor, "storage.schema")?;
    if schema != VECTOR_STORAGE_IDENTITY_SCHEMA_V1 {
        return Err(index_corrupted(
            path,
            format!(
                "v2 vector-storage schema {schema} is unsupported; expected {VECTOR_STORAGE_IDENTITY_SCHEMA_V1}"
            ),
        ));
    }
    let format = read_canonical_text(path, bytes, &mut cursor, "storage.format")?;
    if format != "fsvi-v2" {
        return Err(index_corrupted(
            path,
            "v2 storage identity must name exactly fsvi-v2",
        ));
    }
    let quantization = read_canonical_u8(path, bytes, &mut cursor, "storage.quantization")?;
    let (expected_quantization_wire, quantization_format) = match expected_quantization {
        Quantization::F32 => (1, QuantizationFormat::F32),
        Quantization::F16 => (2, QuantizationFormat::F16),
    };
    if quantization != expected_quantization_wire {
        return Err(index_corrupted(
            path,
            "v2 storage identity quantization disagrees with the fixed header",
        ));
    }
    let endianness = read_canonical_text(path, bytes, &mut cursor, "storage.endianness")?;
    if endianness != "little-endian" {
        return Err(index_corrupted(
            path,
            "v2 storage identity must be canonical little-endian",
        ));
    }
    let normalization =
        read_canonical_text(path, bytes, &mut cursor, "storage.vector_normalization")?;
    let normalization = normalization.to_owned();
    let dimension = read_canonical_u32(path, bytes, &mut cursor, "storage.dimension")?;
    if dimension != expected_dimension {
        return Err(index_corrupted(
            path,
            "v2 storage identity dimension disagrees with the fixed header",
        ));
    }
    if cursor != bytes.len() {
        return Err(index_corrupted(
            path,
            "v2 storage identity canonical bytes contain a trailing suffix",
        ));
    }
    let identity = VectorStorageIdentityV1 {
        schema_version: schema,
        format: format.to_owned(),
        quantization: quantization_format,
        endianness: endianness.to_owned(),
        vector_normalization: normalization,
        dimension,
    };
    identity.validate().map_err(|error| {
        index_corrupted(
            path,
            format!("v2 vector-storage identity is incomplete or invalid: {error}"),
        )
    })?;
    if identity.canonical_bytes() != bytes {
        return Err(index_corrupted(
            path,
            "v2 vector-storage bytes are not the exact canonical encoding",
        ));
    }
    Ok(identity.vector_normalization)
}

fn read_canonical_bytes<'a>(
    path: &Path,
    data: &'a [u8],
    cursor: &mut usize,
    field: &str,
) -> SearchResult<&'a [u8]> {
    let len_u64 = u64::from_be_bytes(read_array::<8>(
        path,
        data,
        cursor,
        &format!("{field}.len"),
    )?);
    let len = usize::try_from(len_u64)
        .map_err(|_| index_corrupted(path, format!("{field} length does not fit in usize")))?;
    read_slice(path, data, cursor, len, field)
}

fn read_canonical_text<'a>(
    path: &Path,
    data: &'a [u8],
    cursor: &mut usize,
    field: &str,
) -> SearchResult<&'a str> {
    let bytes = read_canonical_bytes(path, data, cursor, field)?;
    std::str::from_utf8(bytes)
        .map_err(|error| index_corrupted(path, format!("{field} is not UTF-8: {error}")))
}

fn read_canonical_u8(
    path: &Path,
    data: &[u8],
    cursor: &mut usize,
    field: &str,
) -> SearchResult<u8> {
    Ok(read_array::<1>(path, data, cursor, field)?[0])
}

fn read_canonical_u16(
    path: &Path,
    data: &[u8],
    cursor: &mut usize,
    field: &str,
) -> SearchResult<u16> {
    Ok(u16::from_be_bytes(read_array::<2>(
        path, data, cursor, field,
    )?))
}

fn read_canonical_u32(
    path: &Path,
    data: &[u8],
    cursor: &mut usize,
    field: &str,
) -> SearchResult<u32> {
    Ok(u32::from_be_bytes(read_array::<4>(
        path, data, cursor, field,
    )?))
}

fn read_canonical_u64(
    path: &Path,
    data: &[u8],
    cursor: &mut usize,
    field: &str,
) -> SearchResult<u64> {
    Ok(u64::from_be_bytes(read_array::<8>(
        path, data, cursor, field,
    )?))
}
