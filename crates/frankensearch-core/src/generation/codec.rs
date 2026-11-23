use std::fmt::Write as _;

use sha2::{Digest, Sha256};

use crate::SearchError;

use super::activation_manifest::GenerationComponentReceiptV1;
use super::authority_core::GenerationAuthorityErrorV1;
use super::embedding_identity::MAX_IDENTITY_FIELD_BYTES;

#[derive(Debug)]
pub(super) struct CanonicalEncoder {
    pub(super) bytes: Vec<u8>,
}

impl CanonicalEncoder {
    pub(super) fn new(domain: &[u8]) -> Self {
        let mut encoder = Self { bytes: Vec::new() };
        encoder.bytes(domain);
        encoder
    }

    pub(super) fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    pub(super) fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    pub(super) fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    pub(super) fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    pub(super) fn usize(&mut self, value: usize) {
        self.u64(u64::try_from(value).unwrap_or(u64::MAX));
    }

    pub(super) fn bytes(&mut self, value: &[u8]) {
        self.usize(value.len());
        self.bytes.extend_from_slice(value);
    }

    pub(super) fn text(&mut self, value: &str) {
        self.bytes(value.as_bytes());
    }

    pub(super) fn option<T>(&mut self, value: Option<&T>, encode: impl FnOnce(&T, &mut Self)) {
        match value {
            Some(value) => {
                self.u8(1);
                encode(value, self);
            }
            None => self.u8(0),
        }
    }

    pub(super) fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

/// Bounds-first reader for the fixed subset of canonical fields used by the
/// activation-manifest codec. It never allocates from a decoded length.
pub(super) struct CanonicalDecoder<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> CanonicalDecoder<'a> {
    pub(super) const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    pub(super) fn u8(&mut self, field: &'static str) -> Result<u8, GenerationAuthorityErrorV1> {
        Ok(self.take(1, field)?[0])
    }

    pub(super) fn u16(&mut self, field: &'static str) -> Result<u16, GenerationAuthorityErrorV1> {
        Ok(u16::from_be_bytes(self.array(field)?))
    }

    pub(super) fn u64(&mut self, field: &'static str) -> Result<u64, GenerationAuthorityErrorV1> {
        Ok(u64::from_be_bytes(self.array(field)?))
    }

    pub(super) fn array<const N: usize>(
        &mut self,
        field: &'static str,
    ) -> Result<[u8; N], GenerationAuthorityErrorV1> {
        let mut result = [0_u8; N];
        result.copy_from_slice(self.take(N, field)?);
        Ok(result)
    }

    pub(super) fn fixed_bytes<const N: usize>(
        &mut self,
        field: &'static str,
    ) -> Result<[u8; N], GenerationAuthorityErrorV1> {
        let bytes = self.bytes(field, N)?;
        if bytes.len() != N {
            return Err(GenerationAuthorityErrorV1::InvalidField { field });
        }
        let mut result = [0_u8; N];
        result.copy_from_slice(bytes);
        Ok(result)
    }

    pub(super) fn bytes(
        &mut self,
        field: &'static str,
        maximum_len: usize,
    ) -> Result<&'a [u8], GenerationAuthorityErrorV1> {
        let length = usize::try_from(self.u64(field)?)
            .map_err(|_| GenerationAuthorityErrorV1::InvalidField { field })?;
        if length > maximum_len {
            return Err(GenerationAuthorityErrorV1::InvalidField { field });
        }
        self.take(length, field)
    }

    pub(super) fn component(
        &mut self,
        field: &'static str,
    ) -> Result<GenerationComponentReceiptV1, GenerationAuthorityErrorV1> {
        let byte_len = self.u64(field)?;
        let sha256 = self.fixed_bytes(field)?;
        Ok(GenerationComponentReceiptV1 { byte_len, sha256 })
    }

    pub(super) fn finish(self) -> Result<(), GenerationAuthorityErrorV1> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.canonical_bytes",
            })
        }
    }

    pub(super) fn take(
        &mut self,
        len: usize,
        field: &'static str,
    ) -> Result<&'a [u8], GenerationAuthorityErrorV1> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(GenerationAuthorityErrorV1::InvalidField { field })?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(GenerationAuthorityErrorV1::InvalidField { field })?;
        self.offset = end;
        Ok(value)
    }
}

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(&mut hex, "{byte:02x}");
    }
    hex
}

pub(super) fn validate_schema(field: &str, actual: u16, expected: u16) -> Result<(), SearchError> {
    if actual == expected {
        return Ok(());
    }
    Err(identity_error(
        field,
        &actual.to_string(),
        &format!("unsupported schema; expected {expected}"),
    ))
}

pub(super) fn validate_identity_text(field: &str, value: &str) -> Result<(), SearchError> {
    if value.len() > MAX_IDENTITY_FIELD_BYTES {
        return Err(identity_error(
            field,
            "redacted-oversized",
            "field exceeds the bounded identity size",
        ));
    }
    if value.chars().any(char::is_control) {
        return Err(identity_error(
            field,
            "redacted-control-character",
            "field must not contain control characters",
        ));
    }
    if value.trim().is_empty() {
        return Err(identity_error(field, value, "must not be empty"));
    }
    Ok(())
}

pub(super) fn validate_optional_identity_text(field: &str, value: &str) -> Result<(), SearchError> {
    if value.len() > MAX_IDENTITY_FIELD_BYTES {
        return Err(identity_error(
            field,
            "redacted-oversized",
            "field exceeds the bounded identity size",
        ));
    }
    if value.chars().any(char::is_control) {
        return Err(identity_error(
            field,
            "redacted-control-character",
            "field must not contain control characters",
        ));
    }
    Ok(())
}

pub(super) fn validate_sha256(field: &str, value: &str) -> Result<(), SearchError> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Ok(());
    }
    Err(identity_error(
        field,
        "redacted-invalid-sha256",
        "must be lowercase 64-character SHA-256",
    ))
}

pub(super) fn identity_error(field: &str, value: &str, reason: &str) -> SearchError {
    let bounded_value = if value.len() <= 128 {
        value.to_owned()
    } else {
        format!("sha256:{}", sha256_hex(value.as_bytes()))
    };
    SearchError::InvalidConfig {
        field: format!("embedding_identity.{field}"),
        value: bounded_value,
        reason: reason.to_owned(),
    }
}
