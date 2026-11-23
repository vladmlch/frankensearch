use std::fs;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::{Path, PathBuf};

use frankensearch_core::{SearchError, SearchResult};

use crate::{FsviUpgradeRequired, VectorMetadata};

use super::{VECTOR_INDEX_FALLBACK_FILENAME, VECTOR_INDEX_FAST_FILENAME};

pub(super) fn resolve_fast_path(dir: &Path) -> SearchResult<PathBuf> {
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    if fast_path.exists() {
        return Ok(fast_path);
    }

    let fallback_path = dir.join(VECTOR_INDEX_FALLBACK_FILENAME);
    if fallback_path.exists() {
        return Ok(fallback_path);
    }

    Err(SearchError::IndexCandidatesNotFound {
        paths: vec![fast_path, fallback_path],
    })
}

// ---------------------------------------------------------------------------
// Read-only tier observation (bd-9xuj C4-write r2)
// ---------------------------------------------------------------------------

/// What [`observe_tier`] proved about one on-disk FSVI tier artifact.
///
/// Every variant is established WITHOUT any mutable open: no WAL sidecar is
/// ever deleted or truncated, no file is memory-mapped writable, and no byte
/// of the artifact or its directory changes. Access metadata is the one
/// exception — see [`observe_tier`] for the exact atime/symlink caveat of
/// the ordinary read-only fallback. This is the classification carrier for
/// pre-drain refresh admission (the r2 repair of the C4-write NO-GO's
/// mutable-`VectorIndex::open`-during-classification hazard).
#[derive(Debug)]
pub enum FsviTierObservation {
    /// Recognized legacy FSVI v1 bytes, header-parsed only.
    V1(FsviV1Observation),
    /// Identity-complete FSVI v2 header. Content admission (digest
    /// recomputation, sealed-owner retention) still happens through
    /// [`VectorIndex::open_admitted_v2`](crate::VectorIndex::open_admitted_v2); this variant only proves the
    /// header parses as identity-complete v2.
    V2IdentityComplete(Box<VectorMetadata>),
    /// A newer FSVI schema than this reader supports.
    UpgradeRequired(FsviUpgradeRequired),
}

/// Header-level, read-only observation of a legacy FSVI v1 tier.
///
/// Record FLAGS are deliberately not inspected (that would require reading
/// the record table): `record_count` counts live AND tombstoned rows, so
/// [`Self::retains_content`] is conservative — an all-tombstoned v1 tier
/// reads as retaining content and fails closed at admission seams. Restoring
/// flag-level precision needs a read-only record-table inspector in the
/// crate root (deferred to the observational-open train; see
/// `VectorIndex::inspect`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsviV1Observation {
    /// Total record slots in the main slab (live + tombstoned).
    pub record_count: usize,
    /// Decodable WAL entries whose compaction generation matches the main
    /// slab. `0` when the sidecar is absent, empty, STALE (generation
    /// mismatch), or when only a corrupt trailer follows zero valid batches.
    /// Unlike `VectorIndex::open`, observing a stale or trailer-corrupt WAL
    /// never deletes or truncates it.
    pub active_wal_records: usize,
    /// Whether a WAL sidecar file exists at all (any content state).
    pub wal_sidecar_present: bool,
}

impl FsviV1Observation {
    /// Conservative content-retention signal for admission seams.
    ///
    /// True when the main slab has any record slot (live or tombstoned) or
    /// the WAL retains generation-matched appends. Fail-closed by
    /// construction: it can claim retention for content-free artifacts
    /// (all-tombstoned slabs) but never the reverse.
    #[must_use]
    pub const fn retains_content(&self) -> bool {
        self.record_count > 0 || self.active_wal_records > 0
    }
}

/// Open one tier artifact strictly read-only, preserving atime when the
/// platform allows.
///
/// Prefers the crate's `O_NOATIME | O_NOFOLLOW | O_CLOEXEC` opener (the same
/// one exact v2 admission uses). Where that is denied or unsupported (non-
/// owner files, non-Linux targets, symlinked finals) this falls back to an
/// ordinary read-only [`fs::File::open`], whose only observable effects are
/// a possible atime update and symlink traversal — never weaker than
/// `VectorIndex::inspect`'s unconditional ordinary open, and never
/// write-capable.
fn open_tier_readonly(path: &Path) -> SearchResult<fs::File> {
    crate::open_readonly_noatime_nofollow(path)
        .or_else(|_| fs::File::open(path).map_err(SearchError::Io))
}

/// Classify one on-disk FSVI tier artifact without writes.
///
/// Exact guarantee: nothing is written, truncated, or deleted, and no
/// writable open or mapping occurs — bytes, directory entries, sizes, and
/// mtimes are invariant across the call.
///
/// The one deliberate non-guarantee is access metadata (r3 claim precision,
/// audits #8366/#8367): the preferred opener is the no-atime/no-follow fast
/// path (`O_NOATIME | O_NOFOLLOW | O_CLOEXEC`), but where that open is
/// denied or unsupported the documented fallback is an ordinary read-only
/// `File::open` — and the v1 WAL sidecar is read via `wal::read_wal`'s
/// ordinary open — either of which may update atime and follow symlinks.
///
/// Contrast with the two open paths this deliberately is not:
/// - [`VectorIndex::open`](crate::VectorIndex::open) (v1) opens WRITE-capable, deletes a stale WAL
///   sidecar, and truncates a corrupt WAL trailer — it must never run during
///   classification;
/// - [`VectorIndex::inspect`](crate::VectorIndex::inspect) is header-only but discards the parsed v1
///   metadata, so callers cannot distinguish an empty v1 seed from a
///   content-retaining one without a mutable open.
///
/// This function parses the SAME headers through the SAME crate parsers
/// (`parse_header` / `parse_v2_header`), reads the v1 WAL sidecar through
/// the same read-only `wal::read_wal`, and applies the same staleness
/// predicate `VectorIndex::open` uses — but performs no deletion, no
/// truncation, and no writable mapping. v2 recognition here is HEADER-ONLY:
/// content admission still requires [`VectorIndex::open_admitted_v2`](crate::VectorIndex::open_admitted_v2).
///
/// # Errors
///
/// Returns [`SearchError::IndexNotFound`] for a missing path, I/O errors,
/// and [`SearchError::IndexCorrupted`] for bad magic, malformed or
/// CRC-drifted headers, WAL header corruption, or unsupported historical
/// versions — the same typed outcomes the corresponding open paths produce.
pub fn observe_tier(path: &Path) -> SearchResult<FsviTierObservation> {
    if !path.exists() {
        return Err(SearchError::IndexNotFound {
            path: path.to_path_buf(),
        });
    }
    let mut file = open_tier_readonly(path)?;
    let file_len = file.metadata().map_err(SearchError::Io)?.len();
    let mut prefix = [0_u8; 6];
    crate::read_exact_index_bytes(path, &mut file, &mut prefix, "magic and version")?;
    if prefix[..4] != crate::FSVI_MAGIC {
        return Err(crate::index_corrupted(
            path,
            format!(
                "bad magic bytes: expected {:?}, found {:?}",
                crate::FSVI_MAGIC,
                &prefix[..4]
            ),
        ));
    }
    let version = u16::from_le_bytes([prefix[4], prefix[5]]);
    match version {
        crate::FSVI_VERSION => {
            let bounded_len = usize::try_from(file_len)
                .unwrap_or(usize::MAX)
                .min(crate::FSVI_V1_MAX_HEADER_BYTES);
            file.seek(SeekFrom::Start(0)).map_err(SearchError::Io)?;
            let mut header = Vec::with_capacity(bounded_len);
            file.take(u64::try_from(bounded_len).unwrap_or(u64::MAX))
                .read_to_end(&mut header)
                .map_err(SearchError::Io)?;
            let (metadata, _header_len) = crate::parse_header(path, &header)?;
            Ok(FsviTierObservation::V1(observe_v1_wal(path, &metadata)?))
        }
        crate::FSVI_V2_VERSION => {
            let mut encoded_size = [0_u8; 4];
            crate::read_exact_index_bytes(path, &mut file, &mut encoded_size, "v2 header_size")?;
            let header_size = usize::try_from(u32::from_le_bytes(encoded_size)).map_err(|_| {
                crate::index_corrupted(path, "v2 header_size does not fit in usize")
            })?;
            crate::validate_v2_header_size(path, header_size)?;
            if u64::try_from(header_size).is_ok_and(|size| size > file_len) {
                return Err(crate::index_corrupted(
                    path,
                    format!(
                        "v2 header is truncated: declared {header_size} bytes, file has {file_len}"
                    ),
                ));
            }
            file.seek(SeekFrom::Start(0)).map_err(SearchError::Io)?;
            let mut header = vec![0_u8; header_size];
            crate::read_exact_index_bytes(path, &mut file, &mut header, "v2 header")?;
            let (metadata, _header_len) = crate::parse_v2_header(path, &header)?;
            Ok(FsviTierObservation::V2IdentityComplete(Box::new(metadata)))
        }
        found if found > crate::FSVI_V2_VERSION => {
            Ok(FsviTierObservation::UpgradeRequired(FsviUpgradeRequired {
                found_version: found,
                supported_version: crate::FSVI_V2_VERSION,
            }))
        }
        found => Err(crate::index_corrupted(
            path,
            format!("unsupported historical FSVI schema version {found}"),
        )),
    }
}

/// Read-only WAL observation for a v1 tier: same parser and same staleness
/// predicate as `VectorIndex::open`, minus the deletion/truncation side
/// effects.
fn observe_v1_wal(path: &Path, metadata: &VectorMetadata) -> SearchResult<FsviV1Observation> {
    let wal_path = crate::wal::wal_path_for(path);
    let wal_sidecar_present = wal_path.exists();
    let (entries, wal_compaction_gen, valid_len) =
        crate::wal::read_wal(&wal_path, metadata.dimension, metadata.quantization)?;
    // Staleness predicate mirrored from `VectorIndex::open` (the single
    // other consumer of `read_wal`'s generation output). A stale sidecar's
    // entries belong to a dead generation: they are not counted as retained
    // content, but the FILE is left exactly as found.
    let is_stale = if valid_len > 0 {
        if wal_compaction_gen == 0 {
            metadata.compaction_gen > 0
        } else {
            wal_compaction_gen != crate::next_generation(metadata.compaction_gen)
        }
    } else {
        false
    };
    let active_wal_records = if is_stale { 0 } else { entries.len() };
    Ok(FsviV1Observation {
        record_count: metadata.record_count,
        active_wal_records,
        wal_sidecar_present,
    })
}
