use std::fs::File;
use std::path::Path;

use frankensearch_core::{SearchError, SearchResult};
use sha2::{Digest, Sha256};

use crate::stats::{SegmentStats, SegmentStatsProvider};

use super::admission::{read_writer_lock_record, writer_lock_record_names_live_owner};
use super::constants::MAX_MANIFEST_BYTES;
use super::errors::{KeeperError, ManifestCodecError};
use super::manifest::{Manifest, ManifestSource};
use super::manifest_validation::usize_to_u64;
use super::publish::{parse_quarantined_segment_name, parse_quarantined_sidecar_segment_name};
use super::slot_io::open_manifest_slot;
use super::snapshot::KeeperSnapshot;
use super::writer::KeeperWriter;

pub(in crate::keeper) struct ByteCursor<'a> {
    pub(in crate::keeper) bytes: &'a [u8],
    position: usize,
}

// ============================================================================
// Segment statistics (visibility contract: published_generation,
// last_publish_unix from the MANIFEST v2 freshness witness, live_writer from
// the D1 LOCK record + POSIX liveness).
// ============================================================================

/// Best-effort total of Quill-managed bytes under `directory`.
///
/// Sums FSLX segments, MANIFEST slots and repair sidecars, and lifecycle
/// metadata (LOCK). Entries that vanish or cannot be stat'ed mid-walk are
/// skipped: this is a status number, never an accounting authority.
// Engine-managed filenames are canonical lowercase by construction; a
// case-insensitive compare would mis-count foreign files the engine never
// wrote.
#[allow(clippy::case_sensitive_file_extension_comparisons)]
pub(in crate::keeper) fn managed_disk_bytes(directory: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return 0;
    };
    let mut total = 0_u64;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let managed = (name.starts_with("seg-") && name.ends_with(".fslx"))
            || parse_quarantined_segment_name(name).is_some()
            || parse_quarantined_sidecar_segment_name(name).is_some()
            || matches!(name, "MANIFEST" | "MANIFEST.prev" | "LOCK")
            || name.ends_with(".fec");
        if !managed {
            continue;
        }
        if let Ok(metadata) = entry.metadata() {
            if metadata.is_file() {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    total
}

pub(in crate::keeper) fn hash_exact_snapshot_frame(
    hasher: &mut Sha256,
    role: &[u8],
    bytes: &[u8],
) -> Result<(), KeeperError> {
    let role_len = u16::try_from(role.len()).map_err(|_| KeeperError::ExactSnapshotReceipt {
        detail: "exact lexical snapshot frame role exceeded u16".to_owned(),
    })?;
    let byte_len = u64::try_from(bytes.len()).map_err(|_| KeeperError::ExactSnapshotReceipt {
        detail: "exact lexical snapshot frame length exceeded u64".to_owned(),
    })?;
    hasher.update(role_len.to_le_bytes());
    hasher.update(role);
    hasher.update(byte_len.to_le_bytes());
    hasher.update(bytes);
    Ok(())
}

pub(in crate::keeper) fn exact_snapshot_manifest_bytes(
    snapshot: &KeeperSnapshot,
) -> Result<Vec<u8>, KeeperError> {
    let directory =
        snapshot
            .directory
            .as_deref()
            .ok_or_else(|| KeeperError::ExactSnapshotReceipt {
                detail: "in-memory snapshot has no immutable MANIFEST descriptor".to_owned(),
            })?;
    let name = match snapshot.loaded.source {
        ManifestSource::Current => "MANIFEST",
        ManifestSource::PreviousAfterMissingCurrent
        | ManifestSource::PreviousAfterCorruptCurrent => "MANIFEST.prev",
        ManifestSource::InMemory => {
            return Err(KeeperError::ExactSnapshotReceipt {
                detail: "in-memory snapshot has no immutable MANIFEST descriptor".to_owned(),
            });
        }
    };
    let path = directory.join(name);
    let file = open_manifest_slot(&path).map_err(|source| KeeperError::Io {
        operation: "open exact lexical snapshot manifest",
        path: path.clone(),
        source,
    })?;
    let byte_len = file.metadata().map_err(|source| KeeperError::Io {
        operation: "stat exact lexical snapshot manifest",
        path: path.clone(),
        source,
    })?;
    if byte_len.len() > usize_to_u64(MAX_MANIFEST_BYTES) {
        return Err(KeeperError::ExactSnapshotReceipt {
            detail: "selected MANIFEST exceeded the receipt byte limit".to_owned(),
        });
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(usize::try_from(byte_len.len()).unwrap_or(MAX_MANIFEST_BYTES))
        .map_err(|error| KeeperError::ExactSnapshotReceipt {
            detail: format!("could not allocate selected MANIFEST bytes: {error}"),
        })?;
    file.take(usize_to_u64(MAX_MANIFEST_BYTES).saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|source| KeeperError::Io {
            operation: "read exact lexical snapshot manifest",
            path: path.clone(),
            source,
        })?;
    if bytes.len() > MAX_MANIFEST_BYTES {
        return Err(KeeperError::ExactSnapshotReceipt {
            detail: "selected MANIFEST exceeded the receipt byte limit while reading".to_owned(),
        });
    }
    let decoded =
        Manifest::from_bytes(&bytes).map_err(|error| KeeperError::ExactSnapshotReceipt {
            detail: format!("selected MANIFEST no longer decodes: {error}"),
        })?;
    if decoded != snapshot.loaded.manifest {
        return Err(KeeperError::ExactSnapshotReceipt {
            detail: "selected MANIFEST changed since the snapshot was opened".to_owned(),
        });
    }
    Ok(bytes)
}

/// Whether a live writer currently holds `directory`'s LOCK.
///
/// Reads the D1 LOCK record and applies the POSIX `kill(pid, 0)` liveness
/// rule. Linux additionally requires the recorded process-start nonce to
/// match the currently observed identity for that pid; an unavailable start
/// witness stays conservatively live. Any read/parse failure reports no live
/// writer. Non-Unix targets cannot prove liveness, so a valid record alone
/// decides there.
pub(in crate::keeper) fn detect_live_writer(directory: &Path) -> bool {
    let lock_path = directory.join("LOCK");
    let Ok(mut file) = File::open(&lock_path) else {
        return false;
    };
    let Ok(Some(record)) = read_writer_lock_record(&lock_path, &mut file) else {
        return false;
    };
    writer_lock_record_names_live_owner(record)
}

impl SegmentStatsProvider for KeeperSnapshot {
    fn segment_stats(&self) -> SearchResult<SegmentStats> {
        let manifest = &self.loaded.manifest;
        let (managed_disk_bytes, live_writer) =
            self.directory.as_ref().map_or((0, false), |directory| {
                (managed_disk_bytes(directory), detect_live_writer(directory))
            });
        let live_docs =
            usize::try_from(self.live_doc_count).map_err(|_| SearchError::SubsystemError {
                subsystem: "quill",
                source: "Keeper live-document count does not fit usize".into(),
            })?;
        let tombstones =
            usize::try_from(self.tombstone_count).map_err(|_| SearchError::SubsystemError {
                subsystem: "quill",
                source: "Keeper tombstone count does not fit usize".into(),
            })?;
        Ok(SegmentStats {
            schema_id: manifest.schema_id,
            published_generation: manifest.generation,
            sealed_segments: self.segments.len(),
            // The searchable delta segment is E5; nothing pre-delta exists yet.
            delta_segments: 0,
            live_docs,
            tombstones,
            managed_disk_bytes,
            delta_memory_bytes: 0,
            last_publish_unix: (manifest.last_publish_unix_s != 0)
                .then_some(manifest.last_publish_unix_s),
            live_writer,
            degraded: self.is_degraded(),
            quarantined_segments: self.quarantined_segments.len(),
            estimated_missing_docs: self.estimated_missing_docs(),
            unknown_missing_doc_segments: self
                .quarantined_segments
                .iter()
                .filter(|segment| segment.estimated_missing_docs.is_none())
                .count(),
        })
    }
}

impl SegmentStatsProvider for KeeperWriter {
    fn segment_stats(&self) -> SearchResult<SegmentStats> {
        let mut stats = self
            .snapshot()
            .map_err(SearchError::from)?
            .segment_stats()?;
        // The writer holds LOCK by construction; no probe is needed.
        stats.live_writer = true;
        Ok(stats)
    }
}

impl<'a> ByteCursor<'a> {
    pub(in crate::keeper) const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    pub(in crate::keeper) const fn position(&self) -> usize {
        self.position
    }

    pub(in crate::keeper) const fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    pub(in crate::keeper) fn take(
        &mut self,
        length: usize,
    ) -> Result<&'a [u8], ManifestCodecError> {
        let end =
            self.position
                .checked_add(length)
                .ok_or_else(|| ManifestCodecError::Truncated {
                    offset: self.position,
                    needed: length,
                    remaining: self.remaining(),
                })?;
        let Some(bytes) = self.bytes.get(self.position..end) else {
            return Err(ManifestCodecError::Truncated {
                offset: self.position,
                needed: length,
                remaining: self.remaining(),
            });
        };
        self.position = end;
        Ok(bytes)
    }

    pub(in crate::keeper) fn u8(&mut self) -> Result<u8, ManifestCodecError> {
        Ok(self.take(1)?[0])
    }

    pub(in crate::keeper) fn u16(&mut self) -> Result<u16, ManifestCodecError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    pub(in crate::keeper) fn u32(&mut self) -> Result<u32, ManifestCodecError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    pub(in crate::keeper) fn u64(&mut self) -> Result<u64, ManifestCodecError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }
}
