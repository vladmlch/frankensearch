use std::io;
use std::sync::Arc;

use crate::schema::SchemaDescriptor;

use super::admission::WriterAdmissionInner;
use super::artifact_install::{
    append_path_suffix, remove_stale_generation_claim, retire_regular_artifact,
};
use super::errors::KeeperError;
use super::manifest_load::{load_manifest_pair, scan_generation_claims};
use super::slot_io::{ManifestSlot, read_manifest_slot};
use super::writer::WriterProtection;

pub(in crate::keeper) fn recover_writer_directory(
    admission: &Arc<WriterAdmissionInner>,
    schema: SchemaDescriptor,
    protection: &WriterProtection,
) -> Result<(), KeeperError> {
    admission.ensure_directory_identity()?;
    recover_interrupted_generation_claims(admission)?;
    recover_corrupt_primary_slot(admission)?;
    let _ = schema;
    let _ = protection;
    admission.ensure_directory_identity()
}

fn recover_interrupted_generation_claims(
    admission: &Arc<WriterAdmissionInner>,
) -> Result<(), KeeperError> {
    let recovered_generation = match load_manifest_pair(&admission.directory) {
        Ok(loaded) => loaded.manifest.generation,
        Err(KeeperError::IndexNotFound { .. }) => 0,
        Err(error) => return Err(error),
    };
    let next_generation = recovered_generation.checked_add(1);
    for (path, generation) in scan_generation_claims(&admission.directory)? {
        if generation > recovered_generation && Some(generation) != next_generation {
            return Err(KeeperError::InvalidRecoveryClaim {
                path,
                recovered: recovered_generation,
                claimed: generation,
            });
        }
        if Some(generation) == next_generation {
            retire_interrupted_manifest_temp(admission, generation)?;
        }
        remove_stale_generation_claim(admission, &path)?;
    }
    Ok(())
}

fn retire_interrupted_manifest_temp(
    admission: &WriterAdmissionInner,
    generation: u64,
) -> Result<(), KeeperError> {
    let source = admission
        .directory
        .join(format!(".tmp-manifest-{generation}"));
    let retired = admission.directory.join(format!(
        ".tmp-abandoned-manifest-{generation}-{:016x}",
        admission.record.pid_start_nonce
    ));
    let changed = retire_regular_artifact(admission, &source, &retired)?;
    let source_sidecar = append_path_suffix(&source, ".fec");
    let retired_sidecar = append_path_suffix(&retired, ".fec");
    let changed = retire_regular_artifact(admission, &source_sidecar, &retired_sidecar)? || changed;
    if changed {
        admission
            .directory_file
            .sync_all()
            .map_err(|source| KeeperError::Io {
                operation: "fsync abandoned MANIFEST temp retirement",
                path: admission.directory.clone(),
                source,
            })?;
    }
    Ok(())
}

fn recover_corrupt_primary_slot(admission: &WriterAdmissionInner) -> Result<(), KeeperError> {
    let current = admission.directory.join("MANIFEST");
    let previous = admission.directory.join("MANIFEST.prev");
    let current_slot = read_manifest_slot(&current)?;
    let previous_slot = read_manifest_slot(&previous)?;
    if matches!(current_slot, ManifestSlot::Invalid(_))
        && matches!(previous_slot, ManifestSlot::Valid(_))
    {
        let retired = admission.directory.join(format!(
            ".tmp-corrupt-manifest-{:016x}",
            admission.record.pid_start_nonce
        ));
        let changed = retire_regular_artifact(admission, &current, &retired)?;
        let current_sidecar = append_path_suffix(&current, ".fec");
        let retired_sidecar = append_path_suffix(&retired, ".fec");
        let changed =
            retire_regular_artifact(admission, &current_sidecar, &retired_sidecar)? || changed;
        if changed {
            admission
                .directory_file
                .sync_all()
                .map_err(|source| KeeperError::Io {
                    operation: "fsync corrupt MANIFEST retirement",
                    path: admission.directory.clone(),
                    source,
                })?;
        }
        return Ok(());
    }

    // A crash between the two renames above leaves MANIFEST retired while its
    // sidecar remains canonical; complete that interrupted retirement once.
    if matches!(current_slot, ManifestSlot::Missing)
        && matches!(previous_slot, ManifestSlot::Valid(_))
    {
        recover_interrupted_corrupt_retirement(admission)?;
    }
    Ok(())
}

/// Complete an interrupted corrupt-primary retirement (bd-qxce).
///
/// Corrupt-primary retirement performs two renames — `MANIFEST` into its
/// no-replace quarantine, then `MANIFEST.fec` — followed by one directory
/// fsync. A crash between the renames leaves the sidecar at its canonical
/// name while the `.tmp-corrupt-manifest-*` quarantine records the deliberate
/// retirement. Future durable opens would otherwise reconsider that orphan
/// sidecar on every admission, repeating the repair attempt the retiring
/// writer already rejected. When the retirement evidence exists, retire the
/// orphan sidecar once, deterministically, into its own no-replace
/// quarantine. A canonical sidecar without retirement evidence is a
/// legitimate crash survivor and is left untouched.
fn recover_interrupted_corrupt_retirement(
    admission: &WriterAdmissionInner,
) -> Result<(), KeeperError> {
    let sidecar = admission.directory.join("MANIFEST.fec");
    let sidecar_exists = match std::fs::symlink_metadata(&sidecar) {
        Ok(metadata) if metadata.file_type().is_file() => true,
        Ok(_) => {
            return Err(KeeperError::Io {
                operation: "inspect interrupted-retirement sidecar",
                path: sidecar.clone(),
                source: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "interrupted-retirement sidecar is not a regular file",
                ),
            });
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => false,
        Err(error) => {
            return Err(KeeperError::Io {
                operation: "inspect interrupted-retirement sidecar",
                path: sidecar.clone(),
                source: error,
            });
        }
    };
    if !sidecar_exists {
        return Ok(());
    }
    let mut retirement_evidence = false;
    let entries = std::fs::read_dir(&admission.directory).map_err(|source| KeeperError::Io {
        operation: "scan for corrupt-retirement evidence",
        path: admission.directory.clone(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| KeeperError::Io {
            operation: "read corrupt-retirement evidence entry",
            path: admission.directory.clone(),
            source,
        })?;
        if entry
            .file_name()
            .to_str()
            .is_some_and(|name| name.starts_with(".tmp-corrupt-manifest-"))
        {
            retirement_evidence = true;
            break;
        }
    }
    if !retirement_evidence {
        return Ok(());
    }
    let retired = admission.directory.join(format!(
        ".tmp-corrupt-manifest-fec-{:016x}",
        admission.record.pid_start_nonce
    ));
    if retire_regular_artifact(admission, &sidecar, &retired)? {
        admission
            .directory_file
            .sync_all()
            .map_err(|source| KeeperError::Io {
                operation: "fsync interrupted corrupt-sidecar retirement",
                path: admission.directory.clone(),
                source,
            })?;
    }
    Ok(())
}
