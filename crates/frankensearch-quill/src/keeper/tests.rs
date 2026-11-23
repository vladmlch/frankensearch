#![allow(clippy::wildcard_imports)]

use std::ffi::{OsStr, OsString};
use std::fs::File;
#[cfg(unix)]
use std::fs::OpenOptions;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(unix)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime};

use asupersync::Cx;
#[cfg(unix)]
use asupersync::runtime::yield_now;
use asupersync::sync::LockError;
#[cfg(unix)]
use asupersync::sync::{Mutex, OwnedMutexGuard};
#[cfg(unix)]
use asupersync::types::Budget;
#[cfg(unix)]
use asupersync::{LabConfig, LabRuntime};
use frankensearch_core::generation::{
    CanonicalDocsetV1, CommitRange, GenerationComponentRole, SourceCheckpointV1,
};
use frankensearch_core::{DocId, SearchError};
use tempfile::tempdir;

use crate::error::QuillError;
use crate::grimoire::{
    ByteSpan, EncodedTermDictionary, TermDictionaryError, TermInput, TermMetadata,
    TermSectionLengths,
};
use crate::quiver::{
    BlockMaxConcatList, EncodedBlockMax, EncodedIdHashSection, EncodedIdMapSection,
    EncodedNumericSection, EncodedPostingList, IdMapEntryInput, IdMapPresence, NumericEntry,
    NumericFieldInput, Posting, PostingList, ValidatedTermPruningMetadata,
};
use crate::schema::{DEFAULT_SCHEMA, FSFS_CHUNK_SCHEMA};
use crate::segment::{
    EncodedSegment, SectionInput, SectionKind, SegmentHeaderInput, SegmentReader,
};

use super::*;

type TestResult = Result<(), Box<dyn std::error::Error>>;
const EMPTY_TERM_DICTIONARY: [u8; 4] = 0_u32.to_le_bytes();

mod helpers;
use helpers::*;

mod authority_tier_and_snapshot;
mod blue_green_and_freshness;
mod choreography_and_lab;
mod gc_grace_and_slots;
mod gc_witness_and_genesis;
mod manifest_codec_and_pair;
mod publish_adoption_and_preflight;
mod recovery_delete_and_concat;
mod segment_binding_and_termdict;
mod writer_admission;

#[test]
fn packed_engine_version_matches_crate_semver() {
    let major = env!("CARGO_PKG_VERSION_MAJOR")
        .parse::<u8>()
        .expect("crate major fits packed engine version");
    let minor = env!("CARGO_PKG_VERSION_MINOR")
        .parse::<u8>()
        .expect("crate minor fits packed engine version");
    let patch = env!("CARGO_PKG_VERSION_PATCH")
        .parse::<u16>()
        .expect("crate patch fits packed engine version");
    assert_eq!(
        CURRENT_ENGINE_VERSION,
        pack_engine_version(major, minor, patch)
    );
    assert_eq!(
        unpack_engine_version(CURRENT_ENGINE_VERSION),
        (major, minor, patch)
    );
}

#[cfg(any(unix, windows))]
#[test]
fn writer_lock_child_dispatch() {
    let Ok(role) = std::env::var("QUILL_WRITER_LOCK_ROLE") else {
        return;
    };
    let directory = PathBuf::from(
        std::env::var_os("QUILL_WRITER_LOCK_DIRECTORY")
            .expect("child index directory is configured"),
    );
    let control = PathBuf::from(
        std::env::var_os("QUILL_WRITER_LOCK_CONTROL")
            .expect("child control directory is configured"),
    );
    let identifier = std::env::var("QUILL_WRITER_LOCK_ID").expect("child id is configured");
    let result: Result<(), String> = run_with_test_cx(move |cx| async move {
        if role == "contend" {
            let start = control.join("START");
            while !start.exists() {
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        if role == "live_pid" {
            std::fs::write(control.join(format!("ready-{identifier}")), [])
                .map_err(|error| error.to_string())?;
            let release = control.join("RELEASE");
            while !release.exists() {
                std::thread::sleep(Duration::from_millis(10));
            }
            return Ok(());
        }
        match KeeperWriter::open(&cx, &directory, DEFAULT_SCHEMA).await {
            Ok(writer) => {
                if role == "hold_unpublished" {
                    std::fs::write(
                        directory.join("seg-feedfacefeedface.fslx"),
                        b"sealed but not yet published",
                    )
                    .map_err(|error| error.to_string())?;
                }
                let marker = if role == "contend" {
                    control.join(format!("winner-{identifier}"))
                } else {
                    control.join(format!("ready-{identifier}"))
                };
                std::fs::write(&marker, []).map_err(|error| error.to_string())?;
                let release = control.join("RELEASE");
                while !release.exists() {
                    std::thread::sleep(Duration::from_millis(10));
                }
                drop(writer);
                Ok(())
            }
            Err(KeeperError::WriterBusy { .. }) if role == "contend" => {
                std::fs::write(control.join(format!("busy-{identifier}")), [])
                    .map_err(|error| error.to_string())
            }
            Err(error) => Err(error.to_string()),
        }
    });
    result.unwrap_or_else(|error| panic!("writer child failed: {error}"));
}
