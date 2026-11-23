mod append_identity;
mod build_search;
mod persistence;
mod topology_fallback;

use std::cell::Cell;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{SystemTime, UNIX_EPOCH};

use frankensearch_core::{SearchError, SearchResult, VectorHit};

use crate::{Quantization, VectorIndex};

use super::fingerprint::fingerprint_hnsw_sidecar;
use super::persistence::{
    hnsw_generation_prefix, hnsw_sidecar_basename, install_hnsw_metadata,
    persisted_hnsw_sidecar_location,
};
use super::types::{
    HNSW_DEFAULT_EF_SEARCH, HNSW_GENERATION_RECEIPT_FILENAME, HnswConfig, HnswGenerationReceipt,
    HnswIndex, HnswLoadDisposition, HnswMeta,
};

#[derive(Debug)]
pub(super) struct GraphRepairProbe {
    pub(super) label: &'static str,
    pub(super) valid: bool,
    pub(super) live_graphs: Rc<Cell<usize>>,
}

impl GraphRepairProbe {
    pub(super) fn new(label: &'static str, valid: bool, live_graphs: &Rc<Cell<usize>>) -> Self {
        live_graphs.set(live_graphs.get().saturating_add(1));
        Self {
            label,
            valid,
            live_graphs: Rc::clone(live_graphs),
        }
    }
}

impl Drop for GraphRepairProbe {
    fn drop(&mut self) {
        self.live_graphs
            .set(self.live_graphs.get().saturating_sub(1));
    }
}

pub(super) fn temp_path(label: &str, extension: &str) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "frankensearch-hnsw-{label}-{}-{now}.{extension}",
        std::process::id()
    ))
}

pub(super) fn lcg_next(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    u32::try_from((*state >> 32) & u64::from(u32::MAX)).unwrap_or(u32::MAX)
}

pub(super) fn normalized_vector(seed: usize, dimension: usize) -> Vec<f32> {
    let mut state = u64::try_from(seed).unwrap_or(0).wrapping_add(1);
    let mut out = Vec::with_capacity(dimension);
    for _ in 0..dimension {
        let random = lcg_next(&mut state);
        let upper = u16::try_from((random >> 16) & u32::from(u16::MAX)).unwrap_or(u16::MAX);
        let raw = f32::from(upper) / f32::from(u16::MAX);
        out.push((raw * 2.0_f32) - 1.0_f32);
    }
    let norm = out.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut out {
            *value /= norm;
        }
    }
    out
}

pub(super) fn write_index(path: &Path, vectors: &[Vec<f32>]) -> SearchResult<VectorIndex> {
    write_index_with_quantization(path, vectors, Quantization::F32)
}

pub(super) fn write_index_with_quantization(
    path: &Path,
    vectors: &[Vec<f32>],
    quantization: Quantization,
) -> SearchResult<VectorIndex> {
    let dimension = vectors.first().map_or(8, Vec::len);
    let mut writer =
        VectorIndex::create_with_revision(path, "hash", "test", dimension, quantization)?;
    for (idx, vector) in vectors.iter().enumerate() {
        writer.write_record(&format!("doc-{idx:04}"), vector)?;
    }
    writer.finish()?;
    VectorIndex::open(path)
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct ImmutableTreeEntry {
    relative_path: PathBuf,
    kind: &'static str,
    len: u64,
    modified: Option<SystemTime>,
    readonly: bool,
    permission_mode: Option<u32>,
    bytes: Option<Vec<u8>>,
}

pub(super) fn snapshot_immutable_tree(root: &Path) -> std::io::Result<Vec<ImmutableTreeEntry>> {
    fn visit(
        root: &Path,
        path: &Path,
        entries: &mut Vec<ImmutableTreeEntry>,
    ) -> std::io::Result<()> {
        let metadata = std::fs::symlink_metadata(path)?;
        let file_type = metadata.file_type();
        let kind = if file_type.is_file() {
            "file"
        } else if file_type.is_dir() {
            "directory"
        } else if file_type.is_symlink() {
            "symlink"
        } else {
            "other"
        };
        let relative_path = path.strip_prefix(root).unwrap_or(path).to_path_buf();
        let bytes = file_type
            .is_file()
            .then(|| std::fs::read(path))
            .transpose()?;
        #[cfg(unix)]
        let permission_mode = {
            use std::os::unix::fs::PermissionsExt;
            Some(metadata.permissions().mode())
        };
        #[cfg(not(unix))]
        let permission_mode = None;
        entries.push(ImmutableTreeEntry {
            relative_path,
            kind,
            len: metadata.len(),
            modified: metadata.modified().ok(),
            readonly: metadata.permissions().readonly(),
            permission_mode,
            bytes,
        });
        if file_type.is_dir() {
            let mut children = std::fs::read_dir(path)?
                .map(|entry| entry.map(|entry| entry.path()))
                .collect::<std::io::Result<Vec<_>>>()?;
            children.sort();
            for child in children {
                visit(root, &child, entries)?;
            }
        }
        Ok(())
    }

    let mut entries = Vec::new();
    visit(root, root, &mut entries)?;
    entries.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    Ok(entries)
}

pub(super) fn reject_hnsw_metadata_publish(_: &Path, _: &Path, _: &[u8]) -> SearchResult<()> {
    Err(SearchError::Io(std::io::Error::other(
        "injected metadata publication failure",
    )))
}

pub(super) fn install_then_report_parent_sync_failure(
    path: &Path,
    parent: &Path,
    bytes: &[u8],
) -> SearchResult<()> {
    install_hnsw_metadata(path, parent, bytes)?;
    Err(SearchError::Io(std::io::Error::other(
        "injected post-rename parent sync failure",
    )))
}

pub(super) fn ready_generation_paths(
    metadata_path: &Path,
    vector_fingerprint: u64,
) -> Vec<PathBuf> {
    let parent = metadata_path.parent().unwrap_or_else(|| Path::new("."));
    let basename = hnsw_sidecar_basename(metadata_path).expect("metadata basename");
    let prefix = hnsw_generation_prefix(&basename, vector_fingerprint);
    let mut paths: Vec<PathBuf> = std::fs::read_dir(parent)
        .expect("read metadata parent")
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(&prefix))
                && entry
                    .path()
                    .join(HNSW_GENERATION_RECEIPT_FILENAME)
                    .is_file()
        })
        .map(|entry| entry.path())
        .collect();
    paths.sort_unstable();
    paths
}

pub(super) fn install_digest_valid_unloadable_data(generation_path: &Path, basename: &str) {
    let data_path = generation_path.join(format!("{basename}.hnsw.data"));
    std::fs::write(&data_path, b"not loadable HNSW vector data")
        .expect("corrupt native data fixture");
    let receipt_path = generation_path.join(HNSW_GENERATION_RECEIPT_FILENAME);
    let mut receipt: HnswGenerationReceipt =
        serde_json::from_slice(&std::fs::read(&receipt_path).expect("read receipt"))
            .expect("parse receipt");
    receipt.data = fingerprint_hnsw_sidecar(&data_path).expect("fingerprint corrupt data");
    std::fs::write(
        receipt_path,
        serde_json::to_vec(&receipt).expect("serialize updated receipt"),
    )
    .expect("make corrupt pair digest-valid");
}

pub(super) fn recall_at_k(approx: &[VectorHit], exact: &[VectorHit]) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }
    let exact_ids: HashSet<&str> = exact.iter().map(|hit| hit.doc_id.as_str()).collect();
    let overlap = approx
        .iter()
        .filter(|hit| exact_ids.contains(hit.doc_id.as_str()))
        .count();
    f64::from(u32::try_from(overlap).unwrap_or(u32::MAX))
        / f64::from(u32::try_from(exact.len()).unwrap_or(u32::MAX))
}

#[test]
fn save_accepts_bare_relative_metadata_path() {
    const CHILD_ENV: &str = "FRANKENSEARCH_HNSW_RELATIVE_SAVE_CHILD";

    if std::env::var_os(CHILD_ENV).is_some() {
        let ann = HnswIndex::build_from_parts(
            vec!["relative-doc".to_owned()],
            vec![vec![1.0_f32, 0.0]],
            2,
            HnswConfig::default(),
        )
        .expect("build relative-path ANN");
        let path = Path::new("relative.hnsw");
        ann.save(path).expect("save to bare relative path");

        let meta: HnswMeta =
            serde_json::from_slice(&std::fs::read(path).expect("read relative-path metadata"))
                .expect("parse relative-path metadata");
        let (sidecar_parent, basename) =
            persisted_hnsw_sidecar_location(path, &meta).expect("resolve relative sidecars");
        assert!(
            sidecar_parent
                .join(format!("{basename}.hnsw.graph"))
                .is_file()
        );
        assert!(
            sidecar_parent
                .join(format!("{basename}.hnsw.data"))
                .is_file()
        );
        return;
    }

    // The current working directory is process-global. Exercise the bare
    // path in a child test process so parallel tests cannot observe a cwd
    // change while still covering the end-to-end save contract.
    let directory = tempfile::tempdir().expect("relative-path test directory");
    let output =
        std::process::Command::new(std::env::current_exe().expect("current test executable"))
            .arg("hnsw::tests::save_accepts_bare_relative_metadata_path")
            .arg("--exact")
            .arg("--nocapture")
            .current_dir(directory.path())
            .env(CHILD_ENV, "1")
            .output()
            .expect("run relative-path child test");
    assert!(
        output.status.success(),
        "relative-path child failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn current_digest_valid_parser_panic_rebuilds_in_fresh_process() {
    const CHILD_ROOT_ENV: &str = "FRANKENSEARCH_HNSW_PARSER_PANIC_CHILD_ROOT";
    const TEST_NAME: &str =
        "hnsw::tests::current_digest_valid_parser_panic_rebuilds_in_fresh_process";

    if let Some(root) = std::env::var_os(CHILD_ROOT_ENV) {
        let root = PathBuf::from(root);
        let source_path = root.join("source.fsvi");
        let source_index = write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]])
            .expect("source index");
        let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
            .expect("ANN graph");
        let metadata_path = root.join("current.hnsw");
        ann.save(&metadata_path).expect("save native generation");
        let meta: HnswMeta =
            serde_json::from_slice(&std::fs::read(&metadata_path).expect("metadata"))
                .expect("parse metadata");
        let (generation_path, basename) =
            persisted_hnsw_sidecar_location(&metadata_path, &meta).expect("installed generation");
        install_digest_valid_unloadable_data(&generation_path, &basename);

        let (loaded, disposition) = HnswIndex::load_with_disposition(&metadata_path, &source_index)
            .expect("parser panic must degrade to a source rebuild");
        assert_eq!(disposition, HnswLoadDisposition::Rebuilt);
        let hits = loaded
            .knn_search(&[1.0, 0.0], 2, HNSW_DEFAULT_EF_SEARCH)
            .expect("search rebuilt graph");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "doc-0000");
        return;
    }

    let root = temp_path("parser-panic-child", "dir");
    std::fs::create_dir_all(&root).expect("create child fixture root");
    let status = std::process::Command::new(std::env::current_exe().expect("current test exe"))
        .arg("--exact")
        .arg(TEST_NAME)
        .arg("--nocapture")
        .env(CHILD_ROOT_ENV, &root)
        .status()
        .expect("run isolated malformed-parser child");
    assert!(
        status.success(),
        "digest-valid malformed native generation escaped the rebuild boundary: {status}"
    );
}
