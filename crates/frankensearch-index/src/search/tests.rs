use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use frankensearch_core::{SearchError, SearchResult};

use super::*;
use crate::{Quantization, VectorIndex};

mod core_and_two_pass;
mod filters_wal_and_helpers;

pub(super) fn temp_index_path(name: &str) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "frankensearch-index-search-{name}-{}-{now}.fsvi",
        std::process::id()
    ))
}

pub(super) fn write_index(path: &std::path::Path, rows: &[(&str, Vec<f32>)]) -> SearchResult<()> {
    let dimension =
        rows.first()
            .map(|(_, vec)| vec.len())
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "rows".to_owned(),
                value: "[]".to_owned(),
                reason: "rows must not be empty".to_owned(),
            })?;
    let mut writer =
        VectorIndex::create_with_revision(path, "hash", "test", dimension, Quantization::F16)?;
    for (doc_id, vector) in rows {
        writer.write_record(doc_id, vector)?;
    }
    writer.finish()
}

pub(super) fn create_rows(vectors: &[Vec<f32>]) -> Vec<(String, Vec<f32>)> {
    vectors
        .iter()
        .enumerate()
        .map(|(idx, vector)| (format!("doc-{idx:03}"), vector.clone()))
        .collect()
}

pub(super) fn hit_ids(hits: &[VectorHit]) -> Vec<String> {
    hits.iter().map(|hit| hit.doc_id.to_string()).collect()
}
