use std::fs;
use std::path::{Path, PathBuf};

use frankensearch_core::{SearchError, SearchResult};

/// Explicit filesystem layout for opening a [`TwoTierIndex`](super::TwoTierIndex).
///
/// Use this when a consumer owns the index naming convention instead of using
/// the default `vector.fast.idx` / `vector.quality.idx` layout. Every configured
/// role must resolve to a distinct artifact. When the `ann` feature is enabled,
/// omitting an ANN sidecar path explicitly disables ANN for that tier even if
/// the configured record-count threshold is met. ANN paths are writable:
/// opening an index may create or replace those sidecars. Callers must keep all
/// configured artifact directories trusted and stable for the duration of
/// [`TwoTierIndex::open_with_paths`](super::TwoTierIndex::open_with_paths).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TwoTierIndexPaths {
    pub(super) fast_index: PathBuf,
    pub(super) quality_index: Option<PathBuf>,
    #[cfg(feature = "ann")]
    pub(super) fast_ann: Option<PathBuf>,
    #[cfg(feature = "ann")]
    pub(super) quality_ann: Option<PathBuf>,
}

impl TwoTierIndexPaths {
    /// Create an explicit layout with a required fast index and no quality tier.
    #[must_use]
    pub fn new(fast_index: impl Into<PathBuf>) -> Self {
        Self {
            fast_index: fast_index.into(),
            quality_index: None,
            #[cfg(feature = "ann")]
            fast_ann: None,
            #[cfg(feature = "ann")]
            quality_ann: None,
        }
    }

    /// Add an explicit quality-tier index path.
    #[must_use]
    pub fn with_quality_index(mut self, quality_index: impl Into<PathBuf>) -> Self {
        self.quality_index = Some(quality_index.into());
        self
    }

    /// Add an explicit fast-tier ANN sidecar path.
    #[cfg(feature = "ann")]
    #[must_use]
    pub fn with_fast_ann(mut self, fast_ann: impl Into<PathBuf>) -> Self {
        self.fast_ann = Some(fast_ann.into());
        self
    }

    /// Add an explicit quality-tier ANN sidecar path.
    #[cfg(feature = "ann")]
    #[must_use]
    pub fn with_quality_ann(mut self, quality_ann: impl Into<PathBuf>) -> Self {
        self.quality_ann = Some(quality_ann.into());
        self
    }

    /// Required fast-tier FSVI path.
    #[must_use]
    pub fn fast_index(&self) -> &Path {
        &self.fast_index
    }

    /// Optional quality-tier FSVI path.
    #[must_use]
    pub fn quality_index(&self) -> Option<&Path> {
        self.quality_index.as_deref()
    }

    /// Optional fast-tier ANN sidecar path.
    #[cfg(feature = "ann")]
    #[must_use]
    pub fn fast_ann(&self) -> Option<&Path> {
        self.fast_ann.as_deref()
    }

    /// Optional quality-tier ANN sidecar path.
    #[cfg(feature = "ann")]
    #[must_use]
    pub fn quality_ann(&self) -> Option<&Path> {
        self.quality_ann.as_deref()
    }

    /// Freeze every relative artifact path against the process's current
    /// directory.
    ///
    /// This preserves the original path components (including symlinks and
    /// `..`) so the operating system, rather than lexical path rewriting,
    /// determines their meaning.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the current directory cannot be read.
    pub fn into_absolute(self) -> SearchResult<Self> {
        let current_dir = std::env::current_dir()?;
        self.into_absolute_from(&current_dir)
    }

    /// Freeze every relative artifact path against one captured absolute base.
    ///
    /// This is useful when a higher-level constructor must resolve this layout
    /// and additional paths against the exact same current-directory snapshot.
    /// As with [`Self::into_absolute`], path components are preserved for the
    /// operating system to resolve.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when `base` is not absolute.
    pub fn into_absolute_from(mut self, base: &Path) -> SearchResult<Self> {
        if !base.is_absolute() {
            return Err(SearchError::InvalidConfig {
                field: "index_paths_base".to_owned(),
                value: base.display().to_string(),
                reason: "the explicit-path resolution base must be absolute".to_owned(),
            });
        }
        self.fast_index = make_path_absolute(base, self.fast_index);
        self.quality_index = self
            .quality_index
            .map(|path| make_path_absolute(base, path));
        #[cfg(feature = "ann")]
        {
            self.fast_ann = self.fast_ann.map(|path| make_path_absolute(base, path));
            self.quality_ann = self.quality_ann.map(|path| make_path_absolute(base, path));
        }
        Ok(self)
    }
}

fn make_path_absolute(current_dir: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        current_dir.join(path)
    }
}

fn canonical_path_identity(path: &Path) -> SearchResult<PathBuf> {
    let absolute = TwoTierIndexPaths::new(path).into_absolute()?.fast_index;
    let mut cursor = absolute.as_path();
    let mut missing_components = Vec::new();

    loop {
        match fs::canonicalize(cursor) {
            Ok(mut canonical) => {
                for component in missing_components.iter().rev() {
                    canonical.push(component);
                }
                return Ok(canonical);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let Some(file_name) = cursor.file_name() else {
                    return Err(SearchError::Io(error));
                };
                missing_components.push(file_name.to_os_string());
                let Some(parent) = cursor.parent() else {
                    return Err(SearchError::Io(error));
                };
                cursor = parent;
            }
            Err(error) => return Err(SearchError::Io(error)),
        }
    }
}

pub(super) fn paths_alias(left: &Path, right: &Path) -> SearchResult<bool> {
    let left_identity = canonical_path_identity(left)?;
    let right_identity = canonical_path_identity(right)?;
    if left_identity == right_identity {
        return Ok(true);
    }

    match crate::file_identity::is_same_file(left, right) {
        Ok(is_same) => Ok(is_same),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(SearchError::Io(error)),
    }
}

fn reject_final_symlink(role: &str, path: &Path) -> SearchResult<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(SearchError::InvalidConfig {
            field: "index_paths".to_owned(),
            value: format!("{role}={}", path.display()),
            reason: "configured artifact roles must not be final-component symlinks".to_owned(),
        }),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(SearchError::Io(error)),
    }
}

pub(super) fn validate_index_paths(paths: &TwoTierIndexPaths) -> SearchResult<()> {
    #[cfg(feature = "ann")]
    if paths.quality_ann.is_some() && paths.quality_index.is_none() {
        return Err(SearchError::InvalidConfig {
            field: "quality_ann".to_owned(),
            value: paths
                .quality_ann
                .as_deref()
                .map_or_else(String::new, |path| path.display().to_string()),
            reason: "a quality ANN sidecar requires a quality index path".to_owned(),
        });
    }

    let mut roles = vec![("fast_index", paths.fast_index())];
    if let Some(path) = paths.quality_index() {
        roles.push(("quality_index", path));
    }
    #[cfg(feature = "ann")]
    if let Some(path) = paths.fast_ann() {
        roles.push(("fast_ann", path));
    }
    #[cfg(feature = "ann")]
    if let Some(path) = paths.quality_ann() {
        roles.push(("quality_ann", path));
    }

    for (role, path) in &roles {
        reject_final_symlink(role, path)?;
    }

    for (index, (left_role, left_path)) in roles.iter().enumerate() {
        for (right_role, right_path) in &roles[index + 1..] {
            if paths_alias(left_path, right_path)? {
                return Err(SearchError::InvalidConfig {
                    field: "index_paths".to_owned(),
                    value: format!(
                        "{left_role}={}, {right_role}={}",
                        left_path.display(),
                        right_path.display()
                    ),
                    reason: "each index and ANN role must reference a distinct artifact".to_owned(),
                });
            }
        }
    }

    Ok(())
}

#[cfg(feature = "ann")]
pub(super) fn validate_ann_save_lock_identities(
    paths: &TwoTierIndexPaths,
    fast_ann_enabled: bool,
    quality_ann_enabled: bool,
) -> SearchResult<()> {
    let configured_roles = {
        let mut roles = vec![("fast_index", paths.fast_index())];
        if let Some(path) = paths.quality_index() {
            roles.push(("quality_index", path));
        }
        if let Some(path) = paths.fast_ann() {
            roles.push(("fast_ann", path));
        }
        if let Some(path) = paths.quality_ann() {
            roles.push(("quality_ann", path));
        }
        roles
    };

    let mut lock_roles = Vec::with_capacity(2);
    for (role, path, enabled) in [
        ("fast_ann_lock", paths.fast_ann(), fast_ann_enabled),
        ("quality_ann_lock", paths.quality_ann(), quality_ann_enabled),
    ] {
        let Some(path) = path.filter(|_| enabled) else {
            continue;
        };
        let lock_path = crate::hnsw::hnsw_save_lock_artifact_path(path).map_err(|error| {
            SearchError::InvalidConfig {
                field: role.to_owned(),
                value: path.display().to_string(),
                reason: error.to_string(),
            }
        })?;
        for (artifact_role, artifact_path) in &configured_roles {
            if paths_alias(&lock_path, artifact_path)? {
                return Err(SearchError::InvalidConfig {
                    field: "index_paths".to_owned(),
                    value: format!(
                        "{role}={}, {artifact_role}={}",
                        lock_path.display(),
                        artifact_path.display()
                    ),
                    reason: "an ANN save-lock artifact must not alias any configured index or ANN \
                             role"
                        .to_owned(),
                });
            }
        }
        lock_roles.push((role, lock_path));
    }

    for (index, (left_role, left_path)) in lock_roles.iter().enumerate() {
        for (right_role, right_path) in &lock_roles[index + 1..] {
            if paths_alias(left_path, right_path)? {
                return Err(SearchError::InvalidConfig {
                    field: "index_paths".to_owned(),
                    value: format!(
                        "{left_role}={}, {right_role}={}",
                        left_path.display(),
                        right_path.display()
                    ),
                    reason: "configured ANN roles resolve to the same filesystem artifact"
                        .to_owned(),
                });
            }
        }
    }

    // Materializing the persistent lock artifacts turns still-missing,
    // potentially case/normalization-equivalent ANN leaves into real files.
    // Comparing those files delegates collation to the mounted filesystem
    // instead of guessing from the operating system.
    let mut materialized = Vec::with_capacity(lock_roles.len());
    for (role, lock_path) in lock_roles {
        let identity = crate::hnsw::materialize_hnsw_save_lock_artifact(&lock_path)?;
        materialized.push((role, lock_path, identity));
    }
    for (index, (left_role, left_path, left_identity)) in materialized.iter().enumerate() {
        for (right_role, right_path, right_identity) in &materialized[index + 1..] {
            if left_identity == right_identity
                || crate::file_identity::is_same_file(left_path, right_path)
                    .map_err(SearchError::Io)?
            {
                return Err(SearchError::InvalidConfig {
                    field: "index_paths".to_owned(),
                    value: format!(
                        "{left_role}={}, {right_role}={}",
                        left_path.display(),
                        right_path.display()
                    ),
                    reason: "configured ANN roles alias under the mounted filesystem's path \
                             comparison rules"
                        .to_owned(),
                });
            }
        }
    }

    Ok(())
}

#[cfg(feature = "ann")]
pub(super) fn validate_ann_persistence_paths(
    paths: &TwoTierIndexPaths,
    fast_ann_enabled: bool,
    quality_ann_enabled: bool,
) -> SearchResult<()> {
    validate_index_paths(paths)?;
    validate_ann_save_lock_identities(paths, fast_ann_enabled, quality_ann_enabled)
}
