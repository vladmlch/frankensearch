//! bd-ctzo C5: validate the machine-readable activation call-site census.
//!
//! `docs/CTZO_ACTIVATION_CENSUS.json` names every PRODUCTION entry point that
//! was migrated onto typed owner-backed activation. A checked-in list proves
//! nothing on its own — it rots the moment someone adds a call site — so this
//! suite is the proof half:
//!
//! - every named symbol still exists, in the file the census names, in that
//!   file's PRODUCTION half (above `#[cfg(test)] mod tests`);
//! - the number of production call sites of the activation constructor and of
//!   the coverage receipt matches the census exactly, so adding or deleting
//!   one without updating the census fails here;
//! - the new types are consumed OUTSIDE `frankensearch-core`, which is the
//!   half of C5 that distinguishes "defined" from "used".
//!
//! The census is read from the workspace source tree, not from a build
//! artifact, because the claim is about what the shipped source does.

use std::path::{Path, PathBuf};

/// Workspace root, derived from this crate's manifest directory.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/<crate> sits two levels below the workspace root")
        .to_path_buf()
}

fn read(relative: &str) -> String {
    let path = workspace_root().join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!("census names {relative}, which could not be read: {error}");
    })
}

/// Collect non-test `.rs` files under `dir` recursively in deterministic sorted order.
fn collect_production_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|error| {
        panic!(
            "census could not read submodule directory {}: {error}",
            dir.display()
        );
    });
    for entry in entries {
        let entry = entry.unwrap_or_else(|error| {
            panic!(
                "census failed reading directory entry under {}: {error}",
                dir.display()
            );
        });
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if path.is_dir() {
            if name == "tests" {
                continue;
            }
            collect_production_rs_files(&path, out);
        } else if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("rs"))
        {
            let is_test_file =
                path.file_stem()
                    .and_then(|stem| stem.to_str())
                    .is_some_and(|stem| {
                        stem == "tests" || stem.ends_with("_tests") || stem.ends_with(".tests")
                    });
            if !is_test_file {
                out.push(path);
            }
        }
    }
    out.sort();
}

/// Skip comments, string literals, and char literals starting at `idx`,
/// returning the index of the next code byte if `bytes[idx]` opened one.
fn skip_non_code_token(bytes: &[u8], idx: usize) -> Option<usize> {
    if bytes[idx] == b'/' && bytes.get(idx + 1) == Some(&b'/') {
        let mut cursor = idx + 2;
        while cursor < bytes.len() && bytes[cursor] != b'\n' {
            cursor += 1;
        }
        return Some(cursor);
    }
    if bytes[idx] == b'/' && bytes.get(idx + 1) == Some(&b'*') {
        let mut cursor = idx + 2;
        let mut depth = 1_usize;
        while cursor + 1 < bytes.len() && depth > 0 {
            if bytes[cursor] == b'/' && bytes[cursor + 1] == b'*' {
                depth += 1;
                cursor += 2;
            } else if bytes[cursor] == b'*' && bytes[cursor + 1] == b'/' {
                depth -= 1;
                cursor += 2;
            } else {
                cursor += 1;
            }
        }
        return Some(cursor);
    }
    if bytes[idx] == b'"' {
        let mut cursor = idx + 1;
        while cursor < bytes.len() {
            if bytes[cursor] == b'\\' {
                cursor = (cursor + 2).min(bytes.len());
            } else if bytes[cursor] == b'"' {
                return Some(cursor + 1);
            } else {
                cursor += 1;
            }
        }
        return Some(bytes.len());
    }
    if bytes[idx] == b'\'' {
        if bytes.get(idx + 1) == Some(&b'\\') {
            let mut cursor = idx + 3;
            while cursor < bytes.len() && bytes[cursor] != b'\'' && bytes[cursor] != b'\n' {
                cursor += 1;
            }
            return Some((cursor + 1).min(bytes.len()));
        }
        if bytes.get(idx + 2) == Some(&b'\'') {
            return Some(idx + 3);
        }
    }
    None
}

/// Advance past one `#[cfg(test)]` Rust item (an out-of-line `mod tests;` /
/// `const ...;` ending in `;` at depth 0, or a braced item `mod tests { ... }`
/// / `fn ... { ... }`).
fn skip_cfg_test_item(bytes: &[u8], start: usize) -> usize {
    let mut cursor = start;
    let mut attr_depth = 0_usize;
    while cursor < bytes.len() {
        if let Some(next) = skip_non_code_token(bytes, cursor) {
            cursor = next;
            continue;
        }
        if bytes[cursor] == b'[' {
            attr_depth += 1;
        } else if bytes[cursor] == b']' {
            attr_depth = attr_depth.saturating_sub(1);
            if attr_depth == 0 {
                cursor += 1;
                break;
            }
        }
        cursor += 1;
    }

    let mut brace_depth = 0_usize;
    let mut bracket_depth = 0_usize;
    let mut saw_brace = false;
    while cursor < bytes.len() {
        if let Some(next) = skip_non_code_token(bytes, cursor) {
            cursor = next;
            continue;
        }
        match bytes[cursor] {
            b'[' => bracket_depth += 1,
            b']' => bracket_depth = bracket_depth.saturating_sub(1),
            b'{' if bracket_depth == 0 => {
                brace_depth += 1;
                saw_brace = true;
            }
            b'}' if bracket_depth == 0 => {
                brace_depth = brace_depth.saturating_sub(1);
                if saw_brace && brace_depth == 0 {
                    return cursor + 1;
                }
            }
            b';' if bracket_depth == 0 && brace_depth == 0 && !saw_brace => {
                return cursor + 1;
            }
            _ => {}
        }
        cursor += 1;
    }
    bytes.len()
}

/// The production half of a source file: everything outside `#[cfg(test)]`
/// items (handles inline `#[cfg(test)] mod tests { ... }`, out-of-line
/// `#[cfg(test)] mod tests;`, and individual `#[cfg(test)]` items without
/// truncating production code below them). Consuming a symbol only from a
/// test would satisfy "the symbol exists" while leaving production unmigrated,
/// which is exactly the confusion C5 exists to prevent.
fn production_half(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = String::with_capacity(source.len());
    let mut keep_from = 0_usize;
    let mut cursor = 0_usize;

    while cursor < bytes.len() {
        if let Some(next) = skip_non_code_token(bytes, cursor) {
            cursor = next;
            continue;
        }
        if bytes[cursor..].starts_with(b"#[cfg(test)]") {
            out.push_str(&source[keep_from..cursor]);
            cursor = skip_cfg_test_item(bytes, cursor);
            keep_from = cursor;
            continue;
        }
        cursor += 1;
    }

    if keep_from < bytes.len() {
        out.push_str(&source[keep_from..]);
    }
    out
}

/// Read the production half of `relative` plus, when a matching `<stem>/`
/// directory exists alongside `<stem>.rs`, the production half of every
/// non-test `.rs` submodule sorted deterministically.
fn read_production_subtree(relative: &str) -> String {
    let root_path = workspace_root().join(relative);
    let root_source = read(relative);
    let mut combined = production_half(&root_source);

    let sibling_dir = root_path.with_extension("");
    if sibling_dir.is_dir() {
        let mut subfiles = Vec::new();
        collect_production_rs_files(&sibling_dir, &mut subfiles);
        for subfile in subfiles {
            let sub_source = std::fs::read_to_string(&subfile).unwrap_or_else(|error| {
                panic!(
                    "census could not read submodule {}: {error}",
                    subfile.display()
                );
            });
            combined.push('\n');
            combined.push_str(&production_half(&sub_source));
        }
    }

    combined
}

fn census() -> serde_json::Value {
    serde_json::from_str(&read("docs/CTZO_ACTIVATION_CENSUS.json"))
        .expect("the census must be valid JSON")
}

#[test]
fn every_named_entry_point_exists_in_production_source() {
    let census = census();
    let entries = census["entry_points"]
        .as_array()
        .expect("entry_points is an array");
    assert!(
        entries.len() >= 8,
        "the census lost entries; it named 8 migrated entry points"
    );

    for entry in entries {
        let file = entry["file"].as_str().expect("file");
        let symbol = entry["symbol"].as_str().expect("symbol");
        let production = read_production_subtree(file);
        assert!(
            production.contains(symbol),
            "census names `{symbol}` in {file}, but it is absent from that file's \
             production half — either the symbol was renamed or it now lives only in tests"
        );
        assert!(
            file.contains("/src/"),
            "a census entry must name a production source file, got {file}"
        );
    }
}

#[test]
fn production_call_site_counts_match_the_census_exactly() {
    let census = census();
    for site in census["production_call_sites"]
        .as_array()
        .expect("production_call_sites is an array")
    {
        let file = site["file"].as_str().expect("file");
        let callee = site["callee"].as_str().expect("callee");
        let expected = usize::try_from(site["count"].as_u64().expect("count")).expect("count fits");

        let production = read_production_subtree(file);
        // The definition line is not a call site; count only invocations.
        let actual = production
            .lines()
            .filter(|line| {
                let trimmed = line.trim_start();
                !trimmed.starts_with("///")
                    && !trimmed.starts_with("//")
                    && !trimmed.starts_with("pub fn ")
                    && !trimmed.starts_with("fn ")
                    && line.contains(callee)
            })
            .count();
        assert_eq!(
            actual, expected,
            "{file} has {actual} production call sites of `{callee}` but the census says \
             {expected}; update docs/CTZO_ACTIVATION_CENSUS.json in the same change"
        );
    }
}

#[test]
fn the_activation_types_are_consumed_outside_their_defining_crate() {
    let census = census();
    let definition_crate = census["definition_crate"]
        .as_str()
        .expect("definition_crate");
    assert_eq!(definition_crate, "frankensearch-core");

    let consumers: Vec<&str> = census["entry_points"]
        .as_array()
        .expect("entry_points")
        .iter()
        .filter_map(|entry| entry["crate"].as_str())
        .filter(|krate| *krate != definition_crate)
        .collect();
    assert!(
        consumers.contains(&"frankensearch-index"),
        "the index crate must consume the activation types"
    );
    assert!(
        consumers.contains(&"frankensearch-fusion"),
        "the fusion crate must consume the activation types"
    );

    // And the strong form: a type defined in core is named in a production
    // file of another crate. `docs` alone would not prove consumption.
    let searcher = read_production_subtree("crates/frankensearch-fusion/src/searcher.rs");
    assert!(
        searcher.contains("TieredQueryEmbeddings"),
        "the async searcher must consume TieredQueryEmbeddings in production code"
    );
    let sync_searcher = read_production_subtree("crates/frankensearch-fusion/src/sync_searcher.rs");
    assert!(
        sync_searcher.contains("TieredQueryEmbeddings"),
        "the sync searcher must consume TieredQueryEmbeddings in production code"
    );
}

/// The census must not claim the downstream work bd-ctzo explicitly excludes.
#[test]
fn the_census_claims_no_downstream_work() {
    let raw = read("docs/CTZO_ACTIVATION_CENSUS.json").to_lowercase();
    for forbidden in [
        "fsfs persistence",
        "raw api retirement",
        "daemon enforcement",
        "recovery",
        "cass",
    ] {
        assert!(
            !raw.contains(forbidden),
            "the census must not claim `{forbidden}`; bd-ctzo scopes that downstream"
        );
    }
}
