use std::env;
use std::ffi::OsStr;
use std::path::PathBuf;
use std::process::Command;

use sha2::{Digest, Sha256};

fn main() {
    for name in [
        "CARGO_HOME",
        "CARGO_PROFILE_RELEASE_PERF_LTO",
        "CARGO_PROFILE_RELEASE_PERF_CODEGEN_UNITS",
        "CARGO_PROFILE_RELEASE_PERF_OPT_LEVEL",
        "CARGO_ENCODED_RUSTFLAGS",
        "OPT_LEVEL",
        "PROFILE",
        "DEBUG",
        "RUSTC",
        "TARGET",
        "HOST",
        "CARGO_CFG_TARGET_FEATURE",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }

    emit("PROFILE_DIR", profile_directory());
    emit("PROFILE_FAMILY", required_env("PROFILE"));
    emit("OPT_LEVEL", required_env("OPT_LEVEL"));
    emit("DEBUG_INFO", required_env("DEBUG"));
    emit("HOST", required_env("HOST"));
    emit("TARGET", required_env("TARGET"));
    emit(
        "TARGET_FEATURES",
        env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default(),
    );
    emit(
        "LTO",
        env::var("CARGO_PROFILE_RELEASE_PERF_LTO").unwrap_or_else(|_| "<unset>".to_owned()),
    );
    emit(
        "CODEGEN_UNITS",
        env::var("CARGO_PROFILE_RELEASE_PERF_CODEGEN_UNITS")
            .unwrap_or_else(|_| "<unset>".to_owned()),
    );
    emit(
        "PROFILE_OPT_LEVEL",
        env::var("CARGO_PROFILE_RELEASE_PERF_OPT_LEVEL").unwrap_or_else(|_| "<unset>".to_owned()),
    );

    let rustc = required_env("RUSTC");
    let rustc_vv = Command::new(&rustc)
        .arg("-Vv")
        .output()
        .unwrap_or_else(|error| panic!("failed to execute {rustc:?} -Vv: {error}"));
    assert!(
        rustc_vv.status.success(),
        "{rustc:?} -Vv failed: {}",
        String::from_utf8_lossy(&rustc_vv.stderr)
    );
    emit("RUSTC_VV_SHA256", sha256_bytes(&rustc_vv.stdout));
    emit(
        "RUSTFLAGS_SHA256",
        sha256_bytes(
            env::var("CARGO_ENCODED_RUSTFLAGS")
                .unwrap_or_default()
                .as_bytes(),
        ),
    );

    emit("CANDIDATE_SOURCE_SHA256", "<feature-disabled>");
    emit("BASELINE_SOURCE_SHA256", "<feature-disabled>");
}

fn emit(name: &str, value: impl AsRef<str>) {
    let value = value.as_ref();
    assert!(
        !value.contains('\r') && !value.contains('\n'),
        "build attestation {name} contains a newline"
    );
    println!("cargo:rustc-env=FRANKENSEARCH_HNSW_{name}={value}");
}

fn required_env(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| panic!("Cargo did not provide required build input {name}"))
}

fn profile_directory() -> String {
    let out_dir = PathBuf::from(required_env("OUT_DIR"));
    let components: Vec<_> = out_dir.components().collect();
    let build = components
        .iter()
        .position(|component| component.as_os_str() == OsStr::new("build"))
        .unwrap_or_else(|| panic!("OUT_DIR has no build component: {}", out_dir.display()));
    assert!(build > 0, "OUT_DIR has no profile component");
    components[build - 1]
        .as_os_str()
        .to_string_lossy()
        .into_owned()
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex_bytes(&Sha256::digest(bytes))
}

fn hex_bytes(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}
