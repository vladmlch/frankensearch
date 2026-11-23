use super::{
    GENERATION_ROOT_MAX_CONTROL_FILE_BYTES, GENERATION_ROOT_MAX_FILE_BYTES,
    GenerationFileExpectation, GenerationRootErrorKind, GenerationRootStage,
};

#[test]
fn file_expectation_enforces_roles_digests_and_role_specific_limits() {
    let exact = GenerationFileExpectation::immutable(7, [0x5a; 32])
        .expect("small immutable expectation should be admitted");
    assert_eq!(exact.byte_len(), 7);
    assert_eq!(exact.sha256(), Some([0x5a; 32]));
    assert_eq!(exact.role(), super::GenerationFileRole::ImmutableArtifact);

    let control = GenerationFileExpectation::control(8)
        .expect("small control-file expectation should be admitted");
    assert_eq!(control.byte_len(), 8);
    assert_eq!(control.sha256(), None);
    assert_eq!(control.role(), super::GenerationFileRole::MutableControl);

    GenerationFileExpectation::control(GENERATION_ROOT_MAX_CONTROL_FILE_BYTES)
        .expect("the exact control repeated-hash ceiling must be accepted");
    GenerationFileExpectation::immutable(GENERATION_ROOT_MAX_FILE_BYTES, [0; 32])
        .expect("the exact immutable owned-image ceiling must be accepted");

    let control_limit = GenerationFileExpectation::control(
        GENERATION_ROOT_MAX_CONTROL_FILE_BYTES.saturating_add(1),
    )
    .expect_err("control length above its repeated-hash ceiling must fail");
    assert_eq!(control_limit.kind(), GenerationRootErrorKind::ResourceLimit);
    assert_eq!(
        control_limit.expected(),
        Some(GENERATION_ROOT_MAX_CONTROL_FILE_BYTES)
    );

    let artifact_limit = GenerationFileExpectation::immutable(
        GENERATION_ROOT_MAX_FILE_BYTES.saturating_add(1),
        [0; 32],
    )
    .expect_err("immutable length above the owned-image ceiling must fail");
    assert_eq!(
        artifact_limit.kind(),
        GenerationRootErrorKind::ResourceLimit
    );
    assert_eq!(
        artifact_limit.expected(),
        Some(GENERATION_ROOT_MAX_FILE_BYTES)
    );
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn confined_routes_are_bounded_unambiguous_and_non_utf8_safe() {
    use super::{
        ConfinedGenerationPath, GENERATION_ROOT_MAX_COMPONENT_BYTES,
        GENERATION_ROOT_MAX_COMPONENTS, GENERATION_ROOT_MAX_ROUTE_BYTES,
        QualifiedGenerationDirectory as QualifiedGenerationRoot,
    };
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    use std::path::{Path, PathBuf};

    for invalid in ["", ".", "..", "/absolute", "a/", "a//b", "a/./b", "a/../b"] {
        let error = ConfinedGenerationPath::parse(Path::new(invalid))
            .expect_err("ambiguous confined route must fail");
        assert_eq!(error.kind(), GenerationRootErrorKind::InvalidRoute);
        assert_eq!(error.stage(), GenerationRootStage::ParseRelativeRoute);
    }

    let nul = Path::new(OsStr::from_bytes(b"a\0b"));
    assert_eq!(
        ConfinedGenerationPath::parse(nul)
            .expect_err("NUL-containing route must fail")
            .kind(),
        GenerationRootErrorKind::InvalidRoute
    );

    let max_component = PathBuf::from("x".repeat(GENERATION_ROOT_MAX_COMPONENT_BYTES));
    assert_eq!(
        ConfinedGenerationPath::parse(&max_component)
            .expect("maximum component length should be accepted")
            .component_count(),
        1
    );
    let oversized_component =
        PathBuf::from("x".repeat(GENERATION_ROOT_MAX_COMPONENT_BYTES.saturating_add(1)));
    assert_eq!(
        ConfinedGenerationPath::parse(&oversized_component)
            .expect_err("oversized component must fail")
            .kind(),
        GenerationRootErrorKind::InvalidRoute
    );

    let max_components = std::iter::repeat_n("x", GENERATION_ROOT_MAX_COMPONENTS)
        .collect::<Vec<_>>()
        .join("/");
    assert_eq!(
        ConfinedGenerationPath::parse(Path::new(&max_components))
            .expect("maximum component count should be accepted")
            .component_count(),
        GENERATION_ROOT_MAX_COMPONENTS
    );
    let too_many_components =
        std::iter::repeat_n("x", GENERATION_ROOT_MAX_COMPONENTS.saturating_add(1))
            .collect::<Vec<_>>()
            .join("/");
    assert_eq!(
        ConfinedGenerationPath::parse(Path::new(&too_many_components))
            .expect_err("too many components must fail")
            .kind(),
        GenerationRootErrorKind::ResourceLimit
    );

    let non_utf8 = Path::new(OsStr::from_bytes(&[0xff, b'x']));
    assert_eq!(
        ConfinedGenerationPath::parse(non_utf8)
            .expect("non-UTF-8 bytes are valid Unix component bytes")
            .component_count(),
        1
    );

    let exact_relative = [
        std::iter::repeat_n("x".repeat(255), 15).collect::<Vec<_>>(),
        vec!["x".repeat(254), "x".to_owned()],
    ]
    .concat()
    .join("/");
    assert_eq!(exact_relative.len(), GENERATION_ROOT_MAX_ROUTE_BYTES);
    assert_eq!(
        ConfinedGenerationPath::parse(Path::new(&exact_relative))
            .expect("the exact relative-route byte ceiling must be accepted")
            .component_count(),
        17
    );
    let oversized_relative = format!("{exact_relative}x");
    let relative_error = ConfinedGenerationPath::parse(Path::new(&oversized_relative))
        .expect_err("one byte above the relative-route ceiling must fail");
    assert_eq!(relative_error.kind(), GenerationRootErrorKind::InvalidRoute);
    assert_eq!(
        relative_error.stage(),
        GenerationRootStage::ParseRelativeRoute
    );
    assert_eq!(
        relative_error.expected(),
        Some(GENERATION_ROOT_MAX_ROUTE_BYTES as u64)
    );
    assert_eq!(
        relative_error.observed(),
        Some(GENERATION_ROOT_MAX_ROUTE_BYTES.saturating_add(1) as u64)
    );

    let exact_absolute_body = [
        std::iter::repeat_n("x".repeat(255), 15).collect::<Vec<_>>(),
        vec!["x".repeat(253), "x".to_owned()],
    ]
    .concat()
    .join("/");
    let exact_absolute = format!("/{exact_absolute_body}");
    assert_eq!(exact_absolute.len(), GENERATION_ROOT_MAX_ROUTE_BYTES);
    let exact_absolute_error = QualifiedGenerationRoot::admit(Path::new(&exact_absolute))
        .expect_err("the parsed but nonexistent exact-limit route must fail during open");
    assert_eq!(
        exact_absolute_error.stage(),
        GenerationRootStage::OpenRootRoute,
        "the exact absolute-route ceiling must pass parsing"
    );
    let oversized_absolute = format!("{exact_absolute}x");
    let absolute_error = QualifiedGenerationRoot::admit(Path::new(&oversized_absolute))
        .expect_err("one byte above the absolute-route ceiling must fail during parsing");
    assert_eq!(absolute_error.kind(), GenerationRootErrorKind::InvalidRoute);
    assert_eq!(absolute_error.stage(), GenerationRootStage::ParseRootRoute);
    assert_eq!(
        absolute_error.expected(),
        Some(GENERATION_ROOT_MAX_ROUTE_BYTES as u64)
    );
    assert_eq!(
        absolute_error.observed(),
        Some(GENERATION_ROOT_MAX_ROUTE_BYTES.saturating_add(1) as u64)
    );
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
#[test]
fn unsupported_targets_return_typed_platform_errors_without_path_access() {
    use super::{
        ConfinedGenerationPath, GenerationRootAnchorLayout, GenerationRootReadGuard,
        GenerationRootResult, QualifiedGenerationDirectory,
        QualifiedGenerationRoot as NamedGenerationRoot,
    };
    use std::path::Path;

    let missing = Path::new("this-path-must-never-be-opened");
    let confined = ConfinedGenerationPath::parse(missing)
        .expect_err("unsupported target must fail before parsing or I/O");
    assert_eq!(
        confined.kind(),
        GenerationRootErrorKind::UnsupportedPlatform
    );
    assert_eq!(confined.stage(), GenerationRootStage::PlatformGate);

    let absolute_missing = Path::new("/this-path-must-never-be-opened");
    let directory = QualifiedGenerationDirectory::admit(absolute_missing)
        .expect_err("unsupported target must fail before opening the path");
    assert_eq!(
        directory.kind(),
        GenerationRootErrorKind::UnsupportedPlatform
    );
    assert_eq!(directory.stage(), GenerationRootStage::PlatformGate);

    let layout = GenerationRootAnchorLayout::new(8)
        .expect("the public named-root layout should remain platform-independent");
    let named = NamedGenerationRoot::admit(absolute_missing, layout)
        .expect_err("the public named-root wrapper must fail before anchor path access");
    assert_eq!(named.kind(), GenerationRootErrorKind::UnsupportedPlatform);
    assert_eq!(named.stage(), GenerationRootStage::PlatformGate);

    let _read_guard_surface: for<'root> fn(
        &'root NamedGenerationRoot,
    )
        -> GenerationRootResult<GenerationRootReadGuard<'root>> = NamedGenerationRoot::read_guard;
}

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos_arm64;
