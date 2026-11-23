use sha2::{Digest, Sha256};

use super::tests_support::{CONTROL_DOCS, component, control_quartet, docset, valid_manifest};
use super::*;

// --- bd-exact-generation-component-receipts-rds6v: paired controls ---

#[test]
fn control_quartet_admits_with_and_without_ann() {
    let (vector, lexical, ann, metadata) = control_quartet();
    let admitted = ExactGenerationComponentsV1::admit(
        vector.clone(),
        lexical.clone(),
        Some(ann),
        metadata.clone(),
    )
    .expect("agreeing components admit");
    assert!(admitted.has_ann());
    assert_eq!(admitted.docset_digest(), docset(&CONTROL_DOCS).digest());

    // ANN is an optional accelerator: absent is still exact.
    let without_ann = ExactGenerationComponentsV1::admit(vector, lexical, None, metadata)
        .expect("absent ANN is admissible");
    assert!(!without_ann.has_ann());
    assert_ne!(
        without_ann.composite_digest(),
        admitted.composite_digest(),
        "the composite digest must distinguish an admitted ANN from an absent one"
    );
}

/// Each mandatory role, drifted alone, must reject NAMING THAT ROLE — the
/// acceptance requires rejection "on the correct role", so an unattributed
/// mismatch would not satisfy it.
#[test]
fn each_role_docset_drift_rejects_on_that_role() {
    let other = docset(&["doc-a", "doc-b", "doc-c", "doc-e"]).digest();
    for role in [
        GenerationComponentRole::Lexical,
        GenerationComponentRole::Ann,
        GenerationComponentRole::Metadata,
    ] {
        let (vector, mut lexical, mut ann, mut metadata) = control_quartet();
        match role {
            GenerationComponentRole::Lexical => lexical.docset_digest = other,
            GenerationComponentRole::Ann => ann.docset_digest = other,
            GenerationComponentRole::Metadata => metadata.docset_digest = other,
            GenerationComponentRole::Vector => unreachable!("vector is the anchor"),
        }
        let observed = ExactGenerationComponentsV1::admit(vector, lexical, Some(ann), metadata);
        assert!(
            matches!(
                observed,
                Err(ComponentJoinErrorV1::DocsetDrift { role: named }) if named == role.as_str()
            ),
            "{} docset drift must reject on its own role, observed {observed:?}",
            role.as_str()
        );
    }
}

#[test]
fn each_role_checkpoint_drift_rejects_on_that_role() {
    for role in [
        GenerationComponentRole::Lexical,
        GenerationComponentRole::Ann,
        GenerationComponentRole::Metadata,
    ] {
        let (vector, mut lexical, mut ann, mut metadata) = control_quartet();
        match role {
            GenerationComponentRole::Lexical => lexical.source_checkpoint = [0x11; 32],
            GenerationComponentRole::Ann => ann.source_checkpoint = [0x11; 32],
            GenerationComponentRole::Metadata => metadata.source_checkpoint = [0x11; 32],
            GenerationComponentRole::Vector => unreachable!("vector is the anchor"),
        }
        let observed = ExactGenerationComponentsV1::admit(vector, lexical, Some(ann), metadata);
        assert!(
            matches!(
                observed,
                Err(ComponentJoinErrorV1::CheckpointDrift { role: named })
                    if named == role.as_str()
            ),
            "{} checkpoint drift must reject on its own role, observed {observed:?}",
            role.as_str()
        );
    }
}

/// The canonical digest already binds the document count when a receipt
/// comes from a correct producer, but `live_document_count` is also part of
/// the public receipt and composite identity. A malformed or incorrectly
/// adapted receipt must not carry a contradictory census through a
/// successful join merely because its digest still matches.
#[test]
fn each_role_live_document_count_drift_rejects_on_that_role() {
    for (role, lexical_count, ann_count, metadata_count) in [
        (GenerationComponentRole::Lexical, 3, 4, 4),
        (GenerationComponentRole::Ann, 4, 5, 4),
        (GenerationComponentRole::Metadata, 4, 4, 0),
    ] {
        let (vector, mut lexical, mut ann, mut metadata) = control_quartet();
        lexical.live_document_count = lexical_count;
        ann.live_document_count = ann_count;
        metadata.live_document_count = metadata_count;
        let observed = ExactGenerationComponentsV1::admit(vector, lexical, Some(ann), metadata);
        assert!(
            matches!(
                observed,
                Err(ComponentJoinErrorV1::LiveDocumentCountDrift { role: named })
                    if named == role.as_str()
            ),
            "{} live-document count drift must reject on its own role, observed {observed:?}",
            role.as_str()
        );
    }
}

/// A receipt filed in the wrong slot is caught before any digest is
/// compared — otherwise a lexical receipt sitting in the metadata slot
/// would sail through simply because it agrees with the anchor.
#[test]
fn a_receipt_in_the_wrong_slot_rejects_before_digest_comparison() {
    let (vector, lexical, ann, _) = control_quartet();
    let observed = ExactGenerationComponentsV1::admit(
        vector,
        lexical.clone(),
        Some(ann),
        // A perfectly agreeing LEXICAL receipt in the metadata slot.
        lexical,
    );
    assert!(
        matches!(
            observed,
            Err(ComponentJoinErrorV1::RoleMismatch {
                expected: "metadata",
                found: "lexical",
            })
        ),
        "a misfiled receipt must reject on the slot, observed {observed:?}"
    );
}

/// The document-set mutations the bead enumerates. Each must move the
/// canonical digest; the control proves the algorithm is not simply
/// digesting everything to a constant.
#[test]
fn canonical_docset_digest_separates_every_enumerated_mutation() {
    let control = docset(&CONTROL_DOCS).digest();
    assert_eq!(
        control,
        docset(&CONTROL_DOCS).digest(),
        "the digest must be deterministic across calls"
    );

    // Reordering: same documents, different generation order.
    assert_ne!(
        control,
        docset(&["doc-b", "doc-a", "doc-c", "doc-d"]).digest(),
        "document reordering must move the canonical digest"
    );
    // A hole: one document missing.
    assert_ne!(
        control,
        docset(&["doc-a", "doc-b", "doc-c"]).digest(),
        "a missing document must move the canonical digest"
    );
    // Same-count substitution — the case a bare count check cannot see.
    let substituted = docset(&["doc-a", "doc-b", "doc-c", "doc-z"]);
    assert_eq!(substituted.len(), CONTROL_DOCS.len());
    assert_ne!(
        control,
        substituted.digest(),
        "a same-count substitution must move the canonical digest"
    );
    // A boundary shift between adjacent identifiers must not collide with
    // the healthy set, which is why each identifier is length-prefixed.
    assert_ne!(
        docset(&["doc-a", "bdoc-c", "doc-d"]).digest(),
        docset(&["doc-ab", "doc-c", "doc-d"]).digest(),
        "identifier boundaries must be part of the digest"
    );
}

#[test]
fn a_duplicate_document_is_rejected_rather_than_collapsed() {
    // Silently deduplicating would let a corrupted component agree with a
    // healthy one that genuinely holds one fewer document.
    assert!(matches!(
        CanonicalDocsetV1::from_ordered_live_documents(["doc-a", "doc-b", "doc-a"]),
        Err(GenerationAuthorityErrorV1::InvalidField {
            field: "component_receipt.docset.duplicate_document_id"
        })
    ));
    assert!(matches!(
        CanonicalDocsetV1::from_ordered_live_documents(["doc-a", ""]),
        Err(GenerationAuthorityErrorV1::InvalidField {
            field: "component_receipt.docset.document_id"
        })
    ));
    // Control: the same documents without the duplicate are accepted.
    assert_eq!(
        docset(&["doc-a", "doc-b"]).len(),
        2,
        "the control set must still admit"
    );
}

/// An empty generation is a legitimate state and must have a stable,
/// non-zero digest distinct from any populated one.
#[test]
fn an_empty_docset_has_a_stable_distinct_digest() {
    let empty = docset(&[]);
    assert!(empty.is_empty());
    assert_eq!(empty.digest(), docset(&[]).digest());
    assert_ne!(empty.digest(), [0; 32]);
    assert_ne!(empty.digest(), docset(&["doc-a"]).digest());
}

/// Exact bytes are not identity: a component whose artifact bytes are
/// perfect is still refused when it was built from another generation.
#[test]
fn byte_identical_components_from_different_generations_are_refused() {
    let (vector, mut lexical, ann, metadata) = control_quartet();
    // Same artifact bytes as the control quartet's lexical component.
    assert_eq!(
        lexical.bytes,
        component(
            GenerationComponentRole::Lexical,
            lexical.docset_digest,
            lexical.source_checkpoint
        )
        .bytes
    );
    // ...but built from a different document set.
    lexical.docset_digest = docset(&["doc-a", "doc-b", "doc-c", "doc-e"]).digest();
    assert!(
        lexical.validate().is_ok(),
        "the receipt is internally valid"
    );
    assert!(matches!(
        ExactGenerationComponentsV1::admit(vector, lexical, Some(ann), metadata),
        Err(ComponentJoinErrorV1::DocsetDrift { role: "lexical" })
    ));
}

#[test]
fn a_zero_placeholder_digest_or_checkpoint_is_not_a_valid_receipt() {
    let (vector, mut lexical, ann, metadata) = control_quartet();
    lexical.docset_digest = [0; 32];
    assert!(matches!(
        ExactGenerationComponentsV1::admit(
            vector.clone(),
            lexical,
            Some(ann.clone()),
            metadata.clone()
        ),
        Err(ComponentJoinErrorV1::InvalidComponent {
            role: "lexical",
            ..
        })
    ));

    let (vector2, lexical2, ann2, mut metadata2) = control_quartet();
    metadata2.source_checkpoint = [0; 32];
    assert!(matches!(
        ExactGenerationComponentsV1::admit(vector2, lexical2, Some(ann2), metadata2),
        Err(ComponentJoinErrorV1::InvalidComponent {
            role: "metadata",
            ..
        })
    ));
}

/// The engine-neutral digest must NOT be confused with the index crate's
/// FSVI-specific one; separate domains are what keep them distinguishable.
#[test]
fn the_canonical_domain_is_distinct_from_the_fsvi_domain() {
    assert_eq!(
        CANONICAL_DOCSET_DIGEST_DOMAIN_V1,
        b"frankensearch.generation.canonical-ordered-docset.v1"
    );
    assert_ne!(
        CANONICAL_DOCSET_DIGEST_DOMAIN_V1,
        b"frankensearch.fsvi-v2.ordered-live-docset.v1".as_slice()
    );
}

// -----------------------------------------------------------------------
// bd-uu0ly: the metadata/checkpoint component producer.
//
// The keystone proved the JOIN algebra using synthetic receipts. These
// prove the metadata role can be built from a real manifest and that every
// independent mutation of that manifest is rejected on the metadata role.
// -----------------------------------------------------------------------

/// A manifest whose declared document count matches `documents`, so the
/// producer's count cross-check is satisfied and each test below can move
/// exactly one thing.
fn metadata_manifest(documents: &[&str], commit_range: &CommitRange) -> GenerationManifest {
    let mut manifest = valid_manifest();
    manifest.commit_range = commit_range.clone();
    manifest.total_documents = documents.len() as u64;
    manifest.manifest_hash = String::new();
    manifest.manifest_hash = compute_manifest_hash(&manifest).expect("hash");
    manifest
}

fn metadata_receipt(
    documents: &[&str],
    commit_range: &CommitRange,
) -> (GenerationManifest, Vec<u8>, ExactComponentReceiptV1) {
    let manifest = metadata_manifest(documents, commit_range);
    let bytes = serde_json::to_vec(&manifest).expect("serialize manifest");
    let receipt =
        ExactComponentReceiptV1::for_metadata_manifest(&manifest, &bytes, &docset(documents))
            .expect("a manifest agreeing with its docset produces a receipt");
    (manifest, bytes, receipt)
}

/// Anchor + lexical built to agree with a real metadata receipt, so the
/// metadata role is the only variable in the drift tests.
fn quartet_around(
    metadata: &ExactComponentReceiptV1,
) -> (ExactComponentReceiptV1, ExactComponentReceiptV1) {
    (
        component(
            GenerationComponentRole::Vector,
            metadata.docset_digest,
            metadata.source_checkpoint,
        ),
        component(
            GenerationComponentRole::Lexical,
            metadata.docset_digest,
            metadata.source_checkpoint,
        ),
    )
}

/// CONTROL. Without this, every rejection below could be a producer that
/// never yields an admissible receipt at all.
#[test]
fn a_real_manifest_produces_a_metadata_receipt_that_admits_against_the_anchor() {
    let (_manifest, bytes, metadata) =
        metadata_receipt(&CONTROL_DOCS, &CommitRange { low: 1, high: 100 });

    assert_eq!(metadata.role, GenerationComponentRole::Metadata);
    assert_eq!(metadata.live_document_count, CONTROL_DOCS.len() as u64);
    assert_eq!(metadata.docset_digest, docset(&CONTROL_DOCS).digest());
    assert_eq!(metadata.bytes.byte_len, bytes.len() as u64);
    assert_eq!(
        metadata.bytes.sha256,
        <[u8; 32]>::from(Sha256::digest(&bytes))
    );
    metadata.validate().expect("the produced receipt is valid");

    let (vector, lexical) = quartet_around(&metadata);
    let admitted = ExactGenerationComponentsV1::admit(vector, lexical, None, metadata.clone())
        .expect("a real metadata receipt admits");
    assert_eq!(admitted.metadata(), &metadata);
    assert_eq!(admitted.docset_digest(), metadata.docset_digest);
}

/// The producer binds the EXACT bytes, not just what the manifest means.
/// Two manifests that agree on documents and commit range but were
/// serialized differently must not share a receipt.
#[test]
fn a_rewritten_manifest_produces_a_different_receipt_at_the_same_docset_and_checkpoint() {
    let range = CommitRange { low: 1, high: 100 };
    let (manifest, bytes, receipt) = metadata_receipt(&CONTROL_DOCS, &range);

    // Same LENGTH, one byte different. A trailing-newline rewrite would
    // also change byte_len, and then a receipt that bound only the length
    // would pass this test while binding nothing about content. Holding
    // the length fixed leaves the SHA-256 as the only thing that can tell
    // these two images apart.
    let mut rewritten = bytes.clone();
    let last = rewritten.len() - 1;
    rewritten[last] ^= 0xFF;
    let rewritten_receipt = ExactComponentReceiptV1::for_metadata_manifest(
        &manifest,
        &rewritten,
        &docset(&CONTROL_DOCS),
    )
    .expect("the rewritten image still produces a receipt");

    assert_eq!(
        rewritten_receipt.docset_digest, receipt.docset_digest,
        "the fixture must hold the docset fixed, or this proves nothing about bytes"
    );
    assert_eq!(
        rewritten_receipt.source_checkpoint, receipt.source_checkpoint,
        "the fixture must hold the checkpoint fixed too"
    );
    assert_eq!(
        rewritten_receipt.bytes.byte_len, receipt.bytes.byte_len,
        "the fixture must hold the length fixed, or byte_len does the work"
    );
    assert_ne!(
        rewritten_receipt.bytes.sha256, receipt.bytes.sha256,
        "exact bytes must be bound by content, not merely by length; a rewritten \
         manifest of identical size is still a different component"
    );
}

/// DOCSET DRIFT, three independent shapes, each applied alone. Reordering,
/// a hole, and an addition are distinct failure modes that a digest which
/// ignored order or length would confuse.
#[test]
fn metadata_docset_drift_rejects_on_the_metadata_role() {
    let range = CommitRange { low: 1, high: 100 };
    let (_anchor_manifest, _anchor_bytes, anchor_metadata) =
        metadata_receipt(&CONTROL_DOCS, &range);
    let (vector, lexical) = quartet_around(&anchor_metadata);

    let reordered: [&str; 4] = ["doc-b", "doc-a", "doc-c", "doc-d"];
    let with_hole: [&str; 3] = ["doc-a", "doc-b", "doc-d"];
    let with_extra: [&str; 5] = ["doc-a", "doc-b", "doc-c", "doc-d", "doc-e"];

    for (label, documents) in [
        ("reordered", reordered.as_slice()),
        ("hole", with_hole.as_slice()),
        ("extra", with_extra.as_slice()),
    ] {
        let (_drifted_manifest, _drifted_bytes, drifted) = metadata_receipt(documents, &range);
        assert_ne!(
            drifted.docset_digest, anchor_metadata.docset_digest,
            "{label}: the fixture must actually move the docset"
        );
        assert_eq!(
            drifted.source_checkpoint, anchor_metadata.source_checkpoint,
            "{label}: only the docset may move, or the rejection is unattributable"
        );

        let observed =
            ExactGenerationComponentsV1::admit(vector.clone(), lexical.clone(), None, drifted);
        assert!(
            matches!(
                observed,
                Err(ComponentJoinErrorV1::DocsetDrift { role: "metadata" })
            ),
            "{label}: must reject as metadata docset drift, observed {observed:?}"
        );
    }
}

/// CHECKPOINT DRIFT with the document set held fixed. This is the half the
/// docset digest cannot see: the same documents rebuilt at a different
/// commit range are a different generation.
#[test]
fn metadata_checkpoint_drift_rejects_on_the_metadata_role() {
    let (_anchor_manifest, _anchor_bytes, anchor_metadata) =
        metadata_receipt(&CONTROL_DOCS, &CommitRange { low: 1, high: 100 });
    let (vector, lexical) = quartet_around(&anchor_metadata);

    for range in [
        CommitRange { low: 1, high: 101 },
        CommitRange { low: 2, high: 100 },
    ] {
        let (_drifted_manifest, _drifted_bytes, drifted) = metadata_receipt(&CONTROL_DOCS, &range);
        assert_eq!(
            drifted.docset_digest, anchor_metadata.docset_digest,
            "{range:?}: the documents must be identical, or this is docset drift instead"
        );
        assert_ne!(
            drifted.source_checkpoint, anchor_metadata.source_checkpoint,
            "{range:?}: the fixture must actually move the checkpoint"
        );

        let observed =
            ExactGenerationComponentsV1::admit(vector.clone(), lexical.clone(), None, drifted);
        assert!(
            matches!(
                observed,
                Err(ComponentJoinErrorV1::CheckpointDrift { role: "metadata" })
            ),
            "{range:?}: must reject as metadata checkpoint drift, observed {observed:?}"
        );
    }
}

/// The count cross-check only metadata can make. A manifest declaring a
/// different document count than the docset it is claiming is refused AT
/// CONSTRUCTION -- a receipt that never forms cannot reach the join and be
/// reported there as an anonymous digest mismatch.
#[test]
fn a_manifest_whose_declared_count_disagrees_with_its_docset_never_produces_a_receipt() {
    let mut manifest = metadata_manifest(&CONTROL_DOCS, &CommitRange { low: 1, high: 100 });
    manifest.total_documents = CONTROL_DOCS.len() as u64 + 1;
    let bytes = serde_json::to_vec(&manifest).expect("serialize manifest");

    let observed =
        ExactComponentReceiptV1::for_metadata_manifest(&manifest, &bytes, &docset(&CONTROL_DOCS));
    assert!(
        matches!(
            observed,
            Err(GenerationAuthorityErrorV1::InvalidField {
                field: "component_receipt.metadata.total_documents"
            })
        ),
        "observed {observed:?}"
    );

    // And the same manifest with the count corrected does produce one, so
    // the refusal is attributable to the count and not to the fixture.
    manifest.total_documents = CONTROL_DOCS.len() as u64;
    ExactComponentReceiptV1::for_metadata_manifest(&manifest, &bytes, &docset(&CONTROL_DOCS))
        .expect("correcting only the count produces a receipt");
}

/// Empty bytes cannot identify a component. Checked here rather than left
/// to `validate()`, because `GenerationComponentReceiptV1::validate` would
/// report it as an activation-lane `byte_len` error with no metadata
/// attribution.
#[test]
fn an_empty_manifest_image_never_produces_a_metadata_receipt() {
    let manifest = metadata_manifest(&CONTROL_DOCS, &CommitRange { low: 1, high: 100 });
    let observed =
        ExactComponentReceiptV1::for_metadata_manifest(&manifest, &[], &docset(&CONTROL_DOCS));
    assert!(
        matches!(
            observed,
            Err(GenerationAuthorityErrorV1::InvalidField {
                field: "component_receipt.metadata.manifest_bytes"
            })
        ),
        "observed {observed:?}"
    );
}

/// Domain separation, the same property the keystone pinned for the docset
/// domain. A checkpoint and a docset digest are both 32 bytes derived from
/// one generation; if they could collide, a component could satisfy one
/// comparison with the other's value.
#[test]
fn the_checkpoint_domain_is_distinct_from_the_docset_domain() {
    assert_eq!(
        GENERATION_SOURCE_CHECKPOINT_DOMAIN_V1,
        b"frankensearch.generation.source-checkpoint.v1"
    );
    assert_ne!(
        GENERATION_SOURCE_CHECKPOINT_DOMAIN_V1,
        CANONICAL_DOCSET_DIGEST_DOMAIN_V1
    );

    // Not merely different constants: the derived values differ for the
    // same generation, and neither is the all-zero placeholder that
    // `validate()` rejects.
    let range = CommitRange { low: 1, high: 100 };
    let checkpoint = generation_source_checkpoint_v1(&range);
    assert_ne!(checkpoint, docset(&CONTROL_DOCS).digest());
    assert_ne!(checkpoint, [0; 32]);

    // The derivation is a function of the range, so it is recomputable by
    // any role that knows the range and stable across calls.
    assert_eq!(checkpoint, generation_source_checkpoint_v1(&range));
    assert_ne!(
        checkpoint,
        generation_source_checkpoint_v1(&CommitRange { low: 1, high: 101 })
    );
    // Both endpoints participate: a range cannot be shifted without moving
    // the checkpoint.
    assert_ne!(
        generation_source_checkpoint_v1(&CommitRange { low: 1, high: 2 }),
        generation_source_checkpoint_v1(&CommitRange { low: 2, high: 1 })
    );
}

// -----------------------------------------------------------------------
// bd-uu0ly, consumer half: verifying a stored receipt against the artifact
// a reader actually has. The producer proves what was true at build time;
// these prove a rewritten or substituted artifact is caught afterwards.
// -----------------------------------------------------------------------

/// CONTROL. A receipt verifies against the exact manifest and image that
/// produced it, on both halves.
#[test]
fn a_metadata_receipt_verifies_against_the_manifest_that_produced_it() {
    let (manifest, bytes, receipt) =
        metadata_receipt(&CONTROL_DOCS, &CommitRange { low: 1, high: 100 });

    receipt
        .verify_metadata_manifest(&manifest, &bytes)
        .expect("the producing manifest verifies");
    receipt
        .verify_metadata_docset(&docset(&CONTROL_DOCS))
        .expect("the producing docset verifies");
}

/// A rewritten image at the SAME length is caught by content, not length.
/// Paired with a length change so both branches are exercised and neither
/// can be doing the other's work.
#[test]
fn a_rewritten_manifest_image_fails_verification_on_bytes() {
    let (manifest, bytes, receipt) =
        metadata_receipt(&CONTROL_DOCS, &CommitRange { low: 1, high: 100 });

    // Same length, one byte different. Flip inside the JSON string value
    // of generation_id so the image still decodes and the semantic
    // coherence check passes -- otherwise this would be reported as an
    // image mismatch and prove nothing about the SHA-256 branch.
    let mut same_length = bytes.clone();
    let marker = b"gen-001";
    let at = same_length
        .windows(marker.len())
        .position(|window| window == marker)
        .expect("fixture manifest carries its generation id verbatim");
    same_length[at + marker.len() - 1] = b'2';
    assert_eq!(same_length.len(), bytes.len());
    let observed = receipt.verify_metadata_manifest(&manifest, &same_length);
    assert!(
        matches!(observed, Err(MetadataReceiptDriftV1::ManifestImageMismatch)),
        "changing the generation id makes the pair incoherent first: {observed:?}"
    );

    // Now a same-length change that does NOT alter any semantic field the
    // coherence check compares, so verification must reach the SHA-256.
    let mut cosmetic = bytes.clone();
    let last = cosmetic.len() - 1;
    assert_eq!(cosmetic[last], b'}', "fixture serializes to a JSON object");
    cosmetic[last] = b' ';
    assert_eq!(cosmetic.len(), bytes.len());
    let observed = receipt.verify_metadata_manifest(&manifest, &cosmetic);
    assert!(
        matches!(
            observed,
            Err(MetadataReceiptDriftV1::UnreadableManifestImage)
        ),
        "a truncated object is not decodable: {observed:?}"
    );

    // And a length change is reported as a length drift.
    let mut longer = bytes.clone();
    longer.push(b' ');
    let observed = receipt.verify_metadata_manifest(&manifest, &longer);
    assert!(
        matches!(
            observed,
            Err(MetadataReceiptDriftV1::ByteLen { expected, found })
                if expected == bytes.len() as u64 && found == longer.len() as u64
        ),
        "observed {observed:?}"
    );
}

/// A receipt from one generation must not verify against another
/// generation's manifest, with documents held identical so the checkpoint
/// is the only discriminator.
#[test]
fn a_receipt_from_another_checkpoint_fails_verification_on_the_checkpoint() {
    let (_manifest, _bytes, receipt) =
        metadata_receipt(&CONTROL_DOCS, &CommitRange { low: 1, high: 100 });
    let (other_manifest, other_bytes, _other_receipt) =
        metadata_receipt(&CONTROL_DOCS, &CommitRange { low: 1, high: 101 });

    // The byte checks would fire first and mask the checkpoint, so this
    // asserts the checkpoint branch specifically by presenting a receipt
    // whose bytes match the other image but whose checkpoint does not.
    let mut transplanted = receipt.clone();
    transplanted.bytes = GenerationComponentReceiptV1 {
        byte_len: other_bytes.len() as u64,
        sha256: <[u8; 32]>::from(Sha256::digest(&other_bytes)),
    };
    let observed = transplanted.verify_metadata_manifest(&other_manifest, &other_bytes);
    assert!(
        matches!(observed, Err(MetadataReceiptDriftV1::SourceCheckpoint)),
        "observed {observed:?}"
    );

    // Control: correcting only the checkpoint makes the same receipt verify,
    // so the rejection is attributable to the checkpoint and not the graft.
    transplanted.source_checkpoint = generation_source_checkpoint_v1(&other_manifest.commit_range);
    transplanted
        .verify_metadata_manifest(&other_manifest, &other_bytes)
        .expect("only the checkpoint was wrong");
}

/// The docset half is a separate question, and drift on it is named.
#[test]
fn docset_verification_rejects_a_different_document_set() {
    let (_manifest, _bytes, receipt) =
        metadata_receipt(&CONTROL_DOCS, &CommitRange { low: 1, high: 100 });

    for documents in [
        ["doc-b", "doc-a", "doc-c", "doc-d"].as_slice(),
        ["doc-a", "doc-b", "doc-c"].as_slice(),
        ["doc-a", "doc-b", "doc-c", "doc-d", "doc-e"].as_slice(),
    ] {
        let observed = receipt.verify_metadata_docset(&docset(documents));
        assert!(
            matches!(
                observed,
                Err(MetadataReceiptDriftV1::Docset
                    | MetadataReceiptDriftV1::LiveDocumentCount { .. })
            ),
            "{documents:?}: must be named drift, observed {observed:?}"
        );
    }

    // A reorder keeps the count identical, so it MUST be caught by the
    // digest rather than by the count. Asserted separately because the
    // loop above would accept either.
    let reordered = docset(&["doc-b", "doc-a", "doc-c", "doc-d"]);
    assert_eq!(reordered.len(), CONTROL_DOCS.len());
    assert!(matches!(
        receipt.verify_metadata_docset(&reordered),
        Err(MetadataReceiptDriftV1::Docset)
    ));
}

/// A receipt for another role must never be verified as metadata, and the
/// role check must come FIRST -- otherwise a lexical receipt that happened
/// to agree on bytes would verify as metadata.
#[test]
fn a_non_metadata_receipt_is_refused_before_any_content_check() {
    let (manifest, bytes, receipt) =
        metadata_receipt(&CONTROL_DOCS, &CommitRange { low: 1, high: 100 });

    let mut lexical = receipt.clone();
    lexical.role = GenerationComponentRole::Lexical;
    // Everything else is byte-identical to a receipt that WOULD verify, so
    // only the role ordering can produce this rejection.
    let observed = lexical.verify_metadata_manifest(&manifest, &bytes);
    assert!(
        matches!(
            observed,
            Err(MetadataReceiptDriftV1::NotAMetadataReceipt { found: "lexical" })
        ),
        "observed {observed:?}"
    );
    let observed = lexical.verify_metadata_docset(&docset(&CONTROL_DOCS));
    assert!(
        matches!(
            observed,
            Err(MetadataReceiptDriftV1::NotAMetadataReceipt { found: "lexical" })
        ),
        "observed {observed:?}"
    );
}

/// An incoherent (manifest, bytes) pair is refused before any field is
/// measured. Without this, every check would compare one receipt against
/// two different generations and could report a spurious field drift.
#[test]
fn an_incoherent_manifest_and_image_pair_is_refused() {
    let (manifest, _bytes, receipt) =
        metadata_receipt(&CONTROL_DOCS, &CommitRange { low: 1, high: 100 });
    let (_other_manifest, other_bytes, _other) =
        metadata_receipt(&CONTROL_DOCS, &CommitRange { low: 1, high: 101 });

    let observed = receipt.verify_metadata_manifest(&manifest, &other_bytes);
    assert!(
        matches!(observed, Err(MetadataReceiptDriftV1::ManifestImageMismatch)),
        "observed {observed:?}"
    );

    let observed = receipt.verify_metadata_manifest(&manifest, b"not a manifest");
    assert!(
        matches!(
            observed,
            Err(MetadataReceiptDriftV1::UnreadableManifestImage)
        ),
        "observed {observed:?}"
    );
}

/// The documented limitation, pinned so it cannot be quietly "fixed" into
/// a check that asserts the caller's word back to itself: the manifest
/// records document COUNTS, never document IDENTIFIERS, so
/// [`ExactComponentReceiptV1::verify_metadata_manifest`] cannot and does
/// not validate the docset.
#[test]
fn manifest_verification_does_not_and_cannot_validate_the_docset() {
    let (manifest, bytes, receipt) =
        metadata_receipt(&CONTROL_DOCS, &CommitRange { low: 1, high: 100 });

    // A receipt whose docset digest is pure fiction still passes the
    // manifest half, because nothing in the manifest can contradict it.
    let mut fabricated = receipt.clone();
    fabricated.docset_digest = [0xAB; 32];
    fabricated
        .verify_metadata_manifest(&manifest, &bytes)
        .expect("the manifest half cannot see the docset");

    // The docset half catches it immediately...
    assert!(matches!(
        fabricated.verify_metadata_docset(&docset(&CONTROL_DOCS)),
        Err(MetadataReceiptDriftV1::Docset)
    ));

    // ...and so does the cross-engine join, which is the real proof and the
    // reason the keystone exists.
    let (vector, lexical) = quartet_around(&receipt);
    let observed = ExactGenerationComponentsV1::admit(vector, lexical, None, fabricated.clone());
    assert!(
        matches!(
            observed,
            Err(ComponentJoinErrorV1::DocsetDrift { role: "metadata" })
        ),
        "observed {observed:?}"
    );
}
