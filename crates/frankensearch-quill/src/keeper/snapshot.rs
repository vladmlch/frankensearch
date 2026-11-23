use std::io;
use std::path::{Path, PathBuf};

use asupersync::Cx;
use frankensearch_core::DocId;
use frankensearch_core::generation::{
    CanonicalDocsetV1, ExactComponentReceiptV1, GenerationComponentReceiptV1,
    GenerationComponentRole, SourceCheckpointV1,
};
use sha2::{Digest, Sha256};

use crate::schema::SchemaDescriptor;
use crate::segment::EncodedSegment;

use super::compaction::build_compaction;
use super::concat_merge::{build_concat_merge, recovery_retryable};
use super::errors::{
    CompactionPolicy, CompactionReport, KeeperError, LiveDocumentFloor, PublishIntent,
};
use super::gc::canonical_segment_name;
use super::manifest::{LoadedManifest, Manifest, ManifestSource};
use super::manifest_load::{
    VerifiedProposalSegment, load_manifest_pair, open_manifest_segments, reuse_verified_segments,
    validate_loaded_schema, validate_recovery_claims,
};
use super::manifest_validation::validate_manifest_successor;
use super::publish::discover_quarantined_segments;
use super::recovered_segment::{QuarantinedSegment, RecoveredSegment, ResolvedDocumentId};
use super::stats_and_cursor::{exact_snapshot_manifest_bytes, hash_exact_snapshot_frame};
use super::writer::KeeperWriter;

/// A read-only, internally consistent Keeper snapshot.
///
/// `open` performs recovery by selecting the admitted MANIFEST slot and
/// validating every referenced segment. It never repairs, renames, or removes
/// filesystem entries. Keeping this value alive retains every immutable mapped
/// or owned byte source as well.
#[derive(Clone)]
pub struct KeeperSnapshot {
    pub(in crate::keeper) directory: Option<PathBuf>,
    pub(in crate::keeper) schema: SchemaDescriptor,
    pub(in crate::keeper) loaded: LoadedManifest,
    pub(in crate::keeper) segments: Vec<RecoveredSegment>,
    pub(in crate::keeper) segments_by_seal_desc: Vec<usize>,
    pub(in crate::keeper) at_seal_doc_count: u64,
    pub(in crate::keeper) tombstone_count: u64,
    pub(in crate::keeper) live_doc_count: u64,
    pub(in crate::keeper) quarantined_segments: Vec<QuarantinedSegment>,
}

/// Re-openable authority for one immutable Quill lexical snapshot receipt.
///
/// The descriptor binds a durable directory, its compile-time schema, the
/// shared source checkpoint, and the exact receipt observed at capture time.
/// It deliberately does not consult `CURRENT`, garbage collection, or writer
/// claims: reopening validates the named lexical state and rejects any changed
/// component instead of silently following a newer publication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactLexicalSnapshotDescriptor {
    directory: PathBuf,
    schema: SchemaDescriptor,
    /// Held as the derived type, not raw bytes (bd-z4zr3), so a reopen
    /// re-derives against the same generation rather than re-presenting an
    /// array whose provenance the descriptor can no longer vouch for.
    source_checkpoint: SourceCheckpointV1,
    receipt: ExactComponentReceiptV1,
}

impl ExactLexicalSnapshotDescriptor {
    /// The exact lexical component receipt captured from the immutable snapshot.
    #[must_use]
    pub const fn receipt(&self) -> &ExactComponentReceiptV1 {
        &self.receipt
    }

    /// Reopen the captured durable snapshot and require its receipt to match.
    ///
    /// # Errors
    ///
    /// Returns an error when the directory can no longer be opened as the
    /// captured snapshot, or when its exact artifact, document set, or source
    /// checkpoint differs from the descriptor-bound receipt.
    pub fn reopen(&self) -> Result<KeeperSnapshot, KeeperError> {
        let snapshot = KeeperSnapshot::open(&self.directory, self.schema)?;
        let observed = snapshot.exact_lexical_component_receipt(self.source_checkpoint)?;
        if observed != self.receipt {
            return Err(KeeperError::ExactSnapshotReceipt {
                detail: "descriptor reopen observed a different lexical component".to_owned(),
            });
        }
        Ok(snapshot)
    }
}

impl KeeperSnapshot {
    fn from_parts(
        directory: Option<PathBuf>,
        schema: SchemaDescriptor,
        loaded: LoadedManifest,
        segments: Vec<RecoveredSegment>,
    ) -> Result<Self, KeeperError> {
        let error_path = directory
            .clone()
            .unwrap_or_else(|| PathBuf::from("<in-memory>"));
        let mut segments_by_seal_desc = Vec::new();
        segments_by_seal_desc
            .try_reserve_exact(segments.len())
            .map_err(|error| KeeperError::Io {
                operation: "allocate segment probe order",
                path: error_path,
                source: io::Error::other(error.to_string()),
            })?;
        segments_by_seal_desc.extend(0..segments.len());
        segments_by_seal_desc
            .sort_unstable_by_key(|&index| std::cmp::Reverse(segments[index].manifest.seal_seq));

        let at_seal_doc_count = segments
            .iter()
            .map(|segment| u64::from(segment.at_seal_doc_count()))
            .sum();
        let tombstone_count = segments.iter().map(RecoveredSegment::tombstone_count).sum();
        let live_doc_count = segments
            .iter()
            .map(|segment| u64::from(segment.doc_count()))
            .sum();
        let quarantined_segments = directory
            .as_deref()
            .map(|directory| discover_quarantined_segments(directory, &loaded.manifest))
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            directory,
            schema,
            loaded,
            segments,
            segments_by_seal_desc,
            at_seal_doc_count,
            tombstone_count,
            live_doc_count,
            quarantined_segments,
        })
    }

    /// Open one durable snapshot without mutating the index directory.
    ///
    /// # Errors
    ///
    /// Returns typed not-found, manifest, schema, segment, or metadata errors.
    pub fn open(
        directory: impl AsRef<Path>,
        schema: SchemaDescriptor,
    ) -> Result<Self, KeeperError> {
        let directory = directory.as_ref();
        match Self::open_once(directory, schema) {
            Err(error) if recovery_retryable(&error) => Self::open_once(directory, schema),
            result => result,
        }
    }

    /// Emit the exact Quill lexical component receipt for this immutable snapshot.
    ///
    /// The byte receipt is a domain-separated SHA-256 over the selected
    /// MANIFEST generation and every retained immutable FSLX image in manifest
    /// order. The document-set digest independently materializes every live
    /// IDMAP identity in physical generation order, so a tombstone, a hole, a
    /// duplicate, or a reordered row cannot be hidden by the component hash.
    ///
    /// # Errors
    ///
    /// Returns an error for an all-zero checkpoint, an invalid document-set
    /// identity, or arithmetic overflow while framing the exact byte receipt.
    pub fn exact_lexical_component_receipt(
        &self,
        source_checkpoint: SourceCheckpointV1,
    ) -> Result<ExactComponentReceiptV1, KeeperError> {
        let source_checkpoint = source_checkpoint.to_bytes();
        // Retained despite the newtype (bd-z4zr3): a derived checkpoint cannot
        // be all-zero for any commit range, so this is now unreachable through
        // the public path. It stays as the fail-closed floor for the wire form,
        // which is still a bare [u8; 32] in the receipt.
        if source_checkpoint == [0; 32] {
            return Err(KeeperError::ExactSnapshotReceipt {
                detail: "source checkpoint must not be all zeroes".to_owned(),
            });
        }

        let manifest = exact_snapshot_manifest_bytes(self)?;
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch.quill.exact-lexical-snapshot.v1");
        hash_exact_snapshot_frame(&mut hasher, b"manifest", &manifest)?;
        let mut byte_len =
            u64::try_from(manifest.len()).map_err(|_| KeeperError::ExactSnapshotReceipt {
                detail: "MANIFEST length does not fit the component receipt".to_owned(),
            })?;
        let mut document_ids = Vec::new();

        for segment in &self.segments {
            let manifest = segment.manifest();
            hasher.update(manifest.segment_id.to_le_bytes());
            let bytes = segment.reader.source_bytes();
            hash_exact_snapshot_frame(&mut hasher, b"segment", bytes)?;
            byte_len = byte_len
                .checked_add(u64::try_from(bytes.len()).map_err(|_| {
                    KeeperError::ExactSnapshotReceipt {
                        detail: "segment length does not fit the component receipt".to_owned(),
                    }
                })?)
                .ok_or_else(|| KeeperError::ExactSnapshotReceipt {
                    detail: "lexical component byte length overflowed".to_owned(),
                })?;

            for global_docid in manifest.docid_lo..manifest.docid_hi {
                let global_docid =
                    u32::try_from(global_docid).map_err(|_| KeeperError::ExactSnapshotReceipt {
                        detail: "manifest named a non-u32 Quill document id".to_owned(),
                    })?;
                if let Some(document_id) = segment.materialize_document_id(global_docid) {
                    document_ids.push(document_id.to_string());
                }
            }
        }

        let docset =
            CanonicalDocsetV1::from_ordered_live_documents(document_ids).map_err(|error| {
                KeeperError::ExactSnapshotReceipt {
                    detail: format!("canonical lexical document set was rejected: {error}"),
                }
            })?;
        if u64::try_from(docset.len()).map_err(|_| KeeperError::ExactSnapshotReceipt {
            detail: "lexical live document count does not fit the component receipt".to_owned(),
        })? != self.live_doc_count
        {
            return Err(KeeperError::ExactSnapshotReceipt {
                detail: "materialized lexical document count disagrees with snapshot count"
                    .to_owned(),
            });
        }

        Ok(ExactComponentReceiptV1 {
            role: GenerationComponentRole::Lexical,
            bytes: GenerationComponentReceiptV1 {
                byte_len,
                sha256: hasher.finalize().into(),
            },
            docset_digest: docset.digest(),
            live_document_count: self.live_doc_count,
            source_checkpoint,
        })
    }

    /// Capture a descriptor that can only reopen this exact lexical snapshot.
    ///
    /// # Errors
    ///
    /// In-memory snapshots have no durable descriptor and are rejected. Durable
    /// snapshots return a receipt that [`ExactLexicalSnapshotDescriptor::reopen`]
    /// revalidates rather than following a newer publication.
    pub fn exact_lexical_snapshot_descriptor(
        &self,
        source_checkpoint: SourceCheckpointV1,
    ) -> Result<ExactLexicalSnapshotDescriptor, KeeperError> {
        let directory =
            self.directory
                .clone()
                .ok_or_else(|| KeeperError::ExactSnapshotReceipt {
                    detail: "in-memory snapshot has no descriptor-backed reopen path".to_owned(),
                })?;
        Ok(ExactLexicalSnapshotDescriptor {
            directory,
            schema: self.schema,
            source_checkpoint,
            receipt: self.exact_lexical_component_receipt(source_checkpoint)?,
        })
    }

    fn open_once(directory: &Path, schema: SchemaDescriptor) -> Result<Self, KeeperError> {
        Self::open_once_reusing(directory, schema, Vec::new())
    }

    /// [`Self::open_once`], reusing segments a publication preflight verified
    /// under the same writer admission (see [`super::manifest_load::reuse_verified_segments`]). The
    /// MANIFEST pair is still read from disk as authority.
    pub(in crate::keeper) fn open_once_reusing(
        directory: &Path,
        schema: SchemaDescriptor,
        verified: Vec<VerifiedProposalSegment>,
    ) -> Result<Self, KeeperError> {
        let expected_schema_id = schema
            .schema_id()
            .map_err(|source| KeeperError::InvalidSchema { source })?;
        let loaded = load_manifest_pair(directory)?;
        validate_loaded_schema(directory, expected_schema_id, &loaded)?;
        validate_recovery_claims(directory, &loaded)?;

        let segments = if verified.is_empty() {
            open_manifest_segments(directory, &loaded.manifest.segments, schema)?
        } else {
            reuse_verified_segments(directory, &loaded.manifest.segments, schema, verified)?
        };

        let snapshot = Self::from_parts(Some(directory.to_path_buf()), schema, loaded, segments)?;
        if !snapshot.quarantined_segments.is_empty() {
            tracing::warn!(
                target: crate::tracing_conventions::TARGET,
                event = "quill.keeper.quarantine_persisted",
                directory = %directory.display(),
                quarantined_segments = snapshot.quarantined_segments.len(),
                estimated_missing_docs = snapshot.estimated_missing_docs(),
                unknown_missing_doc_segments = snapshot
                    .quarantined_segments
                    .iter()
                    .filter(|segment| segment.estimated_missing_docs.is_none())
                    .count(),
                "Quill opened with retained quarantined segments; results must be surfaced as degraded"
            );
        }
        Ok(snapshot)
    }

    /// Create a genesis index or open an existing index with the same schema.
    ///
    /// Creation is a writer operation: it acquires the cross-process admission,
    /// publishes genesis through an `O_EXCL` generation claim, and releases the
    /// writer capability before returning this read-only snapshot.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, I/O, schema mismatch, publication, or open
    /// errors. Existing durable state is never rebuilt on mismatch.
    pub async fn create(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
    ) -> Result<Self, KeeperError> {
        let writer = KeeperWriter::create(cx, directory, schema).await?;
        Ok(writer.snapshot.clone())
    }

    /// Acquire the sole cross-process writer capability for this index.
    ///
    /// Reader [`Self::open`] remains lock-free and filesystem-touchless.
    /// Writer open acquires `LOCK` before interrupted-publish recovery and GC.
    ///
    /// # Errors
    ///
    /// Returns a typed busy, corrupt-lock, recovery, schema, or I/O failure.
    pub async fn open_writer(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
    ) -> Result<KeeperWriter, KeeperError> {
        KeeperWriter::open(cx, directory, schema).await
    }

    /// Construct an owned-buffer genesis snapshot without creating files.
    ///
    /// [`Self::publish_owned_segments`] attaches committed owned FSLX bytes;
    /// recovery and garbage collection remain no-ops for this backend.
    ///
    /// # Errors
    ///
    /// Returns an invalid-schema error when the descriptor is not canonical.
    pub fn in_memory(schema: SchemaDescriptor) -> Result<Self, KeeperError> {
        let schema_id = schema
            .schema_id()
            .map_err(|source| KeeperError::InvalidSchema { source })?;
        Self::from_parts(
            None,
            schema,
            LoadedManifest {
                manifest: Manifest::empty(1, schema_id, 0),
                source: ManifestSource::InMemory,
            },
            Vec::new(),
        )
    }

    /// Construct an owned-buffer snapshot at one exact generation for a
    /// facade-boundary test. Shipping callers always start at genesis and use
    /// [`Self::next_manifest`] for successors.
    #[cfg(test)]
    pub(crate) fn in_memory_with_generation_for_test(
        schema: SchemaDescriptor,
        generation: u64,
    ) -> Result<Self, KeeperError> {
        let schema_id = schema
            .schema_id()
            .map_err(|source| KeeperError::InvalidSchema { source })?;
        Self::from_parts(
            None,
            schema,
            LoadedManifest {
                manifest: Manifest::empty(generation, schema_id, 0),
                source: ManifestSource::InMemory,
            },
            Vec::new(),
        )
    }

    /// Publish owned FSLX segments into one immutable in-memory successor.
    ///
    /// `proposed` must be the exact next MANIFEST generation. Retained
    /// segments reuse their immutable byte backing while adopting only the
    /// successor's validated tombstone state. Every newly referenced segment
    /// must have exactly one matching `EncodedSegment`; unreferenced or
    /// duplicate inputs are rejected. The current snapshot is never changed,
    /// so encoded bytes remain invisible until this method returns the fully
    /// validated successor.
    ///
    /// The proposal's tombstones and segment removals are taken as the
    /// caller's explicit deletion intent ([`PublishIntent::ExplicitDeletion`]).
    /// A proposal derived from an additive or replacing mutation should go
    /// through [`Self::publish_owned_segments_with_intent`] with
    /// [`PublishIntent::PreserveLiveDocuments`] so a proposal that lost its
    /// replacement rows is refused instead of published (gh#45).
    ///
    /// # Errors
    ///
    /// Returns a typed backend, MANIFEST-transition, segment-witness, or
    /// identity-section error. Durable snapshots must publish through
    /// [`super::writer::KeeperWriter`] instead.
    pub fn publish_owned_segments(
        &self,
        proposed: &Manifest,
        encoded_segments: Vec<EncodedSegment>,
    ) -> Result<Self, KeeperError> {
        self.publish_owned_segments_with_intent(
            proposed,
            encoded_segments,
            PublishIntent::ExplicitDeletion,
        )
    }

    /// [`Self::publish_owned_segments`] with an explicit [`PublishIntent`].
    ///
    /// # Errors
    ///
    /// As [`Self::publish_owned_segments`], plus
    /// [`KeeperError::LiveDocumentFloor`] when `intent` is
    /// [`PublishIntent::PreserveLiveDocuments`] and the proposal would serve
    /// fewer live documents than this snapshot. The refusal happens before any
    /// state changes; this snapshot stays authoritative.
    pub fn publish_owned_segments_with_intent(
        &self,
        proposed: &Manifest,
        encoded_segments: Vec<EncodedSegment>,
        intent: PublishIntent,
    ) -> Result<Self, KeeperError> {
        if self.directory.is_some() || self.loaded.source != ManifestSource::InMemory {
            return Err(KeeperError::InvalidTransition {
                detail: "owned segment publication requires an in-memory Keeper snapshot"
                    .to_owned(),
            });
        }
        proposed
            .validate()
            .map_err(|source| KeeperError::InvalidManifest { source })?;
        validate_manifest_successor(
            &self.loaded.manifest,
            proposed,
            intent.live_document_floor(),
        )?;

        let mut owned_by_id = Vec::new();
        owned_by_id
            .try_reserve_exact(encoded_segments.len())
            .map_err(|error| KeeperError::InvalidTransition {
                detail: format!("owned segment inventory allocation failed: {error}"),
            })?;
        for encoded in encoded_segments {
            owned_by_id.push((encoded.header().segment_id, Some(encoded)));
        }
        owned_by_id.sort_unstable_by_key(|entry| entry.0);
        if let Some(duplicate) = owned_by_id
            .windows(2)
            .find(|pair| pair[0].0 == pair[1].0)
            .map(|pair| pair[0].0)
        {
            return Err(KeeperError::InvalidTransition {
                detail: format!("owned segment {duplicate:#018x} was supplied more than once"),
            });
        }

        let mut current_by_id = Vec::new();
        current_by_id
            .try_reserve_exact(self.segments.len())
            .map_err(|error| KeeperError::InvalidTransition {
                detail: format!("current segment inventory allocation failed: {error}"),
            })?;
        current_by_id.extend(0..self.segments.len());
        current_by_id.sort_unstable_by_key(|&index| self.segments[index].manifest.segment_id);

        let mut segments = Vec::new();
        segments
            .try_reserve_exact(proposed.segments.len())
            .map_err(|error| KeeperError::InvalidTransition {
                detail: format!("successor segment inventory allocation failed: {error}"),
            })?;
        for manifest_segment in &proposed.segments {
            let segment_id = manifest_segment.segment_id;
            let current = current_by_id
                .binary_search_by_key(&segment_id, |&index| {
                    self.segments[index].manifest.segment_id
                })
                .ok()
                .map(|index| current_by_id[index]);
            let supplied = owned_by_id
                .binary_search_by_key(&segment_id, |entry| entry.0)
                .ok();

            if let Some(current) = current {
                if supplied.is_some() {
                    return Err(KeeperError::InvalidTransition {
                        detail: format!(
                            "retained segment {segment_id:#018x} also supplied replacement bytes"
                        ),
                    });
                }
                segments.push(self.segments[current].rebind(manifest_segment.clone())?);
                continue;
            }

            let supplied = supplied.ok_or_else(|| KeeperError::InvalidTransition {
                detail: format!(
                    "new manifest segment {segment_id:#018x} has no supplied owned bytes"
                ),
            })?;
            let encoded =
                owned_by_id[supplied]
                    .1
                    .take()
                    .ok_or_else(|| KeeperError::InvalidTransition {
                        detail: format!(
                            "owned segment {segment_id:#018x} was consumed more than once"
                        ),
                    })?;
            let path = PathBuf::from("<in-memory>").join(canonical_segment_name(segment_id));
            segments.push(RecoveredSegment::bind_owned(
                path,
                manifest_segment.clone(),
                encoded,
                self.schema,
            )?);
        }

        if let Some(unreferenced) = owned_by_id
            .iter()
            .find_map(|(segment_id, encoded)| encoded.is_some().then_some(*segment_id))
        {
            return Err(KeeperError::InvalidTransition {
                detail: format!(
                    "supplied owned segment {unreferenced:#018x} is absent from the successor manifest"
                ),
            });
        }

        Self::from_parts(
            None,
            self.schema,
            LoadedManifest {
                manifest: proposed.clone(),
                source: ManifestSource::InMemory,
            },
            segments,
        )
    }

    /// Replace one exact caller-ordered manifest run with a Q1 concat merge.
    ///
    /// This owned-buffer entry point is primarily useful for deterministic
    /// testing and offline assembly. Durable indexes must use
    /// [`super::writer::KeeperWriter::concat_merge`], which runs construction on the runtime's
    /// blocking lane and preserves writer admission through publication.
    ///
    /// # Errors
    ///
    /// Rejects fewer than two sources, any list that is not one uninterrupted
    /// slice of the current manifest, output-id collision, unknown extension
    /// sections, corrupt source codecs, duplicate-live external identifiers,
    /// or an invalid successor publication.
    pub fn concat_merge_owned(
        &self,
        source_segment_ids: &[u64],
        output_segment_id: u64,
        created_unix_s: i64,
    ) -> Result<Self, KeeperError> {
        let artifact =
            build_concat_merge(self, source_segment_ids, output_segment_id, created_unix_s)?;
        // A merge rewrites rows; it never removes a live one, so it is held to
        // the live-document floor like any other additive publication.
        self.publish_owned_segments_with_intent(
            &artifact.manifest,
            vec![artifact.encoded],
            PublishIntent::PreserveLiveDocuments,
        )
    }

    /// Fold every segment above `policy` into a new immutable snapshot.
    ///
    /// This owned-buffer entry point is useful for deterministic tests and
    /// in-memory indexes. Durable indexes use [`super::writer::KeeperWriter::compact`] so the
    /// replacement files are synced while still unreachable, followed by one
    /// atomic MANIFEST publication.
    ///
    /// # Errors
    ///
    /// Rejects an invalid policy, corrupt or unsupported source sections,
    /// rewrite/resource failures, or an invalid successor publication.
    pub fn compact_owned(
        &self,
        policy: CompactionPolicy,
        created_unix_s: i64,
    ) -> Result<(Self, CompactionReport), KeeperError> {
        let artifact = build_compaction(self, policy, created_unix_s)?;
        if !artifact.report.changed() {
            return Ok((self.clone(), artifact.report));
        }
        // Compaction folds tombstones into the rewritten segments; every live
        // row survives, so the live-document floor holds here as well.
        let snapshot = self.publish_owned_segments_with_intent(
            &artifact.manifest,
            artifact.encoded,
            PublishIntent::PreserveLiveDocuments,
        )?;
        Ok((snapshot, artifact.report))
    }

    /// Durable index directory, or `None` for an in-memory snapshot.
    #[must_use]
    pub fn directory(&self) -> Option<&Path> {
        self.directory.as_deref()
    }

    /// Compile-time schema admitted by this snapshot.
    #[must_use]
    pub const fn schema(&self) -> SchemaDescriptor {
        self.schema
    }

    /// Selected MANIFEST and its recovery provenance.
    #[must_use]
    pub const fn loaded_manifest(&self) -> &LoadedManifest {
        &self.loaded
    }

    /// Retained corrupt artifacts omitted from the active MANIFEST.
    #[must_use]
    pub fn quarantined_segments(&self) -> &[QuarantinedSegment] {
        &self.quarantined_segments
    }

    /// Whether this snapshot must be surfaced as degraded.
    #[must_use]
    pub const fn is_degraded(&self) -> bool {
        !self.quarantined_segments.is_empty()
    }

    /// Saturating sum of known at-seal rows from quarantined segments.
    ///
    /// Unknown historical rows are excluded and remain visible through
    /// [`Self::quarantined_segments`].
    #[must_use]
    pub fn estimated_missing_docs(&self) -> u64 {
        self.quarantined_segments
            .iter()
            .filter_map(|segment| segment.estimated_missing_docs)
            .fold(0_u64, u64::saturating_add)
    }

    /// Referenced immutable segments in ascending Q1 range order.
    #[must_use]
    pub fn segments(&self) -> &[RecoveredSegment] {
        &self.segments
    }

    /// Total estimated payload bytes of validated TERMDICT metadata across
    /// every segment bound in this snapshot.
    ///
    /// Payload estimate only (see
    /// [`super::recovered_segment::RecoveredSegment::term_dictionary_metadata_payload_bytes`]): it
    /// excludes `Arc` control-block overhead, alignment, and allocator slack,
    /// so it is not an exact memory claim. A snapshot holds each segment
    /// exactly once, so this sum never double-counts inside ONE snapshot.
    /// Successive snapshot generations that share a backing also share one
    /// metadata allocation, and each generation reports that shared payload
    /// in full: summing across concurrently held snapshots double-counts it.
    #[must_use]
    pub fn term_dictionary_metadata_payload_bytes(&self) -> u64 {
        self.segments.iter().fold(0_u64, |total, segment| {
            total.saturating_add(
                u64::try_from(segment.term_dictionary_metadata_payload_bytes()).unwrap_or(u64::MAX),
            )
        })
    }

    /// Physical at-seal rows retained for BM25 statistics until compaction.
    #[must_use]
    pub const fn at_seal_doc_count(&self) -> u64 {
        self.at_seal_doc_count
    }

    /// Tombstones paired with this immutable snapshot generation.
    #[must_use]
    pub const fn tombstone_count(&self) -> u64 {
        self.tombstone_count
    }

    /// Public live document count, cached during snapshot construction.
    #[must_use]
    pub const fn doc_count(&self) -> u64 {
        self.live_doc_count
    }

    /// Whether one global document id is visible in this snapshot.
    #[must_use]
    pub fn is_live(&self, global_docid: u32) -> bool {
        self.live_segment_for_docid(global_docid).is_some()
    }

    /// Select the immutable live segment that owns one global document id.
    ///
    /// Manifest validation keeps segment ranges ordered and disjoint, so the
    /// predecessor of the first range starting after `global_docid` is the
    /// only possible owner. The live-doc check rejects range holes and
    /// tombstones without materializing the external identifier.
    pub(crate) fn live_segment_for_docid(&self, global_docid: u32) -> Option<&RecoveredSegment> {
        let global_docid_u64 = u64::from(global_docid);
        let insertion = self
            .segments
            .partition_point(|segment| segment.manifest.docid_lo <= global_docid_u64);
        insertion
            .checked_sub(1)
            .and_then(|index| self.segments.get(index))
            .filter(|segment| crate::argus::LiveDocs::is_live(*segment, global_docid))
    }

    /// Materialize one live winner's external identifier via its IDMAP slice.
    ///
    /// Segment selection is logarithmic in the manifest segment count; the
    /// selected segment performs constant-time checked offset reads without
    /// reparsing the IDMAP span.
    #[must_use]
    pub fn materialize_document_id(&self, global_docid: u32) -> Option<DocId> {
        let global_docid_u64 = u64::from(global_docid);
        let insertion = self
            .segments
            .partition_point(|segment| segment.manifest.docid_lo <= global_docid_u64);
        insertion
            .checked_sub(1)
            .and_then(|index| self.segments.get(index))
            .and_then(|segment| segment.materialize_document_id(global_docid))
    }

    /// Resolve an external identifier against current sealed segments.
    ///
    /// Probes descend by `seal_seq`, continue after tombstoned hits, and keep
    /// scanning after the first live hit so duplicate-live upsert corruption is
    /// diagnosed rather than silently masked.
    ///
    /// # Errors
    ///
    /// Returns a typed identity-section or multiple-live-row failure.
    pub fn resolve_document_id(
        &self,
        document_id: &str,
    ) -> Result<Option<ResolvedDocumentId>, KeeperError> {
        self.resolve_document_id_in(&self.loaded.manifest, document_id)
    }

    /// Clone exactly one next-generation MANIFEST for staged writer changes.
    ///
    /// # Errors
    ///
    /// Returns a typed generation-exhaustion failure.
    pub fn next_manifest(&self) -> Result<Manifest, KeeperError> {
        let mut manifest = self.loaded.manifest.clone();
        manifest.generation =
            manifest
                .generation
                .checked_add(1)
                .ok_or(KeeperError::GenerationExhausted {
                    current: manifest.generation,
                })?;
        Ok(manifest)
    }

    /// Stage an idempotent sealed-document deletion into one next MANIFEST.
    ///
    /// Missing and already-deleted identifiers return `false` without changing
    /// `proposed`. Multiple calls compose in the same generation; publication
    /// remains the caller's explicit [`super::writer::KeeperWriter::publish`] operation.
    ///
    /// # Errors
    ///
    /// Rejects a non-successor proposal, corrupt identity state, tombstone-on-
    /// hole attempts, allocation failure, or the multiple-live upsert invariant.
    pub fn delete_document(
        &self,
        proposed: &mut Manifest,
        document_id: &str,
    ) -> Result<bool, KeeperError> {
        validate_staged_manifest(&self.loaded.manifest, proposed)?;
        if let Some(segment) = proposed.segments.iter().find(|segment| {
            self.loaded
                .manifest
                .segments
                .binary_search_by_key(&segment.docid_lo, |current| current.docid_lo)
                .map_or(true, |index| {
                    self.loaded.manifest.segments[index].segment_id != segment.segment_id
                })
        }) {
            return Err(KeeperError::InvalidTransition {
                detail: format!(
                    "stage deletes before adding replacement segment {:#018x}",
                    segment.segment_id
                ),
            });
        }
        let Some(resolved) = self.resolve_document_id_in(proposed, document_id)? else {
            return Ok(false);
        };
        let mut staged = proposed.clone();
        let segment = staged
            .segments
            .iter_mut()
            .find(|segment| segment.segment_id == resolved.segment_id)
            .ok_or_else(|| KeeperError::InvalidTransition {
                detail: format!(
                    "resolved segment {:#018x} is absent from staged manifest",
                    resolved.segment_id
                ),
            })?;
        let before = segment.tombstones.cardinality();
        if !segment
            .insert_tombstone(resolved.global_docid)
            .map_err(|source| KeeperError::InvalidManifest { source })?
        {
            return Ok(false);
        }
        validate_staged_manifest(&self.loaded.manifest, &staged)?;
        tracing::trace!(
            target: crate::tracing_conventions::TARGET,
            phase = "keeper.delete_document",
            generation = staged.generation,
            segment_id = resolved.segment_id,
            seal_seq = resolved.seal_seq,
            tombstones_before = before,
            tombstones_after = before + 1,
            "staged sealed-document tombstone"
        );
        *proposed = staged;
        Ok(true)
    }

    /// Stage a valid empty generation while preserving identity and watermark.
    ///
    /// This does not remove segment files; ordinary two-slot publication,
    /// snapshot lifetime, and writer-locked grace-window GC retain ownership of
    /// physical reclamation.
    ///
    /// # Errors
    ///
    /// Rejects a proposal that is not this snapshot's next generation.
    pub fn delete_all(&self, proposed: &mut Manifest) -> Result<(), KeeperError> {
        validate_staged_manifest(&self.loaded.manifest, proposed)?;
        let mut staged = proposed.clone();
        staged.segments.clear();
        staged.field_stats.clear();
        validate_staged_manifest(&self.loaded.manifest, &staged)?;
        *proposed = staged;
        Ok(())
    }

    fn resolve_document_id_in(
        &self,
        manifest: &Manifest,
        document_id: &str,
    ) -> Result<Option<ResolvedDocumentId>, KeeperError> {
        let mut live: Option<ResolvedDocumentId> = None;
        for &index in &self.segments_by_seal_desc {
            let segment = &self.segments[index];
            let Ok(manifest_index) = manifest
                .segments
                .binary_search_by_key(&segment.manifest.docid_lo, |candidate| candidate.docid_lo)
            else {
                continue;
            };
            let manifest_segment = &manifest.segments[manifest_index];
            if manifest_segment.segment_id != segment.manifest.segment_id {
                return Err(KeeperError::InvalidTransition {
                    detail: format!(
                        "staged segment range [{}, {}) changed identity from {:#018x} to {:#018x}",
                        segment.manifest.docid_lo,
                        segment.manifest.docid_hi,
                        segment.manifest.segment_id,
                        manifest_segment.segment_id
                    ),
                });
            }
            let Some((global_docid, content_hash)) = segment.lookup_document_id(document_id)?
            else {
                tracing::trace!(
                    target: crate::tracing_conventions::TARGET,
                    phase = "keeper.idhash_probe",
                    generation = manifest.generation,
                    segment_id = segment.manifest.segment_id,
                    seal_seq = segment.manifest.seal_seq,
                    query_len = document_id.len(),
                    outcome = "miss",
                    "probed sealed-segment identity"
                );
                continue;
            };
            if manifest_segment.tombstones.contains(global_docid) {
                tracing::trace!(
                    target: crate::tracing_conventions::TARGET,
                    phase = "keeper.idhash_probe",
                    generation = manifest.generation,
                    segment_id = segment.manifest.segment_id,
                    seal_seq = segment.manifest.seal_seq,
                    query_len = document_id.len(),
                    outcome = "tombstoned",
                    "continued after tombstoned identity hit"
                );
                continue;
            }
            let resolved = ResolvedDocumentId {
                segment_id: segment.manifest.segment_id,
                seal_seq: segment.manifest.seal_seq,
                global_docid,
                content_hash,
            };
            if let Some(first) = live {
                return Err(KeeperError::MultipleLiveDocumentIds {
                    directory: self
                        .directory
                        .clone()
                        .unwrap_or_else(|| PathBuf::from("<in-memory>")),
                    first_global_docid: first.global_docid,
                    duplicate_global_docid: global_docid,
                });
            }
            tracing::trace!(
                target: crate::tracing_conventions::TARGET,
                phase = "keeper.idhash_probe",
                generation = manifest.generation,
                segment_id = segment.manifest.segment_id,
                seal_seq = segment.manifest.seal_seq,
                query_len = document_id.len(),
                outcome = "live",
                "resolved live sealed-document identity"
            );
            live = Some(resolved);
        }
        Ok(live)
    }
}

impl crate::argus::LiveDocs for KeeperSnapshot {
    fn is_live(&self, global_docid: u32) -> bool {
        self.is_live(global_docid)
    }
}

impl crate::argus::LiveDocs for Manifest {
    fn is_live(&self, global_docid: u32) -> bool {
        let global_docid_u64 = u64::from(global_docid);
        let insertion = self
            .segments
            .partition_point(|segment| segment.docid_lo <= global_docid_u64);
        insertion
            .checked_sub(1)
            .and_then(|index| self.segments.get(index))
            .is_some_and(|segment| segment.is_live(global_docid))
    }
}

pub(in crate::keeper) fn validate_staged_manifest(
    previous: &Manifest,
    proposed: &Manifest,
) -> Result<(), KeeperError> {
    proposed
        .validate()
        .map_err(|source| KeeperError::InvalidManifest { source })?;
    // Staging tombstones into a proposal is not publishing it: the
    // live-document floor is decided when the proposal is offered to
    // `publish_with_intent` / `publish_owned_segments_with_intent`.
    validate_manifest_successor(previous, proposed, LiveDocumentFloor::Waived)
}

pub(in crate::keeper) fn manifest_matches_proposal(
    installed: &Manifest,
    proposed: &Manifest,
) -> bool {
    if proposed.last_publish_unix_s == 0 {
        if installed.last_publish_unix_s <= 0 {
            return false;
        }
        let mut normalized = installed.clone();
        normalized.last_publish_unix_s = 0;
        &normalized == proposed
    } else {
        installed == proposed
    }
}
