use crate::keeper::{
    CompactionPolicy, CompactionReport, GarbageCollectionOptions, GarbageCollectionReport,
    KeeperError, KeeperSnapshot, LiveDocumentFloor, MANIFEST_FLAG_BULK_MODE_IN_PROGRESS,
    PublicationReadState, PublishIntent, TierMergePolicy, plan_tier_merge,
};
use crate::scribe::{DocIdAllocator, DocIdSpan};
use asupersync::Cx;
use asupersync::runtime::spawn_blocking;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Instant;
use tracing::Instrument;

use super::checkpoint::LifecycleTrigger;
#[cfg(feature = "conformance-internals")]
use super::conformance::ConformanceCancellationStage;
use super::error_results::{
    MAX_GLOBAL_DOCID_EXCLUSIVE, QuillIndexError, check_cancel, invalid_state,
    record_snapshot_fields, wall_clock_unix_s,
};
use super::ingest_doc::{IngestBatch, write_canonical_metadata};
use super::parallel_ingest::{
    FANOUT_WAVE_SHARD_DOCUMENTS, IngestParallelismPolicy, ParallelBatchObservation,
    PendingIdentity, ScribeShardState, ValidatedBatchAdmission, accumulate_shard_run,
};
use super::snapshot::SnapshotPublisher;
use super::writer::{IndexBackend, QuillWriterState};

impl QuillWriterState {
    /// Admit or refuse a whole batch before ANY of it is accumulated
    /// (bd-quill-rejected-ingest-publishes-partial-batch-aihri).
    ///
    /// This is deliberately PURE: it inserts nothing, allocates no document
    /// ids, and touches no shard. That is what makes a refused batch a no-op —
    /// there is no partial state to roll back, and no rollback API to need. The
    /// serial route used to run this check per document *inside* its
    /// accumulation loop, so a batch whose last document was a duplicate left
    /// every earlier document staged, and the next `commit()` published them.
    ///
    /// Purity is also load-bearing for correctness, not just for style: both
    /// parallel routes run before the serial/fan-out dispatch and re-check
    /// `uncommitted_ids` themselves, so an admission pass that inserted its ids
    /// up front would make every document of a legal batch look like a
    /// duplicate to them.
    ///
    /// The predicate reproduces the routes' original one exactly, including its
    /// precedence: an id already staged in `uncommitted_ids` is a duplicate
    /// unconditionally, while an id live in the published snapshot is a
    /// duplicate only when it is not being replaced. In-batch repeats are
    /// refused even for replacement ids, which is what the per-document version
    /// did by inserting as it went.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, empty-id, or duplicate-id failures.
    pub(super) fn validate_batch_admission<'a>(
        &self,
        cx: &Cx,
        documents: IngestBatch<'a>,
        replacement_ids: &BTreeSet<&str>,
    ) -> Result<ValidatedBatchAdmission<'a>, QuillIndexError> {
        let authority = self.authority_snapshot()?;
        let mut batch_ids = BTreeSet::new();
        for document in documents {
            check_cancel(cx, "index admission")?;
            if document.id().is_empty() {
                return Err(invalid_state("document id must be nonempty"));
            }
            if !batch_ids.insert(document.id())
                || self.uncommitted_ids.contains(document.id())
                || authority.resolve_document_id(document.id())?.is_some()
                    && !replacement_ids.contains(document.id())
            {
                return Err(invalid_state(format!(
                    "duplicate live document id {:?}",
                    document.id()
                )));
            }
        }
        Ok(ValidatedBatchAdmission { documents })
    }

    /// Route and accumulate one bounded batch into a single shard.
    ///
    /// This is the original single-shard path, retained verbatim: it is the
    /// only path taken when `deterministic_ingest` resolves the router to one
    /// shard, so single-shard ingest is bit-identical to before the fan-out.
    ///
    /// Admission is a precondition here: every document is already known to be
    /// nonempty and unique against the batch, the uncommitted set and the
    /// published snapshot, because
    /// [`Self::validate_batch_admission`] ran before any state was touched.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, accumulation, flush, or publication
    /// failures.
    pub(super) async fn index_batch_serial(
        &mut self,
        cx: &Cx,
        documents: IngestBatch<'_>,
        allow_automatic_publication: bool,
    ) -> Result<(usize, usize), QuillIndexError> {
        let shard_id = self.shard_router.route_batch();
        let document_count = u32::try_from(documents.len())
            .map_err(|_| invalid_state("ingest batch document count does not fit u32"))?;
        let allocated = self
            .docid_allocator
            .alloc_batch(shard_id, document_count)
            .map_err(|error| invalid_state(error.to_string()))?;
        self.next_lease_base = self.docid_allocator.watermark();
        let mut arena_bytes_used_high_water = self
            .shards
            .iter()
            .map(|shard| shard.accumulator.bytes_used())
            .max()
            .unwrap_or(0);
        let mut arena_bytes_reserved_high_water = self
            .shards
            .iter()
            .map(|shard| shard.accumulator.bytes_reserved())
            .max()
            .unwrap_or(0);
        let mut document_index = 0_usize;
        for (span_index, span) in allocated.spans().iter().copied().enumerate() {
            let lease_changed = self.shards[shard_id]
                .current_lease_base
                .is_some_and(|base| base != span.lease_base);
            if lease_changed {
                self.flush_shard(cx, shard_id, LifecycleTrigger::LeaseBoundary)
                    .await?;
            }
            self.shards[shard_id].current_lease_base = Some(span.lease_base);
            for span_offset in 0..span.len {
                let document = documents
                    .get(document_index)
                    .ok_or_else(|| invalid_state("shard document index is outside the batch"))?;
                #[cfg(test)]
                if self.cancel_ingest_after_documents == Some(document_index) {
                    cx.set_cancel_requested(true);
                }
                document_index += 1;
                check_cancel(cx, "index")?;
                let doc_ord = span
                    .ord_start
                    .checked_add(span_offset)
                    .ok_or_else(|| invalid_state("lease-relative document ordinal overflow"))?;
                let global_docid = span
                    .lease_base
                    .checked_add(u64::from(doc_ord))
                    .filter(|docid| *docid < MAX_GLOBAL_DOCID_EXCLUSIVE)
                    .ok_or_else(|| invalid_state("global Q1 document-id space exhausted"))?;

                let accumulated = {
                    let shard = &mut self.shards[shard_id];
                    document.write_canonical_metadata(&mut shard.scratch_metadata)?;
                    let accumulated = document.accumulate(
                        &mut shard.accumulator,
                        doc_ord,
                        global_docid,
                        &shard.scratch_metadata,
                    )?;
                    let content_hash = document.content_hash(&shard.scratch_metadata)?;
                    shard.identities.push(PendingIdentity {
                        doc_ord,
                        document_id: document.id().to_owned(),
                        content_hash,
                    });
                    accumulated
                };
                arena_bytes_used_high_water =
                    arena_bytes_used_high_water.max(accumulated.bytes_used);
                arena_bytes_reserved_high_water =
                    arena_bytes_reserved_high_water.max(accumulated.bytes_reserved);
                self.uncommitted_ids.insert(document.id().to_owned());
                self.unpublished_since.get_or_insert_with(Instant::now);

                if self.shards[shard_id]
                    .accumulator
                    .should_flush(self.config.scribe_shard_budget_bytes)
                {
                    self.flush_shard(cx, shard_id, LifecycleTrigger::ArenaBudget)
                        .await?;
                    self.shards[shard_id].current_lease_base = Some(span.lease_base);
                    if allow_automatic_publication {
                        self.publish_bulk_cadence_if_due(cx).await?;
                    }
                }
            }
            if span_index + 1 < allocated.spans().len() {
                self.flush_shard(cx, shard_id, LifecycleTrigger::LeaseBoundary)
                    .await?;
                if allow_automatic_publication {
                    self.publish_bulk_cadence_if_due(cx).await?;
                }
            }
        }
        debug_assert_eq!(document_index, documents.len());
        Ok((arena_bytes_used_high_water, arena_bytes_reserved_high_water))
    }

    /// Accumulate one bounded batch across every ingest shard concurrently.
    ///
    /// The shards are already shared-nothing — each owns its arena, term
    /// interner and identity list — but production routed a *whole* batch to
    /// one shard, so they never ran at the same time. This partitions the
    /// batch instead, so `shard_count` accumulators fill in parallel.
    ///
    /// Three things must stay serial and do:
    ///
    /// - **Id claiming.** The uncommitted-id set is shared, so ids are claimed
    ///   first, in input order. Admission itself already ran for the whole
    ///   batch in [`Self::validate_batch_admission`], before any route was
    ///   chosen, so both routes refuse the same document with the same message.
    /// - **Docid allocation.** Leases are per shard but the allocator is
    ///   shared, so every span is allocated before the parallel region opens.
    /// - **Sealing.** A flush is `async` and mutates shared publication state,
    ///   so budget and lease-boundary seals happen between waves.
    ///
    /// Work is issued in waves of at most [`FANOUT_WAVE_SHARD_DOCUMENTS`] per
    /// shard so a shard's arena cannot overshoot its byte budget by more than
    /// one wave before the seal check runs.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, accumulation, flush, or publication
    /// failures.
    pub(super) async fn index_batch_fanout(
        &mut self,
        cx: &Cx,
        documents: IngestBatch<'_>,
        allow_automatic_publication: bool,
        shard_count: usize,
        observation: &ParallelBatchObservation,
    ) -> Result<(usize, usize), QuillIndexError> {
        // Admission already ran, for the whole batch, before any state was
        // touched (`validate_batch_admission`). Claiming the ids is all that
        // is left, and it happens in input order.
        for document in documents {
            check_cancel(cx, "index")?;
            self.uncommitted_ids.insert(document.id().to_owned());
        }
        if !documents.is_empty() {
            self.unpublished_since.get_or_insert_with(Instant::now);
        }

        let mut arena_bytes_used_high_water = self
            .shards
            .iter()
            .map(|shard| shard.accumulator.bytes_used())
            .max()
            .unwrap_or(0);
        let mut arena_bytes_reserved_high_water = self
            .shards
            .iter()
            .map(|shard| shard.accumulator.bytes_reserved())
            .max()
            .unwrap_or(0);

        let mut wave_start = 0_usize;
        while wave_start < documents.len() {
            check_cancel(cx, "index")?;
            let wave_len = (documents.len() - wave_start)
                .min(FANOUT_WAVE_SHARD_DOCUMENTS.saturating_mul(shard_count));
            let wave = documents
                .try_slice(wave_start..wave_start + wave_len)
                .ok_or_else(|| invalid_state("fan-out wave is outside the batch"))?;
            wave_start += wave_len;

            // Contiguous partition in ascending shard order: every shard sees
            // strictly increasing document ordinals, which the accumulator
            // requires, and the assignment is a pure function of the batch.
            let per_shard = wave.len() / shard_count;
            let remainder = wave.len() % shard_count;
            let mut allocations: Vec<(usize, usize, Vec<DocIdSpan>)> =
                Vec::with_capacity(shard_count);
            let mut cursor = 0_usize;
            for shard in 0..shard_count {
                let len = per_shard + usize::from(shard < remainder);
                if len == 0 {
                    continue;
                }
                let count = u32::try_from(len)
                    .map_err(|_| invalid_state("fan-out shard document count does not fit u32"))?;
                let allocated = self
                    .docid_allocator
                    .alloc_batch(shard, count)
                    .map_err(|error| invalid_state(error.to_string()))?;
                allocations.push((shard, cursor, allocated.spans().to_vec()));
                cursor += len;
            }
            debug_assert_eq!(cursor, wave.len());
            self.next_lease_base = self.docid_allocator.watermark();

            // A shard whose allocation crossed a lease boundary needs an R1 cut
            // between its spans, so spans are consumed in rounds; all but the
            // last round is rare (a lease is 65,536 ordinals wide).
            let rounds = allocations
                .iter()
                .map(|(_, _, spans)| spans.len())
                .max()
                .unwrap_or(0);
            let mut consumed = vec![0_usize; allocations.len()];

            for round in 0..rounds {
                // Seal any shard whose lease rolled before its ordinals are used.
                for (shard, _, spans) in &allocations {
                    let Some(span) = spans.get(round) else {
                        continue;
                    };
                    let lease_changed = self.shards[*shard]
                        .current_lease_base
                        .is_some_and(|base| base != span.lease_base);
                    if lease_changed {
                        self.flush_shard(cx, *shard, LifecycleTrigger::LeaseBoundary)
                            .await?;
                    }
                    self.shards[*shard].current_lease_base = Some(span.lease_base);
                }

                // Shared-nothing parallel accumulate. Disjoint `&mut` shard
                // borrows are carved out in ascending shard order, scoped so
                // the shard-vector borrow ends before the seal pass below.
                let outcomes = {
                    let mut work: Vec<(&mut ScribeShardState, IngestBatch<'_>, DocIdSpan)> =
                        Vec::with_capacity(allocations.len());
                    let mut rest: &mut [ScribeShardState] = &mut self.shards;
                    let mut taken = 0_usize;
                    for (index, (shard, start, spans)) in allocations.iter().enumerate() {
                        let Some(span) = spans.get(round) else {
                            continue;
                        };
                        let (_, tail) = std::mem::take(&mut rest).split_at_mut(*shard - taken);
                        let (head, tail) = tail.split_at_mut(1);
                        rest = tail;
                        taken = *shard + 1;
                        let begin = *start + consumed[index];
                        let end = begin + span.len as usize;
                        let assigned = wave.try_slice(begin..end).ok_or_else(|| {
                            invalid_state("shard span is outside the fan-out wave")
                        })?;
                        work.push((&mut head[0], assigned, *span));
                    }
                    work.into_par_iter()
                        .map(|(state, assigned, span)| {
                            let shard_guard = observation.enter_shard();
                            let run = accumulate_shard_run(state, assigned, span);
                            if run.is_ok() {
                                shard_guard.completed();
                            }
                            run
                        })
                        .collect::<Vec<_>>()
                };
                for outcome in outcomes {
                    let (bytes_used, bytes_reserved) = outcome?;
                    arena_bytes_used_high_water = arena_bytes_used_high_water.max(bytes_used);
                    arena_bytes_reserved_high_water =
                        arena_bytes_reserved_high_water.max(bytes_reserved);
                }

                for (index, (_, _, spans)) in allocations.iter().enumerate() {
                    if let Some(span) = spans.get(round) {
                        consumed[index] += span.len as usize;
                    }
                }

                // Seal shards that filled their budget, or that still have a
                // span waiting behind a lease boundary. Lease cuts remain
                // ordered because a later span cannot be consumed until its
                // predecessor is installed. Budget cuts are independent, so
                // build every crossing shard together instead of serializing
                // several full accumulated-segment encodes on this one batch.
                let mut budget_flushes = Vec::new();
                budget_flushes
                    .try_reserve_exact(allocations.len())
                    .map_err(|_| invalid_state("could not reserve budget flush set"))?;
                for (shard, _, spans) in &allocations {
                    let Some(span) = spans.get(round) else {
                        continue;
                    };
                    let boundary_pending = spans.len() > round + 1;
                    let budget_reached = self.shards[*shard]
                        .accumulator
                        .should_flush(self.config.scribe_shard_budget_bytes);
                    if !boundary_pending && !budget_reached {
                        continue;
                    }
                    if boundary_pending {
                        self.flush_shard(cx, *shard, LifecycleTrigger::LeaseBoundary)
                            .await?;
                        if allow_automatic_publication {
                            self.publish_bulk_cadence_if_due(cx).await?;
                        }
                    } else {
                        budget_flushes.push((*shard, span.lease_base));
                    }
                }
                let mut budget_shards = Vec::new();
                budget_shards
                    .try_reserve_exact(budget_flushes.len())
                    .map_err(|_| invalid_state("could not reserve parallel budget flush plan"))?;
                budget_shards.extend(budget_flushes.iter().map(|(shard, _)| *shard));
                self.flush_selected_shards(
                    cx,
                    &budget_shards,
                    LifecycleTrigger::ArenaBudget,
                    allow_automatic_publication,
                )
                .await?;
                for (shard, lease_base) in budget_flushes {
                    self.shards[shard].current_lease_base = Some(lease_base);
                }
            }
        }

        Ok((arena_bytes_used_high_water, arena_bytes_reserved_high_water))
    }

    /// Seal the scalar accumulator and atomically publish the next MANIFEST.
    ///
    /// # Errors
    ///
    /// Returns typed flush, segment-install, manifest-transition, durability,
    /// or cancellation failures. Installed-but-unreferenced segments remain
    /// invisible and are retained for a publication retry.
    pub(super) async fn commit(&mut self, cx: &Cx) -> Result<&KeeperSnapshot, QuillIndexError> {
        self.commit_with_trigger(cx, LifecycleTrigger::ExplicitFlush)
            .await
    }

    pub(super) async fn commit_with_trigger(
        &mut self,
        cx: &Cx,
        trigger: LifecycleTrigger,
    ) -> Result<&KeeperSnapshot, QuillIndexError> {
        check_cancel(cx, "commit")?;
        if self.pending_delta_seal.is_some() {
            return Err(invalid_state(
                "resume the retained Delta seal before scalar commit",
            ));
        }
        if self.has_active_deltas() {
            return Err(invalid_state(
                "scalar commit cannot run while process-local Delta epochs are active",
            ));
        }
        self.flush_all_shards(cx, trigger).await?;
        self.publish_pending_segments(cx, trigger).await?;
        if !self.config.bulk_load_mode {
            self.apply_tier_policy(cx).await?;
        }
        self.ingest_retry_required = false;
        self.authority_snapshot()
    }

    pub(super) async fn publish_bulk_cadence_if_due(
        &mut self,
        cx: &Cx,
    ) -> Result<(), QuillIndexError> {
        if self.config.bulk_load_mode
            && self.pending_segments.len() >= self.config.bulk_publish_segment_cadence
        {
            self.publish_pending_segments(cx, LifecycleTrigger::BulkCadence)
                .await?;
        }
        Ok(())
    }

    async fn publish_pending_segments(
        &mut self,
        cx: &Cx,
        trigger: LifecycleTrigger,
    ) -> Result<(), QuillIndexError> {
        check_cancel(cx, "commit publish")?;
        if self.pending_segments.is_empty()
            && self.pending_replacement_manifest.is_none()
            && self.pending_manifest.is_none()
        {
            return Ok(());
        }

        self.prepare_pending_manifest()?;
        let manifest = self
            .pending_manifest
            .as_ref()
            .expect("nonempty pending segments have a retained MANIFEST proposal")
            .clone();
        let prepared_publication = self
            .published_snapshot
            .prepare_sealed_manifest(self.schema, &manifest)?;
        Self::preflight_prepared_publication(&prepared_publication)?;
        let commit_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::KEEPER_LIFECYCLE,
            phase = "publish",
            trigger = trigger.as_str(),
            action = "publish",
            generation = manifest.generation,
            segment_count = manifest.segments.len(),
            doc_count = manifest.field_stats.first().map_or(0, |row| row.doc_count),
            result_count = manifest.segments.len(),
            duration_us = tracing::field::Empty,
        );
        let _commit_timer = crate::tracing_conventions::StageTimer::new(&commit_span);
        let instrumented = commit_span.clone();
        async {
            #[cfg(feature = "conformance-internals")]
            self.reader
                .conformance_controller
                .checkpoint(ConformanceCancellationStage::CommitPublication, cx);
            check_cancel(cx, "commit publish")?;
            let open_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::KEEPER_OPEN,
                phase = "open.committed",
                durability = matches!(&self.backend, IndexBackend::Durable(_)),
                generation = tracing::field::Empty,
                segment_count = tracing::field::Empty,
                doc_count = tracing::field::Empty,
                duration_us = tracing::field::Empty,
            );
            let _open_timer = crate::tracing_conventions::StageTimer::new(&open_span);
            let instrumented = open_span.clone();
            async {
                let published: Result<(), KeeperError> = match &mut self.backend {
                    IndexBackend::Durable(writer) => writer
                        .publish_with_intent(cx, &manifest, PublishIntent::PreserveLiveDocuments)
                        .await
                        .map(drop),
                    IndexBackend::Memory(snapshot) => snapshot
                        .publish_owned_segments_with_intent(
                            &manifest,
                            self.pending_owned_segments.clone(),
                            PublishIntent::PreserveLiveDocuments,
                        )
                        .map(|successor| *snapshot = successor),
                };
                if let Err(error) = published {
                    if matches!(error, KeeperError::LiveDocumentFloor { .. }) {
                        // A scalar commit only ever adds rows or replaces them
                        // one for one, so a proposal that would shrink the live
                        // set is a replacement whose rows never landed. No
                        // retry can repair it: the tombstones are already
                        // staged and scalar indexing is refused while a
                        // proposal is retained, so the only thing a retry
                        // could publish is the same deletion. Drop the whole
                        // transaction and leave the published generation
                        // authoritative; installed-but-unreferenced segments
                        // are ordinary grace-window GC candidates (gh#45).
                        self.discard_scalar_transaction();
                    }
                    return Err(error.into());
                }
                // `writer.publish` may have advanced Keeper before the local
                // ArcSwap installs the matching successor. Use only the
                // proven Keeper view for this synchronous bridge; strict
                // authority resumes immediately after installation below.
                let authority = Arc::new(self.proven_authority_snapshot()?.clone());
                SnapshotPublisher::validate_prepared_sealed(
                    authority.as_ref(),
                    &prepared_publication,
                )?;
                self.reader
                    .install_validated_prepared_sealed(authority, prepared_publication)?;
                record_snapshot_fields(&open_span, self.authority_snapshot()?);
                Ok::<(), QuillIndexError>(())
            }
            .instrument(instrumented)
            .await?;
            Ok::<(), QuillIndexError>(())
        }
        .instrument(instrumented)
        .await?;
        self.finish_published_pending_segments();
        Ok(())
    }

    async fn apply_tier_policy(&mut self, cx: &Cx) -> Result<(), QuillIndexError> {
        loop {
            let policy = TierMergePolicy::from_config(&self.config);
            let plan = plan_tier_merge(
                &self
                    .authority_snapshot()?
                    .loaded_manifest()
                    .manifest
                    .segments,
                policy,
            )?;
            let Some(plan) = plan else {
                return Ok(());
            };
            let output_segment_id =
                self.derive_policy_segment_id(&plan.source_segment_ids, b"tier")?;
            let created_unix_s = self.created_unix_s()?;
            let policy_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::KEEPER_LIFECYCLE,
                phase = "merge",
                trigger = LifecycleTrigger::TierFanout.as_str(),
                action = "concat_merge",
                tier = ?plan.tier,
                source_count = plan.source_segment_ids.len(),
                docid_lo = plan.docid_lo,
                docid_hi = plan.docid_hi,
                hole_ratio = plan.hole_ratio,
                segment_id = output_segment_id,
                duration_us = tracing::field::Empty,
            );
            let _policy_timer = crate::tracing_conventions::StageTimer::new(&policy_span);
            self.concat_merge(
                cx,
                &plan.source_segment_ids,
                output_segment_id,
                created_unix_s,
            )
            .instrument(policy_span)
            .await?;
        }
    }

    pub(super) async fn finish_bulk_load(
        &mut self,
        cx: &Cx,
    ) -> Result<&KeeperSnapshot, QuillIndexError> {
        if !self.config.bulk_load_mode {
            return Err(invalid_state(
                "finish_bulk_load requires bulk_load_mode in QuillConfig",
            ));
        }
        self.commit_with_trigger(cx, LifecycleTrigger::BulkFinish)
            .await?;
        let source_segment_ids = self
            .authority_snapshot()?
            .loaded_manifest()
            .manifest
            .segments
            .iter()
            .map(|segment| segment.segment_id)
            .collect::<Vec<_>>();
        if source_segment_ids.len() > 1 {
            let output_segment_id =
                self.derive_policy_segment_id(&source_segment_ids, b"bulk-finish")?;
            let created_unix_s = self.created_unix_s()?;
            let bulk_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::KEEPER_LIFECYCLE,
                phase = "merge",
                trigger = LifecycleTrigger::BulkFinish.as_str(),
                action = "concat_merge",
                source_count = source_segment_ids.len(),
                segment_id = output_segment_id,
                duration_us = tracing::field::Empty,
            );
            let _bulk_timer = crate::tracing_conventions::StageTimer::new(&bulk_span);
            self.concat_merge(cx, &source_segment_ids, output_segment_id, created_unix_s)
                .instrument(bulk_span)
                .await?;
        }
        self.publish_bulk_completion(cx).await?;
        self.reader.config.bulk_load_mode = false;
        self.authority_snapshot()
    }

    async fn publish_bulk_completion(&mut self, cx: &Cx) -> Result<(), QuillIndexError> {
        let current = &self.authority_snapshot()?.loaded_manifest().manifest;
        if current.flags & MANIFEST_FLAG_BULK_MODE_IN_PROGRESS == 0 {
            return Ok(());
        }
        let mut manifest = self.authority_snapshot()?.next_manifest()?;
        manifest.flags &= !MANIFEST_FLAG_BULK_MODE_IN_PROGRESS;
        manifest.last_publish_unix_s = 0;
        if matches!(&self.backend, IndexBackend::Durable(_)) {
            manifest.last_publish_unix_s = wall_clock_unix_s()?;
        }
        let prepared = self
            .published_snapshot
            .prepare_sealed_manifest(self.schema, &manifest)?;
        Self::preflight_prepared_publication(&prepared)?;
        check_cancel(cx, "bulk completion publish")?;
        match &mut self.backend {
            IndexBackend::Durable(writer) => {
                writer
                    .publish_with_intent(cx, &manifest, PublishIntent::PreserveLiveDocuments)
                    .await?;
            }
            IndexBackend::Memory(snapshot) => {
                *snapshot = snapshot.publish_owned_segments_with_intent(
                    &manifest,
                    Vec::new(),
                    PublishIntent::PreserveLiveDocuments,
                )?;
            }
        }
        let authority = Arc::new(self.proven_authority_snapshot()?.clone());
        SnapshotPublisher::validate_prepared_sealed(authority.as_ref(), &prepared)?;
        self.reader
            .install_validated_prepared_sealed(authority, prepared)?;
        Ok(())
    }

    /// Replace one exact committed manifest run with a Q1 concat merge.
    ///
    /// The caller supplies source ids in current manifest order plus a
    /// collision-free output identity. Policy-driven candidate selection and
    /// identity generation remain Keeper E3.7 concerns; this method enforces
    /// that no accumulator or staged publication can race the structural
    /// replacement.
    ///
    /// # Errors
    ///
    /// Rejects uncommitted state, cancellation, a nonconsecutive source run,
    /// codec/invariant failures, or durable publication failure.
    pub(super) async fn concat_merge(
        &mut self,
        cx: &Cx,
        source_segment_ids: &[u64],
        output_segment_id: u64,
        created_unix_s: i64,
    ) -> Result<&KeeperSnapshot, QuillIndexError> {
        check_cancel(cx, "concat merge")?;
        if self.has_uncommitted_changes() {
            return Err(invalid_state(
                "concat merge requires a fully committed scalar index",
            ));
        }
        // A cross-shard concat can cover an earlier shard's unused lease
        // tail. Retire leases in that case so future seals cannot overlap
        // the merged hull. When every next ID is beyond all published
        // intervals, however, keeping the live leases is safe and avoids
        // introducing burned gaps after each continuous-writer tier merge.
        let published_hi = self
            .authority_snapshot()?
            .loaded_manifest()
            .manifest
            .segments
            .last()
            .map_or(0, |segment| segment.docid_hi);
        let retirement_required = (0..self.docid_allocator.shard_count()).any(|shard| {
            self.docid_allocator
                .live_lease(shard)
                .is_some_and(|(base, next)| {
                    base.checked_add(u64::from(next))
                        .is_none_or(|next_docid| next_docid < published_hi)
                })
        });
        if retirement_required {
            self.retire_ingest_leases()?;
        }
        let next_seal_seq = self
            .next_seal_seq
            .checked_add(1)
            .ok_or_else(|| invalid_state("seal sequence exhausted"))?;
        let prepared_publication = self
            .published_snapshot
            .prepare_equivalent_sealed_successor()?;
        Self::preflight_prepared_publication(&prepared_publication)?;
        match &mut self.backend {
            IndexBackend::Durable(writer) => {
                writer
                    .concat_merge(cx, source_segment_ids, output_segment_id, created_unix_s)
                    .await?;
            }
            IndexBackend::Memory(snapshot) => {
                check_cancel(cx, "concat merge")?;
                let source_snapshot = snapshot.clone();
                let mut source_ids = Vec::new();
                source_ids
                    .try_reserve_exact(source_segment_ids.len())
                    .map_err(|_| invalid_state("could not allocate concat-merge source ids"))?;
                source_ids.extend_from_slice(source_segment_ids);
                let successor = spawn_blocking(move || {
                    source_snapshot.concat_merge_owned(
                        &source_ids,
                        output_segment_id,
                        created_unix_s,
                    )
                })
                .await?;
                check_cancel(cx, "concat merge")?;
                *snapshot = successor;
            }
        }
        let authority = Arc::new(self.proven_authority_snapshot()?.clone());
        SnapshotPublisher::validate_prepared_sealed(authority.as_ref(), &prepared_publication)?;
        self.reader
            .install_validated_prepared_sealed(authority, prepared_publication)?;
        self.next_seal_seq = next_seal_seq;
        self.authority_snapshot()
    }

    fn retire_ingest_leases(&mut self) -> Result<(), QuillIndexError> {
        if self
            .shards
            .iter()
            .any(|shard| shard.accumulator.document_count() != 0)
        {
            return Err(invalid_state(
                "cannot retire ingest leases while a Scribe shard is nonempty",
            ));
        }
        let burn = self.docid_allocator.end_session();
        self.next_lease_base = self.next_lease_base.max(burn.final_watermark);
        self.docid_allocator =
            DocIdAllocator::open(self.next_lease_base, self.shard_router.shard_count())
                .map_err(|error| invalid_state(error.to_string()))?;
        for shard in &mut self.shards {
            shard.current_lease_base = None;
        }
        Ok(())
    }

    /// Fold tombstones into immutable positional holes for eligible segments.
    ///
    /// Every surviving global document id is preserved. The complete set of
    /// replacement files is built and synced before one successor MANIFEST is
    /// published, so interruption before that boundary leaves the old snapshot
    /// authoritative. A policy-boundary equality is a no-op.
    ///
    /// # Errors
    ///
    /// Rejects uncommitted scalar or Delta state, invalid policy values,
    /// cancellation, source/codec failures, and durable publication failures.
    pub(super) async fn compact(
        &mut self,
        cx: &Cx,
        policy: CompactionPolicy,
    ) -> Result<CompactionReport, QuillIndexError> {
        check_cancel(cx, "compaction")?;
        if self.has_uncommitted_changes() {
            return Err(invalid_state(
                "compaction requires a fully committed scalar index",
            ));
        }
        let authority = self.authority_snapshot()?;
        let authority_generation = authority.loaded_manifest().manifest.generation;
        let authority_segment_count = authority.segments().len();
        let created_unix_s = self.created_unix_s()?;
        let compact_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::KEEPER_COMPACT,
            phase = "compact",
            durability = matches!(&self.backend, IndexBackend::Durable(_)),
            generation = authority_generation,
            segment_count = authority_segment_count,
            result_count = tracing::field::Empty,
            output_bytes = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _compact_timer = crate::tracing_conventions::StageTimer::new(&compact_span);
        let instrumented = compact_span.clone();
        let report = async {
            match &mut self.backend {
                IndexBackend::Durable(writer) => Ok::<CompactionReport, QuillIndexError>(
                    writer.compact(cx, policy, created_unix_s).await?,
                ),
                IndexBackend::Memory(snapshot) => {
                    let source = snapshot.clone();
                    let (successor, report) =
                        spawn_blocking(move || source.compact_owned(policy, created_unix_s))
                            .await?;
                    check_cancel(cx, "compaction")?;
                    PublicationReadState::validate_generation(
                        successor.loaded_manifest().manifest.generation,
                    )?;
                    *snapshot = successor;
                    Ok(report)
                }
            }
        }
        .instrument(instrumented)
        .await?;
        compact_span.record("result_count", report.compacted_segments);
        compact_span.record("output_bytes", report.output_bytes);
        if report.changed() {
            let authority = self.proven_authority_snapshot()?;
            let prepared_publication = self
                .published_snapshot
                .prepare_sealed_manifest(self.schema, &authority.loaded_manifest().manifest)?;
            SnapshotPublisher::validate_prepared_sealed(authority, &prepared_publication)?;
            self.reader.install_validated_prepared_sealed(
                Arc::new(authority.clone()),
                prepared_publication,
            )?;
            self.next_seal_seq = self
                .authority_snapshot()?
                .segments()
                .iter()
                .map(|segment| segment.manifest().seal_seq)
                .max()
                .unwrap_or(0);
        }
        Ok(report)
    }

    /// Reclaim segment files no durable MANIFEST slot references.
    ///
    /// Runs the Keeper's writer-owned garbage sweep under the held admission.
    /// A file is unlinked only when neither `MANIFEST` nor `MANIFEST.prev`
    /// references it and a full grace period has elapsed since the
    /// publication that retired it (its durable retirement receipt), so a
    /// reader still holding the previous generation keeps working: it had the
    /// whole grace period to open its segments, and a segment it has already
    /// mapped survives the unlink. The same sweep runs once at writer open;
    /// this entry point lets a long-lived or frequently publishing writer
    /// reclaim without reopening, since a merge's folded inputs become
    /// collectable only after later publications age them out. `options` of
    /// `None` uses the writer's configured grace window.
    ///
    /// # Errors
    ///
    /// Rejects uncommitted state (a staged-but-unpublished segment is
    /// indistinguishable from an orphan on disk) and cancellation, and
    /// returns the Keeper's typed recovery, identity, metadata, or unlink
    /// failure otherwise. Nothing is removed on any error path.
    pub(super) async fn collect_garbage(
        &mut self,
        cx: &Cx,
        options: Option<GarbageCollectionOptions>,
    ) -> Result<GarbageCollectionReport, QuillIndexError> {
        check_cancel(cx, "garbage collection")?;
        if self.has_uncommitted_changes() {
            return Err(invalid_state(
                "garbage collection requires a fully committed index",
            ));
        }
        match (&mut self.backend, options) {
            (IndexBackend::Durable(writer), None) => Ok(writer.collect_garbage(cx).await?),
            (IndexBackend::Durable(writer), Some(options)) => {
                Ok(writer.collect_garbage_with(cx, options).await?)
            }
            (IndexBackend::Memory(_), _) => Ok(GarbageCollectionReport {
                removed: Vec::new(),
            }),
        }
    }

    pub(super) async fn upsert_documents_batch(
        &mut self,
        cx: &Cx,
        documents: IngestBatch<'_>,
    ) -> Result<(), QuillIndexError> {
        check_cancel(cx, "upsert documents")?;
        // Publish an unrelated pending batch before replacing a live document.
        //
        // The refusal below exists because the replacement MANIFEST is derived
        // from the PUBLISHED snapshot, so pending shard state that has not been
        // published yet cannot participate in it. That is a reason to publish
        // first, not a reason to reject the call: `LexicalWrite` is the
        // engine-neutral surface, and the pinned Tantivy backend accepts a
        // replacement with writes outstanding, so refusing here made an
        // ordinary interleaving succeed on one backend and fail on the other.
        //
        // Deliberately narrow, and gated on ORDINARY pending work only.
        //
        // A scalar retry sentinel does not mean "there is work to publish", it
        // means "a prior mutation ended ambiguously and must be reconciled by
        // an explicit commit before more is accepted". Publishing from here
        // would resolve that state as a side effect of an unrelated call —
        // committing a possibly-partial ingest and clearing the guard that
        // exists to stop exactly that. Every such sentinel therefore keeps its
        // existing typed refusal instead:
        //
        //   * `ingest_retry_required` — a prior scalar ingest returned after
        //     mutation may have begun;
        //   * `staged_flush`          — a seal is installed but not committed;
        //   * `pending_manifest` / `pending_replacement_manifest` — a retained
        //     publication proposal awaiting its retry;
        //   * `pending_delta_seal`    — a retained Delta seal awaiting resume.
        //
        // An active Delta epoch likewise falls through untouched, so its own
        // typed refusal still reports the real reason in the original order.
        let scalar_retry_pending = self.ingest_retry_required
            || self.staged_flush.is_some()
            || self.pending_manifest.is_some()
            || self.pending_replacement_manifest.is_some()
            || self.pending_delta_seal.is_some();
        if !scalar_retry_pending
            && self.has_uncommitted_changes()
            && !self.has_active_deltas()
            && self.batch_replaces_published_document(cx, documents)?
        {
            // Refuse a malformed batch BEFORE publishing anything. Publication
            // is a side effect of accepting this call, so a batch that
            // admission will reject anyway must not be able to publish
            // unrelated pending work on its way to failing
            // (bd-quill-rejected-ingest-publishes-partial-batch-aihri): a
            // rejected ingest stages nothing, and that has to keep holding for
            // the publish-first path.
            //
            // Only the commit-INVARIANT conditions belong here. Whether an id
            // is a duplicate or a legal replacement is decided against the
            // published snapshot, and publishing is exactly what changes that
            // answer, so those are left to the full admission pass that runs
            // after the commit against the re-derived snapshot.
            Self::validate_commit_invariant_batch_shape(cx, documents)?;
            self.commit(cx).await?;
        }
        let snapshot = self.authority_snapshot()?;
        let mut manifest = snapshot.next_manifest()?;
        let mut replacement_ids = BTreeSet::new();
        for document in documents {
            check_cancel(cx, "upsert document batch")?;
            if snapshot.delete_document(&mut manifest, document.id())? {
                replacement_ids.insert(document.id());
            }
        }
        if replacement_ids.is_empty() {
            return self.index_documents_batch(cx, documents).await;
        }
        // `ingest_retry_required` is checked HERE, next to the pending-state
        // refusal, because `has_uncommitted_changes` does not include it: a
        // cancelled ingest can arm the retry guard while leaving no accumulated
        // documents, no staged flush and no retained proposal. Such a writer
        // looks perfectly clean to the check below.
        //
        // Reaching the installation on that state loses data. The staged
        // manifest already carries the tombstone for the document being
        // replaced, so installing it and only then failing in
        // `index_documents_with_replacements` — which does check the guard —
        // leaves a retained replacement proposal that is a DELETION with no
        // replacement rows behind it. The explicit retry commit the guard is
        // asking for then publishes exactly that, and the live document is gone.
        //
        // Refusing before the installation keeps the retry state inert: nothing
        // is staged, so the retry commit has nothing to publish.
        if self.ingest_retry_required || self.has_uncommitted_changes() {
            let message = if self.pending_replacement_manifest.is_some() {
                "a retained replacement MANIFEST awaits commit retry"
            } else if self.ingest_retry_required {
                "a prior scalar mutation requires commit retry before replacing live documents"
            } else {
                "uncommitted changes must be committed before replacing live documents"
            };
            return Err(invalid_state(message));
        }
        if self.has_active_deltas() {
            return Err(invalid_state(
                "upsert documents cannot mutate a frozen process-local Delta epoch",
            ));
        }

        self.pending_replacement_manifest = Some(manifest);
        if let Err(error) = self
            .index_documents_with_replacements(
                cx,
                documents,
                &replacement_ids,
                false,
                IngestParallelismPolicy::Adaptive,
            )
            .await
        {
            // The batch did not land — admission refused it, it was cancelled
            // or a shard failed partway through — but the proposal above
            // already carries a tombstone for every identity it meant to
            // replace. Left in place, that proposal is a deletion with no rows
            // behind it, and the explicit `commit()` the caller is told to
            // issue next would publish exactly that (gh#45). The guard above
            // proved the writer clean before this call, so discarding the whole
            // scalar transaction restores precisely the state the caller
            // started from: the previous generation authoritative, nothing
            // staged, and the retry guard disarmed because there is nothing
            // to retry.
            self.discard_scalar_transaction();
            return Err(error);
        }
        self.flush_all_shards(cx, LifecycleTrigger::ExplicitFlush)
            .await?;
        self.prepare_pending_manifest()?;
        self.publish_pending_segments(cx, LifecycleTrigger::ExplicitFlush)
            .await?;
        if !self.config.bulk_load_mode {
            self.apply_tier_policy(cx).await?;
        }
        Ok(())
    }

    /// The admission conditions that publishing cannot change.
    ///
    /// An empty identifier and an in-batch repeat are invalid against every
    /// possible published state, so they can be refused before an implicit
    /// publication without consulting the snapshot. Everything else admission
    /// checks — an id already staged in `uncommitted_ids`, or live in the
    /// published snapshot without being replaced — is exactly what a commit
    /// resolves, so deciding it here would refuse batches that are legal once
    /// the pending work is published.
    ///
    /// Documents are walked in admission's own order and produce admission's
    /// own typed errors, so a batch rejected here is rejected the same way it
    /// would have been one step later.
    pub(super) fn validate_commit_invariant_batch_shape(
        cx: &Cx,
        documents: IngestBatch<'_>,
    ) -> Result<(), QuillIndexError> {
        let mut batch_ids = BTreeSet::new();
        for document in documents {
            check_cancel(cx, "upsert admission preflight")?;
            if document.id().is_empty() {
                return Err(invalid_state("document id must be nonempty"));
            }
            if !batch_ids.insert(document.id()) {
                return Err(invalid_state(format!(
                    "duplicate live document id {:?}",
                    document.id()
                )));
            }
        }
        Ok(())
    }

    /// Whether any document in the batch replaces one that is already
    /// published.
    ///
    /// Deliberately PURE, for the same reason
    /// [`Self::validate_batch_admission`] is: it resolves identifiers against
    /// the published snapshot and touches no shard, allocator, manifest or
    /// uncommitted-id state. That is what makes it safe to consult BEFORE
    /// deciding to publish pending work — a batch that turns out to replace
    /// nothing leaves the writer exactly as it found it, so ordinary
    /// accumulate-before-commit is unaffected.
    fn batch_replaces_published_document(
        &self,
        cx: &Cx,
        documents: IngestBatch<'_>,
    ) -> Result<bool, QuillIndexError> {
        let snapshot = self.authority_snapshot()?;
        for document in documents {
            check_cancel(cx, "upsert replacement probe")?;
            if snapshot.resolve_document_id(document.id())?.is_some() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(super) async fn delete_document(
        &mut self,
        cx: &Cx,
        document_id: &str,
    ) -> Result<bool, QuillIndexError> {
        Ok(self.delete_documents(cx, &[document_id]).await? != 0)
    }

    pub(super) async fn delete_documents(
        &mut self,
        cx: &Cx,
        document_ids: &[&str],
    ) -> Result<usize, QuillIndexError> {
        check_cancel(cx, "delete document")?;
        if self.has_uncommitted_changes() {
            return Err(invalid_state(
                "delete_document requires a fully committed scalar index",
            ));
        }
        if self.has_active_deltas() {
            return Err(invalid_state(
                "delete_document cannot mutate a frozen process-local Delta epoch",
            ));
        }
        let snapshot = self.authority_snapshot()?;
        let mut manifest = snapshot.next_manifest()?;
        manifest.last_publish_unix_s = 0;
        let mut deleted = 0_usize;
        for &document_id in document_ids {
            check_cancel(cx, "delete document batch")?;
            if snapshot.delete_document(&mut manifest, document_id)? {
                deleted = deleted.saturating_add(1);
            }
        }
        if deleted == 0 {
            return Ok(0);
        }
        let prepared = self
            .published_snapshot
            .prepare_sealed_manifest(self.schema, &manifest)?;
        Self::preflight_prepared_publication(&prepared)?;
        check_cancel(cx, "delete document publish")?;
        // The caller named every identity being removed: this is the explicit
        // deletion the live-document floor exists to distinguish from a
        // replacement that lost its rows (gh#45).
        match &mut self.backend {
            IndexBackend::Durable(writer) => {
                writer
                    .publish_with_intent(cx, &manifest, PublishIntent::ExplicitDeletion)
                    .await?;
            }
            IndexBackend::Memory(snapshot) => {
                *snapshot = snapshot.publish_owned_segments_with_intent(
                    &manifest,
                    Vec::new(),
                    PublishIntent::ExplicitDeletion,
                )?;
            }
        }
        let authority = Arc::new(self.proven_authority_snapshot()?.clone());
        SnapshotPublisher::validate_prepared_sealed(authority.as_ref(), &prepared)?;
        self.reader
            .install_validated_prepared_sealed(authority, prepared)?;
        Ok(deleted)
    }

    pub(super) async fn delete_all(&mut self, cx: &Cx) -> Result<(), QuillIndexError> {
        check_cancel(cx, "delete all")?;
        if self.has_uncommitted_changes() {
            return Err(invalid_state(
                "delete_all requires a fully committed scalar index",
            ));
        }
        if self.has_active_deltas() {
            return Err(invalid_state(
                "delete_all cannot mutate frozen process-local Delta epochs",
            ));
        }
        let snapshot = self.authority_snapshot()?;
        let mut manifest = snapshot.next_manifest()?;
        manifest.last_publish_unix_s = 0;
        snapshot.delete_all(&mut manifest)?;
        let prepared = self
            .published_snapshot
            .prepare_sealed_manifest(self.schema, &manifest)?;
        Self::preflight_prepared_publication(&prepared)?;
        check_cancel(cx, "delete all publish")?;
        // `delete_all` is the explicit opt-in for publishing a generation with
        // no live documents; every other scalar publication is held to the
        // live-document floor (gh#45).
        match &mut self.backend {
            IndexBackend::Durable(writer) => {
                writer
                    .publish_with_intent(cx, &manifest, PublishIntent::ExplicitDeletion)
                    .await?;
            }
            IndexBackend::Memory(snapshot) => {
                *snapshot = snapshot.publish_owned_segments_with_intent(
                    &manifest,
                    Vec::new(),
                    PublishIntent::ExplicitDeletion,
                )?;
            }
        }
        let authority = Arc::new(self.proven_authority_snapshot()?.clone());
        SnapshotPublisher::validate_prepared_sealed(authority.as_ref(), &prepared)?;
        self.reader
            .install_validated_prepared_sealed(authority, prepared)?;
        Ok(())
    }
}
