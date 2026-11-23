use crate::scribe::{
    ColumnarAccumulator, DEFAULT_ARENA_CHUNK_BYTES, DocIdSpan, FrankensearchTokenizer,
};
use asupersync::Cx;
use frankensearch_core::IndexableDocument;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tracing::Instrument;

use super::checkpoint::LifecycleTrigger;
#[cfg(feature = "conformance-internals")]
use super::conformance::ConformanceCancellationStage;
use super::error_results::{
    MAX_GLOBAL_DOCID_EXCLUSIVE, QuillIndexError, check_cancel, invalid_state,
};
use super::ingest_doc::{IngestBatch, SchemaDocument};
use super::parallel_ingest::{
    FANOUT_MIN_SHARD_DOCUMENTS, FANOUT_WAVE_SHARD_DOCUMENTS, INTERNAL_PARALLEL_INGEST_SHARDS,
    IngestParallelismPolicy, MIN_ARENA_CHUNK_BYTES, PARALLEL_ARENA_BUDGET_DIVISOR,
    ParallelBatchObservation, ParallelBudgetAdmission, ParallelDocumentBudgetBound,
    ParallelIngestCheckpoint, ParallelIngestReceipt, ParallelIngestRoute, ParallelIngestTerminal,
    ParallelObservationRecorder, ParallelShardAssignment, ParallelShardWork, PendingIdentity,
    ScribeShardState, ShardFlushPlan, ValidatedBatchAdmission, catch_parallel_ingest_worker,
    parallel_arena_chunk_bytes, parallel_document_logical_upper_bound, plan_parallel_ingest,
    retain_prepared_metadata_within_budget,
};
use super::writer::{IndexBackend, QuillWriterState};

impl QuillWriterState {
    pub(super) fn parallel_budget_admission(
        &self,
        cx: &Cx,
        documents: IngestBatch<'_>,
        active_shards: usize,
    ) -> Result<Option<ParallelBudgetAdmission>, QuillIndexError> {
        // A schema wider than the five-field shipping shape has no parallel
        // budget model, so a schema batch keeps the scalar route by
        // construction rather than by a per-document refusal.
        let IngestBatch::Default(documents) = documents else {
            return Ok(None);
        };
        // The first budget-safe activation accepts only a pristine Scribe
        // generation. Retained leases or logical rows from an earlier routed
        // batch would make the scalar counterfactual depend on prior shard
        // selection, so those cases deliberately keep the scalar path.
        if self.shards.iter().any(|shard| {
            shard.accumulator.document_count() != 0
                || !shard.identities.is_empty()
                || shard.current_lease_base.is_some()
        }) {
            return Ok(None);
        }

        let Some(initial_logical_bytes) = self.shards.iter().try_fold(0_usize, |total, shard| {
            total.checked_add(shard.accumulator.bytes_used())
        }) else {
            return Ok(None);
        };
        let Some(initial_reserved_bytes) = self.shards.iter().try_fold(0_usize, |total, shard| {
            total.checked_add(shard.accumulator.bytes_reserved())
        }) else {
            return Ok(None);
        };

        let logical_budget_bytes = self.config.scribe_shard_budget_bytes;
        if initial_logical_bytes >= logical_budget_bytes {
            return Ok(None);
        }

        #[cfg(feature = "conformance-internals")]
        let conformance_serial = self.reader.conformance_controller.is_armed();
        #[cfg(not(feature = "conformance-internals"))]
        let conformance_serial = false;
        #[cfg(test)]
        let mut budget_worker_slots = 0_usize;

        let (batch_logical_upper_bound, prepared_metadata) = if conformance_serial {
            let mut analyzer = FrankensearchTokenizer::default();
            let mut admitted_batch_bytes = 0_usize;
            for document in documents {
                #[cfg(feature = "conformance-internals")]
                self.reader
                    .conformance_controller
                    .checkpoint(ConformanceCancellationStage::ParallelBudgetAdmission, cx);
                check_cancel(cx, "parallel budget admission")?;
                let Some(consumed_before_document) =
                    initial_logical_bytes.checked_add(admitted_batch_bytes)
                else {
                    return Ok(None);
                };
                let Some(remaining_budget) =
                    logical_budget_bytes.checked_sub(consumed_before_document)
                else {
                    return Ok(None);
                };
                let Some(document_bound) = parallel_document_logical_upper_bound(
                    self.schema,
                    &mut analyzer,
                    document,
                    remaining_budget,
                )?
                else {
                    return Ok(None);
                };
                let ParallelDocumentBudgetBound::Within {
                    logical_upper_bound: document_upper_bound,
                    canonical_metadata: _,
                } = document_bound
                else {
                    return Ok(None);
                };
                let Some(charged) = admitted_batch_bytes.checked_add(document_upper_bound) else {
                    return Ok(None);
                };
                admitted_batch_bytes = charged;
                if initial_logical_bytes
                    .checked_add(admitted_batch_bytes)
                    .is_none_or(|projected| projected >= logical_budget_bytes)
                {
                    return Ok(None);
                }
            }
            (admitted_batch_bytes, None)
        } else {
            check_cancel(cx, "parallel budget admission")?;
            let mut document_bounds = Vec::new();
            let documents_per_wave = active_shards
                .checked_mul(FANOUT_WAVE_SHARD_DOCUMENTS)
                .ok_or_else(|| invalid_state("parallel budget wave width overflow"))?;
            document_bounds
                .try_reserve_exact(documents.len().min(documents_per_wave))
                .map_err(|_| invalid_state("could not reserve parallel budget document bounds"))?;
            let mut prepared_metadata = Vec::new();
            prepared_metadata
                .try_reserve_exact(documents.len())
                .map_err(|_| invalid_state("could not reserve prepared parallel metadata"))?;
            let retained_metadata_bytes = AtomicUsize::new(0);
            let prepared_metadata_cache_ceiling = logical_budget_bytes
                .checked_sub(initial_logical_bytes)
                .expect("initial logical use is below the configured budget");
            #[cfg(test)]
            let budget_observation = ParallelBatchObservation::default();
            #[cfg(test)]
            if self.force_ingest_overlap_rendezvous {
                budget_observation.arm_rendezvous(active_shards);
            }
            let mut admitted_batch_bytes = 0_usize;
            for wave in documents.chunks(documents_per_wave) {
                document_bounds.clear();
                document_bounds.resize_with(wave.len(), || None);
                let documents_per_worker = wave.len().div_ceil(active_shards).max(1);
                document_bounds
                    .par_chunks_mut(documents_per_worker)
                    .zip(wave.par_chunks(documents_per_worker))
                    .for_each(|(slots, chunk)| {
                        #[cfg(test)]
                        let shard_guard = budget_observation.enter_shard();
                        let mut analyzer = FrankensearchTokenizer::default();
                        for (slot, document) in slots.iter_mut().zip(chunk) {
                            #[cfg(test)]
                            self.parallel_budget_documents_analyzed
                                .fetch_add(1, Ordering::AcqRel);
                            let document_bound = parallel_document_logical_upper_bound(
                                self.schema,
                                &mut analyzer,
                                document,
                                logical_budget_bytes,
                            )
                            .map(|bound| {
                                bound.map(|bound| match bound {
                                    ParallelDocumentBudgetBound::Within {
                                        logical_upper_bound,
                                        canonical_metadata,
                                    } => ParallelDocumentBudgetBound::Within {
                                        logical_upper_bound,
                                        canonical_metadata: canonical_metadata.and_then(
                                            |metadata| {
                                                retain_prepared_metadata_within_budget(
                                                    metadata,
                                                    &retained_metadata_bytes,
                                                    prepared_metadata_cache_ceiling,
                                                )
                                            },
                                        ),
                                    },
                                    ParallelDocumentBudgetBound::ReachesCeiling => {
                                        ParallelDocumentBudgetBound::ReachesCeiling
                                    }
                                })
                            });
                            *slot = Some(document_bound);
                        }
                        #[cfg(test)]
                        shard_guard.completed();
                    });
                for document_bound in &mut document_bounds {
                    let Some(document_bound) = document_bound.take() else {
                        return Err(invalid_state(
                            "parallel budget worker did not publish its document bound",
                        ));
                    };
                    let Some(document_bound) = document_bound? else {
                        return Ok(None);
                    };
                    let ParallelDocumentBudgetBound::Within {
                        logical_upper_bound: document_upper_bound,
                        canonical_metadata,
                    } = document_bound
                    else {
                        return Ok(None);
                    };
                    let Some(charged) = admitted_batch_bytes.checked_add(document_upper_bound)
                    else {
                        return Ok(None);
                    };
                    admitted_batch_bytes = charged;
                    if initial_logical_bytes
                        .checked_add(admitted_batch_bytes)
                        .is_none_or(|projected| projected >= logical_budget_bytes)
                    {
                        return Ok(None);
                    }
                    prepared_metadata.push(canonical_metadata);
                }
            }
            check_cancel(cx, "parallel budget admission")?;
            #[cfg(test)]
            {
                budget_worker_slots = budget_observation.snapshot().distinct_worker_slots;
            }
            (admitted_batch_bytes, Some(prepared_metadata))
        };
        let Some(projected_logical_upper_bound) =
            initial_logical_bytes.checked_add(batch_logical_upper_bound)
        else {
            return Ok(None);
        };
        debug_assert!(projected_logical_upper_bound < logical_budget_bytes);

        Ok(Some(ParallelBudgetAdmission {
            logical_budget_bytes,
            initial_logical_bytes,
            initial_reserved_bytes,
            batch_logical_upper_bound,
            projected_logical_upper_bound,
            arena_chunk_bytes: parallel_arena_chunk_bytes(batch_logical_upper_bound, active_shards),
            prepared_metadata,
            #[cfg(test)]
            budget_worker_slots,
        }))
    }

    /// Route and accumulate one bounded batch into the production Scribe shard set.
    ///
    /// A budget or lease boundary seals an immutable segment, but no newly
    /// installed segment becomes searchable until [`Self::commit`] publishes
    /// its successor MANIFEST.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, duplicate-id, accumulation, flush, or
    /// publication failures. A failed prior MANIFEST publish must be retried
    /// with `commit` before more documents are accepted.
    pub(super) async fn index_documents(
        &mut self,
        cx: &Cx,
        documents: &[IndexableDocument],
    ) -> Result<(), QuillIndexError> {
        self.index_documents_batch(cx, IngestBatch::Default(documents))
            .await
    }

    /// Route and accumulate one bounded batch already projected onto this
    /// index's compiled schema.
    ///
    /// # Errors
    ///
    /// Identical to [`Self::index_documents`].
    pub(super) async fn index_schema_documents(
        &mut self,
        cx: &Cx,
        documents: &[SchemaDocument],
    ) -> Result<(), QuillIndexError> {
        self.index_documents_batch(cx, IngestBatch::Schema(documents))
            .await
    }

    pub(super) async fn index_documents_batch(
        &mut self,
        cx: &Cx,
        documents: IngestBatch<'_>,
    ) -> Result<(), QuillIndexError> {
        let replacement_ids = BTreeSet::new();
        self.index_documents_with_replacements(
            cx,
            documents,
            &replacement_ids,
            true,
            IngestParallelismPolicy::Adaptive,
        )
        .await
    }

    /// Index through the production scalar accumulator without parallel-planner
    /// fanout so pruning conformance can request a specific leaf geometry.
    ///
    /// Shipping ingest keeps its adaptive shared-nothing and fixed-width
    /// internal fanout. This feature-gated seam exists only because Salej
    /// treats the number and size of sealed leaves as evidence. Scalar lease,
    /// arena-budget, visibility-publication, and tier-merge rules remain live,
    /// so proof callers must validate the realized topology before using it as
    /// evidence.
    ///
    /// # Errors
    ///
    /// Returns the same typed cancellation, duplicate-ID, accumulation,
    /// flush, or publication failures as [`Self::index_documents`].
    #[cfg(feature = "pruning-conformance")]
    pub(super) async fn index_documents_with_scalar_topology_conformance(
        &mut self,
        cx: &Cx,
        documents: &[IndexableDocument],
    ) -> Result<(), QuillIndexError> {
        let documents = IngestBatch::Default(documents);
        let replacement_ids = BTreeSet::new();
        self.index_documents_with_replacements(
            cx,
            documents,
            &replacement_ids,
            true,
            IngestParallelismPolicy::ScalarTopologyConformance,
        )
        .await
    }

    async fn try_index_documents_internal_parallel(
        &mut self,
        cx: &Cx,
        admission: ValidatedBatchAdmission<'_>,
        replacement_ids: &BTreeSet<&str>,
        allow_automatic_publication: bool,
        observation: &ParallelBatchObservation,
    ) -> Result<Option<ParallelIngestReceipt>, QuillIndexError> {
        let documents = admission.documents;
        if !replacement_ids.is_empty()
            || !self.shard_router.is_deterministic()
            || !matches!(&self.backend, IndexBackend::Memory(_))
            || self.shards.iter().any(|state| {
                state.accumulator.document_count() != 0 || !state.identities.is_empty()
            })
        {
            return Ok(None);
        }

        // Use a fixed logical width so deterministic-ingest output does not
        // change with the host's available parallelism. Rayon executes the
        // same independent jobs serially when installed on a one-thread pool.
        let plan = plan_parallel_ingest(
            documents.len(),
            INTERNAL_PARALLEL_INGEST_SHARDS,
            INTERNAL_PARALLEL_INGEST_SHARDS,
        )?;
        if plan.route == ParallelIngestRoute::Serial {
            return Ok(None);
        }
        let Some(initial_logical_bytes) = self.shards.iter().try_fold(0_usize, |total, shard| {
            total.checked_add(shard.accumulator.bytes_used())
        }) else {
            return Ok(None);
        };
        let Some(initial_reserved_bytes) = self.shards.iter().try_fold(0_usize, |total, shard| {
            total.checked_add(shard.accumulator.bytes_reserved())
        }) else {
            return Ok(None);
        };
        let logical_budget_bytes = self.config.scribe_shard_budget_bytes;
        let Some(available_logical_bytes) = logical_budget_bytes.checked_sub(initial_logical_bytes)
        else {
            return Ok(None);
        };
        if available_logical_bytes == 0 {
            return Ok(None);
        }
        let arena_chunk_bytes = available_logical_bytes
            .saturating_div(plan.active_shards.max(1))
            .saturating_div(PARALLEL_ARENA_BUDGET_DIVISOR)
            .clamp(MIN_ARENA_CHUNK_BYTES, DEFAULT_ARENA_CHUNK_BYTES);

        let document_count = u32::try_from(documents.len())
            .map_err(|_| invalid_state("ingest batch document count does not fit u32"))?;
        let prior_grant_count = self.docid_allocator.lease_grants().len();
        let mut planned_allocator = self.docid_allocator.speculative_clone();
        let allocated = planned_allocator
            .alloc_batch(0, document_count)
            .map_err(|error| invalid_state(error.to_string()))?;
        let [batch_span] = allocated.spans() else {
            // The scalar path already owns exact lease-boundary cuts. Keep it
            // as the fallback for the infrequent batch that crosses one.
            return Ok(None);
        };

        let mut work = Vec::new();
        work.try_reserve_exact(plan.active_shards)
            .map_err(|_| invalid_state("could not reserve internal parallel ingest work"))?;
        for (shard, range) in plan.ranges.iter().copied().enumerate() {
            let range_start = u32::try_from(range.start)
                .map_err(|_| invalid_state("parallel range start does not fit u32"))?;
            let range_len = u32::try_from(range.len())
                .map_err(|_| invalid_state("parallel range length does not fit u32"))?;
            let ord_start = batch_span
                .ord_start
                .checked_add(range_start)
                .ok_or_else(|| invalid_state("parallel range document ordinal overflow"))?;
            work.push(ParallelShardWork {
                shard,
                assignment: ParallelShardAssignment {
                    document_start: range.start,
                    document_end: range.end,
                    span: DocIdSpan {
                        lease_base: batch_span.lease_base,
                        ord_start,
                        len: range_len,
                    },
                },
                state: ScribeShardState {
                    accumulator: ColumnarAccumulator::with_arena_chunk_size(
                        self.schema,
                        arena_chunk_bytes,
                    )?,
                    identities: Vec::new(),
                    current_lease_base: Some(batch_span.lease_base),
                    scratch_metadata: Vec::new(),
                },
                prepared_metadata: None,
            });
        }

        let checkpoint = ParallelIngestCheckpoint::new(&self.reader);
        #[cfg(test)]
        if self.force_ingest_overlap_rendezvous {
            observation.arm_rendezvous(plan.active_shards);
        }
        let completed = work
            .into_par_iter()
            .map(|work| {
                let mut work = work;
                // Started/in-flight/worker-id are recorded HERE, before any
                // fallible step, and the guard decrements on every exit — so a
                // shard that panics or is cancelled still shows up as started
                // and never as completed.
                let shard_guard = observation.enter_shard();
                catch_parallel_ingest_worker(work.shard, || {
                    Self::accumulate_parallel_shard(
                        &mut work.state,
                        documents,
                        None,
                        #[cfg(test)]
                        None,
                        work.assignment,
                        cx,
                        &checkpoint,
                    )
                })?;
                shard_guard.completed();
                Ok(work)
            })
            .collect::<Result<Vec<_>, QuillIndexError>>()?;
        checkpoint.check(cx)?;
        #[cfg(test)]
        let worker_witness = observation.snapshot();

        let created_unix_s = self.created_unix_s()?;
        let mut reserved_segment_ids = BTreeSet::new();
        let mut flush_plans = Vec::new();
        flush_plans
            .try_reserve_exact(completed.len())
            .map_err(|_| invalid_state("could not reserve internal parallel flush plans"))?;
        for (offset, completed_shard) in completed.iter().enumerate() {
            let segment_id = self.derive_segment_id_for_state_avoiding(
                &completed_shard.state,
                batch_span.lease_base,
                created_unix_s,
                &reserved_segment_ids,
            )?;
            reserved_segment_ids.insert(segment_id);
            let offset = u64::try_from(offset)
                .map_err(|_| invalid_state("parallel shard count does not fit u64"))?;
            let seal_seq = self
                .next_seal_seq
                .checked_add(offset)
                .ok_or_else(|| invalid_state("seal sequence exhausted"))?;
            flush_plans.push(ShardFlushPlan {
                shard: completed_shard.shard,
                state: &completed_shard.state,
                segment_id,
                lease_docid_base: batch_span.lease_base,
                created_unix_s,
                seal_seq,
            });
        }
        let built = flush_plans
            .par_iter()
            .map(Self::build_planned_shard_flush)
            .collect::<Result<Vec<_>, _>>()?;
        checkpoint.check(cx)?;

        let mut pending_field_stats = self.pending_field_stats.clone();
        for completed_shard in &completed {
            let shard_document_count =
                u32::try_from(completed_shard.state.accumulator.document_count())
                    .map_err(|_| invalid_state("segment document count does not fit u32"))?;
            for field in completed_shard.state.accumulator.fields() {
                let entry = pending_field_stats
                    .entry(field.field_ord())
                    .or_insert((0, 0));
                entry.0 = entry
                    .0
                    .checked_add(field.total_tokens())
                    .ok_or_else(|| invalid_state("pending field token count overflow"))?;
                entry.1 = entry
                    .1
                    .checked_add(shard_document_count)
                    .ok_or_else(|| invalid_state("pending field document count overflow"))?;
            }
        }
        let batch_logical_bytes = completed.iter().fold(0_usize, |total, shard| {
            total.saturating_add(shard.state.accumulator.bytes_used())
        });
        let Some(projected_logical_bytes) = initial_logical_bytes.checked_add(batch_logical_bytes)
        else {
            return Ok(None);
        };
        if projected_logical_bytes >= logical_budget_bytes {
            // The workers mutated only local accumulators. Discarding them and
            // taking the scalar path preserves the exclusive logical ceiling
            // without tokenizing every document twice on the admitted path.
            return Ok(None);
        }
        let arena_bytes_used_high_water = projected_logical_bytes;
        let arena_bytes_reserved_high_water = completed
            .iter()
            .fold(initial_reserved_bytes, |total, shard| {
                total.saturating_add(shard.state.accumulator.bytes_reserved())
            });
        let next_seal_seq = built
            .last()
            .map_or(self.next_seal_seq, |flush| flush.next_seal_seq);
        self.pending_segments
            .try_reserve(built.len())
            .map_err(|_| invalid_state("could not reserve pending segment bookkeeping"))?;
        self.pending_owned_segments
            .try_reserve(built.len())
            .map_err(|_| invalid_state("could not reserve owned segment bookkeeping"))?;

        // All fallible accumulation, encoding, accounting, and reservations
        // finished against local state. Install the generation as one writer
        // mutation so a failed worker leaves the original batch retryable.
        planned_allocator.commit_speculative_grants(prior_grant_count);
        self.ingest_retry_required = true;
        self.docid_allocator = planned_allocator;
        self.next_lease_base = self.docid_allocator.watermark();
        for flush in built {
            self.pending_segments.push(flush.manifest_segment);
            self.pending_owned_segments.push(flush.encoded);
        }
        self.pending_field_stats = pending_field_stats;
        self.next_seal_seq = next_seal_seq;
        self.uncommitted_ids
            .extend(documents.iter().map(|document| document.id().to_owned()));
        self.unpublished_since.get_or_insert_with(Instant::now);

        if allow_automatic_publication {
            self.publish_bulk_cadence_if_due(cx).await?;
        }
        let visibility_due = self.unpublished_since.is_some_and(|started| {
            started.elapsed() >= Duration::from_millis(self.config.max_visibility_lag_ms)
        });
        if allow_automatic_publication && visibility_due {
            self.commit_with_trigger(cx, LifecycleTrigger::VisibilityLag)
                .await?;
        }

        Ok(Some(ParallelIngestReceipt {
            planner_version: plan.planner_version,
            route: plan.route,
            configured_width: plan.configured_width,
            verified_pool_capacity: rayon::current_num_threads(),
            eligible_shards: plan.eligible_shards,
            active_shards: plan.active_shards,
            logical_budget_bytes,
            initial_logical_bytes,
            batch_logical_upper_bound: batch_logical_bytes,
            projected_logical_upper_bound: projected_logical_bytes,
            arena_chunk_bytes,
            arena_bytes_used_high_water,
            arena_bytes_reserved_high_water,
            #[cfg(test)]
            worker_witness,
        }))
    }

    pub(super) async fn try_index_documents_parallel(
        &mut self,
        cx: &Cx,
        admission: ValidatedBatchAdmission<'_>,
        replacement_ids: &BTreeSet<&str>,
        allow_automatic_publication: bool,
        observation: &ParallelBatchObservation,
    ) -> Result<Option<ParallelIngestReceipt>, QuillIndexError> {
        let documents = admission.documents;
        let shard_count = self.shards.len();
        let verified_pool_capacity = rayon::current_num_threads();
        let plan = plan_parallel_ingest(documents.len(), shard_count, verified_pool_capacity)?;
        if !replacement_ids.is_empty()
            || self.shard_router.is_deterministic()
            || plan.route == ParallelIngestRoute::Serial
        {
            return Ok(None);
        }

        let active_shard_count = plan.active_shards;
        let Some(mut budget_admission) =
            self.parallel_budget_admission(cx, documents, active_shard_count)?
        else {
            return Ok(None);
        };
        #[cfg(test)]
        observation.record_prepared_metadata(
            budget_admission
                .prepared_metadata
                .as_ref()
                .map_or(0, Vec::len),
        );
        let mut prepared_metadata = budget_admission
            .prepared_metadata
            .take()
            .map(Vec::into_iter);
        let mut planned_router = self.shard_router.clone();
        let prior_grant_count = self.docid_allocator.lease_grants().len();
        let mut planned_allocator = self.docid_allocator.speculative_clone();
        let mut selected_shards = vec![false; shard_count];
        let mut work = Vec::new();
        work.try_reserve_exact(active_shard_count)
            .map_err(|_| invalid_state("could not reserve parallel ingest worker plans"))?;
        for range in &plan.ranges {
            let shard = planned_router.route_batch();
            let selected = selected_shards.get_mut(shard).ok_or_else(|| {
                invalid_state("parallel ingest planner selected an unknown shard")
            })?;
            if std::mem::replace(selected, true) {
                return Err(invalid_state(
                    "parallel ingest planner selected one shard more than once",
                ));
            }
            let state = self
                .shards
                .get(shard)
                .ok_or_else(|| invalid_state("parallel ingest shard state is missing"))?;
            if state.accumulator.document_count() != 0 || !state.identities.is_empty() {
                return Ok(None);
            }
            let live_lease = planned_allocator.live_lease(shard);
            if let (Some(current), Some((live, _))) = (state.current_lease_base, live_lease)
                && current != live
            {
                return Ok(None);
            }
            let count = range.len();
            let count_u32 = u32::try_from(count)
                .map_err(|_| invalid_state("parallel shard document count does not fit u32"))?;
            if count_u32 > planned_allocator.contiguous_capacity(shard) {
                return Ok(None);
            }
            let shard_prepared_metadata = if let Some(metadata) = prepared_metadata.as_mut() {
                let mut shard_metadata = Vec::new();
                shard_metadata.try_reserve_exact(count).map_err(|_| {
                    invalid_state("could not reserve parallel shard prepared metadata")
                })?;
                for _ in 0..count {
                    shard_metadata.push(metadata.next().ok_or_else(|| {
                        invalid_state("prepared metadata ended before the parallel plan")
                    })?);
                }
                Some(shard_metadata)
            } else {
                None
            };
            let allocated = planned_allocator
                .alloc_batch(shard, count_u32)
                .map_err(|error| invalid_state(error.to_string()))?;
            let [span] = allocated.spans() else {
                return Err(invalid_state(
                    "parallel shard allocation unexpectedly crossed a Q1 lease",
                ));
            };
            work.push(ParallelShardWork {
                shard,
                assignment: ParallelShardAssignment {
                    document_start: range.start,
                    document_end: range.end,
                    span: *span,
                },
                state: ScribeShardState {
                    accumulator: ColumnarAccumulator::with_arena_chunk_size(
                        self.schema,
                        budget_admission.arena_chunk_bytes,
                    )?,
                    identities: Vec::new(),
                    current_lease_base: Some(span.lease_base),
                    scratch_metadata: Vec::new(),
                },
                prepared_metadata: shard_prepared_metadata,
            });
        }
        if prepared_metadata
            .as_mut()
            .is_some_and(|metadata| metadata.next().is_some())
        {
            return Err(invalid_state(
                "prepared metadata extends beyond the parallel plan",
            ));
        }

        let checkpoint = ParallelIngestCheckpoint::new(&self.reader);
        #[cfg(test)]
        if self.force_ingest_overlap_rendezvous {
            observation.arm_rendezvous(plan.active_shards);
        }
        let completed = work
            .into_par_iter()
            .map(|work| {
                let mut work = work;
                // Started/in-flight/worker-id are recorded HERE, before any
                // fallible step, and the guard decrements on every exit — so a
                // shard that panics or is cancelled still shows up as started
                // and never as completed.
                let shard_guard = observation.enter_shard();
                catch_parallel_ingest_worker(work.shard, || {
                    let prepared_metadata = work.prepared_metadata.take();
                    Self::accumulate_parallel_shard(
                        &mut work.state,
                        documents,
                        prepared_metadata,
                        #[cfg(test)]
                        Some(observation),
                        work.assignment,
                        cx,
                        &checkpoint,
                    )
                })?;
                shard_guard.completed();
                Ok(work)
            })
            .collect::<Result<Vec<_>, QuillIndexError>>()?;
        checkpoint.check(cx)?;
        #[cfg(test)]
        let worker_witness = observation.snapshot();

        let mut replacements = Vec::new();
        replacements
            .try_reserve_exact(shard_count)
            .map_err(|_| invalid_state("could not reserve parallel ingest commit table"))?;
        replacements.resize_with(shard_count, || None);
        for completed_shard in completed {
            let slot = replacements
                .get_mut(completed_shard.shard)
                .ok_or_else(|| invalid_state("parallel ingest completed an unknown shard"))?;
            if slot.replace(completed_shard.state).is_some() {
                return Err(invalid_state(
                    "parallel ingest completed one shard more than once",
                ));
            }
        }

        let Some(arena_bytes_used_high_water) = self.shards.iter().zip(&replacements).try_fold(
            0_usize,
            |total, (current, replacement)| {
                total.checked_add(replacement.as_ref().map_or_else(
                    || current.accumulator.bytes_used(),
                    |state| state.accumulator.bytes_used(),
                ))
            },
        ) else {
            return Err(invalid_state(
                "parallel ingest aggregate logical accounting overflowed",
            ));
        };
        if arena_bytes_used_high_water > budget_admission.projected_logical_upper_bound
            || arena_bytes_used_high_water >= budget_admission.logical_budget_bytes
        {
            return Err(invalid_state(
                "parallel ingest exceeded its pre-worker logical budget proof",
            ));
        }
        let Some(speculative_reserved_bytes) =
            replacements.iter().try_fold(0_usize, |total, replacement| {
                replacement.as_ref().map_or(Some(total), |state| {
                    total.checked_add(state.accumulator.bytes_reserved())
                })
            })
        else {
            return Err(invalid_state(
                "parallel ingest speculative reserve accounting overflowed",
            ));
        };
        let Some(arena_bytes_reserved_high_water) = budget_admission
            .initial_reserved_bytes
            .checked_add(speculative_reserved_bytes)
        else {
            return Err(invalid_state(
                "parallel ingest aggregate reserve accounting overflowed",
            ));
        };

        // This is the first mutation of writer-owned state. Every fallible
        // validation, allocation, worker, cancellation, and panic boundary
        // above operates only on local plans, so a failure remains exactly
        // retryable without committing allocator, router, or shard progress.
        planned_allocator.commit_speculative_grants(prior_grant_count);
        self.ingest_retry_required = true;
        for (state, replacement) in self.shards.iter_mut().zip(replacements) {
            if let Some(replacement) = replacement {
                *state = replacement;
            }
        }
        self.shard_router = planned_router;
        self.docid_allocator = planned_allocator;
        self.next_lease_base = self.docid_allocator.watermark();

        self.uncommitted_ids
            .extend(documents.iter().map(|document| document.id().to_owned()));
        self.unpublished_since.get_or_insert_with(Instant::now);
        let visibility_due = self.unpublished_since.is_some_and(|started| {
            started.elapsed() >= Duration::from_millis(self.config.max_visibility_lag_ms)
        });
        if allow_automatic_publication && visibility_due {
            self.commit_with_trigger(cx, LifecycleTrigger::VisibilityLag)
                .await?;
        }
        Ok(Some(ParallelIngestReceipt {
            planner_version: plan.planner_version,
            route: plan.route,
            configured_width: plan.configured_width,
            verified_pool_capacity: plan.verified_pool_capacity,
            eligible_shards: plan.eligible_shards,
            active_shards: active_shard_count,
            logical_budget_bytes: budget_admission.logical_budget_bytes,
            initial_logical_bytes: budget_admission.initial_logical_bytes,
            batch_logical_upper_bound: budget_admission.batch_logical_upper_bound,
            projected_logical_upper_bound: budget_admission.projected_logical_upper_bound,
            arena_chunk_bytes: budget_admission.arena_chunk_bytes,
            arena_bytes_used_high_water,
            arena_bytes_reserved_high_water,
            #[cfg(test)]
            worker_witness,
        }))
    }

    fn accumulate_parallel_shard(
        state: &mut ScribeShardState,
        documents: IngestBatch<'_>,
        prepared_metadata: Option<Vec<Option<Vec<u8>>>>,
        #[cfg(test)] metadata_observation: Option<&ParallelBatchObservation>,
        assignment: ParallelShardAssignment,
        cx: &Cx,
        checkpoint: &ParallelIngestCheckpoint,
    ) -> Result<(), QuillIndexError> {
        checkpoint.check(cx)?;
        let shard_documents = documents
            .try_slice(assignment.document_start..assignment.document_end)
            .ok_or_else(|| invalid_state("parallel shard document range is outside the batch"))?;
        let mut prepared_metadata = prepared_metadata.map(Vec::into_iter);
        if shard_documents.len()
            != usize::try_from(assignment.span.len)
                .map_err(|_| invalid_state("parallel shard span length does not fit usize"))?
        {
            return Err(invalid_state(
                "parallel shard document range differs from its Q1 span",
            ));
        }
        if prepared_metadata
            .as_ref()
            .is_some_and(|metadata| metadata.len() != shard_documents.len())
        {
            return Err(invalid_state(
                "prepared metadata range differs from its parallel document range",
            ));
        }
        state
            .identities
            .try_reserve(shard_documents.len())
            .map_err(|_| invalid_state("could not reserve parallel shard identities"))?;
        for (document_offset, document) in shard_documents.iter().enumerate() {
            checkpoint.check(cx)?;
            let span_offset = u32::try_from(document_offset)
                .map_err(|_| invalid_state("parallel shard offset does not fit u32"))?;
            let doc_ord = assignment
                .span
                .ord_start
                .checked_add(span_offset)
                .ok_or_else(|| invalid_state("lease-relative document ordinal overflow"))?;
            let global_docid = assignment
                .span
                .lease_base
                .checked_add(u64::from(doc_ord))
                .filter(|docid| *docid < MAX_GLOBAL_DOCID_EXCLUSIVE)
                .ok_or_else(|| invalid_state("global Q1 document-id space exhausted"))?;
            let mut fallback_metadata = Vec::new();
            let prepared_metadata_for_document = if let Some(prepared) = prepared_metadata.as_mut()
            {
                Some(
                    prepared
                        .next()
                        .ok_or_else(|| invalid_state("prepared metadata document is missing"))?,
                )
            } else {
                None
            };
            let metadata = match prepared_metadata_for_document {
                Some(Some(ref metadata)) => {
                    #[cfg(test)]
                    if let Some(observation) = metadata_observation {
                        observation.record_reused_metadata();
                    }
                    metadata.as_slice()
                }
                Some(None) | None => {
                    document.write_canonical_metadata(&mut fallback_metadata)?;
                    fallback_metadata.as_slice()
                }
            };
            document.accumulate(&mut state.accumulator, doc_ord, global_docid, metadata)?;
            let content_hash = document.content_hash(metadata)?;
            state.identities.push(PendingIdentity {
                doc_ord,
                document_id: document.id().to_owned(),
                content_hash,
            });
        }
        Ok(())
    }

    pub(super) async fn index_documents_with_replacements(
        &mut self,
        cx: &Cx,
        documents: IngestBatch<'_>,
        replacement_ids: &BTreeSet<&str>,
        allow_automatic_publication: bool,
        parallelism_policy: IngestParallelismPolicy,
    ) -> Result<(), QuillIndexError> {
        let ingest_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::SCRIBE_INGEST,
            phase = "ingest",
            doc_count = documents.len(),
            result_count = tracing::field::Empty,
            parallel_planner_version = tracing::field::Empty,
            parallel_route = tracing::field::Empty,
            parallel_configured_width = tracing::field::Empty,
            parallel_verified_pool_capacity = tracing::field::Empty,
            parallel_eligible_shards = tracing::field::Empty,
            parallel_active_shards = tracing::field::Empty,
            parallel_logical_budget_bytes = tracing::field::Empty,
            parallel_initial_logical_bytes = tracing::field::Empty,
            parallel_batch_logical_upper_bound = tracing::field::Empty,
            parallel_projected_logical_upper_bound = tracing::field::Empty,
            parallel_arena_chunk_bytes = tracing::field::Empty,
            parallel_run_seq = tracing::field::Empty,
            parallel_pool_threads = tracing::field::Empty,
            parallel_started_shards = tracing::field::Empty,
            parallel_completed_shards = tracing::field::Empty,
            parallel_distinct_worker_slots = tracing::field::Empty,
            parallel_unidentified_shards = tracing::field::Empty,
            parallel_peak_shards_in_flight = tracing::field::Empty,
            parallel_pool_local_worker_ids = tracing::field::Empty,
            parallel_identity_degraded = tracing::field::Empty,
            parallel_shared_nothing_declined = tracing::field::Empty,
            ingest_operation_succeeded = tracing::field::Empty,
            parallel_terminal = tracing::field::Empty,
            arena_bytes_used_high_water = tracing::field::Empty,
            arena_bytes_reserved_high_water = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _ingest_timer = crate::tracing_conventions::StageTimer::new(&ingest_span);
        let instrumented = ingest_span.clone();
        async move {
            // Created before the FIRST early exit — cancellation, pending-seal,
            // active-delta, retry-required, empty-batch and admission refusals
            // all used to return before any observation existed and so emitted
            // nothing at all.
            let observation = ParallelBatchObservation::default();
            let _observation_recorder = ParallelObservationRecorder {
                span: ingest_span.clone(),
                observation: &observation,
            };
            let outcome: Result<(), QuillIndexError> = async {
                check_cancel(cx, "index")?;
                if self.pending_delta_seal.is_some() {
                    return Err(invalid_state(
                        "resume the retained Delta seal before scalar indexing",
                    ));
                }
                if self.has_active_deltas() {
                    return Err(invalid_state(
                        "scalar indexing cannot run while process-local Delta epochs are active",
                    ));
                }
                if self.ingest_retry_required
                    || self.staged_flush.is_some()
                    || self.pending_manifest.is_some()
                    || (self.pending_replacement_manifest.is_some() && replacement_ids.is_empty())
                {
                    return Err(invalid_state(
                        "a prior scalar mutation requires commit retry before indexing",
                    ));
                }
                if documents.is_empty() {
                    observation.set_terminal(ParallelIngestTerminal::NoWork);
                    return Ok(());
                }
                // Admit or refuse the WHOLE batch before any route touches shard,
                // allocator or uncommitted-id state, and before the commit-retry
                // guard is armed, so a refused batch leaves the writer exactly as
                // it found it (bd-quill-rejected-ingest-publishes-partial-batch-aihri).
                let admission = self.validate_batch_admission(cx, documents, replacement_ids)?;
                let parallel_receipt = match parallelism_policy {
                    IngestParallelismPolicy::Adaptive => {
                        let internal = self
                            .try_index_documents_internal_parallel(
                                cx,
                                admission,
                                replacement_ids,
                                allow_automatic_publication,
                                &observation,
                            )
                            .await;
                        let routed = match internal {
                            Ok(Some(receipt)) => Ok(Some(receipt)),
                            Ok(None) => {
                                self.try_index_documents_parallel(
                                    cx,
                                    admission,
                                    replacement_ids,
                                    allow_automatic_publication,
                                    &observation,
                                )
                                .await
                            }
                            Err(error) => Err(error),
                        };
                        // No terminal is set here. `Ok(None)` means the routes
                        // DECLINED and the batch still has the whole scalar path
                        // ahead of it, so promoting success at this point would
                        // let a later error, cancel, or panic publish a success
                        // terminal. The recorder classifies cancellation on every
                        // exit; success is promoted only at the true end.
                        routed?
                    }
                    #[cfg(feature = "pruning-conformance")]
                    IngestParallelismPolicy::ScalarTopologyConformance => {
                        ingest_span.record("parallel_route", "scalar_topology_conformance");
                        None
                    }
                };
                if let Some(receipt) = parallel_receipt {
                    ingest_span.record(
                        "result_count",
                        u64::try_from(documents.len()).unwrap_or(u64::MAX),
                    );
                    ingest_span.record(
                        "parallel_planner_version",
                        u64::from(receipt.planner_version),
                    );
                    ingest_span.record("parallel_route", receipt.route.as_str());
                    ingest_span.record(
                        "parallel_configured_width",
                        u64::try_from(receipt.configured_width).unwrap_or(u64::MAX),
                    );
                    ingest_span.record(
                        "parallel_verified_pool_capacity",
                        u64::try_from(receipt.verified_pool_capacity).unwrap_or(u64::MAX),
                    );
                    ingest_span.record(
                        "parallel_eligible_shards",
                        u64::try_from(receipt.eligible_shards).unwrap_or(u64::MAX),
                    );
                    ingest_span.record(
                        "parallel_active_shards",
                        u64::try_from(receipt.active_shards).unwrap_or(u64::MAX),
                    );
                    ingest_span.record(
                        "parallel_logical_budget_bytes",
                        u64::try_from(receipt.logical_budget_bytes).unwrap_or(u64::MAX),
                    );
                    ingest_span.record(
                        "parallel_initial_logical_bytes",
                        u64::try_from(receipt.initial_logical_bytes).unwrap_or(u64::MAX),
                    );
                    ingest_span.record(
                        "parallel_batch_logical_upper_bound",
                        u64::try_from(receipt.batch_logical_upper_bound).unwrap_or(u64::MAX),
                    );
                    ingest_span.record(
                        "parallel_projected_logical_upper_bound",
                        u64::try_from(receipt.projected_logical_upper_bound).unwrap_or(u64::MAX),
                    );
                    ingest_span.record(
                        "parallel_arena_chunk_bytes",
                        u64::try_from(receipt.arena_chunk_bytes).unwrap_or(u64::MAX),
                    );
                    ingest_span.record(
                        "arena_bytes_used_high_water",
                        u64::try_from(receipt.arena_bytes_used_high_water).unwrap_or(u64::MAX),
                    );
                    ingest_span.record(
                        "arena_bytes_reserved_high_water",
                        u64::try_from(receipt.arena_bytes_reserved_high_water).unwrap_or(u64::MAX),
                    );
                    #[cfg(test)]
                    {
                        debug_assert_eq!(
                            receipt.worker_witness,
                            observation.snapshot(),
                            "committed parallel receipt must match the batch observation"
                        );
                    }
                    self.ingest_retry_required = false;
                    // The whole operation is finished here: the fan-out published
                    // a receipt and nothing further can fail.
                    observation.set_terminal(ParallelIngestTerminal::FanoutCompleted);
                    return Ok(());
                }
                // No fan-out was published. Any shard counts already on the
                // observation belong to an attempt whose work was DISCARDED, so
                // the terminal must say so rather than read as a completed
                // fan-out.
                // A decline is a FACT about routing, not an ending: the whole
                // scalar/fallback path still lies ahead. Terminal stays `Aborted`
                // until the typed result or a real success, so a fallback panic or
                // a dropped future cannot publish a stale success-shaped terminal.
                observation
                    .shared_nothing_declined
                    .store(true, Ordering::Release);
                // The parallel route arms this guard only at its transactional
                // commit boundary. The serial route still mutates the live router
                // and allocator directly, so arm it immediately before doing so.
                self.ingest_retry_required = true;
                // Fan out only when every participating shard gets a segment's
                // worth of documents: each sealed segment carries its own term
                // dictionary, so splitting a small batch widely trades real bytes
                // and search-time segment count for parallelism that is not there.
                let fanout_shards = self
                    .shard_router
                    .shard_count()
                    .min(documents.len() / FANOUT_MIN_SHARD_DOCUMENTS);
                let mut used_fallback_fanout = false;
                let (arena_bytes_used_high_water, arena_bytes_reserved_high_water) =
                    if fanout_shards >= 2 {
                        // Rayon-parallel too: it gets the same shard witness as the
                        // shared-nothing route, and its own terminal, so it can never
                        // be published as serial.
                        used_fallback_fanout = true;
                        self.index_batch_fanout(
                            cx,
                            documents,
                            allow_automatic_publication,
                            fanout_shards,
                            &observation,
                        )
                        .await?
                    } else {
                        self.index_batch_serial(cx, documents, allow_automatic_publication)
                            .await?
                    };
                let visibility_due = self.unpublished_since.is_some_and(|started| {
                    started.elapsed() >= Duration::from_millis(self.config.max_visibility_lag_ms)
                });
                if allow_automatic_publication && visibility_due {
                    self.commit_with_trigger(cx, LifecycleTrigger::VisibilityLag)
                        .await?;
                }
                ingest_span.record(
                    "result_count",
                    u64::try_from(documents.len()).unwrap_or(u64::MAX),
                );
                ingest_span.record(
                    "arena_bytes_used_high_water",
                    u64::try_from(arena_bytes_used_high_water).unwrap_or(u64::MAX),
                );
                ingest_span.record(
                    "arena_bytes_reserved_high_water",
                    u64::try_from(arena_bytes_reserved_high_water).unwrap_or(u64::MAX),
                );
                self.ingest_retry_required = false;
                observation.set_terminal(if used_fallback_fanout {
                    ParallelIngestTerminal::FanoutFallbackCompleted
                } else {
                    ParallelIngestTerminal::SerialCompleted
                });
                Ok(())
            }
            .await;
            // Typed classification, before the recorder publishes on drop.
            if let Err(error) = &outcome {
                observation.set_terminal(ParallelIngestTerminal::from_error(error));
            }
            outcome
        }
        .instrument(instrumented)
        .await
    }
}
