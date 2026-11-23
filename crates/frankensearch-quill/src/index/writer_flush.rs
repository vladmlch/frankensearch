use crate::keeper::{CURRENT_ENGINE_VERSION, MANIFEST_FLAG_BULK_MODE_IN_PROGRESS, Manifest};
use crate::scribe::{
    FlushDocumentInput, FlushMode, FlushSegmentInput, flush_accumulator_with_mode,
};
use asupersync::Cx;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::collections::BTreeSet;
use tracing::Instrument;
use xxhash_rust::xxh3::{Xxh3, xxh3_64};

use super::checkpoint::LifecycleTrigger;
use super::error_results::{QuillIndexError, check_cancel, invalid_state, wall_clock_unix_s};
use super::ingest_doc::{manifest_segment, merge_field_stats};
use super::parallel_ingest::{BuiltShardFlush, ScribeShardState, ShardFlushPlan, StagedFlush};
use super::writer::{IndexBackend, QuillWriterState};

impl QuillWriterState {
    pub(super) fn prepare_pending_manifest(&mut self) -> Result<(), QuillIndexError> {
        if self.pending_manifest.is_some() {
            return Ok(());
        }
        let manifest = if let Some(manifest) = &self.pending_replacement_manifest {
            manifest.clone()
        } else {
            self.authority_snapshot()?.next_manifest()?
        };
        self.prepare_pending_manifest_from(manifest)?;
        self.pending_replacement_manifest = None;
        Ok(())
    }

    fn prepare_pending_manifest_from(
        &mut self,
        mut manifest: Manifest,
    ) -> Result<(), QuillIndexError> {
        if self.pending_manifest.is_some() {
            return Ok(());
        }
        manifest
            .segments
            .extend(self.pending_segments.iter().cloned());
        manifest
            .segments
            .sort_unstable_by_key(|segment| segment.docid_lo);
        manifest.docid_high_watermark = self.next_lease_base;
        if self.config.bulk_load_mode {
            manifest.flags |= MANIFEST_FLAG_BULK_MODE_IN_PROGRESS;
        } else {
            manifest.flags &= !MANIFEST_FLAG_BULK_MODE_IN_PROGRESS;
        }
        manifest.field_stats = merge_field_stats(&manifest.field_stats, &self.pending_field_stats)?;
        if matches!(&self.backend, IndexBackend::Durable(_)) {
            // Retain one exact, explicitly stamped image until publication
            // succeeds. A crash-shaped retry may encounter the prior attempt's
            // synced `.tmp-manifest-N`, whose bytes must remain reusable even
            // after the wall clock advances.
            manifest.last_publish_unix_s = wall_clock_unix_s()?;
        }
        self.pending_manifest = Some(manifest);
        Ok(())
    }

    pub(super) async fn flush_shard(
        &mut self,
        cx: &Cx,
        requested_shard: usize,
        trigger: LifecycleTrigger,
    ) -> Result<(), QuillIndexError> {
        loop {
            let shard = self
                .staged_flush
                .as_ref()
                .map_or(requested_shard, |staged| staged.shard);
            if self.staged_flush.is_none()
                && self.shards[requested_shard].accumulator.document_count() == 0
            {
                return Ok(());
            }
            let state = &self.shards[shard];
            let seal_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::KEEPER_LIFECYCLE,
                phase = "seal",
                trigger = trigger.as_str(),
                action = "seal",
                shard_id = shard,
                segment_id = tracing::field::Empty,
                doc_count = state.accumulator.document_count(),
                token_count = state.accumulator.token_count(),
                result_count = tracing::field::Empty,
                output_bytes = tracing::field::Empty,
                arena_bytes_used_high_water = state.accumulator.bytes_used(),
                arena_bytes_reserved_high_water = state.accumulator.bytes_reserved(),
                duration_us = tracing::field::Empty,
            );
            let _seal_timer = crate::tracing_conventions::StageTimer::new(&seal_span);
            let instrumented = seal_span.clone();
            async {
                check_cancel(cx, "flush")?;
                self.prepare_shard_flush(shard)?;
                if let Some(staged) = self.staged_flush.as_ref() {
                    seal_span.record("segment_id", staged.manifest_segment.segment_id);
                    seal_span.record("result_count", u64::from(staged.manifest_segment.doc_count));
                    seal_span.record("output_bytes", staged.encoded.file_len());
                }
                self.install_staged_flush(cx).await
            }
            .instrument(instrumented)
            .await?;
            if shard == requested_shard {
                return Ok(());
            }
        }
    }

    pub(super) async fn flush_all_shards(
        &mut self,
        cx: &Cx,
        trigger: LifecycleTrigger,
    ) -> Result<(), QuillIndexError> {
        if let Some(shard) = self.staged_flush.as_ref().map(|staged| staged.shard) {
            self.flush_shard(cx, shard, trigger).await?;
        }

        let nonempty = self
            .shards
            .iter()
            .enumerate()
            .filter_map(|(shard, state)| (state.accumulator.document_count() != 0).then_some(shard))
            .collect::<Vec<_>>();
        self.flush_selected_shards(cx, &nonempty, trigger, false)
            .await
    }

    pub(super) async fn flush_selected_shards(
        &mut self,
        cx: &Cx,
        shards: &[usize],
        trigger: LifecycleTrigger,
        allow_automatic_publication: bool,
    ) -> Result<(), QuillIndexError> {
        // Bulk cadence is part of the segment-identity contract: a cadence
        // publication advances the authority generation used by the next
        // segment ID. Keep that mode on the original ordered seal/publish
        // sequence. Ordinary ingest has no publication between these installs,
        // so its independent encodes may be built together without changing
        // their identities or seal sequence.
        if shards.len() < 2
            || rayon::current_num_threads() < 2
            || (allow_automatic_publication && self.config.bulk_load_mode)
        {
            for &shard in shards {
                self.flush_shard(cx, shard, trigger).await?;
                if allow_automatic_publication {
                    self.publish_bulk_cadence_if_due(cx).await?;
                }
            }
            return Ok(());
        }

        check_cancel(cx, "parallel flush")?;
        let built = {
            let plans = self.plan_parallel_shard_flushes(shards)?;
            plans
                .par_iter()
                .map(Self::build_planned_shard_flush)
                .collect::<Result<Vec<_>, _>>()?
        };

        for built in built {
            check_cancel(cx, "parallel flush install")?;
            let shard = built.shard;
            let state = &self.shards[shard];
            let seal_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::KEEPER_LIFECYCLE,
                phase = "seal",
                trigger = trigger.as_str(),
                action = "install_parallel_build",
                shard_id = shard,
                segment_id = built.manifest_segment.segment_id,
                doc_count = state.accumulator.document_count(),
                token_count = state.accumulator.token_count(),
                result_count = u64::from(built.manifest_segment.doc_count),
                output_bytes = built.encoded.file_len(),
                arena_bytes_used_high_water = state.accumulator.bytes_used(),
                arena_bytes_reserved_high_water = state.accumulator.bytes_reserved(),
                duration_us = tracing::field::Empty,
            );
            let _seal_timer = crate::tracing_conventions::StageTimer::new(&seal_span);
            let instrumented = seal_span.clone();
            async {
                self.stage_built_shard_flush(built)?;
                self.install_staged_flush(cx).await
            }
            .instrument(instrumented)
            .await?;
            if allow_automatic_publication {
                self.publish_bulk_cadence_if_due(cx).await?;
            }
        }
        Ok(())
    }

    fn plan_parallel_shard_flushes<'a>(
        &'a self,
        shards: &[usize],
    ) -> Result<Vec<ShardFlushPlan<'a>>, QuillIndexError> {
        let mut plans = Vec::new();
        plans
            .try_reserve_exact(shards.len())
            .map_err(|_| invalid_state("could not reserve parallel shard flush plans"))?;
        let mut reserved_segment_ids = BTreeSet::new();
        for (offset, &shard) in shards.iter().enumerate() {
            let state = &self.shards[shard];
            let lease_docid_base = state
                .current_lease_base
                .ok_or_else(|| invalid_state("nonempty accumulator has no Q1 lease"))?;
            let created_unix_s = self.created_unix_s()?;
            let segment_id = self.derive_segment_id_avoiding(
                shard,
                lease_docid_base,
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
            plans.push(ShardFlushPlan {
                shard,
                state,
                segment_id,
                lease_docid_base,
                created_unix_s,
                seal_seq,
            });
        }
        Ok(plans)
    }

    pub(super) fn build_planned_shard_flush(
        plan: &ShardFlushPlan<'_>,
    ) -> Result<BuiltShardFlush, QuillIndexError> {
        let state = plan.state;
        let seal_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::KEEPER_LIFECYCLE,
            phase = "seal",
            trigger = "parallel_batch",
            action = "build_shared_nothing",
            shard_id = plan.shard,
            segment_id = plan.segment_id,
            doc_count = state.accumulator.document_count(),
            token_count = state.accumulator.token_count(),
            result_count = tracing::field::Empty,
            output_bytes = tracing::field::Empty,
            arena_bytes_used_high_water = state.accumulator.bytes_used(),
            arena_bytes_reserved_high_water = state.accumulator.bytes_reserved(),
            duration_us = tracing::field::Empty,
        );
        let _seal_timer = crate::tracing_conventions::StageTimer::new(&seal_span);
        let _entered = seal_span.enter();
        let documents = state
            .identities
            .iter()
            .map(|identity| {
                FlushDocumentInput::new(
                    identity.doc_ord,
                    &identity.document_id,
                    identity.content_hash,
                )
            })
            .collect::<Vec<_>>();
        let encoded = flush_accumulator_with_mode(
            &state.accumulator,
            FlushSegmentInput {
                segment_id: plan.segment_id,
                lease_docid_base: plan.lease_docid_base,
                created_unix_s: plan.created_unix_s,
                engine_version: CURRENT_ENGINE_VERSION,
                documents: &documents,
            },
            FlushMode::Automatic,
        )?;
        let manifest_segment = manifest_segment(&encoded, plan.seal_seq);
        let next_seal_seq = plan
            .seal_seq
            .checked_add(1)
            .ok_or_else(|| invalid_state("seal sequence exhausted"))?;
        seal_span.record("result_count", u64::from(manifest_segment.doc_count));
        seal_span.record("output_bytes", encoded.file_len());
        Ok(BuiltShardFlush {
            shard: plan.shard,
            encoded,
            manifest_segment,
            next_seal_seq,
        })
    }

    fn stage_built_shard_flush(&mut self, built: BuiltShardFlush) -> Result<(), QuillIndexError> {
        if self.staged_flush.is_some() {
            return Err(invalid_state(
                "cannot stage a parallel shard build while another flush is retained",
            ));
        }
        let shard = built.shard;
        let document_count = u32::try_from(self.shards[shard].accumulator.document_count())
            .map_err(|_| invalid_state("segment document count does not fit u32"))?;
        let mut pending_field_stats = self.pending_field_stats.clone();
        for field in self.shards[shard].accumulator.fields() {
            let entry = pending_field_stats
                .entry(field.field_ord())
                .or_insert((0, 0));
            entry.0 = entry
                .0
                .checked_add(field.total_tokens())
                .ok_or_else(|| invalid_state("pending field token count overflow"))?;
            entry.1 = entry
                .1
                .checked_add(document_count)
                .ok_or_else(|| invalid_state("pending field document count overflow"))?;
        }
        self.pending_segments
            .try_reserve(1)
            .map_err(|_| invalid_state("could not reserve pending segment bookkeeping"))?;
        if matches!(&self.backend, IndexBackend::Memory(_)) {
            self.pending_owned_segments
                .try_reserve(1)
                .map_err(|_| invalid_state("could not reserve owned segment bookkeeping"))?;
        }
        self.staged_flush = Some(StagedFlush {
            shard,
            encoded: built.encoded,
            manifest_segment: built.manifest_segment,
            pending_field_stats,
            next_seal_seq: built.next_seal_seq,
        });
        Ok(())
    }

    pub(super) fn prepare_shard_flush(&mut self, shard: usize) -> Result<(), QuillIndexError> {
        if self.staged_flush.is_some() || self.shards[shard].accumulator.document_count() == 0 {
            return Ok(());
        }
        let lease_docid_base = self.shards[shard]
            .current_lease_base
            .ok_or_else(|| invalid_state("nonempty accumulator has no Q1 lease"))?;
        let created_unix_s = self.created_unix_s()?;
        let segment_id = self.derive_segment_id(shard, lease_docid_base, created_unix_s)?;
        let documents = self.shards[shard]
            .identities
            .iter()
            .map(|identity| {
                FlushDocumentInput::new(
                    identity.doc_ord,
                    &identity.document_id,
                    identity.content_hash,
                )
            })
            .collect::<Vec<_>>();
        let encoded = flush_accumulator_with_mode(
            &self.shards[shard].accumulator,
            FlushSegmentInput {
                segment_id,
                lease_docid_base,
                created_unix_s,
                engine_version: CURRENT_ENGINE_VERSION,
                documents: &documents,
            },
            FlushMode::Scalar,
        )?;
        let manifest_segment = manifest_segment(&encoded, self.next_seal_seq);
        let document_count = u32::try_from(self.shards[shard].accumulator.document_count())
            .map_err(|_| invalid_state("segment document count does not fit u32"))?;
        let mut pending_field_stats = self.pending_field_stats.clone();
        for field in self.shards[shard].accumulator.fields() {
            let entry = pending_field_stats
                .entry(field.field_ord())
                .or_insert((0, 0));
            entry.0 = entry
                .0
                .checked_add(field.total_tokens())
                .ok_or_else(|| invalid_state("pending field token count overflow"))?;
            entry.1 = entry
                .1
                .checked_add(document_count)
                .ok_or_else(|| invalid_state("pending field document count overflow"))?;
        }
        let next_seal_seq = self
            .next_seal_seq
            .checked_add(1)
            .ok_or_else(|| invalid_state("seal sequence exhausted"))?;
        self.pending_segments
            .try_reserve(1)
            .map_err(|_| invalid_state("could not reserve pending segment bookkeeping"))?;
        if matches!(&self.backend, IndexBackend::Memory(_)) {
            self.pending_owned_segments
                .try_reserve(1)
                .map_err(|_| invalid_state("could not reserve owned segment bookkeeping"))?;
        }
        self.staged_flush = Some(StagedFlush {
            shard,
            encoded,
            manifest_segment,
            pending_field_stats,
            next_seal_seq,
        });
        Ok(())
    }

    async fn install_staged_flush(&mut self, cx: &Cx) -> Result<(), QuillIndexError> {
        let Some(staged) = self.staged_flush.as_ref() else {
            return Ok(());
        };
        if let IndexBackend::Durable(writer) = &mut self.backend {
            let directory = writer
                .retained_snapshot_for_bookkeeping()
                .directory()
                .expect("KeeperWriter always owns a durable directory");
            let pending = staged.encoded.write_temp_retryable(directory)?;
            writer.publish_segment(cx, pending).await?;
        }
        let StagedFlush {
            shard,
            encoded,
            manifest_segment,
            pending_field_stats,
            next_seal_seq,
        } = self
            .staged_flush
            .take()
            .expect("staged flush remains present until installation succeeds");
        if matches!(&self.backend, IndexBackend::Memory(_)) {
            self.pending_owned_segments.push(encoded);
        }
        self.pending_segments.push(manifest_segment);
        self.pending_field_stats = pending_field_stats;
        self.next_seal_seq = next_seal_seq;
        self.shards[shard].accumulator.reset();
        self.shards[shard].identities.clear();
        self.shards[shard].current_lease_base = None;
        Ok(())
    }

    fn derive_segment_id(
        &self,
        shard: usize,
        lease_base: u64,
        created_unix_s: i64,
    ) -> Result<u64, QuillIndexError> {
        self.derive_segment_id_avoiding(shard, lease_base, created_unix_s, &BTreeSet::new())
    }

    fn derive_segment_id_avoiding(
        &self,
        shard: usize,
        lease_base: u64,
        created_unix_s: i64,
        reserved_segment_ids: &BTreeSet<u64>,
    ) -> Result<u64, QuillIndexError> {
        self.derive_segment_id_for_state_avoiding(
            &self.shards[shard],
            lease_base,
            created_unix_s,
            reserved_segment_ids,
        )
    }

    pub(super) fn derive_segment_id_for_state_avoiding(
        &self,
        state: &ScribeShardState,
        lease_base: u64,
        created_unix_s: i64,
        reserved_segment_ids: &BTreeSet<u64>,
    ) -> Result<u64, QuillIndexError> {
        let authority = self.authority_snapshot()?;
        let generation = authority
            .loaded_manifest()
            .manifest
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid_state("manifest generation exhausted"))?;
        let schema_id = self.schema.schema_id()?;
        let mut batch_hasher = Xxh3::new();
        for identity in &state.identities {
            batch_hasher.update(&identity.content_hash.to_le_bytes());
        }
        let batch_digest = batch_hasher.digest();
        for salt in 0_u64..=u64::from(u16::MAX) {
            let mut preimage = [0_u8; 52];
            preimage[..8].copy_from_slice(&schema_id.to_le_bytes());
            preimage[8..16].copy_from_slice(&generation.to_le_bytes());
            preimage[16..24].copy_from_slice(&lease_base.to_le_bytes());
            preimage[24..32].copy_from_slice(&created_unix_s.to_le_bytes());
            preimage[32..36].copy_from_slice(&CURRENT_ENGINE_VERSION.to_le_bytes());
            preimage[36..44].copy_from_slice(&batch_digest.to_le_bytes());
            preimage[44..].copy_from_slice(&salt.to_le_bytes());
            let candidate = xxh3_64(&preimage);
            let collision = reserved_segment_ids.contains(&candidate)
                || authority
                    .loaded_manifest()
                    .manifest
                    .segments
                    .iter()
                    .chain(&self.pending_segments)
                    .any(|segment| segment.segment_id == candidate);
            if !collision {
                return Ok(candidate);
            }
        }
        Err(invalid_state(
            "could not derive a collision-free segment id",
        ))
    }

    pub(super) fn derive_policy_segment_id(
        &self,
        source_segment_ids: &[u64],
        domain: &[u8],
    ) -> Result<u64, QuillIndexError> {
        let authority = self.authority_snapshot()?;
        let generation = authority
            .loaded_manifest()
            .manifest
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid_state("manifest generation exhausted"))?;
        let schema_id = self.schema.schema_id()?;
        let mut source_hasher = Xxh3::new();
        source_hasher.update(domain);
        for source in source_segment_ids {
            source_hasher.update(&source.to_le_bytes());
        }
        let source_digest = source_hasher.digest();
        for salt in 0_u64..=u64::from(u16::MAX) {
            let mut preimage = [0_u8; 40];
            preimage[..8].copy_from_slice(&schema_id.to_le_bytes());
            preimage[8..16].copy_from_slice(&generation.to_le_bytes());
            preimage[16..24].copy_from_slice(&source_digest.to_le_bytes());
            preimage[24..32].copy_from_slice(&self.next_seal_seq.to_le_bytes());
            preimage[32..].copy_from_slice(&salt.to_le_bytes());
            let candidate = xxh3_64(&preimage);
            if authority
                .loaded_manifest()
                .manifest
                .segments
                .iter()
                .all(|segment| segment.segment_id != candidate)
            {
                return Ok(candidate);
            }
        }
        Err(invalid_state(
            "could not derive a collision-free policy segment id",
        ))
    }

    pub(super) fn created_unix_s(&self) -> Result<i64, QuillIndexError> {
        if self.config.deterministic_ingest {
            return Ok(0);
        }
        wall_clock_unix_s()
    }
}
