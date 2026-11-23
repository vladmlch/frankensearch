use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

use frankensearch_core::config::ZeroSignalState;
use frankensearch_core::{SearchError, SearchResult};
use half::f16;
use tracing::debug;

use crate::wal::{self, CompactionStats};

#[cfg(unix)]
use super::codec::sync_parent_directory;
use super::codec::{
    align_up, build_header_prefix, build_v2_header, crc32, fnv1a_hash, fsvi_v2_config_error,
    fsvi_v2_header_len, is_tombstoned_flags, next_generation, ordered_docset_digest,
    remove_durability_sidecar, temporary_output_path, vector_signal_usable, write_vector_slab,
    write_vector_slab_v2,
};
use super::types::{
    FsviRecordFlags, FsviV2IdentityBinding, Quantization, RECORD_FLAG_TOMBSTONE, RECORD_SIZE_BYTES,
    RecordEntry, TOMBSTONE_VACUUM_THRESHOLD, VECTOR_ALIGN_BYTES, VacuumStats, VectorIndex,
};

impl VectorIndex {
    /// Tombstone (soft-delete) a document by `doc_id`.
    ///
    /// Returns `Ok(true)` when a live record was marked deleted, and `Ok(false)`
    /// when the document does not exist or is already tombstoned.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Io` for filesystem write/sync failures and
    /// `SearchError::IndexCorrupted` if the on-disk record table is malformed.
    pub fn soft_delete(&mut self, doc_id: &str) -> SearchResult<bool> {
        self.soft_delete_batch(&[doc_id]).map(|count| count > 0)
    }

    /// Tombstone a batch of document ids.
    ///
    /// Returns the number of records that transitioned from live -> deleted.
    ///
    /// # Errors
    ///
    /// Returns the first IO/corruption error encountered while updating flags.
    pub fn soft_delete_batch(&mut self, doc_ids: &[&str]) -> SearchResult<usize> {
        self.ensure_legacy_mutation_format("soft_delete_batch")?;
        let mut deleted = 0usize;
        let mut wal_changed = false;

        // Track modified main index entries for potential rollback
        let mut modified_main_entries = Vec::new();

        // Use a fast lookup for WAL entries to delete
        let mut to_delete_set = std::collections::HashSet::with_capacity(doc_ids.len());
        for &id in doc_ids {
            to_delete_set.insert(id);
        }

        // 1. Mark all matching records in the main index as tombstoned.
        for &doc_id in doc_ids {
            let doc_id_hash = fnv1a_hash(doc_id.as_bytes());
            if let Some(mut index) = self.find_first_hash_match(doc_id_hash)? {
                while index > 0 {
                    let prev = self.record_at(index - 1)?;
                    if prev.doc_id_hash != doc_id_hash {
                        break;
                    }
                    index -= 1;
                }

                for candidate in index..self.record_count() {
                    let entry = self.record_at(candidate)?;
                    if entry.doc_id_hash != doc_id_hash {
                        break;
                    }
                    if !is_tombstoned_flags(entry.flags) {
                        let candidate_doc_id = self.doc_id_at(candidate)?;
                        if candidate_doc_id == doc_id {
                            let flags = entry.flags | RECORD_FLAG_TOMBSTONE;
                            self.set_record_flags(candidate, flags)?;
                            modified_main_entries.push((candidate, entry.flags));
                            deleted += 1;
                        }
                    }
                }
            }
        }

        // 2. Remove all matching records from WAL entries.
        let original_wal_len = self.wal_entries.len();
        let filtered: Vec<wal::WalEntry> = self
            .wal_entries
            .iter()
            .filter(|entry| !to_delete_set.contains(entry.doc_id.as_str()))
            .cloned()
            .collect();

        let prev_wal = if filtered.len() < original_wal_len {
            deleted += original_wal_len - filtered.len();
            wal_changed = true;
            std::mem::replace(&mut self.wal_entries, filtered)
        } else {
            Vec::new()
        };

        // 3. Rewrite WAL sidecar once if anything was removed.
        if wal_changed {
            if let Err(err) = self.rewrite_wal_sidecar() {
                self.wal_entries = prev_wal;
                // Rollback main index modifications
                for (candidate, original_flags) in modified_main_entries {
                    if let Err(rollback_err) = self.set_record_flags(candidate, original_flags) {
                        tracing::error!(
                            error = %rollback_err,
                            candidate,
                            "failed to rollback main index flag during soft_delete_batch failure"
                        );
                    }
                }
                tracing::error!(
                    error = %err,
                    "failed to rewrite WAL sidecar during batch delete"
                );
                return Err(err);
            }
        }

        if !modified_main_entries.is_empty() {
            self.touch_main_file()?;
            remove_durability_sidecar(&self.path);
        }

        Ok(deleted)
    }

    /// Whether the record at `record_index` is tombstoned.
    #[must_use]
    pub fn is_deleted(&self, record_index: usize) -> bool {
        matches!(
            self.record_at(record_index),
            Ok(entry) if is_tombstoned_flags(entry.flags)
        )
    }

    /// Number of tombstoned records in the main index.
    #[must_use]
    pub fn tombstone_count(&self) -> usize {
        (0..self.record_count())
            .filter(|&index| self.is_deleted(index))
            .count()
    }

    /// Number of live (non-tombstoned) records in the main index.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.record_count().saturating_sub(self.tombstone_count())
    }

    /// Whether the stored vector at `record_index` is usable for similarity
    /// ranking: every component finite and the norm non-zero.
    ///
    /// Undecodable records are reported as unusable rather than erroring —
    /// a vector that cannot be read cannot contribute signal.
    #[must_use]
    pub fn is_vector_usable(&self, record_index: usize) -> bool {
        self.vector_at_f32(record_index)
            .is_ok_and(|vector| vector_signal_usable(&vector))
    }

    /// Compute the zero-signal census for this index generation.
    ///
    /// O(n·dim): decodes every live vector once. Callers classify empty
    /// search results with it lazily — an empty result costs one extra pass
    /// comparable to the scan that just ran, and the hot (non-empty) path
    /// pays nothing.
    #[must_use]
    pub fn zero_signal_state(&self) -> ZeroSignalState {
        let record_count = self.record_count();
        let mut tombstone_count = 0usize;
        let mut usable_vector_count = 0usize;
        for index in 0..record_count {
            if self.is_deleted(index) {
                tombstone_count += 1;
            } else if self.is_vector_usable(index) {
                usable_vector_count += 1;
            }
        }
        ZeroSignalState {
            record_count,
            live_count: record_count - tombstone_count,
            tombstone_count,
            wal_count: self.wal_record_count(),
            usable_vector_count,
        }
    }

    /// Fraction of records that are tombstoned (`tombstones / record_count`).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn tombstone_ratio(&self) -> f64 {
        if self.record_count() == 0 {
            return 0.0;
        }
        self.tombstone_count() as f64 / self.record_count() as f64
    }

    /// Whether the tombstone ratio exceeds the default vacuum threshold.
    #[must_use]
    pub fn needs_vacuum(&self) -> bool {
        self.tombstone_ratio() > TOMBSTONE_VACUUM_THRESHOLD
    }

    /// Rewrite the main index file without tombstoned records.
    ///
    /// WAL entries are preserved and reloaded after the rewrite.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Io` for filesystem failures and
    /// `SearchError::IndexCorrupted` for malformed data.
    pub fn vacuum(&mut self) -> SearchResult<VacuumStats> {
        self.ensure_legacy_mutation_format("vacuum")?;
        let start = Instant::now();
        let records_before = self.record_count();
        let bytes_before = self.data.len();
        let tombstones_before = self.tombstone_count();

        if records_before == 0 || tombstones_before == 0 {
            return Ok(VacuumStats {
                records_before,
                records_after: records_before,
                tombstones_removed: 0,
                bytes_reclaimed: 0,
                duration: start.elapsed(),
            });
        }

        // Collect live entries from main index.
        let mut sources = Vec::with_capacity(records_before - tombstones_before);
        for index in 0..records_before {
            if !self.is_deleted(index) {
                sources.push(MergeSource::Main(index));
            }
        }

        self.rewrite_index(&sources, self.metadata.compaction_gen)?;

        let records_after = self.record_count();
        let bytes_reclaimed = bytes_before.saturating_sub(self.data.len());
        Ok(VacuumStats {
            records_before,
            records_after,
            tombstones_removed: records_before.saturating_sub(records_after),
            bytes_reclaimed,
            duration: start.elapsed(),
        })
    }

    /// Append a single vector to the index via the WAL.
    ///
    /// The vector is immediately searchable. It is written to the WAL
    /// sidecar file for crash safety.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` for wrong embedding lengths
    /// and `SearchError::Io` for filesystem failures.
    pub fn append(&mut self, doc_id: &str, vector: &[f32]) -> SearchResult<()> {
        self.append_batch(&[(doc_id.to_owned(), vector.to_vec())])
    }

    /// Append a batch of vectors to the index via the WAL.
    ///
    /// All vectors in the batch are written atomically to a single WAL
    /// batch (one CRC covers the whole batch).
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` for wrong embedding lengths,
    /// `SearchError::InvalidConfig` for invalid values, and
    /// `SearchError::Io` for filesystem failures.
    pub fn append_batch(&mut self, entries: &[(String, Vec<f32>)]) -> SearchResult<()> {
        self.append_batch_impl(entries)
    }

    /// Bench-only alias retained for the wal_append_dedup_ab harness.
    ///
    /// The "skip redundant dedup" candidate arm no longer exists: since the
    /// log-then-supersede reordering, the resident-WAL dedup is load-bearing
    /// (it feeds the sidecar compaction) and can never be skipped, so both
    /// arms measure the same code path.
    ///
    /// # Errors
    ///
    /// Returns the same validation and I/O errors as [`Self::append_batch`].
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn bench_append_batch_skip_redundant_dedup(
        &mut self,
        entries: &[(String, Vec<f32>)],
    ) -> SearchResult<()> {
        self.append_batch_impl(entries)
    }

    fn append_batch_impl(&mut self, entries: &[(String, Vec<f32>)]) -> SearchResult<()> {
        self.ensure_legacy_mutation_format("append_batch")?;
        if entries.is_empty() {
            return Ok(());
        }

        // Validate all entries before writing anything.
        for (doc_id, vector) in entries {
            if vector.len() != self.dimension() {
                return Err(SearchError::DimensionMismatch {
                    expected: self.dimension(),
                    found: vector.len(),
                });
            }
            if vector.iter().any(|v| !v.is_finite()) {
                return Err(SearchError::InvalidConfig {
                    field: "embedding".to_owned(),
                    value: "<contains non-finite values>".to_owned(),
                    reason: "all embedding values must be finite".to_owned(),
                });
            }
            if !vector_signal_usable(vector) {
                return Err(SearchError::InvalidConfig {
                    field: "embedding".to_owned(),
                    value: "<zero-norm vector>".to_owned(),
                    reason: "embedding norm must be non-zero and finite; a zero vector can never match any query".to_owned(),
                });
            }
            let _ = u16::try_from(doc_id.len()).map_err(|_| SearchError::InvalidConfig {
                field: "doc_id".to_owned(),
                value: doc_id.clone(),
                reason: "doc_id byte length must fit in u16".to_owned(),
            })?;
        }

        let mut wal_entries: Vec<wal::WalEntry> = Vec::with_capacity(entries.len());
        let mut seen = std::collections::HashSet::new();
        for (doc_id, embedding) in entries.iter().rev() {
            if seen.insert(doc_id) {
                wal_entries.push(wal::WalEntry {
                    doc_id: doc_id.clone(),
                    doc_id_hash: fnv1a_hash(doc_id.as_bytes()),
                    embedding: embedding.clone(),
                });
            }
        }
        wal_entries.reverse();

        // DURABILITY ORDER IS LOAD-BEARING (fleet-review critical): the
        // replacement entries must be durably logged BEFORE anything
        // destroys the old copies. The previous ordering ran
        // `soft_delete_batch` first, which durably tombstoned the main-index
        // rows and durably rewrote the WAL sidecar without the doc — so an
        // append_wal_batch failure (ENOSPC is the canonical one) or a crash
        // in between destroyed the old vector with the new one never
        // written: an update executed as delete-then-log. Log-then-supersede
        // is safe at every cut point because `open` deduplicates duplicate
        // WAL doc IDs with last-wins semantics.
        let wal_path = wal::wal_path_for(&self.path);
        wal::append_wal_batch(
            &wal_path,
            &wal_entries,
            self.dimension(),
            self.quantization(),
            next_generation(self.metadata.compaction_gen),
            self.wal_config.fsync_on_write,
        )?;

        // Supersede older resident copies in memory, then admit the new
        // entries (immediately searchable). This dedup is load-bearing: it
        // produces the exact resident set the sidecar compaction below
        // persists.
        let resident_before = self.wal_entries.len();
        for new_entry in &wal_entries {
            self.wal_entries
                .retain(|existing| existing.doc_id != new_entry.doc_id);
        }
        let superseded_resident = self.wal_entries.len() < resident_before;
        self.wal_entries.extend(wal_entries.clone());

        // BEST-EFFORT: compact superseded duplicates out of the durable
        // sidecar (keeps repeated updates from growing it — the goal the
        // old delete-first ordering pursued unsafely). The atomic
        // tmp+rename rewrite includes the entries appended above, and a
        // failure or crash here merely leaves old+new pairs on disk for
        // `open`'s last-wins dedup — so it must never fail the append.
        if superseded_resident {
            if let Err(err) = self.rewrite_wal_sidecar() {
                tracing::warn!(
                    target: "frankensearch.index",
                    path = %self.path.display(),
                    error = %err,
                    "post-append WAL sidecar compaction failed; superseded \
                     duplicates remain until the next rewrite"
                );
            }
        }

        // BEST-EFFORT: Tombstone the old main index entries so they don't pollute the top-K heap.
        // If this crashes before completing, it's fine; they will be resolved out later (though they might steal a top-K slot temporarily).
        for entry in &wal_entries {
            let hash = entry.doc_id_hash;
            if let Ok(Some(mut index)) = self.find_first_hash_match(hash) {
                while index > 0 {
                    if let Ok(prev) = self.record_at(index - 1) {
                        if prev.doc_id_hash != hash {
                            break;
                        }
                        index -= 1;
                    } else {
                        break;
                    }
                }
                for candidate in index..self.record_count() {
                    if let Ok(rec) = self.record_at(candidate) {
                        if rec.doc_id_hash != hash {
                            break;
                        }
                        if !is_tombstoned_flags(rec.flags) {
                            if let Ok(candidate_doc_id) = self.doc_id_at(candidate) {
                                if candidate_doc_id == entry.doc_id {
                                    let flags = rec.flags | RECORD_FLAG_TOMBSTONE;
                                    if let Err(err) = self.set_record_flags(candidate, flags) {
                                        tracing::warn!(
                                            target: "frankensearch.index",
                                            path = %self.path.display(),
                                            candidate_index = candidate,
                                            doc_id = %entry.doc_id,
                                            error = %err,
                                            "WAL replay: failed to tombstone superseded record; \
                                             duplicate may persist until next compaction"
                                        );
                                    }
                                    break;
                                }
                            }
                        }
                    } else {
                        break;
                    }
                }
            }
        }

        debug!(
            target: "frankensearch.index",
            path = %self.path.display(),
            batch_size = entries.len(),
            wal_total = self.wal_entries.len(),
            "appended to WAL"
        );
        Ok(())
    }

    /// Compact the WAL into the main index.
    ///
    /// Rewrites the main index file with all main + WAL records merged,
    /// then removes the WAL sidecar. The index is atomically swapped
    /// (write to tmp, rename over original).
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Io` for filesystem failures and
    /// `SearchError::InvalidConfig` for encoding issues.
    #[allow(clippy::cast_precision_loss)]
    pub fn compact(&mut self) -> SearchResult<CompactionStats> {
        self.ensure_legacy_mutation_format("compact")?;
        let start = Instant::now();
        let main_before = self.record_count();
        let wal_count = self.wal_entries.len();

        if wal_count == 0 {
            return Ok(CompactionStats {
                main_records_before: main_before,
                wal_records: 0,
                total_records_after: main_before,
                elapsed_ms: 0.0,
            });
        }

        let deduped_sources = (|| -> SearchResult<Vec<MergeSource>> {
            #[derive(Clone, Copy)]
            struct SortKey<'a> {
                doc_id_hash: u64,
                doc_id: &'a str,
            }

            #[derive(Clone, Copy)]
            struct KeyedSource<'a> {
                key: SortKey<'a>,
                source: MergeSource,
            }

            // Collect all sources with their sort keys.
            let mut keyed_sources = Vec::with_capacity(main_before + wal_count);
            for i in 0..main_before {
                if !self.is_deleted(i) {
                    let entry = self.record_at(i)?;
                    let doc_id = self.doc_id_at(i)?;
                    keyed_sources.push(KeyedSource {
                        key: SortKey {
                            doc_id_hash: entry.doc_id_hash,
                            doc_id,
                        },
                        source: MergeSource::Main(i),
                    });
                }
            }
            for (idx, entry) in self.wal_entries.iter().enumerate() {
                keyed_sources.push(KeyedSource {
                    key: SortKey {
                        doc_id_hash: entry.doc_id_hash,
                        doc_id: &entry.doc_id,
                    },
                    source: MergeSource::Wal(idx),
                });
            }

            // Sort to ensure binary search property.
            keyed_sources.sort_by(|a, b| {
                a.key
                    .doc_id_hash
                    .cmp(&b.key.doc_id_hash)
                    .then(a.key.doc_id.cmp(b.key.doc_id))
            });

            // Deduplicate sources by doc_id, keeping the latest (WAL over Main).
            // Since `keyed_sources` is sorted, duplicates are adjacent and the stable sort
            // ensures that newer sources (WAL) appear after older sources (Main).
            let mut deduped: Vec<KeyedSource<'_>> = Vec::with_capacity(keyed_sources.len());
            for item in keyed_sources {
                if let Some(last) = deduped.last_mut() {
                    if item.key.doc_id_hash == last.key.doc_id_hash
                        && item.key.doc_id == last.key.doc_id
                    {
                        // Overwrite the older entry with the newer one
                        *last = item;
                        continue;
                    }
                }
                deduped.push(item);
            }

            Ok(deduped
                .into_iter()
                .map(|item| item.source)
                .collect::<Vec<_>>())
        })()?;

        // Perform the rewrite.
        self.rewrite_index(
            &deduped_sources,
            next_generation(self.metadata.compaction_gen),
        )?;

        // After rewrite_index succeeds, clear in-memory WAL state immediately
        // (the data is now in the main index). If remove_wal fails, the stale
        // WAL file on disk will be detected and discarded on next open() via
        // the generation counter.
        self.wal_entries.clear();

        // Then try to remove the WAL file (best-effort).
        let wal_path = wal::wal_path_for(&self.path);
        if let Err(e) = wal::remove_wal(&wal_path) {
            tracing::warn!("failed to remove WAL file after compaction: {e}");
        }

        let elapsed = start.elapsed();
        let stats = CompactionStats {
            main_records_before: main_before,
            wal_records: wal_count,
            total_records_after: self.record_count(),
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        };

        debug!(
            target: "frankensearch.index",
            path = %self.path.display(),
            main_before,
            wal_count,
            total_after = stats.total_records_after,
            elapsed_ms = format_args!("{:.1}", stats.elapsed_ms),
            "compaction complete"
        );
        Ok(stats)
    }

    fn resolve_sort_key<'a>(&'a self, source: &MergeSource) -> SearchResult<(u64, &'a str)> {
        match source {
            MergeSource::Main(idx) => {
                let entry = self.record_at(*idx)?;
                let id = self.doc_id_at(*idx)?;
                Ok((entry.doc_id_hash, id))
            }
            MergeSource::Wal(idx) => {
                let entry = &self.wal_entries[*idx];
                Ok((entry.doc_id_hash, &entry.doc_id))
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn rewrite_index(&mut self, sources: &[MergeSource], new_gen: u8) -> SearchResult<()> {
        self.ensure_legacy_mutation_format("rewrite_index")?;
        let record_count = sources.len();
        let records_bytes = record_count.checked_mul(RECORD_SIZE_BYTES).ok_or_else(|| {
            SearchError::InvalidConfig {
                field: "record_count".to_owned(),
                value: record_count.to_string(),
                reason: "record table size overflow".to_owned(),
            }
        })?;
        let records_bytes_u64 =
            u64::try_from(records_bytes).map_err(|_| SearchError::InvalidConfig {
                field: "record_count".to_owned(),
                value: record_count.to_string(),
                reason: "record table size does not fit in u64".to_owned(),
            })?;

        // Pass 1: Build Record Table and calculate layout.
        // We buffer the Record Table in memory (16 bytes * N).
        // 10M records = 160MB, which is acceptable.
        let mut record_table = Vec::with_capacity(records_bytes);
        let mut current_string_offset = 0u32;
        let mut string_table_len = 0u64;

        for source in sources {
            let (doc_id_hash, doc_id) = self.resolve_sort_key(source)?;
            let doc_id_len = doc_id.len();

            // Validation
            let len_u16 = u16::try_from(doc_id_len).map_err(|_| SearchError::InvalidConfig {
                field: "doc_id_len".to_owned(),
                value: doc_id_len.to_string(),
                reason: "doc_id length exceeds u16".to_owned(),
            })?;
            let len_u32 = u32::from(len_u16);
            let len_u64 = u64::from(len_u16);
            if current_string_offset.checked_add(len_u32).is_none() {
                return Err(SearchError::InvalidConfig {
                    field: "doc_id_offset".to_owned(),
                    value: "overflow".to_owned(),
                    reason: "string table offset exceeds u32".to_owned(),
                });
            }

            // Append to record table
            record_table.extend_from_slice(&doc_id_hash.to_le_bytes());
            record_table.extend_from_slice(&current_string_offset.to_le_bytes());
            record_table.extend_from_slice(&len_u16.to_le_bytes());
            record_table.extend_from_slice(&0u16.to_le_bytes()); // Flags cleared (tombstones gone)

            current_string_offset += len_u32;
            string_table_len += len_u64;
        }

        // Calculate layout
        let provisional_header = build_header_prefix(
            &self.metadata.embedder_id,
            &self.metadata.embedder_revision,
            self.dimension(),
            self.quantization(),
            new_gen,
            self.metadata.publication_nonce,
            record_count,
            0,
        )?;
        let header_len = provisional_header.len() + 4; // + CRC
        let header_len_u64 = u64::try_from(header_len).map_err(|_| SearchError::InvalidConfig {
            field: "header".to_owned(),
            value: header_len.to_string(),
            reason: "header length does not fit in u64".to_owned(),
        })?;

        let pre_vector = header_len_u64
            .checked_add(records_bytes_u64)
            .and_then(|v| v.checked_add(string_table_len))
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "layout".to_owned(),
                value: "overflow".to_owned(),
                reason: "layout offset overflow".to_owned(),
            })?;

        let vectors_offset = align_up(pre_vector, VECTOR_ALIGN_BYTES)?;
        let padding_len = usize::try_from(vectors_offset - pre_vector).map_err(|_| {
            SearchError::InvalidConfig {
                field: "padding_len".to_owned(),
                value: (vectors_offset - pre_vector).to_string(),
                reason: "padding length exceeds usize".to_owned(),
            }
        })?;

        // Open temp file
        let tmp_path = temporary_output_path(&self.path);

        // Helper: perform all I/O into tmp_path, rename atomically, and reload.
        // If anything fails after the temp file is created, we clean it up.
        let result = (|| -> SearchResult<()> {
            let mut file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&tmp_path)?;
            {
                let mut writer = BufWriter::with_capacity(256 * 1024, &mut file);

                // Pass 2: Write Header and Record Table
                let mut header_prefix = build_header_prefix(
                    &self.metadata.embedder_id,
                    &self.metadata.embedder_revision,
                    self.dimension(),
                    self.quantization(),
                    new_gen,
                    self.metadata.publication_nonce,
                    record_count,
                    vectors_offset,
                )?;
                let header_crc = crc32(&header_prefix);
                header_prefix.extend_from_slice(&header_crc.to_le_bytes());

                writer.write_all(&header_prefix)?;
                writer.write_all(&record_table)?;

                // Pass 3: Write String Table
                for source in sources {
                    let (_, doc_id) = self.resolve_sort_key(source)?;
                    writer.write_all(doc_id.as_bytes())?;
                }

                // Padding
                if padding_len > 0 {
                    writer.write_all(&vec![0u8; padding_len])?;
                }

                // Pass 4: Write Vectors
                match self.quantization() {
                    Quantization::F16 => {
                        for source in sources {
                            match source {
                                MergeSource::Main(idx) => {
                                    // Fast path: copy raw bytes
                                    let start = self.vector_start(*idx)?;
                                    let len = self.dimension() * 2;
                                    let bytes = &self.data[start..start + len];
                                    writer.write_all(bytes)?;
                                }
                                MergeSource::Wal(idx) => {
                                    // Slow path: encode
                                    let entry = &self.wal_entries[*idx];
                                    for &val in &entry.embedding {
                                        writer.write_all(&f16::from_f32(val).to_le_bytes())?;
                                    }
                                }
                            }
                        }
                    }
                    Quantization::F32 => {
                        for source in sources {
                            match source {
                                MergeSource::Main(idx) => {
                                    // Fast path: copy raw bytes
                                    let start = self.vector_start(*idx)?;
                                    let len = self.dimension() * 4;
                                    let bytes = &self.data[start..start + len];
                                    writer.write_all(bytes)?;
                                }
                                MergeSource::Wal(idx) => {
                                    // Slow path: encode
                                    let entry = &self.wal_entries[*idx];
                                    for &val in &entry.embedding {
                                        writer.write_all(&val.to_le_bytes())?;
                                    }
                                }
                            }
                        }
                    }
                }
                writer.flush()?;
            }

            file.sync_all()?;
            fs::rename(&tmp_path, &self.path)?;
            #[cfg(unix)]
            sync_parent_directory(&self.path)?;
            remove_durability_sidecar(&self.path);
            Ok(())
        })();

        if result.is_err() {
            // Clean up the temp file on error (best-effort).
            if tmp_path.exists() {
                if let Err(cleanup_err) = fs::remove_file(&tmp_path) {
                    tracing::warn!(
                        "failed to clean up temp file {} after rewrite error: {cleanup_err}",
                        tmp_path.display()
                    );
                }
            }
        }
        result?;

        // A generation bump means the merge sources absorbed the WAL sidecar
        // and the renamed main file is now the durable home of those entries.
        // Remove the sidecar BEFORE reloading: `open` stamps the sidecar
        // against the just-published generation, classifies it as stale (it
        // is merely superseded), and warns "discarding stale/mismatched WAL
        // entries" for data it has already merged. Vacuum keeps the
        // generation, so its still-live sidecar survives the reload.
        if new_gen != self.metadata.compaction_gen {
            let wal_path = wal::wal_path_for(&self.path);
            if let Err(error) = wal::remove_wal(&wal_path) {
                tracing::warn!(
                    path = %wal_path.display(),
                    "failed to remove the superseded WAL before reload: {error}"
                );
            }
        }

        // Reload
        let config = self.wal_config.clone();
        let reloaded = Self::open(&self.path)?;
        self.data = reloaded.data;
        // The new mapping must retain the new inode's lock. Replacing data
        // first destroys the old mapping before releasing its former lock.
        self.map_lock = reloaded.map_lock;
        self.metadata = reloaded.metadata;
        self.records_offset = reloaded.records_offset;
        self.strings_offset = reloaded.strings_offset;
        self.vectors_offset = reloaded.vectors_offset;
        // WAL entries are cleared by caller if compacting, or preserved if vacuuming
        // But vacuum preserves WAL on disk, so open() loads them.
        // Vacuum caller ignores the reloaded WAL entries? No, vacuum preserves them.
        // self.vacuum() impl:
        //   writer.finish()
        //   Self::open() -> loads WAL entries
        //   self.wal_entries = reloaded.wal_entries
        // So we need to update self.wal_entries from reloaded.
        self.wal_entries = reloaded.wal_entries;
        self.wal_config = config;
        // The lazy int8/4-bit slabs quantized the PRE-rewrite vector region;
        // carrying them across the swap scores pass-1 candidates from rows
        // that no longer exist (or reads past a shrunken slab). Reset so the
        // next two-pass search re-quantizes the new region (bd-0ho5p).
        self.vectors_i8 = OnceLock::new();
        self.vectors_nibbles = OnceLock::new();

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(super) struct PendingRecord {
    pub(super) doc_id: String,
    pub(super) doc_id_hash: u64,
    pub(super) flags: u16,
    pub(super) embedding: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
enum MergeSource {
    Main(usize),
    Wal(usize),
}

#[derive(Debug)]
pub struct VectorIndexWriter {
    pub(super) path: PathBuf,
    pub(super) embedder_id: String,
    pub(super) embedder_revision: String,
    pub(super) dimension: usize,
    pub(super) quantization: Quantization,
    pub(super) compaction_gen: u8,
    pub(super) publication_nonce: u16,
    pub(super) records: Vec<PendingRecord>,
    pub(super) identity_v2: Option<FsviV2IdentityBinding>,
}

impl VectorIndexWriter {
    /// Append a single `(doc_id, embedding)` record.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` for wrong embedding lengths
    /// and `SearchError::InvalidConfig` for invalid values.
    pub fn write_record(&mut self, doc_id: &str, embedding: &[f32]) -> SearchResult<()> {
        self.write_record_with_flags(doc_id, embedding, FsviRecordFlags::LIVE)
    }

    /// Append one retained tombstone row to an identity-complete FSVI v2
    /// generation.
    ///
    /// Tombstone rows remain part of the physical vector-content digest but do
    /// not participate in exact search or the ordered live-document-set digest.
    /// Legacy v1 writers reject this operation because their identity contract
    /// cannot attest the resulting membership semantics.
    ///
    /// # Errors
    ///
    /// Returns the same validation errors as [`Self::write_record`] or
    /// [`SearchError::InvalidConfig`] when called on a legacy writer.
    pub fn write_tombstone_record(&mut self, doc_id: &str, embedding: &[f32]) -> SearchResult<()> {
        if self.identity_v2.is_none() {
            return Err(SearchError::InvalidConfig {
                field: "fsvi_v2.record_flags".to_owned(),
                value: "tombstone".to_owned(),
                reason: "retained tombstones require an identity-complete FSVI v2 writer"
                    .to_owned(),
            });
        }
        self.write_record_with_flags(doc_id, embedding, FsviRecordFlags::TOMBSTONE)
    }

    fn write_record_with_flags(
        &mut self,
        doc_id: &str,
        embedding: &[f32],
        flags: FsviRecordFlags,
    ) -> SearchResult<()> {
        if embedding.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: embedding.len(),
            });
        }
        if embedding.iter().any(|value| !value.is_finite()) {
            return Err(SearchError::InvalidConfig {
                field: "embedding".to_owned(),
                value: "<contains non-finite values>".to_owned(),
                reason: "all embedding values must be finite".to_owned(),
            });
        }
        if !vector_signal_usable(embedding) {
            return Err(SearchError::InvalidConfig {
                field: "embedding".to_owned(),
                value: "<zero-norm vector>".to_owned(),
                reason: "embedding norm must be non-zero and finite; a zero vector can never match any query".to_owned(),
            });
        }
        let _ = u16::try_from(doc_id.len()).map_err(|_| SearchError::InvalidConfig {
            field: "doc_id".to_owned(),
            value: doc_id.to_owned(),
            reason: "doc_id byte length must fit in u16".to_owned(),
        })?;
        self.records.push(PendingRecord {
            doc_id: doc_id.into(),
            doc_id_hash: fnv1a_hash(doc_id.as_bytes()),
            flags: flags.bits(),
            embedding: embedding.to_vec(),
        });
        Ok(())
    }

    /// Bench-only access to the owned record handoff used by `TwoTierIndexBuilder`.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` for wrong embedding lengths
    /// and `SearchError::InvalidConfig` for invalid values.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn write_record_owned_for_benchmark(
        &mut self,
        doc_id: String,
        embedding: Vec<f32>,
    ) -> SearchResult<()> {
        if embedding.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: embedding.len(),
            });
        }
        if embedding.iter().any(|value| !value.is_finite()) {
            return Err(SearchError::InvalidConfig {
                field: "embedding".to_owned(),
                value: "<contains non-finite values>".to_owned(),
                reason: "all embedding values must be finite".to_owned(),
            });
        }
        if !vector_signal_usable(&embedding) {
            return Err(SearchError::InvalidConfig {
                field: "embedding".to_owned(),
                value: "<zero-norm vector>".to_owned(),
                reason: "embedding norm must be non-zero and finite; a zero vector can never match any query".to_owned(),
            });
        }
        let _ = u16::try_from(doc_id.len()).map_err(|_| SearchError::InvalidConfig {
            field: "doc_id".to_owned(),
            value: doc_id.clone(),
            reason: "doc_id byte length must fit in u16".to_owned(),
        })?;
        let doc_id_hash = fnv1a_hash(doc_id.as_bytes());
        self.records.push(PendingRecord {
            doc_id,
            doc_id_hash,
            flags: 0,
            embedding,
        });
        Ok(())
    }

    /// Override the compaction generation this writer stamps into the header.
    ///
    /// Required when the finished file will be handed to
    /// [`VectorIndex::install_replacement`] over an existing destination: pass
    /// `next_generation(VectorIndex::peek_compaction_gen(destination)?)` so
    /// the install's rename atomically invalidates any surviving destination
    /// WAL sidecar.
    #[must_use]
    pub const fn with_generation(mut self, generation: u8) -> Self {
        self.compaction_gen = generation;
        self
    }

    /// Stamp the shared publication identity of a multi-tier install
    /// (bd-miio8). Every tier of one publication must carry the same non-zero
    /// nonce so a crash between per-tier installs is detectable at open time.
    #[must_use]
    pub const fn with_publication_nonce(mut self, publication_nonce: u16) -> Self {
        self.publication_nonce = publication_nonce;
        self
    }

    /// Persist the index to disk, including fsync of file and parent directory.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` for layout/encoding failures and
    /// `SearchError::Io` for filesystem write/sync failures.
    #[allow(clippy::too_many_lines)]
    pub fn finish(mut self) -> SearchResult<()> {
        // STABLE sort is load-bearing here: `self.records` can contain duplicate
        // `doc_id`s (soft-delete then rewrite), so the comparator is NOT a strict
        // total order, and stable ordering preserves last-write-wins among dupes
        // (verified by `soft_delete_wal_restores_state_on_rewrite_failure`). Do not
        // switch to `sort_unstable_by`.
        self.records.sort_by(|left, right| {
            left.doc_id_hash
                .cmp(&right.doc_id_hash)
                .then(left.doc_id.cmp(&right.doc_id))
        });
        if self.identity_v2.is_some() {
            if self.records.iter().any(|record| record.doc_id.is_empty()) {
                return Err(SearchError::InvalidConfig {
                    field: "doc_id".to_owned(),
                    value: "<empty>".to_owned(),
                    reason: "identity-complete FSVI v2 document ids must not be empty".to_owned(),
                });
            }
            if self
                .records
                .windows(2)
                .any(|pair| pair[0].doc_id == pair[1].doc_id)
            {
                return Err(SearchError::InvalidConfig {
                    field: "doc_id".to_owned(),
                    value: "<duplicate>".to_owned(),
                    reason:
                        "identity-complete FSVI v2 requires one unique physical row per document id"
                            .to_owned(),
                });
            }
        }

        let record_count = self.records.len();
        let records_bytes = record_count.checked_mul(RECORD_SIZE_BYTES).ok_or_else(|| {
            SearchError::InvalidConfig {
                field: "record_count".to_owned(),
                value: record_count.to_string(),
                reason: "record table size overflow".to_owned(),
            }
        })?;
        let records_bytes_u64 =
            u64::try_from(records_bytes).map_err(|_| SearchError::InvalidConfig {
                field: "record_count".to_owned(),
                value: record_count.to_string(),
                reason: "record table size does not fit in u64".to_owned(),
            })?;

        let mut string_table = Vec::<u8>::new();
        let mut record_entries = Vec::<RecordEntry>::with_capacity(record_count);
        for record in &self.records {
            let offset_u32 =
                u32::try_from(string_table.len()).map_err(|_| SearchError::InvalidConfig {
                    field: "doc_id_offset".to_owned(),
                    value: string_table.len().to_string(),
                    reason: "string table offset exceeds u32".to_owned(),
                })?;
            let doc_id_bytes = record.doc_id.as_bytes();
            let len_u16 =
                u16::try_from(doc_id_bytes.len()).map_err(|_| SearchError::InvalidConfig {
                    field: "doc_id_len".to_owned(),
                    value: doc_id_bytes.len().to_string(),
                    reason: "doc_id length exceeds u16".to_owned(),
                })?;
            string_table.extend_from_slice(doc_id_bytes);
            record_entries.push(RecordEntry {
                doc_id_hash: record.doc_id_hash,
                doc_id_offset: offset_u32,
                doc_id_len: len_u16,
                flags: record.flags,
            });
        }

        let string_table_len_u64 =
            u64::try_from(string_table.len()).map_err(|_| SearchError::InvalidConfig {
                field: "string_table".to_owned(),
                value: string_table.len().to_string(),
                reason: "string table length does not fit in u64".to_owned(),
            })?;

        let header_len = if let Some(binding) = &self.identity_v2 {
            fsvi_v2_header_len(binding)?
        } else {
            build_header_prefix(
                &self.embedder_id,
                &self.embedder_revision,
                self.dimension,
                self.quantization,
                self.compaction_gen,
                self.publication_nonce,
                record_count,
                0,
            )?
            .len()
            .checked_add(4)
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "header".to_owned(),
                value: "overflow".to_owned(),
                reason: "header length overflow".to_owned(),
            })?
        };
        let header_len_u64 = u64::try_from(header_len).map_err(|_| SearchError::InvalidConfig {
            field: "header".to_owned(),
            value: header_len.to_string(),
            reason: "header length does not fit in u64".to_owned(),
        })?;
        let pre_vector = header_len_u64
            .checked_add(records_bytes_u64)
            .and_then(|value| value.checked_add(string_table_len_u64))
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "layout".to_owned(),
                value: format!("{header_len_u64}+{records_bytes_u64}+{string_table_len_u64}"),
                reason: "layout offset overflow".to_owned(),
            })?;
        let vectors_offset = align_up(pre_vector, VECTOR_ALIGN_BYTES)?;
        let padding_len_u64 =
            vectors_offset
                .checked_sub(pre_vector)
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "layout".to_owned(),
                    value: format!("{vectors_offset}-{pre_vector}"),
                    reason: "negative padding detected".to_owned(),
                })?;
        let padding_len =
            usize::try_from(padding_len_u64).map_err(|_| SearchError::InvalidConfig {
                field: "padding".to_owned(),
                value: padding_len_u64.to_string(),
                reason: "padding length does not fit in usize".to_owned(),
            })?;

        let legacy_header = if self.identity_v2.is_none() {
            let mut header = build_header_prefix(
                &self.embedder_id,
                &self.embedder_revision,
                self.dimension,
                self.quantization,
                self.compaction_gen,
                self.publication_nonce,
                record_count,
                vectors_offset,
            )?;
            let header_crc = crc32(&header);
            header.extend_from_slice(&header_crc.to_le_bytes());
            Some(header)
        } else {
            None
        };
        let ordered_docset_digest = self
            .identity_v2
            .as_ref()
            .map(|_| ordered_docset_digest(&self.records));

        let tmp_path = temporary_output_path(&self.path);
        let result = (|| -> SearchResult<()> {
            let mut file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&tmp_path)?;
            let vector_content_digest;
            {
                let mut writer = BufWriter::with_capacity(256 * 1024, &mut file);

                if let Some(header) = &legacy_header {
                    writer.write_all(header)?;
                } else {
                    writer.write_all(&vec![0_u8; header_len])?;
                }
                for entry in &record_entries {
                    writer.write_all(&entry.doc_id_hash.to_le_bytes())?;
                    writer.write_all(&entry.doc_id_offset.to_le_bytes())?;
                    writer.write_all(&entry.doc_id_len.to_le_bytes())?;
                    writer.write_all(&entry.flags.to_le_bytes())?;
                }
                writer.write_all(&string_table)?;
                if padding_len > 0 {
                    writer.write_all(&vec![0_u8; padding_len])?;
                }
                vector_content_digest = if self.identity_v2.is_some() {
                    Some(write_vector_slab_v2(
                        &mut writer,
                        &self.records,
                        self.dimension,
                        self.quantization,
                    )?)
                } else {
                    write_vector_slab(&mut writer, &self.records, self.quantization)?;
                    None
                };
                writer.flush()?;
            }

            if let Some(binding) = &self.identity_v2 {
                let docset_digest = ordered_docset_digest.ok_or_else(|| {
                    fsvi_v2_config_error(
                        "ordered_live_docset_digest",
                        "v2 writer did not compute its docset digest",
                    )
                })?;
                let vector_digest = vector_content_digest.ok_or_else(|| {
                    fsvi_v2_config_error(
                        "vector_content_digest",
                        "v2 writer did not compute its vector digest",
                    )
                })?;
                let header = build_v2_header(
                    binding,
                    self.publication_nonce,
                    record_count,
                    vectors_offset,
                    docset_digest,
                    vector_digest,
                )?;
                if header.len() != header_len {
                    return Err(SearchError::InvalidConfig {
                        field: "fsvi_v2.header_size".to_owned(),
                        value: header.len().to_string(),
                        reason: format!("expected precomputed header size {header_len}"),
                    });
                }
                file.seek(SeekFrom::Start(0))?;
                file.write_all(&header)?;
            }
            file.sync_all()?;
            // Publication over an existing WAL sidecar is refused for EVERY
            // format version, not just v2. A fresh v1 generation restarts at
            // compaction_gen 1, so next_generation(1) collides with the
            // previous generation's sidecar binding and open() would adopt the
            // stale WAL wholesale: deleted documents resurrect and the WAL
            // shadow suppresses the freshly rebuilt main-slab winners
            // (bd-cnby1). Removing the sidecar here instead is not crash-safe
            // in either order (before the rename it destroys acknowledged
            // writes of the still-live old generation; after, a crash between
            // rename and removal re-opens the adoption window), so the only
            // sound contract is refusal: in-place replacement goes through
            // install_replacement, which carries the destination generation
            // forward and supersedes the sidecar under a real authority.
            {
                let wal_path = wal::wal_path_for(&self.path);
                match fs::symlink_metadata(&wal_path) {
                    Ok(_) => {
                        let (field, reason) = if self.identity_v2.is_some() {
                            (
                                "fsvi_v2.wal_sidecar",
                                "identity-complete v2 publication refuses a target with any existing WAL sidecar",
                            )
                        } else {
                            (
                                "wal_sidecar",
                                "publication refuses a target with an existing WAL sidecar: a fresh generation \
                                 would collide with the sidecar's generation binding and silently adopt stale \
                                 entries (deleted documents resurrect); replace an existing index via \
                                 install_replacement, or remove the sidecar explicitly if discarding its \
                                 pending writes is intended",
                            )
                        };
                        return Err(SearchError::InvalidConfig {
                            field: field.to_owned(),
                            value: wal_path.display().to_string(),
                            reason: reason.to_owned(),
                        });
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(SearchError::Io(error)),
                }
            }
            fs::rename(&tmp_path, &self.path)?;
            #[cfg(unix)]
            sync_parent_directory(&self.path)?;
            remove_durability_sidecar(&self.path);
            Ok(())
        })();

        if result.is_err() {
            if tmp_path.exists() {
                if let Err(cleanup_err) = fs::remove_file(&tmp_path) {
                    tracing::warn!(
                        "failed to clean up temp file {} after write error: {cleanup_err}",
                        tmp_path.display()
                    );
                }
            }
        }
        result?;

        debug!(
            target: "frankensearch.index",
            path = %self.path.display(),
            record_count,
            dimension = self.dimension,
            quantization = self.quantization as u8,
            vectors_offset,
            "wrote fsvi index"
        );
        Ok(())
    }
}
