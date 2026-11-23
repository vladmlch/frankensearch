use asupersync::Cx;
use frankensearch_core::error::{SearchError, SearchResult};
use tantivy::Index;
use tantivy::schema::Value as _;
use tracing::instrument;

use crate::execution::{
    execute_query_with_offset, execute_top_k, load_doc, render_snippet_html,
    try_build_snippet_generator,
};
use crate::index::{TantivyIndex, WRITER_HEAP_BYTES, WriterPlan};
use crate::schema::build_schema;
use crate::types::{
    LexicalDocHit, OraclePageHit, OraclePageObservation, OracleQueryObservation, OracleRankedHit,
    OracleSegmentLayout, SnippetConfig, oracle_searcher_sha256,
};

impl TantivyIndex {
    /// Create an in-memory oracle with one deterministic indexing worker.
    ///
    /// This is reserved for differential campaigns that compare Tantivy's
    /// native `DocAddress` tie order across repeated runs. Shipping callers
    /// should use [`Self::in_memory`], which retains Tantivy's default writer
    /// parallelism.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::SubsystemError` if the index writer cannot be
    /// created.
    #[doc(hidden)]
    pub fn in_memory_single_threaded_oracle() -> SearchResult<Self> {
        let (schema, fields) = build_schema();
        let index = Index::create_in_ram(schema.clone());
        Self::from_index_with_writer_threads(
            index,
            schema,
            fields,
            None,
            WRITER_HEAP_BYTES,
            // An ordinary single-threaded oracle, not a benchmark candidate:
            // it really does pin width 1, and claims no screening receipt.
            WriterPlan::PinnedWidth(1),
            #[cfg(feature = "bench-internals")]
            None,
        )
    }

    /// Disable automatic merging for an exact oracle segment-topology proof.
    ///
    /// This method is intentionally separate from the benchmark-only helper:
    /// conformance fixtures use Tantivy's shipping indexing path but must keep
    /// explicit commit boundaries observable for native `DocAddress` ties.
    ///
    /// # Errors
    ///
    /// Returns a typed cancellation or writer-lock error.
    #[doc(hidden)]
    pub async fn oracle_disable_auto_merge(&self, cx: &Cx) -> SearchResult<()> {
        let writer_guard = self
            .writer
            .lock(cx)
            .await
            .map_err(|error| Self::map_writer_lock_error("tantivy.oracle_no_merge", error))?;
        let Some(writer) = writer_guard.as_ref() else {
            return Err(Self::read_only_writer_error("oracle_no_merge"));
        };
        writer.set_merge_policy(Box::new(tantivy::merge_policy::NoMergePolicy));
        Ok(())
    }

    /// Return searchable oracle segments in native `segment_ord` order.
    ///
    /// # Errors
    ///
    /// Returns an invalid-config error if a segment ordinal cannot fit
    /// Tantivy's public address type.
    #[doc(hidden)]
    pub fn oracle_segment_layout(&self) -> SearchResult<Vec<OracleSegmentLayout>> {
        self.reader
            .searcher()
            .segment_readers()
            .iter()
            .enumerate()
            .map(|(segment_ord, segment)| {
                let segment_ord =
                    u32::try_from(segment_ord).map_err(|_| SearchError::InvalidConfig {
                        field: "tantivy.segment_ord".to_owned(),
                        value: segment_ord.to_string(),
                        reason: "segment ordinal must fit in u32".to_owned(),
                    })?;
                Ok(OracleSegmentLayout {
                    segment_ord,
                    max_doc: segment.max_doc(),
                    num_docs: segment.num_docs(),
                })
            })
            .collect()
    }

    /// Observe the shipping parser/search/snippet path for Quill conformance.
    ///
    /// This surface is deliberately feature-gated: it retains the full Tantivy
    /// `DocAddress`, exact count, score bits, and an expanded cutoff tie group.
    /// Shipping consumers should use the ordinary lexical APIs instead.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if query execution or stored-field loading fails.
    #[instrument(skip_all, fields(query = %query, limit = limit, tie_expansion_limit = tie_expansion_limit))]
    pub fn oracle_observe_query(
        &self,
        _cx: &Cx,
        query: &str,
        limit: usize,
        tie_expansion_limit: usize,
        snippet_config: &SnippetConfig,
    ) -> SearchResult<OracleQueryObservation> {
        let query = Self::truncate_query(query);
        let searcher = self.reader.searcher();
        let doc_count = Self::checked_searcher_doc_count(&searcher)?;
        if query.trim().is_empty() {
            return Ok(OracleQueryObservation {
                hits: Vec::new(),
                cutoff_tie_group: Vec::new(),
                cutoff_tie_complete: true,
                total_count: 0,
                doc_count,
                expanded_hits: Vec::new(),
                searcher_sha256: oracle_searcher_sha256(&searcher),
            });
        }

        let parsed = self.parse_query_lenient(query);
        let fetch_limit = if limit == 0 {
            0
        } else {
            limit.saturating_add(tie_expansion_limit)
        };
        // Preserve the exact shipping TopDocs-only path for both `hits` and
        // the expanded tie evidence. Tantivy may choose a different score
        // accumulation or cutoff strategy when `TopDocs` is paired with
        // `Count`, and a larger collector limit may select a different member
        // of an exact-score cutoff tie. Count therefore runs independently,
        // and the expanded query may describe the tie envelope but must never
        // redefine the incumbent result whose latency and rank we compare.
        let native_hits = execute_top_k(&searcher, &*parsed, limit, 0)?;
        let expanded_hits = execute_top_k(&searcher, &*parsed, fetch_limit, 0)?;
        let total_count = execute_query_with_offset(&searcher, &*parsed, 0, 0)?.total_count;
        let snippet_gen =
            try_build_snippet_generator(&searcher, &*parsed, self.fields.content, snippet_config);
        let materialize = |hit: LexicalDocHit| -> SearchResult<OracleRankedHit> {
            let doc = load_doc(&searcher, hit.doc_address)?;
            let doc_id = doc
                .get_first(self.fields.id)
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_owned();
            let snippet = snippet_gen.as_ref().and_then(|generator| {
                render_snippet_html(
                    generator,
                    &doc,
                    &snippet_config.highlight_prefix,
                    &snippet_config.highlight_postfix,
                )
            });
            Ok(OracleRankedHit {
                doc_id,
                score_bits: hit.bm25_score.to_bits(),
                rank: hit.rank,
                segment_ord: hit.doc_address.segment_ord,
                segment_doc_id: hit.doc_address.doc_id,
                snippet,
            })
        };
        let hits = native_hits
            .into_iter()
            .map(&materialize)
            .collect::<SearchResult<Vec<_>>>()?;
        let expanded_hits = expanded_hits
            .into_iter()
            .map(materialize)
            .collect::<SearchResult<Vec<_>>>()?;

        let cutoff_bits = hits.last().map(|hit| hit.score_bits);
        let cutoff_tie_group = cutoff_bits.map_or_else(Vec::new, |cutoff| {
            expanded_hits
                .iter()
                .filter(|hit| {
                    f32::from_bits(hit.score_bits)
                        .total_cmp(&f32::from_bits(cutoff))
                        .is_eq()
                })
                .cloned()
                .collect()
        });
        let cutoff_tie_complete = cutoff_bits.is_none_or(|cutoff| {
            total_count <= fetch_limit
                || expanded_hits.last().is_none_or(|last| {
                    !f32::from_bits(last.score_bits)
                        .total_cmp(&f32::from_bits(cutoff))
                        .is_eq()
                })
        });

        Ok(OracleQueryObservation {
            hits,
            cutoff_tie_group,
            cutoff_tie_complete,
            total_count,
            doc_count,
            expanded_hits,
            searcher_sha256: oracle_searcher_sha256(&searcher),
        })
    }

    /// Observe Tantivy's real exact-count offset-pagination path.
    ///
    /// This method exists only for the replacement conformance harness. It
    /// intentionally uses the same private lenient parser, reader snapshot,
    /// stored-ID materialization, and [`execute_query_with_offset`] collector
    /// as the incumbent engine. The exact count is returned by Tantivy's
    /// [`crate::Count`] collector and is never inferred from the page length.
    ///
    /// # Errors
    ///
    /// Returns a typed search failure when query execution, rank arithmetic,
    /// or stored-document materialization fails.
    #[instrument(skip_all, fields(query = %query, limit = limit, offset = offset))]
    pub fn oracle_observe_page(
        &self,
        _cx: &Cx,
        query: &str,
        limit: usize,
        offset: usize,
    ) -> SearchResult<OraclePageObservation> {
        let query = Self::truncate_query(query);
        let searcher = self.reader.searcher();
        let doc_count = Self::checked_searcher_doc_count(&searcher)?;
        if query.trim().is_empty() {
            return Ok(OraclePageObservation {
                hits: Vec::new(),
                total_count: 0,
                doc_count,
            });
        }

        let parsed = self.parse_query_lenient(query);
        let search_result = execute_query_with_offset(&searcher, &*parsed, limit, offset)?;
        let mut hits = Vec::with_capacity(search_result.hits.len());
        for hit in search_result.hits {
            let doc = load_doc(&searcher, hit.doc_address)?;
            let doc_id = doc
                .get_first(self.fields.id)
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_owned();
            let absolute_rank =
                offset
                    .checked_add(hit.rank)
                    .ok_or_else(|| SearchError::InvalidConfig {
                        field: "tantivy.oracle_page_rank".to_owned(),
                        value: format!("{offset}+{}", hit.rank),
                        reason: "absolute result rank must fit usize".to_owned(),
                    })?;
            hits.push(OraclePageHit {
                doc_id,
                score_bits: hit.bm25_score.to_bits(),
                page_rank: hit.rank,
                absolute_rank,
            });
        }

        Ok(OraclePageObservation {
            hits,
            total_count: search_result.total_count,
            doc_count,
        })
    }
}
