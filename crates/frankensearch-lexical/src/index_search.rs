use std::sync::atomic::Ordering;

use asupersync::Cx;
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::traits::{LexicalCandidateBatch, LexicalHydrationContext, SearchFuture};
use frankensearch_core::types::{DocId, IndexableDocument, ScoreSource, ScoredResult};
use tantivy::collector::{DocSetCollector, TopDocs};
use tantivy::query::{BooleanQuery, Occur, Query, QueryParser, TermQuery};
use tantivy::schema::{IndexRecordOption, Value};
use tantivy::{DocAddress, Searcher, TantivyDocument, Term};
use tracing::{debug, instrument};

use crate::execution::{
    execute_query_with_offset, execute_top_k, load_doc, render_snippet_html, search_guarded,
    try_build_snippet_generator,
};
use crate::index::TantivyIndex;
use crate::query_repair::{
    repair_and_not, repair_boosted_group_negation, repair_negated_conjunction,
};
use crate::types::{
    LexicalDocHit, LexicalHit, LexicalIdHit, QueryExplanation, SnippetConfig, classify_query,
};

/// BM25 boost applied to title field matches.
const TITLE_BOOST: f32 = 2.0;

impl TantivyIndex {
    /// Build a `QueryParser` for BM25 search with title boost.
    pub(super) fn query_parser(&self) -> QueryParser {
        let mut parser =
            QueryParser::for_index(&self.index, vec![self.fields.content, self.fields.title]);
        parser.set_field_boost(self.fields.title, TITLE_BOOST);
        parser
    }

    /// Parse for the SHIPPING search path: lenient, plus the bd-f20ye repair.
    ///
    /// THE TWO ROLES OF THIS CRATE DIVERGE HERE, AND ONLY HERE.
    /// `frankensearch-lexical` is both the shipping Tantivy backend and the
    /// Quill gauntlet's pinned conformance oracle. Every `oracle_observe_*`
    /// method keeps calling [`Self::parse_query_lenient`] and stays bit-faithful
    /// to Tantivy 0.26.1, because the oracle is a pinned COMPARATOR and moving
    /// it would silently move the target Quill is measured against. Every
    /// user-facing `search*` method calls THIS instead, because a boost that
    /// changes boolean MEMBERSHIP is a defect by any reading and shipping it to
    /// users is not defensible.
    ///
    /// The resulting divergence between the two roles is registered as
    /// **DIV-009** in `docs/contracts/quill-divergence-register.md`, disposition
    /// accepted-with-rationale. Do not "unify" these two paths without
    /// superseding that entry: the shipping path is deliberately NOT
    /// bit-faithful to the oracle for this one query shape.
    pub(super) fn parse_query_shipping(&self, query: &str) -> Box<dyn tantivy::query::Query> {
        // ORDER IS LOAD-BEARING. bd-8a2a8 runs first because it decodes whole
        // `AND` chains, and it needs to see the `AND` that bd-eeq0q deletes:
        // after that deletion `a AND NOT b AND c` reads as the unrelated chain
        // `NOT b AND c` and would be normalised to the wrong grouping. Both
        // repairs leave a `-` or `NOT` inside the group they touch, so the
        // bd-f20ye repair still recognises `(a AND NOT b)^2` as negating — it
        // records negation against the innermost open group, which is why
        // bd-8a2a8 emits a whole-level chain without adding parentheses.
        let conjunction = repair_negated_conjunction(query);
        let repaired = repair_and_not(&conjunction);
        self.parse_query_lenient(&repair_boosted_group_negation(&repaired))
    }

    /// Parse a query using lenient mode (never fails, returns best-effort query).
    ///
    /// Unknown field prefixes, unbalanced quotes, and other syntax issues are
    /// silently ignored rather than producing errors. This makes user-facing
    /// search robust against arbitrary input.
    ///
    /// BIT-FAITHFUL TO THE PINNED ORACLE. Callers in the `oracle_observe_*`
    /// family depend on this reproducing Tantivy 0.26.1 exactly, defects
    /// included — see [`Self::parse_query_shipping`] and DIV-009.
    pub(super) fn parse_query_lenient(&self, query: &str) -> Box<dyn tantivy::query::Query> {
        let parser = self.query_parser();
        let (parsed, errors) = parser.parse_query_lenient(query);
        if let Some(first_error) = errors.first() {
            debug!(
                error_count = errors.len(),
                first_error = %first_error,
                "lenient query parse produced warnings"
            );
        }
        parsed
    }

    /// Search with snippet generation and query explanation.
    ///
    /// Returns [`LexicalHit`] results enriched with highlighted snippets
    /// from the content field and a [`QueryExplanation`] indicating how
    /// the query was parsed.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if the query cannot be parsed or search fails.
    #[instrument(skip_all, fields(query = %query, limit = limit))]
    pub fn search_with_snippets(
        &self,
        _cx: &Cx,
        query: &str,
        limit: usize,
        snippet_config: &SnippetConfig,
    ) -> SearchResult<Vec<LexicalHit>> {
        let query = Self::truncate_query(query);
        let explanation = classify_query(query);

        if explanation == QueryExplanation::Empty {
            return Ok(Vec::new());
        }

        let parsed = self.parse_query_shipping(query);

        let searcher = self.reader.searcher();
        let search_result = execute_query_with_offset(&searcher, &*parsed, limit, 0)?;
        let snippet_gen =
            try_build_snippet_generator(&searcher, &*parsed, self.fields.content, snippet_config);

        debug!(
            hits = search_result.hits.len(),
            total_count = search_result.total_count,
            query_type = %explanation,
            "tantivy search_with_snippets completed"
        );

        let mut results = Vec::with_capacity(search_result.hits.len());
        for hit in search_result.hits {
            let doc = load_doc(&searcher, hit.doc_address)?;

            let doc_id = doc
                .get_first(self.fields.id)
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| {
                    debug!("tantivy document missing id field, using empty doc_id");
                    ""
                })
                .to_owned();

            let metadata = doc
                .get_first(self.fields.metadata_json)
                .and_then(|v| v.as_str())
                .and_then(|s| match serde_json::from_str(s) {
                    Ok(val) => Some(val),
                    Err(e) => {
                        debug!(doc_id = %doc_id, error = %e, "failed to deserialize metadata JSON");
                        None
                    }
                });

            // Generate snippet from the document.
            let snippet = snippet_gen.as_ref().and_then(|generator| {
                render_snippet_html(
                    generator,
                    &doc,
                    &snippet_config.highlight_prefix,
                    &snippet_config.highlight_postfix,
                )
            });

            results.push(LexicalHit {
                doc_id,
                bm25_score: hit.bm25_score,
                rank: hit.rank,
                snippet,
                query_type: explanation,
                metadata,
            });
        }

        Ok(results)
    }

    /// Search and return only `(doc_id, score, rank)` rows.
    ///
    /// This avoids metadata JSON decoding and is intended for latency-critical
    /// callers that only require identifiers plus BM25 scores.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if the query cannot be parsed or search fails.
    #[instrument(skip_all, fields(query = %query, limit = limit))]
    pub fn search_doc_ids(
        &self,
        _cx: &Cx,
        query: &str,
        limit: usize,
    ) -> SearchResult<Vec<LexicalIdHit>> {
        let query = Self::truncate_query(query);
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let parsed = self.parse_query_shipping(query);
        let searcher = self.reader.searcher();
        let hits = execute_top_k(&searcher, &*parsed, limit, 0)?;
        self.collect_id_hits(&searcher, hits)
    }

    /// Enumerate every live document identifier in the committed index.
    ///
    /// This is intended for generation reconciliation and repair paths that
    /// cannot trust an external manifest. Results are sorted and deduplicated
    /// so callers can stage deterministic deletes even when an older index
    /// contains duplicate IDs.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if a live stored document cannot be loaded or the
    /// segment ordinal cannot be represented by Tantivy's address type.
    pub fn all_doc_ids(&self) -> SearchResult<Vec<DocId>> {
        let searcher = self.reader.searcher();
        let mut ids = Vec::with_capacity(self.doc_count.load(Ordering::Relaxed));
        for (segment_ord, segment) in searcher.segment_readers().iter().enumerate() {
            let segment_ord =
                u32::try_from(segment_ord).map_err(|_| SearchError::InvalidConfig {
                    field: "tantivy.segment_ord".to_owned(),
                    value: segment_ord.to_string(),
                    reason: "segment ordinal must fit in u32".to_owned(),
                })?;
            for doc_id in 0..segment.max_doc() {
                if segment.is_deleted(doc_id) {
                    continue;
                }
                let doc = load_doc(&searcher, DocAddress::new(segment_ord, doc_id))?;
                if let Some(doc_id) = doc
                    .get_first(self.fields.id)
                    .and_then(|value| value.as_str())
                {
                    ids.push(DocId::from(doc_id));
                }
            }
        }
        ids.sort_unstable();
        ids.dedup();
        Ok(ids)
    }

    /// Export every live committed document for exact-generation shadow replay.
    ///
    /// Results are ordered by document id. Duplicate ids from legacy indexes
    /// collapse deterministically to the last live segment/document address.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if a stored live document cannot be loaded or its
    /// metadata payload is malformed.
    pub fn all_documents(&self) -> SearchResult<Vec<IndexableDocument>> {
        let searcher = self.reader.searcher();
        let mut documents = std::collections::BTreeMap::new();
        for (segment_ord, segment) in searcher.segment_readers().iter().enumerate() {
            let segment_ord =
                u32::try_from(segment_ord).map_err(|_| SearchError::InvalidConfig {
                    field: "tantivy.segment_ord".to_owned(),
                    value: segment_ord.to_string(),
                    reason: "segment ordinal must fit in u32".to_owned(),
                })?;
            for doc_id in 0..segment.max_doc() {
                if segment.is_deleted(doc_id) {
                    continue;
                }
                let doc = load_doc(&searcher, DocAddress::new(segment_ord, doc_id))?;
                let Some(id) = doc
                    .get_first(self.fields.id)
                    .and_then(|value| value.as_str())
                else {
                    continue;
                };
                let content = doc
                    .get_first(self.fields.content)
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_owned();
                let title = doc
                    .get_first(self.fields.title)
                    .and_then(|value| value.as_str())
                    .filter(|value| !value.is_empty())
                    .map(str::to_owned);
                let metadata = doc
                    .get_first(self.fields.metadata_json)
                    .and_then(|value| value.as_str())
                    .map_or_else(
                        || Ok(std::collections::HashMap::new()),
                        |json| {
                            serde_json::from_str(json).map_err(|error| {
                                SearchError::SubsystemError {
                                    subsystem: "tantivy.shadow_export",
                                    source: Box::new(error),
                                }
                            })
                        },
                    )?;
                documents.insert(
                    id.to_owned(),
                    IndexableDocument {
                        id: id.to_owned(),
                        content,
                        title,
                        metadata,
                    },
                );
            }
        }
        Ok(documents.into_values().collect())
    }

    /// Pre-optimization baseline for [`Self::search_doc_ids`] that retains the
    /// discarded total-count collector. This is used only by the `doc_ids_topk`
    /// benchmark so the optimized and counted paths can be compared in one
    /// binary.
    #[doc(hidden)]
    #[instrument(skip_all, fields(query = %query, limit = limit))]
    pub fn search_doc_ids_counted(
        &self,
        _cx: &Cx,
        query: &str,
        limit: usize,
    ) -> SearchResult<Vec<LexicalIdHit>> {
        let query = Self::truncate_query(query);
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let parsed = self.parse_query_shipping(query);
        let searcher = self.reader.searcher();
        let search_result = execute_query_with_offset(&searcher, &*parsed, limit, 0)?;
        self.collect_id_hits(&searcher, search_result.hits)
    }

    /// Pre-optimization baseline for [`Self::search_doc_ids`] that forces the
    /// docstore id-materialization path (ignoring the `ord` fast field + table).
    /// Used only by the `search_doc_ids_materialize` benchmark to A/B the fast
    /// materialization wiring in one binary; not for production use.
    #[doc(hidden)]
    #[instrument(skip_all, fields(query = %query, limit = limit))]
    pub fn search_doc_ids_via_docstore(
        &self,
        _cx: &Cx,
        query: &str,
        limit: usize,
    ) -> SearchResult<Vec<LexicalIdHit>> {
        let query = Self::truncate_query(query);
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let parsed = self.parse_query_shipping(query);
        let searcher = self.reader.searcher();
        let hits = execute_top_k(&searcher, &*parsed, limit, 0)?;
        let mut results = Vec::with_capacity(hits.len());
        for hit in hits {
            results.push(LexicalIdHit {
                doc_id: self.docstore_id(&searcher, hit.doc_address)?,
                bm25_score: hit.bm25_score,
                rank: hit.rank,
            });
        }
        Ok(results)
    }

    /// Read a single hit's `doc_id` from the stored document (the docstore
    /// fallback path: decompresses the stored block to read the `id` field).
    fn docstore_id(&self, searcher: &Searcher, addr: DocAddress) -> SearchResult<DocId> {
        let doc = load_doc(searcher, addr)?;
        Ok(DocId::from(
            doc.get_first(self.fields.id)
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| {
                    debug!("tantivy document missing id field, using empty doc_id");
                    ""
                }),
        ))
    }

    /// Materialize ranked Tantivy hits into [`LexicalIdHit`] rows.
    ///
    /// Fast path: read each hit's dense ordinal from the `ord` `u64` FAST column
    /// (a flat bit-packed read — no stored-document decompress, no dictionary
    /// seek) and resolve it through the in-memory `ord_table`. Falls back
    /// per-hit to [`Self::docstore_id`] for any ordinal the table cannot resolve
    /// (documents written before `ord` existed, a reopened-but-not-rebuilt
    /// table, or a poisoned lock), so results are identical either way.
    // The `ord_table` read guard is intentionally held across the whole hit
    // loop — it is the once-per-call snapshot every hit resolves against.
    #[allow(clippy::significant_drop_tightening)]
    fn collect_id_hits(
        &self,
        searcher: &Searcher,
        hits: Vec<LexicalDocHit>,
    ) -> SearchResult<Vec<LexicalIdHit>> {
        let mut results = Vec::with_capacity(hits.len());

        // Take the table snapshot once; `None` when the field is absent or the
        // table is empty → pure docstore path. `ord` columns are opened lazily
        // per *touched* segment (top-k hits cluster in a few segments, so we
        // avoid paying for segments no hit lands in).
        let table = self.fields.ord.and_then(|_| {
            let table = self.ord_table.read().ok()?;
            if table.is_empty() { None } else { Some(table) }
        });
        // Cache the opened `ord` column per segment in a flat Vec indexed by
        // `segment_ord` — O(1), no per-hit hash — while still opening each
        // column lazily on first touch (outer `None` = not yet opened) so
        // segments no hit lands in are never opened. `segment_ord` is a dense
        // in-range index for hits from this `searcher`. Only sized on the
        // fast-field path (`table` present); the docstore path never indexes it.
        let mut columns: Vec<Option<Option<tantivy::columnar::Column<u64>>>> = Vec::new();
        if table.is_some() {
            columns.resize_with(searcher.segment_readers().len(), || None);
        }

        for hit in hits {
            let addr = hit.doc_address;
            let doc_id = match table.as_ref().and_then(|table| {
                columns[addr.segment_ord as usize]
                    .get_or_insert_with(|| {
                        searcher
                            .segment_reader(addr.segment_ord)
                            .fast_fields()
                            .u64("ord")
                            .ok()
                    })
                    .as_ref()
                    .and_then(|c| c.first(addr.doc_id))
                    .and_then(|ord| usize::try_from(ord).ok())
                    .and_then(|ord| table.get(ord))
                    // Empty = a padded/stale slot (sidecar didn't cover this
                    // ordinal) → fall back to the docstore for the real id.
                    .filter(|id| !id.is_empty())
                    .cloned()
            }) {
                Some(id) => id,
                None => self.docstore_id(searcher, addr)?,
            };

            results.push(LexicalIdHit {
                doc_id,
                bm25_score: hit.bm25_score,
                rank: hit.rank,
            });
        }

        Ok(results)
    }

    /// Hydrate winners from an explicitly supplied searcher.
    ///
    /// A Tantivy `Searcher` holds a fixed segment set, so passing the *same*
    /// searcher that produced the scores is what makes hydration read the
    /// scoring generation rather than the newest one (`bd-8nqz.1`).
    pub(super) fn hydrate_with_searcher(
        &self,
        searcher: &Searcher,
        results: &mut [ScoredResult],
        unscored: bool,
    ) -> SearchResult<()> {
        let clauses: Vec<(Occur, Box<dyn Query>)> = results
            .iter()
            .filter(|result| result.lexical_score.is_some() && result.metadata.is_none())
            .map(|result| {
                let term = Term::from_field_text(self.fields.id, result.doc_id.as_str());
                (
                    Occur::Should,
                    Box::new(TermQuery::new(term, IndexRecordOption::Basic)) as Box<dyn Query>,
                )
            })
            .collect();
        if clauses.is_empty() {
            return Ok(());
        }

        let limit = clauses.len();
        let query = BooleanQuery::new(clauses);
        let doc_addresses: Vec<DocAddress> = if unscored {
            // Every clause is an exact lookup on the unique `id` field. Hydration
            // consumes neither BM25 scores nor hit order, so asking Tantivy to
            // score and heap-sort these documents is pure overhead.
            search_guarded(searcher, &query, &DocSetCollector)?
                .into_iter()
                .collect()
        } else {
            search_guarded(
                searcher,
                &query,
                &TopDocs::with_limit(limit).order_by_score(),
            )?
            .into_iter()
            .map(|(_, doc_address)| doc_address)
            .collect()
        };

        for doc_address in doc_addresses {
            let doc = load_doc(searcher, doc_address)?;
            let doc_id = doc
                .get_first(self.fields.id)
                .and_then(|value| value.as_str())
                .unwrap_or_else(|| {
                    debug!("tantivy document missing id field while hydrating metadata");
                    ""
                });
            let metadata = doc
                .get_first(self.fields.metadata_json)
                .and_then(|value| value.as_str())
                .and_then(|raw| match serde_json::from_str(raw) {
                    Ok(value) => Some(value),
                    Err(error) => {
                        debug!(%doc_id, %error, "failed to deserialize metadata JSON");
                        None
                    }
                });
            if let Some(result) = results
                .iter_mut()
                .find(|result| result.doc_id.as_str() == doc_id)
            {
                result.metadata = metadata;
            }
        }

        Ok(())
    }
}

// ─── LexicalSearch implementation ───────────────────────────────────────────

impl frankensearch_core::traits::LexicalRead for TantivyIndex {
    #[instrument(skip_all, fields(query = %query, limit = limit))]
    fn search<'a>(
        &'a self,
        _cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            let query = Self::truncate_query(query);

            if query.trim().is_empty() || limit == 0 {
                return Ok(Vec::new());
            }

            let parsed = self.parse_query_shipping(query);

            let searcher = self.reader.searcher();
            let top_docs = search_guarded(
                &searcher,
                &*parsed,
                &TopDocs::with_limit(limit).order_by_score(),
            )?;

            debug!(hits = top_docs.len(), "tantivy BM25 search completed");

            let mut results = Vec::with_capacity(top_docs.len());
            for (bm25_score, doc_address) in top_docs {
                let doc: TantivyDocument =
                    searcher
                        .doc(doc_address)
                        .map_err(|e| SearchError::SubsystemError {
                            subsystem: "tantivy",
                            source: Box::new(e),
                        })?;

                let doc_id = doc
                    .get_first(self.fields.id)
                    .and_then(|v| v.as_str())
                    .unwrap_or_else(|| {
                        debug!("tantivy document missing id field, using empty doc_id");
                        ""
                    })
                    .to_owned();

                let metadata = doc
                    .get_first(self.fields.metadata_json)
                    .and_then(|v| v.as_str())
                    .and_then(|s| match serde_json::from_str(s) {
                        Ok(val) => Some(val),
                        Err(e) => {
                            debug!(doc_id = %doc_id, error = %e, "failed to deserialize metadata JSON");
                            None
                        }
                    });

                results.push(ScoredResult {
                    doc_id: doc_id.into(),
                    score: bm25_score,
                    source: ScoreSource::Lexical,
                    index: None,
                    fast_score: None,
                    quality_score: None,
                    lexical_score: Some(bm25_score),
                    rerank_score: None,
                    explanation: None,
                    metadata,
                });
            }

            Ok(results)
        })
    }

    /// Score candidates and pin the searcher that produced them.
    ///
    /// Tantivy's deferred-metadata path skips loading stored fields for the
    /// whole candidate pool and restores them only for the winners. The batch
    /// carries the exact [`Searcher`] used for scoring: a Tantivy searcher
    /// holds a fixed segment set, so it *is* the immutable generation handle
    /// this bead requires, and hydration through it cannot observe a commit
    /// that landed after scoring.
    fn search_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, LexicalCandidateBatch> {
        Box::pin(async move {
            let truncated = Self::truncate_query(query);
            // One searcher for scoring AND hydration. Taking a second one at
            // hydration time is exactly the generation race this pins shut.
            let searcher = self.reader.searcher();
            // An empty query or a zero limit still returns a *pinned* batch
            // rather than an eager one. Keeping the shape uniform means a
            // `None` context always signals caller misuse, so hydration can
            // reject it without having to guess whether the batch was legitimately
            // eager.
            let hits = if truncated.trim().is_empty() || limit == 0 {
                Vec::new()
            } else {
                let parsed = self.parse_query_shipping(truncated);
                execute_top_k(&searcher, &*parsed, limit, 0)?
            };
            let candidates = self
                .collect_id_hits(&searcher, hits)?
                .into_iter()
                .map(|hit| ScoredResult {
                    doc_id: hit.doc_id,
                    score: hit.bm25_score,
                    source: ScoreSource::Lexical,
                    index: None,
                    fast_score: None,
                    quality_score: None,
                    lexical_score: Some(hit.bm25_score),
                    rerank_score: None,
                    explanation: None,
                    metadata: None,
                })
                .collect();
            let _ = cx;
            Ok(LexicalCandidateBatch::deferred(
                candidates,
                LexicalHydrationContext::new("tantivy", Box::new(searcher)),
            ))
        })
    }

    /// Restore winner metadata from the pinned scoring searcher.
    ///
    /// Rejects a missing or foreign context with a typed error rather than
    /// silently reading the current generation.
    fn hydrate_candidates<'a>(
        &'a self,
        _cx: &'a Cx,
        context: Option<&'a LexicalHydrationContext>,
        results: &'a mut [ScoredResult],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            let Some(context) = context else {
                return Err(SearchError::SubsystemError {
                    subsystem: "tantivy.hydration",
                    source: "deferred Tantivy candidates require their batch context; \
                             none was provided"
                        .into(),
                });
            };
            let Some(searcher) = context.downcast_ref::<Searcher>() else {
                return Err(SearchError::SubsystemError {
                    subsystem: "tantivy.hydration",
                    source: format!(
                        "hydration context from backend {:?} is not a Tantivy searcher pin; \
                         refusing cross-engine hydration",
                        context.backend()
                    )
                    .into(),
                });
            };
            self.hydrate_with_searcher(searcher, results, true)
        })
    }

    fn doc_count(&self) -> SearchResult<usize> {
        Self::checked_searcher_doc_count(&self.reader.searcher())
    }
}

impl frankensearch_core::traits::LexicalWrite for TantivyIndex {
    fn index_document<'a>(
        &'a self,
        cx: &'a Cx,
        doc: &'a IndexableDocument,
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            let mut tantivy_doc = self.to_tantivy_doc(doc);

            {
                let writer_guard = self
                    .writer
                    .lock(cx)
                    .await
                    .map_err(|e| Self::map_writer_lock_error("tantivy.index", e))?;
                let Some(writer) = writer_guard.as_ref() else {
                    return Err(Self::read_only_writer_error("index"));
                };

                // Delete any existing document with same ID (upsert semantics).
                let term = Term::from_field_text(self.fields.id, &doc.id);
                writer.delete_term(term);
                self.assign_ord(&mut tantivy_doc, &doc.id);
                writer
                    .add_document(tantivy_doc)
                    .map_err(|e| SearchError::SubsystemError {
                        subsystem: "tantivy",
                        source: Box::new(e),
                    })?;
            }

            self.doc_count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        })
    }

    fn index_documents<'a>(
        &'a self,
        cx: &'a Cx,
        docs: &'a [IndexableDocument],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            {
                let writer_guard = self
                    .writer
                    .lock(cx)
                    .await
                    .map_err(|e| Self::map_writer_lock_error("tantivy.batch_index", e))?;
                let Some(writer) = writer_guard.as_ref() else {
                    return Err(Self::read_only_writer_error("batch_index"));
                };

                for doc in docs {
                    let mut tantivy_doc = self.to_tantivy_doc(doc);
                    // Upsert: delete existing then add.
                    let term = Term::from_field_text(self.fields.id, &doc.id);
                    writer.delete_term(term);
                    self.assign_ord(&mut tantivy_doc, &doc.id);
                    writer
                        .add_document(tantivy_doc)
                        .map_err(|e| SearchError::SubsystemError {
                            subsystem: "tantivy",
                            source: Box::new(e),
                        })?;
                }
            }

            self.doc_count.fetch_add(docs.len(), Ordering::Relaxed);

            debug!(count = docs.len(), "batch indexed documents");
            Ok(())
        })
    }

    fn commit<'a>(&'a self, cx: &'a Cx) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            {
                let mut writer_guard = self
                    .writer
                    .lock(cx)
                    .await
                    .map_err(|e| Self::map_writer_lock_error("tantivy.commit", e))?;
                let Some(writer) = writer_guard.as_mut() else {
                    return Err(Self::read_only_writer_error("commit"));
                };

                writer.commit().map_err(|e| SearchError::SubsystemError {
                    subsystem: "tantivy",
                    source: Box::new(e),
                })?;
            }

            // Reload the reader to pick up committed changes.
            self.reader
                .reload()
                .map_err(|e| SearchError::SubsystemError {
                    subsystem: "tantivy",
                    source: Box::new(e),
                })?;

            // Re-count after commit for accuracy.
            let searcher = self.reader.searcher();
            let actual = usize::try_from(searcher.num_docs()).unwrap_or(usize::MAX);
            self.doc_count.store(actual, Ordering::Relaxed);

            // Persist the ordinal→doc_id table so a reopen keeps the fast path.
            self.persist_ord_table();

            debug!(doc_count = actual, "tantivy commit completed");
            Ok(())
        })
    }
}
