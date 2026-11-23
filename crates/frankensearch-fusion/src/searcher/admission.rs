use asupersync::Cx;
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::{
    EmbeddingIdentityBundleV1, ProducerCompatibilityErrorV1, QuantizationFormat,
};
use frankensearch_core::traits::Embedder;
use frankensearch_index::FsviV2IdentityBinding;

/// Whether this searcher's vectors may be touched by this query at all
/// (bd-core-vector-space-search-guard-ctzo C1).
///
/// Decided ONCE, before the query is embedded and therefore before any
/// vector byte, ANN graph node, or embedding cache entry is read. That
/// ordering is the point: the identity facts this decision needs — the
/// embedder's own bundle and the binding each tier was admitted under — are
/// both available with no inference work done, so a query in the wrong space
/// costs one fingerprint comparison instead of an embed plus a full scan
/// that returns confident nonsense.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SemanticAdmission {
    /// Every tier this query will consult is an admitted FSVI v2 owner whose
    /// retained identity the configured embedder provably produces.
    OwnerBacked {
        /// The quality tier is also owner-backed and its embedder joins.
        /// `false` when no quality tier is retained or no quality embedder is
        /// configured — the query is then admitted for the fast tier alone.
        quality: bool,
    },
    /// This index retains no admitted owner: a v1 or path-opened artifact,
    /// which carries no space identity to join against. Typed and named
    /// rather than silently treated as "compatible": every canonical
    /// production index on disk today is still v1, so this is the ordinary
    /// case and it must stay searchable, but a caller that thinks it is
    /// running under identity enforcement is entitled to know it is not.
    LegacyUnidentified,
}

/// Query-side contracts captured independently before inference or cache access.
/// A missing identity represents only the explicit legacy custom-embedder lane;
/// a malformed or refused declared identity never enters that lane.
pub(super) struct QueryEmbeddingIdentities {
    pub(super) fast: Option<EmbeddingIdentityBundleV1>,
    pub(super) quality: Option<EmbeddingIdentityBundleV1>,
}

/// Observe cancellation at a boundary between synchronous search stages.
///
/// This does not preempt the stage that just ran or the opaque call that
/// preceded it; it stops the next material stage from starting.
pub(super) fn cancellation_checkpoint(cx: &Cx, phase: &'static str) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: phase.to_owned(),
        reason: cx
            .cancel_reason()
            .map_or_else(|| error.to_string(), |reason| reason.to_string()),
    })
}

fn missing_query_identity(error: &SearchError) -> bool {
    matches!(error, SearchError::InvalidConfig { field, .. } if field == "embedder.identity")
}

pub(super) fn capture_query_identity(
    embedder: &dyn Embedder,
    tier: &str,
) -> SearchResult<Option<EmbeddingIdentityBundleV1>> {
    let identity = match embedder.identity() {
        Ok(identity) => identity,
        Err(error) if missing_query_identity(&error) => return Ok(None),
        Err(error) => return Err(error),
    };
    identity.validate()?;
    if usize::try_from(identity.space.dimension).ok() != Some(embedder.dimension()) {
        return Err(SearchError::InvalidConfig {
            field: format!("query_embedding.{tier}.dimension"),
            value: embedder.dimension().to_string(),
            reason: "the configured embedder's dimension disagrees with its identity".to_owned(),
        });
    }
    if identity.storage.quantization != QuantizationFormat::F32
        || !identity.storage.format.starts_with("in-memory-")
        || !matches!(
            identity.storage.endianness.as_str(),
            "native-f32-values" | "native-test-only"
        )
    {
        return Err(SearchError::InvalidConfig {
            field: format!("query_embedding.{tier}.storage"),
            value: "output_contract".to_owned(),
            reason: "query embedders must declare an in-memory native f32 output contract"
                .to_owned(),
        });
    }
    Ok(Some(identity.clone()))
}

fn validate_query_producer(
    embedder: &dyn Embedder,
    expected: Option<&EmbeddingIdentityBundleV1>,
    tier: &str,
) -> SearchResult<()> {
    let unchanged = match (expected, embedder.identity()) {
        (Some(expected), Ok(actual)) => {
            actual == expected
                && usize::try_from(expected.space.dimension).ok() == Some(embedder.dimension())
        }
        (None, Err(error)) => missing_query_identity(&error),
        _ => false,
    };
    if unchanged {
        Ok(())
    } else {
        Err(SearchError::UnverifiableRemoteSpace {
            producer: format!("query_embedding.{tier}"),
            reason: "query producer changed after this request's identity admission".to_owned(),
        })
    }
}

fn validate_query_response_identity(
    actual: &EmbeddingIdentityBundleV1,
    expected: &EmbeddingIdentityBundleV1,
    tier: &str,
) -> SearchResult<()> {
    if actual == expected {
        return Ok(());
    }
    let field = if actual.space != expected.space {
        format!("query_embedding.{tier}.space_identity")
    } else if actual.producer != expected.producer {
        format!("search_activation.{tier}.producer_conformance")
    } else if actual.input != expected.input {
        format!("query_embedding.{tier}.input_identity")
    } else {
        format!("query_embedding.{tier}.storage")
    };
    Err(SearchError::InvalidConfig {
        field,
        value: "bound_response".to_owned(),
        reason: "the actual query response differs from the identity admitted before inference"
            .to_owned(),
    })
}

/// Infer once under a captured request contract. Bound calls deliberately bypass
/// `CachedEmbedder`'s raw-vector cache: advertised identity cannot attest an old
/// raw cache entry or replace the identity attached to a provider response.
pub(super) async fn embed_admitted_query(
    cx: &Cx,
    embedder: &dyn Embedder,
    expected: Option<&EmbeddingIdentityBundleV1>,
    tier: &'static str,
    query: &str,
) -> SearchResult<(Vec<f32>, Option<EmbeddingIdentityBundleV1>)> {
    let before_phase = match tier {
        "fast" => "fast_identity_to_embedding",
        _ => "quality_identity_to_embedding",
    };
    cancellation_checkpoint(cx, before_phase)?;
    let admission = validate_query_producer(embedder, expected, tier);
    cancellation_checkpoint(cx, before_phase)?;
    admission?;
    let after_phase = match tier {
        "fast" => "fast_embed_to_activation",
        _ => "quality_embed_to_prf",
    };
    let response = if expected.is_some() {
        embedder
            .embed_bound(cx, query)
            .await
            .map(|bound| (bound.values, Some(bound.identity)))
    } else {
        embedder.embed(cx, query).await.map(|values| (values, None))
    };
    // Cancellation wins over a late response, provider error or identity drift.
    cancellation_checkpoint(cx, after_phase)?;
    let producer = validate_query_producer(embedder, expected, tier);
    cancellation_checkpoint(cx, after_phase)?;
    producer?;
    let (values, identity) = response?;
    if let (Some(actual), Some(expected)) = (identity, expected) {
        // Compare before validating: even a malformed foreign identity remains
        // a terminal response refusal, not a value-only embedding failure.
        validate_query_response_identity(&actual, expected, tier)?;
        let bound = frankensearch_core::traits::IdentityBoundEmbedding {
            values,
            identity: actual,
        };
        bound.validate()?;
        Ok((bound.values, Some(bound.identity)))
    } else {
        if values.len() != embedder.dimension() {
            return Err(SearchError::DimensionMismatch {
                expected: embedder.dimension(),
                found: values.len(),
            });
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(SearchError::InvalidConfig {
                field: format!("query_embedding.{tier}.values"),
                value: "non_finite".to_owned(),
                reason: "query vectors must contain only finite coordinates".to_owned(),
            });
        }
        Ok((values, None))
    }
}

pub(super) fn terminal_query_embedding_error(error: &SearchError) -> bool {
    matches!(
        error,
        SearchError::Cancelled { .. }
            | SearchError::UnverifiableRemoteSpace { .. }
            | SearchError::InvalidConfig { .. }
            | SearchError::DimensionMismatch { .. }
    )
}

/// Compare a v1 producer fingerprint without manufacturing an admitted owner.
/// Empty, identity-less indexes retain their explicit legacy/control behavior;
/// an identified semantic model needs an index built with its whole identity.
pub(super) fn admit_legacy_tier_embedder(
    embedder: &dyn Embedder,
    identity: Option<&EmbeddingIdentityBundleV1>,
    revision: &str,
    tier: &str,
) -> SearchResult<()> {
    if revision.is_empty() && !embedder.is_semantic() {
        return Ok(());
    }
    let identity = match identity {
        Some(identity) => identity,
        None if revision.is_empty() => return Ok(()),
        None => {
            return Err(SearchError::InvalidConfig {
                field: format!("search_activation.{tier}.producer_revision"),
                value: embedder.id().to_owned(),
                reason: format!("the stored {tier} producer requires a declared query identity"),
            });
        }
    };
    identity.validate()?;
    if revision != identity.fingerprint() {
        return Err(SearchError::InvalidConfig {
            field: format!("search_activation.{tier}.producer_revision"),
            value: revision.to_owned(),
            reason: format!(
                "the stored {tier} vectors do not bind this embedder's complete producer \
                 identity; rebuild the index with this embedder before searching"
            ),
        });
    }
    Ok(())
}

/// Join one configured embedder against the identity its tier was admitted
/// under, with no vector in hand (bd-ctzo C1).
///
/// The refusal field names are deliberately the SAME ones
/// [`frankensearch_index::TwoTierIndex::activate_owner_backed_search`] uses for the post-embed
/// join, because they are the same two failures: the query's space is not the
/// index's space, or its producer is not the index's producer. A caller that
/// greps for `query_embedding.fast.space_identity` finds both.
pub(super) fn admit_tier_embedder(
    embedder: &dyn Embedder,
    query_identity: Option<&EmbeddingIdentityBundleV1>,
    binding: &FsviV2IdentityBinding,
    tier: &str,
) -> SearchResult<()> {
    let query_identity = query_identity.ok_or_else(|| SearchError::InvalidConfig {
        field: format!("search_activation.{tier}.embedder_identity"),
        value: embedder.id().to_owned(),
        reason: format!(
            "the {tier} tier is an admitted FSVI v2 artifact, so its vectors may only be \
             searched by an embedder that declares a complete immutable identity; this one \
             does not"
        ),
    })?;
    match binding
        .frozen_identity()
        .identity
        .verify_exact_producer_with(query_identity)
    {
        Ok(_) => Ok(()),
        Err(ProducerCompatibilityErrorV1::SpaceMismatch) => Err(SearchError::InvalidConfig {
            field: format!("query_embedding.{tier}.space_identity"),
            value: query_identity.space.fingerprint(),
            reason: format!(
                "the configured {tier} embedder produces a different embedding space than the \
                 {tier} index was written in; matching dimensions do not make two spaces \
                 comparable, and this index must be reindexed with this embedder before it \
                 can serve its queries"
            ),
        }),
        // Every remaining variant is a refusal here, including
        // `CertificateRequired` -- the certified-compatible pairing that
        // `verify_producer_conformance` reports as comparison-grade telemetry
        // and that activation likewise refuses. Listing them as one arm keeps
        // a future variant refused by default instead of admitted by omission.
        Err(error) => Err(SearchError::InvalidConfig {
            field: format!("search_activation.{tier}.producer_conformance"),
            value: query_identity.producer.fingerprint(),
            reason: format!(
                "the configured {tier} embedder is not the producer this index's vectors were \
                 written by ({error}); a certified-compatible pairing is comparison-grade \
                 telemetry and never an admission basis"
            ),
        }),
    }
}
