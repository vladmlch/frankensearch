//! Convenience API for building frankensearch indexes in a single method chain.
//!
//! [`IndexBuilder`] handles all the complexity of coordinating embedders,
//! vector index writers, and optional lexical indexing behind a fluent API.
//!
//! # Example
//!
//! ```rust,ignore
//! use frankensearch::IndexBuilder;
//!
//! let stats = IndexBuilder::new("./my_index")
//!     .add_document("doc-1", "Hello world")
//!     .add_document("doc-2", "Distributed consensus algorithms")
//!     .build(&cx)
//!     .await?; // errors if auto-detect finds no semantic model
//!
//! println!("Indexed {} docs in {:.1}ms", stats.doc_count, stats.total_ms);
//! ```

mod builder;
mod embed;
mod lexical;
mod open;
mod types;

#[cfg(test)]
mod tests;

pub use builder::IndexBuilder;
#[cfg(any(feature = "native", feature = "rerank"))]
pub use embed::detect_embedder_stack_with_pool;
pub use open::{open_admitted_v2_sync_with_residual_sidecar_cache, open_hybrid};
pub use types::{
    HybridIndexParts, IndexBuildStats, IndexProgress, IndexSizeBreakdown, LexicalArmReceipt,
    LexicalReaderBackend,
};
