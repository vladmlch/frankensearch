//! Embedder auto-detection and fallback stack assembly.

mod discovery;
#[cfg(all(
    feature = "download",
    any(feature = "model2vec", feature = "fastembed")
))]
mod lazy_download;
mod stack;

#[cfg(test)]
mod remote_intent_tests;
#[cfg(test)]
mod tests;

pub use discovery::DetectOptions;
pub use stack::{
    DimReduceEmbedder, EmbedderStack, ModelAvailabilityDiagnostic, ModelStatus, TwoTierAvailability,
};
