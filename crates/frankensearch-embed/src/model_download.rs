//! Model download system with progress reporting and atomic installation.
//!
//! Downloads model files from `HuggingFace` with SHA-256 verification,
//! atomic installation (rename-over), and progress callbacks.
//!
//! Gated behind the `download` feature flag to keep the core crate network-free.

mod downloader;
mod helpers;
mod types;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod transport_tests;

pub use downloader::ModelDownloader;
pub use types::{
    DEFAULT_MAX_MODEL_ARTIFACT_BYTES, DownloadConfig, DownloadProgress,
    MODEL_ACQUISITION_PROGRESS_SCHEMA_V1, MODEL_ACQUISITION_RECEIPT_SCHEMA_V1,
    MODEL_ACQUISITION_RECOVERY_SCHEMA_V1, ModelAcquisitionCacheReasonV1, ModelAcquisitionOutcomeV1,
    ModelAcquisitionProgressV1, ModelAcquisitionReceiptV1, ModelAcquisitionRecoveryV1,
    ModelAcquisitionRequest, ModelAcquisitionSource, ModelAcquisitionSourceKindV1,
    ModelAcquisitionStageV1, ModelAcquisitionVerificationResultV1, VerifiedModelStageV1,
    diagnose_model_acquisition,
};
