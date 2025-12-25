//! Distributed builds and CI integration
//!
//! This module provides infrastructure for distributed build execution,
//! shared build caching, reproducible builds, and CI/CD integration.
//!
//! # Features
//!
//! - **Remote Build Execution**: Submit build jobs to remote servers
//! - **Shared Build Cache**: Cache build artifacts across machines
//! - **Reproducible Builds**: Ensure bit-for-bit reproducibility
//! - **CI Integration**: Generate CI configurations for GitHub/GitLab
//!
//! # Example
//!
//! ```ignore
//! use sounio::distributed::{client::*, protocol::*};
//!
//! // Connect to build server
//! let client = BuildClient::new(ClientConfig::default());
//! client.connect().await?;
//!
//! // Submit a build job
//! let result = client.build_remote(
//!     Path::new("."),
//!     "x86_64-unknown-linux-gnu",
//!     "release",
//! ).await?;
//! ```

pub mod cache;
pub mod ci;
pub mod client;
pub mod protocol;
pub mod reproducible;
pub mod server;

// Re-exports for convenience
pub use cache::{CacheClient, CacheConfig, CacheError, CacheMetadata, CacheServer, CacheStats};
pub use ci::{github::WorkflowGenerator, gitlab::PipelineGenerator};
pub use client::{BuildClient, ClientConfig, ClientError};
pub use protocol::{
    Artifact, BuildConfig, BuildError, BuildJob, BuildMessage, BuildResult, BuildStats,
    ClientCapabilities, Dependency, DependencyType, Diagnostic, DiagnosticSeverity, ErrorCode,
    FileInfo, JobProgress, JobType, OutputSpec, OutputType, PROTOCOL_VERSION, ProtocolError,
    ServerCapabilities, SourceFile, WorkerInfo, WorkerStatus,
};
pub use reproducible::{
    BuildEnvironment, BuildInputs, BuildProvenance, DependencyInput, ReproducibilityResult,
    SourceInput,
};
pub use server::{BuildServer, ServerConfig, ServerError, ServerStatsSnapshot};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify main types are accessible
        let _ = PROTOCOL_VERSION;
        let _ = ClientConfig::default();
        let _ = ServerConfig::default();
        let _ = CacheConfig::default();
    }

    #[test]
    fn test_protocol_version() {
        assert_eq!(PROTOCOL_VERSION, 1);
    }
}
