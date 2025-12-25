//! CI/CD integration
//!
//! This module provides generators for CI/CD configuration files
//! for various platforms (GitHub Actions, GitLab CI, etc.).

pub mod github;
pub mod gitlab;

pub use github::WorkflowGenerator;
pub use gitlab::PipelineGenerator;
