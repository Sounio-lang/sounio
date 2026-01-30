//! Package Manager for Sounio
//!
//! This module implements the package manager for the Sounio programming language,
//! providing dependency resolution, build orchestration, and registry interaction.
//!
//! # Components
//!
//! - [`manifest`] - Package manifest (d.toml) parsing and validation
//! - [`resolver`] - Dependency version resolution
//! - [`build`] - Build system and incremental compilation
//! - [`registry`] - Package registry interaction
//! - [`cli`] - Command-line interface
//! - [`auth`] - Credentials management
//! - [`cache`] - Package cache management

pub mod auth;
pub mod build;
pub mod cache;
pub mod cli;
pub mod manifest;
pub mod registry;
pub mod resolver;

pub use build::{BuildContext, BuildExecutor, BuildPlan, BuildProfile, BuildResult};
pub use manifest::{Dependency, Manifest, Package, PackageId};
pub use registry::{DefaultRegistry, Registry};
pub use resolver::{Lockfile, Resolution, Resolver};

// Re-export auth types
pub use auth::{AuthError, CredentialsStore, TokenManager};

// Re-export cache types
pub use cache::{CacheEntry, CacheIndex, CacheStats, PackageCache};

// Re-export registry types
pub use registry::{PackageMetadata, PackageSummary, RegistryError, VersionInfo};

#[cfg(feature = "pkg")]
pub use registry::async_client::HttpRegistry;

#[cfg(feature = "pkg")]
pub use cache::create_tarball;
