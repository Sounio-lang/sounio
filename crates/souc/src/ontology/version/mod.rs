//! Ontology Versioning & Evolution
//!
//! This module implements version stability for reproducible builds with
//! ontology dependencies. It provides:
//!
//! - Lock file system (`ontology.lock`) for pinning exact versions
//! - Version constraint resolution (semver-like)
//! - Ontology diffing (detect breaking changes)
//! - Deprecation tracking and warnings
//! - Update checking and migration support
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Version Management                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ontology.lock     │  Exact versions for reproducibility        │
//! │  VersionResolver   │  Constraint solving (^1.0, ~2.3, etc.)     │
//! │  OntologyDiff      │  Detect added/removed/deprecated terms     │
//! │  DeprecationTracker│  Warn on deprecated term usage             │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Lock File Format
//!
//! ```toml
//! # ontology.lock - DO NOT EDIT MANUALLY
//! [metadata]
//! generated = "2025-01-15T10:30:00Z"
//! sounio_version = "0.1.0"
//!
//! [[ontology]]
//! name = "chebi"
//! version = "2024-01-01"
//! checksum = "sha256:abc123..."
//! source = "https://purl.obolibrary.org/obo/chebi.owl"
//!
//! [[ontology]]
//! name = "go"
//! version = "2024-01-15"
//! checksum = "sha256:def456..."
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use sounio::ontology::version::{Manifest, VersionResolver, OntologyDiff};
//!
//! // Load lock file
//! let manifest = Manifest::load("ontology.lock")?;
//!
//! // Resolve constraints
//! let resolver = VersionResolver::new();
//! let versions = resolver.resolve(&constraints)?;
//!
//! // Check for updates
//! let diff = OntologyDiff::compute(&old_version, &new_version)?;
//! if diff.has_breaking_changes() {
//!     warn!("Breaking changes detected!");
//! }
//! ```

pub mod deprecation;
pub mod diff;
pub mod manifest;
pub mod resolver;

pub use deprecation::{
    DeprecatedTerm, DeprecationLevel, DeprecationTracker, DeprecationWarning, Replacement,
};
pub use diff::{ChangeKind, OntologyChange, OntologyDiff, TermChange};
pub use manifest::{Manifest, ManifestError, OntologyEntry, OntologySource as ManifestSource};
pub use resolver::{
    Constraint, ConstraintOp, Resolution, ResolutionError as VersionResolutionError,
    VersionResolver,
};

use std::fmt;

/// Semantic version for ontologies
///
/// Ontology versions can be:
/// - Date-based: "2024-01-15"
/// - Semver-like: "1.2.3"
/// - Release tags: "2024Q1", "v2.0"
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OntologyVersion {
    /// Date-based version (YYYY-MM-DD)
    Date { year: u16, month: u8, day: u8 },
    /// Semantic version
    Semver {
        major: u32,
        minor: u32,
        patch: u32,
        prerelease: Option<String>,
    },
    /// Arbitrary release tag
    Tag(String),
}

impl OntologyVersion {
    /// Parse a version string
    pub fn parse(s: &str) -> Result<Self, VersionParseError> {
        let s = s.trim();

        // Try date format first (YYYY-MM-DD)
        if let Some(date) = Self::try_parse_date(s) {
            return Ok(date);
        }

        // Try semver (X.Y.Z or vX.Y.Z)
        let semver_str = s.strip_prefix('v').unwrap_or(s);
        if let Some(semver) = Self::try_parse_semver(semver_str) {
            return Ok(semver);
        }

        // Fall back to tag
        if s.is_empty() {
            return Err(VersionParseError::Empty);
        }

        Ok(OntologyVersion::Tag(s.to_string()))
    }

    fn try_parse_date(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 3 {
            return None;
        }

        let year: u16 = parts[0].parse().ok()?;
        let month: u8 = parts[1].parse().ok()?;
        let day: u8 = parts[2].parse().ok()?;

        // Validate ranges
        if !(1900..=2100).contains(&year) || !(1..=12).contains(&month) || !(1..=31).contains(&day)
        {
            return None;
        }

        Some(OntologyVersion::Date { year, month, day })
    }

    fn try_parse_semver(s: &str) -> Option<Self> {
        // Split off prerelease if present
        let (version_part, prerelease) = if let Some(idx) = s.find('-') {
            (&s[..idx], Some(s[idx + 1..].to_string()))
        } else {
            (s, None)
        };

        let parts: Vec<&str> = version_part.split('.').collect();
        if parts.len() < 2 || parts.len() > 3 {
            return None;
        }

        let major: u32 = parts[0].parse().ok()?;
        let minor: u32 = parts[1].parse().ok()?;
        let patch: u32 = parts.get(2).and_then(|p| p.parse().ok()).unwrap_or(0);

        Some(OntologyVersion::Semver {
            major,
            minor,
            patch,
            prerelease,
        })
    }

    /// Check if this version is a date version
    pub fn is_date(&self) -> bool {
        matches!(self, OntologyVersion::Date { .. })
    }

    /// Check if this version is semver
    pub fn is_semver(&self) -> bool {
        matches!(self, OntologyVersion::Semver { .. })
    }

    /// Compare versions (returns None if incomparable)
    pub fn compare(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;

        match (self, other) {
            (
                OntologyVersion::Date {
                    year: y1,
                    month: m1,
                    day: d1,
                },
                OntologyVersion::Date {
                    year: y2,
                    month: m2,
                    day: d2,
                },
            ) => Some((y1, m1, d1).cmp(&(y2, m2, d2))),

            (
                OntologyVersion::Semver {
                    major: maj1,
                    minor: min1,
                    patch: p1,
                    prerelease: pre1,
                },
                OntologyVersion::Semver {
                    major: maj2,
                    minor: min2,
                    patch: p2,
                    prerelease: pre2,
                },
            ) => {
                let core_cmp = (maj1, min1, p1).cmp(&(maj2, min2, p2));
                if core_cmp != Ordering::Equal {
                    return Some(core_cmp);
                }
                // Prerelease versions are less than release versions
                match (pre1, pre2) {
                    (None, None) => Some(Ordering::Equal),
                    (Some(_), None) => Some(Ordering::Less),
                    (None, Some(_)) => Some(Ordering::Greater),
                    (Some(a), Some(b)) => Some(a.cmp(b)),
                }
            }

            (OntologyVersion::Tag(t1), OntologyVersion::Tag(t2)) => {
                // Tags are only equal if identical
                if t1 == t2 {
                    Some(Ordering::Equal)
                } else {
                    None
                }
            }

            // Different version types are incomparable
            _ => None,
        }
    }

    /// Check if this version satisfies a constraint
    pub fn satisfies(&self, constraint: &Constraint) -> bool {
        resolver::version_satisfies(self, constraint)
    }
}

impl fmt::Display for OntologyVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OntologyVersion::Date { year, month, day } => {
                write!(f, "{:04}-{:02}-{:02}", year, month, day)
            }
            OntologyVersion::Semver {
                major,
                minor,
                patch,
                prerelease,
            } => {
                write!(f, "{}.{}.{}", major, minor, patch)?;
                if let Some(pre) = prerelease {
                    write!(f, "-{}", pre)?;
                }
                Ok(())
            }
            OntologyVersion::Tag(tag) => write!(f, "{}", tag),
        }
    }
}

impl PartialOrd for OntologyVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.compare(other)
    }
}

/// Error parsing a version string
#[derive(Debug, Clone, thiserror::Error)]
pub enum VersionParseError {
    #[error("Empty version string")]
    Empty,
    #[error("Invalid version format: {0}")]
    InvalidFormat(String),
}

/// Checksum for integrity verification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Checksum {
    /// Algorithm used (sha256, sha512, etc.)
    pub algorithm: String,
    /// Hex-encoded hash value
    pub hash: String,
}

impl Checksum {
    /// Create a new checksum
    pub fn new(algorithm: impl Into<String>, hash: impl Into<String>) -> Self {
        Self {
            algorithm: algorithm.into(),
            hash: hash.into(),
        }
    }

    /// Parse a checksum string (e.g., "sha256:abc123...")
    pub fn parse(s: &str) -> Option<Self> {
        let (algo, hash) = s.split_once(':')?;
        Some(Self {
            algorithm: algo.to_string(),
            hash: hash.to_string(),
        })
    }

    /// Verify a checksum against data
    #[cfg(feature = "crypto")]
    pub fn verify(&self, data: &[u8]) -> bool {
        use sha2::{Digest, Sha256, Sha512};

        let computed = match self.algorithm.as_str() {
            "sha256" => {
                let mut hasher = Sha256::new();
                hasher.update(data);
                hex::encode(hasher.finalize())
            }
            "sha512" => {
                let mut hasher = Sha512::new();
                hasher.update(data);
                hex::encode(hasher.finalize())
            }
            _ => return false,
        };

        computed == self.hash
    }

    /// Placeholder verify for when crypto is disabled
    #[cfg(not(feature = "crypto"))]
    pub fn verify(&self, _data: &[u8]) -> bool {
        // Without crypto feature, we trust the checksum
        true
    }
}

impl fmt::Display for Checksum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.algorithm, self.hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_date_version() {
        let v = OntologyVersion::parse("2024-01-15").unwrap();
        assert!(matches!(
            v,
            OntologyVersion::Date {
                year: 2024,
                month: 1,
                day: 15
            }
        ));
    }

    #[test]
    fn test_parse_semver() {
        let v = OntologyVersion::parse("1.2.3").unwrap();
        assert!(matches!(
            v,
            OntologyVersion::Semver {
                major: 1,
                minor: 2,
                patch: 3,
                prerelease: None
            }
        ));
    }

    #[test]
    fn test_parse_semver_with_v() {
        let v = OntologyVersion::parse("v2.0.0").unwrap();
        assert!(matches!(
            v,
            OntologyVersion::Semver {
                major: 2,
                minor: 0,
                patch: 0,
                ..
            }
        ));
    }

    #[test]
    fn test_parse_semver_with_prerelease() {
        let v = OntologyVersion::parse("1.0.0-beta").unwrap();
        if let OntologyVersion::Semver { prerelease, .. } = v {
            assert_eq!(prerelease, Some("beta".to_string()));
        } else {
            panic!("Expected semver");
        }
    }

    #[test]
    fn test_parse_tag() {
        let v = OntologyVersion::parse("2024Q1").unwrap();
        assert!(matches!(v, OntologyVersion::Tag(t) if t == "2024Q1"));
    }

    #[test]
    fn test_version_comparison_dates() {
        let v1 = OntologyVersion::parse("2024-01-01").unwrap();
        let v2 = OntologyVersion::parse("2024-06-15").unwrap();
        assert!(v1 < v2);
    }

    #[test]
    fn test_version_comparison_semver() {
        let v1 = OntologyVersion::parse("1.0.0").unwrap();
        let v2 = OntologyVersion::parse("1.2.0").unwrap();
        let v3 = OntologyVersion::parse("2.0.0").unwrap();
        assert!(v1 < v2);
        assert!(v2 < v3);
    }

    #[test]
    fn test_prerelease_less_than_release() {
        let pre = OntologyVersion::parse("1.0.0-beta").unwrap();
        let release = OntologyVersion::parse("1.0.0").unwrap();
        assert!(pre < release);
    }

    #[test]
    fn test_checksum_parse() {
        let cs = Checksum::parse("sha256:abc123def456").unwrap();
        assert_eq!(cs.algorithm, "sha256");
        assert_eq!(cs.hash, "abc123def456");
    }

    #[test]
    fn test_checksum_display() {
        let cs = Checksum::new("sha256", "abc123");
        assert_eq!(cs.to_string(), "sha256:abc123");
    }
}
