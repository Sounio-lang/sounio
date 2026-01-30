//! Core Temporal Types
//!
//! Defines the fundamental temporal dimension types for epistemic knowledge.
//!
//! # Temporal Dimension
//!
//! ```text
//! Temporal ::=
//!     | Instant(DateTime)              -- Fixed point in time
//!     | Interval(DateTime, DateTime)   -- Validity period
//!     | Versioned(Version, DateTime)   -- Version + timestamp
//!     | Timeless                       -- Atemporal knowledge
//!     | Decaying(DateTime, DecayFn)    -- With decay function
//! ```
//!
//! # Subtyping Rules
//!
//! ```text
//! Timeless <: Instant(t) for all t
//!     "Atemporal knowledge is valid at any instant"
//!
//! Interval(t₁, t₂) <: Instant(t) if t₁ ≤ t ≤ t₂
//!     "Interval includes its points"
//!
//! Decaying(t₀, f) <: Instant(t) with ε' = ε × f(t - t₀)
//!     "Decayed knowledge has reduced confidence"
//! ```

use super::decay::DecayFunction;
use chrono::{DateTime, Utc};
use std::cmp::Ordering;
use std::fmt;

/// Temporal dimension of knowledge
#[derive(Clone, Debug)]
pub enum Temporal {
    /// Fixed point in time
    Instant(DateTime<Utc>),

    /// Valid during interval [start, end]
    Interval {
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    },

    /// Versioned with semantic version
    Versioned {
        version: Version,
        created: DateTime<Utc>,
        superseded_by: Option<Version>,
    },

    /// Mathematical/logical truth (no decay)
    Timeless,

    /// Decays over time
    Decaying {
        created: DateTime<Utc>,
        decay_fn: DecayFunction,
    },
}

impl Temporal {
    /// Create an instant at the current time
    pub fn now() -> Self {
        Temporal::Instant(Utc::now())
    }

    /// Create a timeless temporal
    pub fn timeless() -> Self {
        Temporal::Timeless
    }

    /// Create an interval
    pub fn interval(start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        Temporal::Interval { start, end }
    }

    /// Create a decaying temporal with given function
    pub fn decaying(decay_fn: DecayFunction) -> Self {
        Temporal::Decaying {
            created: Utc::now(),
            decay_fn,
        }
    }

    /// Get the effective instant for this temporal
    pub fn effective_instant(&self) -> DateTime<Utc> {
        match self {
            Temporal::Instant(t) => *t,
            Temporal::Interval { end, .. } => *end,
            Temporal::Decaying { created, .. } => *created,
            Temporal::Versioned { created, .. } => *created,
            Temporal::Timeless => Utc::now(),
        }
    }

    /// Get the creation time (if applicable)
    pub fn creation_time(&self) -> Option<DateTime<Utc>> {
        match self {
            Temporal::Instant(t) => Some(*t),
            Temporal::Interval { start, .. } => Some(*start),
            Temporal::Decaying { created, .. } => Some(*created),
            Temporal::Versioned { created, .. } => Some(*created),
            Temporal::Timeless => None,
        }
    }

    /// Check if this temporal is valid at a given instant
    pub fn is_valid_at(&self, instant: DateTime<Utc>) -> bool {
        match self {
            Temporal::Timeless => true,
            Temporal::Instant(t) => *t <= instant,
            Temporal::Interval { start, end } => *start <= instant && instant <= *end,
            Temporal::Versioned { superseded_by, .. } => superseded_by.is_none(),
            Temporal::Decaying { .. } => true, // Always valid, just with reduced confidence
        }
    }

    /// Check if this temporal is superseded
    pub fn is_superseded(&self) -> bool {
        matches!(
            self,
            Temporal::Versioned {
                superseded_by: Some(_),
                ..
            }
        )
    }

    /// Combine two temporals for tensor product
    pub fn combine_for_tensor(&self, other: &Temporal) -> Temporal {
        match (self, other) {
            (Temporal::Timeless, t) | (t, Temporal::Timeless) => t.clone(),

            (Temporal::Instant(t1), Temporal::Instant(t2)) => Temporal::Instant((*t1).max(*t2)),

            (
                Temporal::Interval { start: s1, end: e1 },
                Temporal::Interval { start: s2, end: e2 },
            ) => Temporal::Interval {
                start: (*s1).max(*s2),
                end: (*e1).min(*e2),
            },

            (
                Temporal::Decaying {
                    created: c1,
                    decay_fn: d1,
                },
                Temporal::Decaying {
                    created: c2,
                    decay_fn: d2,
                },
            ) => Temporal::Decaying {
                created: (*c1).max(*c2),
                decay_fn: DecayFunction::product(d1, d2),
            },

            // Default: use the more recent instant
            _ => Temporal::Instant(Utc::now()),
        }
    }

    /// Get the newer of two temporals
    pub fn newer(&self, other: &Temporal) -> Temporal {
        let t1 = self.effective_instant();
        let t2 = other.effective_instant();

        if t1 >= t2 {
            self.clone()
        } else {
            other.clone()
        }
    }
}

impl fmt::Display for Temporal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Temporal::Instant(t) => write!(f, "Instant({})", t.format("%Y-%m-%d %H:%M:%S")),
            Temporal::Interval { start, end } => write!(
                f,
                "Interval({} to {})",
                start.format("%Y-%m-%d"),
                end.format("%Y-%m-%d")
            ),
            Temporal::Versioned {
                version,
                superseded_by,
                ..
            } => {
                if let Some(sup) = superseded_by {
                    write!(f, "Versioned({}, superseded by {})", version, sup)
                } else {
                    write!(f, "Versioned({})", version)
                }
            }
            Temporal::Timeless => write!(f, "Timeless"),
            Temporal::Decaying { decay_fn, .. } => write!(f, "Decaying({:?})", decay_fn),
        }
    }
}

/// Semantic versioning for knowledge
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl Version {
    /// Create a new version
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Version {
            major,
            minor,
            patch,
        }
    }

    /// Initial version (1.0.0)
    pub fn initial() -> Self {
        Version::new(1, 0, 0)
    }

    /// Bump major version (breaking change)
    pub fn bump_major(&self) -> Self {
        Version::new(self.major + 1, 0, 0)
    }

    /// Bump minor version (feature addition)
    pub fn bump_minor(&self) -> Self {
        Version::new(self.major, self.minor + 1, 0)
    }

    /// Bump patch version (bug fix)
    pub fn bump_patch(&self) -> Self {
        Version::new(self.major, self.minor, self.patch + 1)
    }

    /// Check if this version is compatible with another (same major)
    pub fn is_compatible_with(&self, other: &Version) -> bool {
        self.major == other.major
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.major.cmp(&other.major) {
            Ordering::Equal => match self.minor.cmp(&other.minor) {
                Ordering::Equal => self.patch.cmp(&other.patch),
                ord => ord,
            },
            ord => ord,
        }
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl std::str::FromStr for Version {
    type Err = VersionParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return Err(VersionParseError::InvalidFormat);
        }

        let major = parts[0]
            .parse()
            .map_err(|_| VersionParseError::InvalidNumber)?;
        let minor = parts[1]
            .parse()
            .map_err(|_| VersionParseError::InvalidNumber)?;
        let patch = parts[2]
            .parse()
            .map_err(|_| VersionParseError::InvalidNumber)?;

        Ok(Version::new(major, minor, patch))
    }
}

/// Error parsing a version string
#[derive(Debug, Clone)]
pub enum VersionParseError {
    InvalidFormat,
    InvalidNumber,
}

impl fmt::Display for VersionParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VersionParseError::InvalidFormat => {
                write!(f, "Invalid version format (expected X.Y.Z)")
            }
            VersionParseError::InvalidNumber => write!(f, "Invalid version number"),
        }
    }
}

impl std::error::Error for VersionParseError {}

/// Version information with metadata
#[derive(Clone, Debug)]
pub struct VersionInfo {
    pub version: Version,
    pub created: DateTime<Utc>,
    pub author: String,
    pub changelog: String,
    pub relation_to_previous: Option<VersionRelation>,
}

impl VersionInfo {
    /// Create a new version info
    pub fn new(
        version: Version,
        author: impl Into<String>,
        changelog: impl Into<String>,
        relation: Option<VersionRelation>,
    ) -> Self {
        VersionInfo {
            version,
            created: Utc::now(),
            author: author.into(),
            changelog: changelog.into(),
            relation_to_previous: relation,
        }
    }

    /// Create initial version info
    pub fn initial(author: impl Into<String>, changelog: impl Into<String>) -> Self {
        VersionInfo::new(
            Version::initial(),
            author,
            changelog,
            Some(VersionRelation::Initial),
        )
    }
}

/// Relation between versions
#[derive(Clone, Debug)]
pub enum VersionRelation {
    /// First version
    Initial,

    /// Supersedes previous version (breaking change)
    Supersedes { previous: Version, reason: String },

    /// Extends previous version (feature addition)
    Extends {
        previous: Version,
        additions: Vec<String>,
    },

    /// Refines previous version (precision improvement)
    Refines {
        previous: Version,
        improvements: Vec<String>,
    },
}

impl VersionRelation {
    /// Create a supersedes relation
    pub fn supersedes(previous: Version, reason: impl Into<String>) -> Self {
        VersionRelation::Supersedes {
            previous,
            reason: reason.into(),
        }
    }

    /// Create an extends relation
    pub fn extends(previous: Version, additions: Vec<String>) -> Self {
        VersionRelation::Extends {
            previous,
            additions,
        }
    }

    /// Create a refines relation
    pub fn refines(previous: Version, improvements: Vec<String>) -> Self {
        VersionRelation::Refines {
            previous,
            improvements,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_ordering() {
        let v1 = Version::new(1, 0, 0);
        let v1_1 = Version::new(1, 1, 0);
        let v2 = Version::new(2, 0, 0);

        assert!(v1 < v1_1);
        assert!(v1_1 < v2);
        assert!(v1 < v2);
    }

    #[test]
    fn test_version_parse() {
        let v: Version = "1.2.3".parse().unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
    }

    #[test]
    fn test_version_bump() {
        let v = Version::new(1, 2, 3);

        let major = v.bump_major();
        assert_eq!(major, Version::new(2, 0, 0));

        let minor = v.bump_minor();
        assert_eq!(minor, Version::new(1, 3, 0));

        let patch = v.bump_patch();
        assert_eq!(patch, Version::new(1, 2, 4));
    }

    #[test]
    fn test_version_compatibility() {
        let v1 = Version::new(1, 0, 0);
        let v1_5 = Version::new(1, 5, 3);
        let v2 = Version::new(2, 0, 0);

        assert!(v1.is_compatible_with(&v1_5));
        assert!(!v1.is_compatible_with(&v2));
    }

    #[test]
    fn test_temporal_instant() {
        let now = Utc::now();
        let t = Temporal::Instant(now);

        assert!(t.is_valid_at(now));
        assert_eq!(t.effective_instant(), now);
    }

    #[test]
    fn test_temporal_timeless() {
        let t = Temporal::Timeless;
        let any_time = Utc::now();

        assert!(t.is_valid_at(any_time));
        assert!(t.creation_time().is_none());
    }

    #[test]
    fn test_temporal_interval() {
        let start = Utc::now() - chrono::Duration::days(10);
        let end = Utc::now() + chrono::Duration::days(10);
        let t = Temporal::interval(start, end);

        assert!(t.is_valid_at(Utc::now()));
        assert!(!t.is_valid_at(start - chrono::Duration::days(1)));
        assert!(!t.is_valid_at(end + chrono::Duration::days(1)));
    }

    #[test]
    fn test_temporal_combine_timeless() {
        let timeless = Temporal::Timeless;
        let instant = Temporal::Instant(Utc::now());

        let combined = timeless.combine_for_tensor(&instant);
        assert!(matches!(combined, Temporal::Instant(_)));
    }
}
