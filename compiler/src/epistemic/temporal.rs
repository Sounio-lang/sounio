//! Temporal indexing: τ context-time
//!
//! Types in Sounio can evolve over context-time.
//! This enables modeling of knowledge that changes based on when/where it's used.
//!
//! # Context-Time
//!
//! Context-time captures not just when, but in what context knowledge is valid.
//! This is inspired by Dynamic Homotopy Type Theory (DHoTT) concepts.
//!
//! # Example
//!
//! ```sounio
//! // Knowledge valid in a specific lab context
//! let measurement: Knowledge[
//!     τ = (2024, MyLab, Experiment001),
//!     ...
//! ] = ...;
//!
//! // Knowledge that changes with ontology versions
//! let classification: Knowledge[
//!     τ = Version(ChEBI, "2024.1"),
//!     ...
//! ] = ...;
//! ```

use std::cmp::Ordering;

/// Context-time index for knowledge
///
/// This captures not just when, but in what context knowledge is valid.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextTime {
    /// Temporal component (when)
    pub temporal: TemporalIndex,

    /// Spatial/institutional component (where)
    pub context: ContextIndex,

    /// Validity bounds
    pub validity: ValidityBounds,
}

impl ContextTime {
    /// Current context-time (compile-time)
    pub fn current() -> Self {
        Self {
            temporal: TemporalIndex::CompileTime,
            context: ContextIndex::Unspecified,
            validity: ValidityBounds::Unbounded,
        }
    }

    /// Runtime-determined context
    pub fn runtime() -> Self {
        Self {
            temporal: TemporalIndex::Runtime,
            context: ContextIndex::Unspecified,
            validity: ValidityBounds::Unbounded,
        }
    }

    /// Specific historical context
    pub fn historical(year: u32, context: &str) -> Self {
        Self {
            temporal: TemporalIndex::Absolute {
                year,
                month: None,
                day: None,
            },
            context: ContextIndex::Named(context.to_string()),
            validity: ValidityBounds::Unbounded,
        }
    }

    /// Version-based context (for evolving ontologies)
    pub fn versioned(ontology: &str, version: &str) -> Self {
        Self {
            temporal: TemporalIndex::Version {
                ontology: ontology.to_string(),
                version: version.to_string(),
            },
            context: ContextIndex::Unspecified,
            validity: ValidityBounds::Unbounded,
        }
    }

    /// Create context with specific validity bounds
    pub fn with_validity(mut self, validity: ValidityBounds) -> Self {
        self.validity = validity;
        self
    }

    /// Create context with named context
    pub fn with_context(mut self, context: &str) -> Self {
        self.context = ContextIndex::Named(context.to_string());
        self
    }

    /// Check if this context-time is before another
    pub fn is_before(&self, other: &ContextTime) -> Option<bool> {
        self.temporal
            .partial_cmp(&other.temporal)
            .map(|o| o == Ordering::Less)
    }

    /// Check if this context-time is valid now
    pub fn is_currently_valid(&self) -> bool {
        match &self.validity {
            ValidityBounds::Unbounded => true,
            ValidityBounds::Until(end) => {
                // Would need actual time comparison
                true
            }
            ValidityBounds::From(start) => {
                // Would need actual time comparison
                true
            }
            ValidityBounds::Range { from, until } => {
                // Would need actual time comparison
                true
            }
            ValidityBounds::Conditional { condition: _ } => {
                // Requires runtime evaluation
                true
            }
        }
    }
}

impl Default for ContextTime {
    fn default() -> Self {
        Self::current()
    }
}

/// Temporal index component
#[derive(Debug, Clone, PartialEq, Default)]
pub enum TemporalIndex {
    /// Known at compile time
    #[default]
    CompileTime,

    /// Determined at runtime
    Runtime,

    /// Absolute timestamp
    Absolute {
        year: u32,
        month: Option<u8>,
        day: Option<u8>,
    },

    /// Relative to another index
    Relative {
        base: Box<TemporalIndex>,
        offset: TemporalOffset,
    },

    /// Version-based (for evolving ontologies)
    Version { ontology: String, version: String },
}

impl TemporalIndex {
    /// Create an absolute date
    pub fn date(year: u32, month: u8, day: u8) -> Self {
        Self::Absolute {
            year,
            month: Some(month),
            day: Some(day),
        }
    }

    /// Create a year-only date
    pub fn year(year: u32) -> Self {
        Self::Absolute {
            year,
            month: None,
            day: None,
        }
    }

    /// Create a relative index
    pub fn relative(base: TemporalIndex, offset: TemporalOffset) -> Self {
        Self::Relative {
            base: Box::new(base),
            offset,
        }
    }
}

impl PartialOrd for TemporalIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (
                TemporalIndex::Absolute {
                    year: y1,
                    month: m1,
                    day: d1,
                },
                TemporalIndex::Absolute {
                    year: y2,
                    month: m2,
                    day: d2,
                },
            ) => match y1.cmp(y2) {
                Ordering::Equal => match (m1, m2) {
                    (Some(m1), Some(m2)) => match m1.cmp(m2) {
                        Ordering::Equal => match (d1, d2) {
                            (Some(d1), Some(d2)) => Some(d1.cmp(d2)),
                            _ => Some(Ordering::Equal),
                        },
                        other => Some(other),
                    },
                    _ => Some(Ordering::Equal),
                },
                other => Some(other),
            },
            (
                TemporalIndex::Version {
                    ontology: o1,
                    version: v1,
                },
                TemporalIndex::Version {
                    ontology: o2,
                    version: v2,
                },
            ) => {
                if o1 == o2 {
                    // Simple version comparison (would need semver parsing)
                    v1.partial_cmp(v2)
                } else {
                    None
                }
            }
            _ => None, // Cannot compare compile-time vs runtime, etc.
        }
    }
}

/// Temporal offset for relative indices
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TemporalOffset {
    pub years: i32,
    pub months: i32,
    pub days: i32,
}

impl TemporalOffset {
    /// Create a new offset
    pub fn new(years: i32, months: i32, days: i32) -> Self {
        Self {
            years,
            months,
            days,
        }
    }

    /// Create offset of N years
    pub fn years(n: i32) -> Self {
        Self {
            years: n,
            months: 0,
            days: 0,
        }
    }

    /// Create offset of N months
    pub fn months(n: i32) -> Self {
        Self {
            years: 0,
            months: n,
            days: 0,
        }
    }

    /// Create offset of N days
    pub fn days(n: i32) -> Self {
        Self {
            years: 0,
            months: 0,
            days: n,
        }
    }
}

/// Context/institutional component
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ContextIndex {
    /// No specific context
    #[default]
    Unspecified,

    /// Named context (lab, institution, project)
    Named(String),

    /// Hierarchical context
    Hierarchical(Vec<String>),

    /// Geographic context
    Geographic {
        country: Option<String>,
        region: Option<String>,
        institution: Option<String>,
    },
}

impl ContextIndex {
    /// Create a named context
    pub fn named(name: impl Into<String>) -> Self {
        Self::Named(name.into())
    }

    /// Create a hierarchical context
    pub fn hierarchical(levels: Vec<String>) -> Self {
        Self::Hierarchical(levels)
    }

    /// Create a geographic context
    pub fn geographic(
        country: Option<&str>,
        region: Option<&str>,
        institution: Option<&str>,
    ) -> Self {
        Self::Geographic {
            country: country.map(String::from),
            region: region.map(String::from),
            institution: institution.map(String::from),
        }
    }
}

/// Validity bounds for knowledge
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ValidityBounds {
    /// Always valid
    #[default]
    Unbounded,

    /// Valid until specific time
    Until(TemporalIndex),

    /// Valid from specific time
    From(TemporalIndex),

    /// Valid within range
    Range {
        from: TemporalIndex,
        until: TemporalIndex,
    },

    /// Conditional validity
    Conditional { condition: String },
}

impl ValidityBounds {
    /// Create bounded validity until a date
    pub fn until(index: TemporalIndex) -> Self {
        Self::Until(index)
    }

    /// Create bounded validity from a date
    pub fn from(index: TemporalIndex) -> Self {
        Self::From(index)
    }

    /// Create bounded validity within a range
    pub fn range(from: TemporalIndex, until: TemporalIndex) -> Self {
        Self::Range { from, until }
    }

    /// Create conditional validity
    pub fn conditional(condition: impl Into<String>) -> Self {
        Self::Conditional {
            condition: condition.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_time_current() {
        let ct = ContextTime::current();
        assert_eq!(ct.temporal, TemporalIndex::CompileTime);
        assert!(ct.is_currently_valid());
    }

    #[test]
    fn test_temporal_index_ordering() {
        let t1 = TemporalIndex::date(2024, 1, 1);
        let t2 = TemporalIndex::date(2024, 6, 15);
        let t3 = TemporalIndex::date(2023, 12, 31);

        assert!(t1 < t2);
        assert!(t3 < t1);
    }

    #[test]
    fn test_version_ordering() {
        let v1 = TemporalIndex::Version {
            ontology: "ChEBI".into(),
            version: "2024.1".into(),
        };
        let v2 = TemporalIndex::Version {
            ontology: "ChEBI".into(),
            version: "2024.2".into(),
        };

        assert!(v1 < v2);
    }

    #[test]
    fn test_context_index() {
        let ctx = ContextIndex::hierarchical(vec![
            "MIT".into(),
            "Koch Institute".into(),
            "Lab 101".into(),
        ]);

        if let ContextIndex::Hierarchical(levels) = ctx {
            assert_eq!(levels.len(), 3);
        } else {
            panic!("Expected hierarchical context");
        }
    }
}
