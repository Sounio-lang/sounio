//! Version Constraint Resolution
//!
//! This module implements semver-like constraint solving for ontology versions.
//! It supports common constraint operators and resolves to the best matching version.
//!
//! # Constraint Syntax
//!
//! - `^1.2.3` - Compatible with 1.2.3 (>=1.2.3, <2.0.0)
//! - `~1.2.3` - Approximately 1.2.3 (>=1.2.3, <1.3.0)
//! - `>=1.2.3` - Greater than or equal
//! - `<2.0.0` - Less than
//! - `=1.2.3` - Exact version
//! - `*` - Any version
//! - `2024-01-*` - Any version in January 2024 (for date-based)
//!
//! # Examples
//!
//! ```rust,ignore
//! use sounio::ontology::version::{VersionResolver, Constraint};
//!
//! let resolver = VersionResolver::new();
//!
//! let available = vec![
//!     OntologyVersion::parse("1.0.0").unwrap(),
//!     OntologyVersion::parse("1.2.0").unwrap(),
//!     OntologyVersion::parse("2.0.0").unwrap(),
//! ];
//!
//! let constraint = Constraint::parse("^1.0.0").unwrap();
//! let best = resolver.resolve_single(&constraint, &available);
//! assert_eq!(best.unwrap().to_string(), "1.2.0");
//! ```

use std::collections::HashMap;

use super::OntologyVersion;

/// A version constraint
#[derive(Debug, Clone)]
pub struct Constraint {
    /// The constraint operator
    pub op: ConstraintOp,
    /// The version to compare against
    pub version: OntologyVersion,
    /// For wildcard constraints on date versions
    pub wildcard_month: Option<u8>,
    pub wildcard_year: Option<u16>,
}

/// Constraint operator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintOp {
    /// Exact version (=1.2.3)
    Exact,
    /// Greater than (>1.2.3)
    Gt,
    /// Greater than or equal (>=1.2.3)
    Gte,
    /// Less than (<1.2.3)
    Lt,
    /// Less than or equal (<=1.2.3)
    Lte,
    /// Compatible with (^1.2.3) - same major version
    Caret,
    /// Approximately (~1.2.3) - same minor version
    Tilde,
    /// Any version (*)
    Any,
    /// Date wildcard (2024-01-*)
    DateWildcard,
}

impl Constraint {
    /// Parse a constraint string
    pub fn parse(s: &str) -> Result<Self, ConstraintParseError> {
        let s = s.trim();

        // Handle wildcard
        if s == "*" {
            return Ok(Constraint {
                op: ConstraintOp::Any,
                version: OntologyVersion::Tag("*".to_string()),
                wildcard_month: None,
                wildcard_year: None,
            });
        }

        // Handle date wildcard (2024-01-* or 2024-*)
        if s.contains('*')
            && s.chars()
                .next()
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false)
        {
            return Self::parse_date_wildcard(s);
        }

        // Parse operator prefix
        let (op, version_str) = if let Some(rest) = s.strip_prefix(">=") {
            (ConstraintOp::Gte, rest)
        } else if let Some(rest) = s.strip_prefix("<=") {
            (ConstraintOp::Lte, rest)
        } else if let Some(rest) = s.strip_prefix('>') {
            (ConstraintOp::Gt, rest)
        } else if let Some(rest) = s.strip_prefix('<') {
            (ConstraintOp::Lt, rest)
        } else if let Some(rest) = s.strip_prefix('^') {
            (ConstraintOp::Caret, rest)
        } else if let Some(rest) = s.strip_prefix('~') {
            (ConstraintOp::Tilde, rest)
        } else if let Some(rest) = s.strip_prefix('=') {
            (ConstraintOp::Exact, rest)
        } else {
            // No operator = exact match
            (ConstraintOp::Exact, s)
        };

        let version = OntologyVersion::parse(version_str.trim())
            .map_err(|_| ConstraintParseError::InvalidVersion(version_str.to_string()))?;

        Ok(Constraint {
            op,
            version,
            wildcard_month: None,
            wildcard_year: None,
        })
    }

    fn parse_date_wildcard(s: &str) -> Result<Self, ConstraintParseError> {
        let parts: Vec<&str> = s.split('-').collect();

        match parts.as_slice() {
            [year, "*"] => {
                // 2024-*
                let year: u16 = year
                    .parse()
                    .map_err(|_| ConstraintParseError::InvalidVersion(s.to_string()))?;
                Ok(Constraint {
                    op: ConstraintOp::DateWildcard,
                    version: OntologyVersion::Date {
                        year,
                        month: 1,
                        day: 1,
                    },
                    wildcard_month: None,
                    wildcard_year: Some(year),
                })
            }
            [year, month, "*"] => {
                // 2024-01-*
                let year: u16 = year
                    .parse()
                    .map_err(|_| ConstraintParseError::InvalidVersion(s.to_string()))?;
                let month: u8 = month
                    .parse()
                    .map_err(|_| ConstraintParseError::InvalidVersion(s.to_string()))?;
                Ok(Constraint {
                    op: ConstraintOp::DateWildcard,
                    version: OntologyVersion::Date {
                        year,
                        month,
                        day: 1,
                    },
                    wildcard_month: Some(month),
                    wildcard_year: Some(year),
                })
            }
            _ => Err(ConstraintParseError::InvalidVersion(s.to_string())),
        }
    }

    /// Check if a version satisfies this constraint
    pub fn is_satisfied_by(&self, version: &OntologyVersion) -> bool {
        version_satisfies(version, self)
    }

    /// Get the minimum version that satisfies this constraint (if bounded)
    pub fn minimum(&self) -> Option<&OntologyVersion> {
        match self.op {
            ConstraintOp::Exact | ConstraintOp::Gte | ConstraintOp::Caret | ConstraintOp::Tilde => {
                Some(&self.version)
            }
            ConstraintOp::Gt => Some(&self.version), // Technically > but close enough
            _ => None,
        }
    }
}

impl std::fmt::Display for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.op {
            ConstraintOp::Exact => write!(f, "={}", self.version),
            ConstraintOp::Gt => write!(f, ">{}", self.version),
            ConstraintOp::Gte => write!(f, ">={}", self.version),
            ConstraintOp::Lt => write!(f, "<{}", self.version),
            ConstraintOp::Lte => write!(f, "<={}", self.version),
            ConstraintOp::Caret => write!(f, "^{}", self.version),
            ConstraintOp::Tilde => write!(f, "~{}", self.version),
            ConstraintOp::Any => write!(f, "*"),
            ConstraintOp::DateWildcard => {
                if let (Some(year), month) = (self.wildcard_year, self.wildcard_month) {
                    if let Some(m) = month {
                        write!(f, "{:04}-{:02}-*", year, m)
                    } else {
                        write!(f, "{:04}-*", year)
                    }
                } else {
                    write!(f, "{}", self.version)
                }
            }
        }
    }
}

/// Error parsing a constraint
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConstraintParseError {
    #[error("Invalid version: {0}")]
    InvalidVersion(String),
    #[error("Invalid operator: {0}")]
    InvalidOperator(String),
    #[error("Empty constraint")]
    Empty,
}

/// Check if a version satisfies a constraint
pub fn version_satisfies(version: &OntologyVersion, constraint: &Constraint) -> bool {
    use std::cmp::Ordering;

    match constraint.op {
        ConstraintOp::Any => true,

        ConstraintOp::Exact => version == &constraint.version,

        ConstraintOp::Gt => version.compare(&constraint.version) == Some(Ordering::Greater),

        ConstraintOp::Gte => {
            matches!(
                version.compare(&constraint.version),
                Some(Ordering::Greater | Ordering::Equal)
            )
        }

        ConstraintOp::Lt => version.compare(&constraint.version) == Some(Ordering::Less),

        ConstraintOp::Lte => {
            matches!(
                version.compare(&constraint.version),
                Some(Ordering::Less | Ordering::Equal)
            )
        }

        ConstraintOp::Caret => {
            // ^1.2.3 means >=1.2.3, <2.0.0
            match (&constraint.version, version) {
                (
                    OntologyVersion::Semver {
                        major: c_maj,
                        minor: c_min,
                        patch: c_patch,
                        ..
                    },
                    OntologyVersion::Semver {
                        major: v_maj,
                        minor: v_min,
                        patch: v_patch,
                        ..
                    },
                ) => {
                    // Same major version and >= constraint version
                    v_maj == c_maj && (*v_min, *v_patch) >= (*c_min, *c_patch)
                }
                (
                    OntologyVersion::Date {
                        year: c_year,
                        month: c_month,
                        day: c_day,
                    },
                    OntologyVersion::Date {
                        year: v_year,
                        month: v_month,
                        day: v_day,
                    },
                ) => {
                    // Same year and >= constraint date
                    v_year == c_year && (*v_month, *v_day) >= (*c_month, *c_day)
                }
                _ => false,
            }
        }

        ConstraintOp::Tilde => {
            // ~1.2.3 means >=1.2.3, <1.3.0
            match (&constraint.version, version) {
                (
                    OntologyVersion::Semver {
                        major: c_maj,
                        minor: c_min,
                        patch: c_patch,
                        ..
                    },
                    OntologyVersion::Semver {
                        major: v_maj,
                        minor: v_min,
                        patch: v_patch,
                        ..
                    },
                ) => {
                    // Same major and minor, patch >= constraint
                    v_maj == c_maj && v_min == c_min && v_patch >= c_patch
                }
                (
                    OntologyVersion::Date {
                        year: c_year,
                        month: c_month,
                        day: c_day,
                    },
                    OntologyVersion::Date {
                        year: v_year,
                        month: v_month,
                        day: v_day,
                    },
                ) => {
                    // Same year and month, day >= constraint
                    v_year == c_year && v_month == c_month && v_day >= c_day
                }
                _ => false,
            }
        }

        ConstraintOp::DateWildcard => {
            if let OntologyVersion::Date { year, month, .. } = version
                && let Some(c_year) = constraint.wildcard_year
            {
                if *year != c_year {
                    return false;
                }
                if let Some(c_month) = constraint.wildcard_month {
                    return *month == c_month;
                }
                return true;
            }
            false
        }
    }
}

/// Result of version resolution
#[derive(Debug, Clone)]
pub struct Resolution {
    /// Resolved versions for each ontology
    pub versions: HashMap<String, OntologyVersion>,
    /// Warnings during resolution
    pub warnings: Vec<String>,
}

/// Error during resolution
#[derive(Debug, Clone, thiserror::Error)]
pub enum ResolutionError {
    #[error("No version satisfies constraint {constraint} for {ontology}")]
    NoSatisfyingVersion {
        ontology: String,
        constraint: String,
    },

    #[error("Conflicting constraints for {ontology}: {constraint1} and {constraint2}")]
    ConflictingConstraints {
        ontology: String,
        constraint1: String,
        constraint2: String,
    },

    #[error("Dependency cycle detected: {cycle}")]
    CyclicDependency { cycle: String },

    #[error("Ontology not found: {0}")]
    OntologyNotFound(String),
}

/// Version resolver for ontology constraints
pub struct VersionResolver {
    /// Available versions per ontology
    available: HashMap<String, Vec<OntologyVersion>>,
    /// Prefer newer versions
    prefer_latest: bool,
}

impl VersionResolver {
    /// Create a new resolver
    pub fn new() -> Self {
        Self {
            available: HashMap::new(),
            prefer_latest: true,
        }
    }

    /// Add available versions for an ontology
    pub fn add_available(&mut self, ontology: impl Into<String>, versions: Vec<OntologyVersion>) {
        let ontology = ontology.into();
        self.available.entry(ontology).or_default().extend(versions);
    }

    /// Set whether to prefer latest versions
    pub fn prefer_latest(mut self, prefer: bool) -> Self {
        self.prefer_latest = prefer;
        self
    }

    /// Resolve a single constraint to the best version
    pub fn resolve_single(
        &self,
        ontology: &str,
        constraint: &Constraint,
    ) -> Result<OntologyVersion, ResolutionError> {
        let available = self
            .available
            .get(ontology)
            .ok_or_else(|| ResolutionError::OntologyNotFound(ontology.to_string()))?;

        let mut matching: Vec<_> = available
            .iter()
            .filter(|v| constraint.is_satisfied_by(v))
            .collect();

        if matching.is_empty() {
            return Err(ResolutionError::NoSatisfyingVersion {
                ontology: ontology.to_string(),
                constraint: constraint.to_string(),
            });
        }

        // Sort by version
        matching.sort_by(|a, b| a.compare(b).unwrap_or(std::cmp::Ordering::Equal));

        // Return latest or earliest depending on preference
        let best = if self.prefer_latest {
            matching.last()
        } else {
            matching.first()
        };

        Ok((*best.unwrap()).clone())
    }

    /// Resolve multiple constraints (one per ontology)
    pub fn resolve(
        &self,
        constraints: &HashMap<String, Vec<Constraint>>,
    ) -> Result<Resolution, ResolutionError> {
        let mut versions = HashMap::new();
        let mut warnings = Vec::new();

        for (ontology, onto_constraints) in constraints {
            // Find versions that satisfy ALL constraints for this ontology
            let available = self
                .available
                .get(ontology)
                .ok_or_else(|| ResolutionError::OntologyNotFound(ontology.to_string()))?;

            let mut matching: Vec<_> = available
                .iter()
                .filter(|v| onto_constraints.iter().all(|c| c.is_satisfied_by(v)))
                .collect();

            if matching.is_empty() {
                // Check for conflicting constraints
                if onto_constraints.len() > 1 {
                    return Err(ResolutionError::ConflictingConstraints {
                        ontology: ontology.clone(),
                        constraint1: onto_constraints[0].to_string(),
                        constraint2: onto_constraints[1].to_string(),
                    });
                }
                return Err(ResolutionError::NoSatisfyingVersion {
                    ontology: ontology.clone(),
                    constraint: onto_constraints
                        .first()
                        .map(|c| c.to_string())
                        .unwrap_or_default(),
                });
            }

            // Sort and select
            matching.sort_by(|a, b| a.compare(b).unwrap_or(std::cmp::Ordering::Equal));

            let selected = if self.prefer_latest {
                matching.last().unwrap()
            } else {
                matching.first().unwrap()
            };

            // Warn if not exact match
            if let Some(min) = onto_constraints.iter().find_map(|c| c.minimum())
                && *selected != min
            {
                warnings.push(format!(
                    "{}: using {} instead of minimum {}",
                    ontology, selected, min
                ));
            }

            versions.insert(ontology.clone(), (*selected).clone());
        }

        Ok(Resolution { versions, warnings })
    }

    /// Resolve with dependency ordering (topological sort)
    pub fn resolve_with_deps(
        &self,
        constraints: &HashMap<String, Vec<Constraint>>,
        dependencies: &HashMap<String, Vec<String>>,
    ) -> Result<Vec<(String, OntologyVersion)>, ResolutionError> {
        // First resolve all versions
        let resolution = self.resolve(constraints)?;

        // Topological sort
        let order = self.topological_sort(&resolution.versions, dependencies)?;

        Ok(order
            .into_iter()
            .filter_map(|name| resolution.versions.get(&name).map(|v| (name, v.clone())))
            .collect())
    }

    fn topological_sort(
        &self,
        versions: &HashMap<String, OntologyVersion>,
        dependencies: &HashMap<String, Vec<String>>,
    ) -> Result<Vec<String>, ResolutionError> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut in_progress = std::collections::HashSet::new();

        fn visit(
            node: &str,
            dependencies: &HashMap<String, Vec<String>>,
            versions: &HashMap<String, OntologyVersion>,
            visited: &mut std::collections::HashSet<String>,
            in_progress: &mut std::collections::HashSet<String>,
            result: &mut Vec<String>,
        ) -> Result<(), ResolutionError> {
            if in_progress.contains(node) {
                return Err(ResolutionError::CyclicDependency {
                    cycle: node.to_string(),
                });
            }
            if visited.contains(node) {
                return Ok(());
            }

            in_progress.insert(node.to_string());

            if let Some(deps) = dependencies.get(node) {
                for dep in deps {
                    if versions.contains_key(dep) {
                        visit(dep, dependencies, versions, visited, in_progress, result)?;
                    }
                }
            }

            in_progress.remove(node);
            visited.insert(node.to_string());
            result.push(node.to_string());
            Ok(())
        }

        for name in versions.keys() {
            visit(
                name,
                dependencies,
                versions,
                &mut visited,
                &mut in_progress,
                &mut result,
            )?;
        }

        Ok(result)
    }
}

impl Default for VersionResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_constraint_exact() {
        let c = Constraint::parse("=1.2.3").unwrap();
        assert_eq!(c.op, ConstraintOp::Exact);
    }

    #[test]
    fn test_parse_constraint_caret() {
        let c = Constraint::parse("^1.2.3").unwrap();
        assert_eq!(c.op, ConstraintOp::Caret);
    }

    #[test]
    fn test_parse_constraint_tilde() {
        let c = Constraint::parse("~1.2.3").unwrap();
        assert_eq!(c.op, ConstraintOp::Tilde);
    }

    #[test]
    fn test_parse_constraint_gte() {
        let c = Constraint::parse(">=2.0.0").unwrap();
        assert_eq!(c.op, ConstraintOp::Gte);
    }

    #[test]
    fn test_parse_constraint_any() {
        let c = Constraint::parse("*").unwrap();
        assert_eq!(c.op, ConstraintOp::Any);
    }

    #[test]
    fn test_parse_date_wildcard() {
        let c = Constraint::parse("2024-01-*").unwrap();
        assert_eq!(c.op, ConstraintOp::DateWildcard);
        assert_eq!(c.wildcard_year, Some(2024));
        assert_eq!(c.wildcard_month, Some(1));
    }

    #[test]
    fn test_satisfies_exact() {
        let c = Constraint::parse("=1.2.3").unwrap();
        let v1 = OntologyVersion::parse("1.2.3").unwrap();
        let v2 = OntologyVersion::parse("1.2.4").unwrap();

        assert!(c.is_satisfied_by(&v1));
        assert!(!c.is_satisfied_by(&v2));
    }

    #[test]
    fn test_satisfies_caret() {
        let c = Constraint::parse("^1.2.0").unwrap();
        let v1 = OntologyVersion::parse("1.2.0").unwrap();
        let v2 = OntologyVersion::parse("1.5.0").unwrap();
        let v3 = OntologyVersion::parse("2.0.0").unwrap();
        let v4 = OntologyVersion::parse("1.1.0").unwrap();

        assert!(c.is_satisfied_by(&v1)); // exact
        assert!(c.is_satisfied_by(&v2)); // same major, higher minor
        assert!(!c.is_satisfied_by(&v3)); // different major
        assert!(!c.is_satisfied_by(&v4)); // lower minor
    }

    #[test]
    fn test_satisfies_tilde() {
        let c = Constraint::parse("~1.2.0").unwrap();
        let v1 = OntologyVersion::parse("1.2.0").unwrap();
        let v2 = OntologyVersion::parse("1.2.5").unwrap();
        let v3 = OntologyVersion::parse("1.3.0").unwrap();

        assert!(c.is_satisfied_by(&v1)); // exact
        assert!(c.is_satisfied_by(&v2)); // same minor, higher patch
        assert!(!c.is_satisfied_by(&v3)); // different minor
    }

    #[test]
    fn test_satisfies_date_wildcard() {
        let c = Constraint::parse("2024-01-*").unwrap();
        let v1 = OntologyVersion::parse("2024-01-01").unwrap();
        let v2 = OntologyVersion::parse("2024-01-31").unwrap();
        let v3 = OntologyVersion::parse("2024-02-01").unwrap();

        assert!(c.is_satisfied_by(&v1));
        assert!(c.is_satisfied_by(&v2));
        assert!(!c.is_satisfied_by(&v3));
    }

    #[test]
    fn test_resolve_single() {
        let mut resolver = VersionResolver::new();
        resolver.add_available(
            "chebi",
            vec![
                OntologyVersion::parse("1.0.0").unwrap(),
                OntologyVersion::parse("1.2.0").unwrap(),
                OntologyVersion::parse("2.0.0").unwrap(),
            ],
        );

        let c = Constraint::parse("^1.0.0").unwrap();
        let result = resolver.resolve_single("chebi", &c).unwrap();

        // Should return 1.2.0 (latest in ^1.x range)
        assert_eq!(result.to_string(), "1.2.0");
    }

    #[test]
    fn test_resolve_multiple() {
        let mut resolver = VersionResolver::new();
        resolver.add_available(
            "chebi",
            vec![
                OntologyVersion::parse("1.0.0").unwrap(),
                OntologyVersion::parse("1.5.0").unwrap(),
            ],
        );
        resolver.add_available(
            "go",
            vec![
                OntologyVersion::parse("2024-01-01").unwrap(),
                OntologyVersion::parse("2024-06-01").unwrap(),
            ],
        );

        let mut constraints = HashMap::new();
        constraints.insert(
            "chebi".to_string(),
            vec![Constraint::parse("^1.0.0").unwrap()],
        );
        constraints.insert("go".to_string(), vec![Constraint::parse("2024-*").unwrap()]);

        let resolution = resolver.resolve(&constraints).unwrap();

        assert_eq!(resolution.versions["chebi"].to_string(), "1.5.0");
        assert_eq!(resolution.versions["go"].to_string(), "2024-06-01");
    }

    #[test]
    fn test_no_satisfying_version() {
        let mut resolver = VersionResolver::new();
        resolver.add_available("chebi", vec![OntologyVersion::parse("1.0.0").unwrap()]);

        let c = Constraint::parse(">=2.0.0").unwrap();
        let result = resolver.resolve_single("chebi", &c);

        assert!(matches!(
            result,
            Err(ResolutionError::NoSatisfyingVersion { .. })
        ));
    }

    #[test]
    fn test_topological_sort() {
        let mut resolver = VersionResolver::new();
        resolver.add_available("a", vec![OntologyVersion::parse("1.0.0").unwrap()]);
        resolver.add_available("b", vec![OntologyVersion::parse("1.0.0").unwrap()]);
        resolver.add_available("c", vec![OntologyVersion::parse("1.0.0").unwrap()]);

        let mut constraints = HashMap::new();
        constraints.insert("a".to_string(), vec![Constraint::parse("*").unwrap()]);
        constraints.insert("b".to_string(), vec![Constraint::parse("*").unwrap()]);
        constraints.insert("c".to_string(), vec![Constraint::parse("*").unwrap()]);

        let mut deps = HashMap::new();
        deps.insert("c".to_string(), vec!["b".to_string()]);
        deps.insert("b".to_string(), vec!["a".to_string()]);

        let result = resolver.resolve_with_deps(&constraints, &deps).unwrap();
        let names: Vec<_> = result.iter().map(|(n, _)| n.as_str()).collect();

        // a should come before b, b before c
        let a_idx = names.iter().position(|&n| n == "a").unwrap();
        let b_idx = names.iter().position(|&n| n == "b").unwrap();
        let c_idx = names.iter().position(|&n| n == "c").unwrap();

        assert!(a_idx < b_idx);
        assert!(b_idx < c_idx);
    }
}
