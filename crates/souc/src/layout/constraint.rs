//! Layout Constraints - Day 39
//!
//! Allows developers to influence layout decisions through annotations:
//! - `#[colocate(a, b)]` - concepts should be in the same cluster
//! - `#[separate(a, b)]` - concepts should NOT be colocated
//! - `#[hot]` / `#[cold]` - force region assignment
//! - `#[explain_layout]` - request detailed layout explanation

use std::collections::HashSet;

use crate::common::Span;

/// Layout constraint from developer annotations
#[derive(Debug, Clone)]
pub enum LayoutConstraint {
    /// These concepts must be in the same cluster
    Colocate {
        concepts: Vec<String>,
        source: ConstraintSource,
    },

    /// These concepts must NOT be in the same cluster
    Separate {
        concepts: Vec<String>,
        source: ConstraintSource,
    },

    /// Force concept to specific region
    ForceRegion {
        concept: String,
        region: ForcedRegion,
        source: ConstraintSource,
    },

    /// Request explanation for this scope
    Explain { scope_name: String, span: Span },
}

/// Forced memory region
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForcedRegion {
    /// Hot: Stack, arena - L1/L2 friendly
    Hot,
    /// Warm: Arena - L2/L3
    Warm,
    /// Cold: Heap - RAM
    Cold,
}

impl std::fmt::Display for ForcedRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForcedRegion::Hot => write!(f, "Hot (L1/L2)"),
            ForcedRegion::Warm => write!(f, "Warm (L2/L3)"),
            ForcedRegion::Cold => write!(f, "Cold (RAM)"),
        }
    }
}

/// Source location of a constraint
#[derive(Debug, Clone)]
pub struct ConstraintSource {
    pub file: String,
    pub line: u32,
    pub column: u32,
    pub attribute: String,
}

impl ConstraintSource {
    pub fn new(
        file: impl Into<String>,
        line: u32,
        column: u32,
        attribute: impl Into<String>,
    ) -> Self {
        Self {
            file: file.into(),
            line,
            column,
            attribute: attribute.into(),
        }
    }

    pub fn from_span(span: &Span, file: &str, attribute: &str) -> Self {
        // Span only contains byte offsets, so we use the start offset as a reference
        // Line/column would need to be computed from the source file
        Self {
            file: file.to_string(),
            line: 0, // Would need source file to compute line
            column: span.start as u32,
            attribute: attribute.to_string(),
        }
    }
}

impl std::fmt::Display for ConstraintSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "#[{}] at {}:{}:{}",
            self.attribute, self.file, self.line, self.column
        )
    }
}

/// Collection of constraints for a compilation unit
#[derive(Debug, Default, Clone)]
pub struct ConstraintSet {
    pub constraints: Vec<LayoutConstraint>,
}

impl ConstraintSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, constraint: LayoutConstraint) {
        self.constraints.push(constraint);
    }

    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    /// Get all colocate constraints involving a concept
    pub fn colocate_partners(&self, concept: &str) -> HashSet<String> {
        let mut partners = HashSet::new();

        for constraint in &self.constraints {
            if let LayoutConstraint::Colocate { concepts, .. } = constraint
                && concepts.iter().any(|c| c == concept)
            {
                partners.extend(concepts.iter().cloned());
            }
        }

        partners.remove(concept);
        partners
    }

    /// Check if two concepts are forced separate
    pub fn must_separate(&self, a: &str, b: &str) -> bool {
        for constraint in &self.constraints {
            if let LayoutConstraint::Separate { concepts, .. } = constraint {
                let has_a = concepts.iter().any(|c| c == a);
                let has_b = concepts.iter().any(|c| c == b);
                if has_a && has_b {
                    return true;
                }
            }
        }
        false
    }

    /// Get forced region for a concept, if any
    pub fn forced_region(&self, concept: &str) -> Option<ForcedRegion> {
        for constraint in &self.constraints {
            if let LayoutConstraint::ForceRegion {
                concept: c, region, ..
            } = constraint
                && c == concept
            {
                return Some(*region);
            }
        }
        None
    }

    /// Get all colocate constraints
    pub fn colocate_constraints(&self) -> impl Iterator<Item = &LayoutConstraint> {
        self.constraints
            .iter()
            .filter(|c| matches!(c, LayoutConstraint::Colocate { .. }))
    }

    /// Get all separate constraints
    pub fn separate_constraints(&self) -> impl Iterator<Item = &LayoutConstraint> {
        self.constraints
            .iter()
            .filter(|c| matches!(c, LayoutConstraint::Separate { .. }))
    }

    /// Get all force region constraints
    pub fn force_region_constraints(&self) -> impl Iterator<Item = &LayoutConstraint> {
        self.constraints
            .iter()
            .filter(|c| matches!(c, LayoutConstraint::ForceRegion { .. }))
    }

    /// Get all explain constraints
    pub fn explain_constraints(&self) -> impl Iterator<Item = &LayoutConstraint> {
        self.constraints
            .iter()
            .filter(|c| matches!(c, LayoutConstraint::Explain { .. }))
    }

    /// Merge another constraint set into this one
    pub fn merge(&mut self, other: ConstraintSet) {
        self.constraints.extend(other.constraints);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_source() -> ConstraintSource {
        ConstraintSource::new("test.sio", 1, 1, "test")
    }

    #[test]
    fn test_colocate_partners() {
        let mut set = ConstraintSet::new();
        set.add(LayoutConstraint::Colocate {
            concepts: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            source: test_source(),
        });

        let partners = set.colocate_partners("A");
        assert!(partners.contains("B"));
        assert!(partners.contains("C"));
        assert!(!partners.contains("A"));
    }

    #[test]
    fn test_must_separate() {
        let mut set = ConstraintSet::new();
        set.add(LayoutConstraint::Separate {
            concepts: vec!["X".to_string(), "Y".to_string()],
            source: test_source(),
        });

        assert!(set.must_separate("X", "Y"));
        assert!(set.must_separate("Y", "X"));
        assert!(!set.must_separate("X", "Z"));
    }

    #[test]
    fn test_forced_region() {
        let mut set = ConstraintSet::new();
        set.add(LayoutConstraint::ForceRegion {
            concept: "hot_data".to_string(),
            region: ForcedRegion::Hot,
            source: test_source(),
        });
        set.add(LayoutConstraint::ForceRegion {
            concept: "cold_data".to_string(),
            region: ForcedRegion::Cold,
            source: test_source(),
        });

        assert_eq!(set.forced_region("hot_data"), Some(ForcedRegion::Hot));
        assert_eq!(set.forced_region("cold_data"), Some(ForcedRegion::Cold));
        assert_eq!(set.forced_region("other"), None);
    }

    #[test]
    fn test_constraint_set_merge() {
        let mut set1 = ConstraintSet::new();
        set1.add(LayoutConstraint::Colocate {
            concepts: vec!["A".to_string(), "B".to_string()],
            source: test_source(),
        });

        let mut set2 = ConstraintSet::new();
        set2.add(LayoutConstraint::Separate {
            concepts: vec!["C".to_string(), "D".to_string()],
            source: test_source(),
        });

        set1.merge(set2);
        assert_eq!(set1.len(), 2);
    }
}
