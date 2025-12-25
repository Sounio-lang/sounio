//! Semantic Prefetch Table: Using ontology relationships to predict data access.
//!
//! The key insight: The compiler knows the ontology. It knows what data will be
//! accessed together. It can emit prefetch hints that hardware can't infer.
//!
//! This module bridges semantic knowledge (ontology relationships) with
//! physical optimization (memory prefetching).

use super::Ontology;
use super::types::Locality;
use std::collections::HashMap;

/// Semantic distance between concepts, normalized to [0.0, 1.0].
/// 0.0 = identical, 1.0 = completely unrelated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SemanticDistance(pub f64);

impl SemanticDistance {
    /// Create a new semantic distance.
    pub fn new(value: f64) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Identical concepts (distance 0).
    pub fn identical() -> Self {
        Self(0.0)
    }

    /// Strongly related concepts.
    pub fn related() -> Self {
        Self(0.2)
    }

    /// Moderately related concepts.
    pub fn moderate() -> Self {
        Self(0.5)
    }

    /// Weakly related concepts.
    pub fn weak() -> Self {
        Self(0.8)
    }

    /// Unrelated concepts.
    pub fn unrelated() -> Self {
        Self(1.0)
    }

    /// Get the raw distance value.
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Check if concepts are close enough to benefit from prefetching.
    pub fn is_prefetchable(&self) -> bool {
        self.0 < 0.5
    }

    /// Convert distance to prefetch priority (inverse relationship).
    pub fn to_priority(&self) -> PrefetchPriority {
        if self.0 < 0.1 {
            PrefetchPriority::Critical
        } else if self.0 < 0.3 {
            PrefetchPriority::High
        } else if self.0 < 0.5 {
            PrefetchPriority::Medium
        } else if self.0 < 0.7 {
            PrefetchPriority::Low
        } else {
            PrefetchPriority::None
        }
    }

    /// Combine distances (for path computation).
    pub fn combine(&self, other: SemanticDistance) -> SemanticDistance {
        // Use max to be conservative - the furthest determines overall distance
        SemanticDistance(self.0.max(other.0))
    }
}

impl Default for SemanticDistance {
    fn default() -> Self {
        Self::unrelated()
    }
}

/// Priority level for prefetch hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrefetchPriority {
    /// No prefetch needed
    None,
    /// Low priority - prefetch if bandwidth available
    Low,
    /// Medium priority - likely to be accessed
    Medium,
    /// High priority - very likely to be accessed soon
    High,
    /// Critical - almost certainly needed immediately
    Critical,
}

impl PrefetchPriority {
    /// Get the LLVM prefetch hint value (0=read, 1=write; locality 0-3).
    pub fn to_llvm_hint(&self) -> Option<(u8, u8)> {
        match self {
            PrefetchPriority::None => None,
            PrefetchPriority::Low => Some((0, 1)), // read, L3
            PrefetchPriority::Medium => Some((0, 2)), // read, L2
            PrefetchPriority::High => Some((0, 3)), // read, L1
            PrefetchPriority::Critical => Some((0, 3)), // read, L1 (keep)
        }
    }
}

/// A prefetch hint for a specific field access.
#[derive(Debug, Clone)]
pub struct PrefetchHint {
    /// The source type/field that triggers this prefetch
    pub source: String,
    /// The target type/field to prefetch
    pub target: String,
    /// Semantic distance between source and target
    pub distance: SemanticDistance,
    /// Priority of this prefetch
    pub priority: PrefetchPriority,
    /// Target locality level for the prefetched data
    pub target_locality: Locality,
    /// Stride for prefetching arrays (in elements)
    pub stride: Option<usize>,
    /// Reason for this prefetch (for diagnostics)
    pub reason: String,
}

impl PrefetchHint {
    /// Create a new prefetch hint.
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        distance: SemanticDistance,
    ) -> Self {
        let priority = distance.to_priority();
        Self {
            source: source.into(),
            target: target.into(),
            distance,
            priority,
            target_locality: Locality::L2,
            stride: None,
            reason: String::new(),
        }
    }

    /// Set the target locality.
    pub fn with_locality(mut self, locality: Locality) -> Self {
        self.target_locality = locality;
        self
    }

    /// Set the stride for array prefetching.
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = Some(stride);
        self
    }

    /// Set the reason.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = reason.into();
        self
    }
}

/// Entry in the prefetch table.
#[derive(Debug, Clone)]
pub struct PrefetchEntry {
    /// The type this entry applies to
    pub type_name: String,
    /// Field-level prefetch hints
    pub field_hints: HashMap<String, Vec<PrefetchHint>>,
    /// Type-level prefetch hints (for entire type access)
    pub type_hints: Vec<PrefetchHint>,
}

impl PrefetchEntry {
    /// Create a new prefetch entry.
    pub fn new(type_name: impl Into<String>) -> Self {
        Self {
            type_name: type_name.into(),
            field_hints: HashMap::new(),
            type_hints: Vec::new(),
        }
    }

    /// Add a field-level hint.
    pub fn add_field_hint(&mut self, field: impl Into<String>, hint: PrefetchHint) {
        self.field_hints.entry(field.into()).or_default().push(hint);
    }

    /// Add a type-level hint.
    pub fn add_type_hint(&mut self, hint: PrefetchHint) {
        self.type_hints.push(hint);
    }

    /// Get hints for a specific field.
    pub fn get_field_hints(&self, field: &str) -> &[PrefetchHint] {
        self.field_hints
            .get(field)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

/// The semantic prefetch table: maps types/fields to prefetch hints.
pub struct PrefetchTable {
    /// Entries indexed by type name
    entries: HashMap<String, PrefetchEntry>,
    /// Global statistics
    stats: PrefetchStats,
}

/// Statistics about prefetch table contents.
#[derive(Debug, Clone, Default)]
pub struct PrefetchStats {
    /// Total number of prefetch hints
    pub total_hints: usize,
    /// Number of high-priority hints
    pub high_priority: usize,
    /// Number of types with hints
    pub types_with_hints: usize,
    /// Average semantic distance
    pub avg_distance: f64,
}

impl PrefetchTable {
    /// Create an empty prefetch table.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            stats: PrefetchStats::default(),
        }
    }

    /// Build a prefetch table from an ontology.
    pub fn from_ontology(ontology: &dyn Ontology) -> Self {
        let mut table = Self::new();

        // Get concepts from the ontology
        let concepts = ontology.search("", 1000);

        // Build relationships
        for concept in &concepts {
            let id = concept.curie().to_string();
            let mut entry = PrefetchEntry::new(&id);

            // Add hints for parent concepts (is-a relationships)
            for parent in ontology.ancestors(&id) {
                let distance = Self::compute_distance(ontology, &id, &parent);
                if distance.is_prefetchable() {
                    let hint = PrefetchHint::new(&id, &parent, distance)
                        .with_reason("parent concept (is-a)");
                    entry.add_type_hint(hint);
                }
            }

            // Add hints for child concepts
            for child in ontology.descendants(&id) {
                let distance = Self::compute_distance(ontology, &id, &child);
                if distance.is_prefetchable() {
                    let hint =
                        PrefetchHint::new(&id, &child, distance).with_reason("child concept");
                    entry.add_type_hint(hint);
                }
            }

            if !entry.type_hints.is_empty() || !entry.field_hints.is_empty() {
                table.entries.insert(id, entry);
            }
        }

        table.update_stats();
        table
    }

    /// Compute semantic distance between two concepts.
    fn compute_distance(ontology: &dyn Ontology, from: &str, to: &str) -> SemanticDistance {
        if from == to {
            return SemanticDistance::identical();
        }

        // Check if direct parent/child
        let ancestors = ontology.ancestors(from);
        if ancestors.iter().any(|a| a == to) {
            // Direct ancestor - compute based on depth
            let depth = ancestors.iter().position(|a| a == to).unwrap_or(0) + 1;
            return SemanticDistance::new(0.1 * depth as f64);
        }

        let descendants = ontology.descendants(from);
        if descendants.iter().any(|d| d == to) {
            let depth = descendants.iter().position(|d| d == to).unwrap_or(0) + 1;
            return SemanticDistance::new(0.1 * depth as f64);
        }

        // Check for common ancestor (siblings)
        for ancestor in &ancestors {
            let to_ancestors = ontology.ancestors(to);
            if to_ancestors.contains(ancestor) {
                return SemanticDistance::new(0.3);
            }
        }

        // No direct relationship found
        SemanticDistance::unrelated()
    }

    /// Add a prefetch entry.
    pub fn add_entry(&mut self, entry: PrefetchEntry) {
        self.entries.insert(entry.type_name.clone(), entry);
        self.update_stats();
    }

    /// Get prefetch hints for a type/field access.
    pub fn get_hints(&self, type_name: &str, field: &str) -> Vec<PrefetchHint> {
        let mut hints = Vec::new();

        if let Some(entry) = self.entries.get(type_name) {
            // Add field-specific hints
            hints.extend(entry.get_field_hints(field).iter().cloned());

            // Add type-level hints
            hints.extend(entry.type_hints.iter().cloned());
        }

        // Sort by priority (highest first)
        hints.sort_by(|a, b| b.priority.cmp(&a.priority));
        hints
    }

    /// Get all hints for a type.
    pub fn get_type_hints(&self, type_name: &str) -> Vec<PrefetchHint> {
        self.entries
            .get(type_name)
            .map(|e| e.type_hints.clone())
            .unwrap_or_default()
    }

    /// Check if a type has any prefetch hints.
    pub fn has_hints(&self, type_name: &str) -> bool {
        self.entries.contains_key(type_name)
    }

    /// Get statistics about the table.
    pub fn stats(&self) -> &PrefetchStats {
        &self.stats
    }

    /// Update statistics.
    fn update_stats(&mut self) {
        let mut total_hints = 0;
        let mut high_priority = 0;
        let mut total_distance = 0.0;

        for entry in self.entries.values() {
            let hints: Vec<_> = entry
                .type_hints
                .iter()
                .chain(entry.field_hints.values().flatten())
                .collect();

            total_hints += hints.len();

            for hint in hints {
                if hint.priority >= PrefetchPriority::High {
                    high_priority += 1;
                }
                total_distance += hint.distance.value();
            }
        }

        self.stats = PrefetchStats {
            total_hints,
            high_priority,
            types_with_hints: self.entries.len(),
            avg_distance: if total_hints > 0 {
                total_distance / total_hints as f64
            } else {
                0.0
            },
        };
    }

    /// Merge another table into this one.
    pub fn merge(&mut self, other: PrefetchTable) {
        for (name, entry) in other.entries {
            if let Some(existing) = self.entries.get_mut(&name) {
                existing.type_hints.extend(entry.type_hints);
                for (field, hints) in entry.field_hints {
                    existing.field_hints.entry(field).or_default().extend(hints);
                }
            } else {
                self.entries.insert(name, entry);
            }
        }
        self.update_stats();
    }

    /// Get all entries.
    pub fn entries(&self) -> impl Iterator<Item = &PrefetchEntry> {
        self.entries.values()
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for PrefetchTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_distance() {
        let identical = SemanticDistance::identical();
        assert_eq!(identical.value(), 0.0);
        assert!(identical.is_prefetchable());
        assert_eq!(identical.to_priority(), PrefetchPriority::Critical);

        let related = SemanticDistance::related();
        assert!(related.is_prefetchable());
        assert_eq!(related.to_priority(), PrefetchPriority::High);

        let unrelated = SemanticDistance::unrelated();
        assert!(!unrelated.is_prefetchable());
        assert_eq!(unrelated.to_priority(), PrefetchPriority::None);
    }

    #[test]
    fn test_distance_clamping() {
        let over = SemanticDistance::new(1.5);
        assert_eq!(over.value(), 1.0);

        let under = SemanticDistance::new(-0.5);
        assert_eq!(under.value(), 0.0);
    }

    #[test]
    fn test_distance_combine() {
        let d1 = SemanticDistance::new(0.2);
        let d2 = SemanticDistance::new(0.5);
        let combined = d1.combine(d2);
        assert_eq!(combined.value(), 0.5);
    }

    #[test]
    fn test_prefetch_priority_llvm() {
        assert_eq!(PrefetchPriority::None.to_llvm_hint(), None);
        assert_eq!(PrefetchPriority::Low.to_llvm_hint(), Some((0, 1)));
        assert_eq!(PrefetchPriority::High.to_llvm_hint(), Some((0, 3)));
    }

    #[test]
    fn test_prefetch_hint_creation() {
        let hint = PrefetchHint::new("Patient", "Diagnosis", SemanticDistance::related())
            .with_locality(Locality::L1)
            .with_stride(8)
            .with_reason("commonly accessed together");

        assert_eq!(hint.source, "Patient");
        assert_eq!(hint.target, "Diagnosis");
        assert_eq!(hint.priority, PrefetchPriority::High);
        assert_eq!(hint.target_locality, Locality::L1);
        assert_eq!(hint.stride, Some(8));
    }

    #[test]
    fn test_prefetch_entry() {
        let mut entry = PrefetchEntry::new("Patient");

        let hint1 = PrefetchHint::new(
            "Patient.diagnosis",
            "Diagnosis",
            SemanticDistance::related(),
        );
        entry.add_field_hint("diagnosis", hint1);

        let hint2 = PrefetchHint::new("Patient", "Person", SemanticDistance::new(0.1));
        entry.add_type_hint(hint2);

        assert_eq!(entry.get_field_hints("diagnosis").len(), 1);
        assert_eq!(entry.get_field_hints("unknown").len(), 0);
        assert_eq!(entry.type_hints.len(), 1);
    }

    #[test]
    fn test_prefetch_table() {
        let mut table = PrefetchTable::new();

        let mut entry = PrefetchEntry::new("Patient");
        entry.add_type_hint(PrefetchHint::new(
            "Patient",
            "Person",
            SemanticDistance::related(),
        ));
        table.add_entry(entry);

        assert!(table.has_hints("Patient"));
        assert!(!table.has_hints("Unknown"));

        let hints = table.get_hints("Patient", "any");
        assert_eq!(hints.len(), 1);
    }

    #[test]
    fn test_table_stats() {
        let mut table = PrefetchTable::new();

        let mut entry1 = PrefetchEntry::new("Type1");
        entry1.add_type_hint(PrefetchHint::new(
            "Type1",
            "Type2",
            SemanticDistance::new(0.1),
        ));
        entry1.add_type_hint(PrefetchHint::new(
            "Type1",
            "Type3",
            SemanticDistance::new(0.5),
        ));
        table.add_entry(entry1);

        let stats = table.stats();
        assert_eq!(stats.total_hints, 2);
        assert_eq!(stats.high_priority, 1); // 0.1 is high priority
        assert_eq!(stats.types_with_hints, 1);
    }

    #[test]
    fn test_table_merge() {
        let mut table1 = PrefetchTable::new();
        let mut entry1 = PrefetchEntry::new("Type1");
        entry1.add_type_hint(PrefetchHint::new(
            "Type1",
            "Type2",
            SemanticDistance::related(),
        ));
        table1.add_entry(entry1);

        let mut table2 = PrefetchTable::new();
        let mut entry2 = PrefetchEntry::new("Type1");
        entry2.add_type_hint(PrefetchHint::new(
            "Type1",
            "Type3",
            SemanticDistance::related(),
        ));
        table2.add_entry(entry2);

        let mut entry3 = PrefetchEntry::new("Type2");
        entry3.add_type_hint(PrefetchHint::new(
            "Type2",
            "Type1",
            SemanticDistance::related(),
        ));
        table2.add_entry(entry3);

        table1.merge(table2);

        assert_eq!(table1.len(), 2);
        assert_eq!(table1.get_type_hints("Type1").len(), 2);
    }
}
