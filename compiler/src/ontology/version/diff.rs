//! Ontology Diffing
//!
//! This module computes differences between ontology versions to detect:
//! - Added terms
//! - Removed terms
//! - Modified terms (label, definition, hierarchy changes)
//! - Deprecated terms
//! - Breaking vs non-breaking changes
//!
//! # Breaking Changes
//!
//! A breaking change is one that could cause existing code to fail:
//! - Removing a term entirely
//! - Changing a term's type/category
//! - Removing a superclass relationship
//!
//! Non-breaking changes:
//! - Adding new terms
//! - Adding synonyms
//! - Adding new superclass relationships
//! - Deprecation (with replacement)
//!
//! # Usage
//!
//! ```rust,ignore
//! use sounio::ontology::version::{OntologyDiff, OntologySnapshot};
//!
//! let old = OntologySnapshot::from_terms(old_terms);
//! let new = OntologySnapshot::from_terms(new_terms);
//!
//! let diff = OntologyDiff::compute(&old, &new);
//!
//! if diff.has_breaking_changes() {
//!     println!("Warning: {} breaking changes detected", diff.breaking_count());
//!     for change in diff.breaking_changes() {
//!         println!("  - {}", change);
//!     }
//! }
//! ```

use std::collections::{HashMap, HashSet};

/// A snapshot of an ontology at a specific version
#[derive(Debug, Clone)]
pub struct OntologySnapshot {
    /// Ontology ID
    pub id: String,
    /// Version string
    pub version: String,
    /// Terms in this version
    pub terms: HashMap<String, SnapshotTerm>,
}

/// A term in the snapshot
#[derive(Debug, Clone)]
pub struct SnapshotTerm {
    /// Term ID (CURIE)
    pub id: String,
    /// Label/name
    pub label: Option<String>,
    /// Definition
    pub definition: Option<String>,
    /// Superclass IDs
    pub superclasses: Vec<String>,
    /// Synonyms
    pub synonyms: Vec<String>,
    /// Is obsolete/deprecated
    pub obsolete: bool,
    /// Replaced by (if obsolete)
    pub replaced_by: Option<String>,
}

impl OntologySnapshot {
    /// Create a new snapshot
    pub fn new(id: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            version: version.into(),
            terms: HashMap::new(),
        }
    }

    /// Add a term to the snapshot
    pub fn add_term(&mut self, term: SnapshotTerm) {
        self.terms.insert(term.id.clone(), term);
    }

    /// Get a term by ID
    pub fn get_term(&self, id: &str) -> Option<&SnapshotTerm> {
        self.terms.get(id)
    }

    /// Get all term IDs
    pub fn term_ids(&self) -> impl Iterator<Item = &str> {
        self.terms.keys().map(|s| s.as_str())
    }

    /// Number of terms
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
}

/// Difference between two ontology versions
#[derive(Debug, Clone)]
pub struct OntologyDiff {
    /// Source ontology ID
    pub ontology_id: String,
    /// Old version
    pub old_version: String,
    /// New version
    pub new_version: String,
    /// All changes
    pub changes: Vec<OntologyChange>,
    /// Summary statistics
    pub stats: DiffStats,
}

/// Statistics about the diff
#[derive(Debug, Clone, Default)]
pub struct DiffStats {
    /// Number of terms added
    pub added: usize,
    /// Number of terms removed
    pub removed: usize,
    /// Number of terms modified
    pub modified: usize,
    /// Number of terms deprecated
    pub deprecated: usize,
    /// Number of breaking changes
    pub breaking: usize,
}

/// A single change in the ontology
#[derive(Debug, Clone)]
pub struct OntologyChange {
    /// Term ID affected
    pub term_id: String,
    /// Type of change
    pub kind: ChangeKind,
    /// Detailed changes (for modifications)
    pub details: Vec<TermChange>,
    /// Is this a breaking change?
    pub breaking: bool,
    /// Human-readable description
    pub description: String,
}

/// Kind of ontology change
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeKind {
    /// Term was added
    Added,
    /// Term was removed
    Removed,
    /// Term was modified
    Modified,
    /// Term was deprecated
    Deprecated,
    /// Term was un-deprecated (rare)
    Undeprecated,
}

impl std::fmt::Display for ChangeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChangeKind::Added => write!(f, "added"),
            ChangeKind::Removed => write!(f, "removed"),
            ChangeKind::Modified => write!(f, "modified"),
            ChangeKind::Deprecated => write!(f, "deprecated"),
            ChangeKind::Undeprecated => write!(f, "undeprecated"),
        }
    }
}

/// Detailed change to a term
#[derive(Debug, Clone)]
pub enum TermChange {
    /// Label changed
    LabelChanged {
        old: Option<String>,
        new: Option<String>,
    },
    /// Definition changed
    DefinitionChanged {
        old: Option<String>,
        new: Option<String>,
    },
    /// Superclass added
    SuperclassAdded(String),
    /// Superclass removed
    SuperclassRemoved(String),
    /// Synonym added
    SynonymAdded(String),
    /// Synonym removed
    SynonymRemoved(String),
    /// Replacement specified
    ReplacedBy(String),
}

impl std::fmt::Display for TermChange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TermChange::LabelChanged { old, new } => {
                write!(
                    f,
                    "label: {:?} -> {:?}",
                    old.as_deref().unwrap_or("(none)"),
                    new.as_deref().unwrap_or("(none)")
                )
            }
            TermChange::DefinitionChanged { .. } => write!(f, "definition changed"),
            TermChange::SuperclassAdded(id) => write!(f, "superclass added: {}", id),
            TermChange::SuperclassRemoved(id) => write!(f, "superclass removed: {}", id),
            TermChange::SynonymAdded(s) => write!(f, "synonym added: {}", s),
            TermChange::SynonymRemoved(s) => write!(f, "synonym removed: {}", s),
            TermChange::ReplacedBy(id) => write!(f, "replaced by: {}", id),
        }
    }
}

impl OntologyDiff {
    /// Compute the difference between two snapshots
    pub fn compute(old: &OntologySnapshot, new: &OntologySnapshot) -> Self {
        let mut changes = Vec::new();
        let mut stats = DiffStats::default();

        let old_ids: HashSet<&str> = old.term_ids().collect();
        let new_ids: HashSet<&str> = new.term_ids().collect();

        // Find added terms
        for id in new_ids.difference(&old_ids) {
            let term = new.get_term(id).unwrap();
            changes.push(OntologyChange {
                term_id: id.to_string(),
                kind: ChangeKind::Added,
                details: vec![],
                breaking: false, // Adding is never breaking
                description: format!(
                    "Added: {} ({})",
                    id,
                    term.label.as_deref().unwrap_or("no label")
                ),
            });
            stats.added += 1;
        }

        // Find removed terms
        for id in old_ids.difference(&new_ids) {
            let term = old.get_term(id).unwrap();
            changes.push(OntologyChange {
                term_id: id.to_string(),
                kind: ChangeKind::Removed,
                details: vec![],
                breaking: true, // Removing is breaking
                description: format!(
                    "Removed: {} ({})",
                    id,
                    term.label.as_deref().unwrap_or("no label")
                ),
            });
            stats.removed += 1;
            stats.breaking += 1;
        }

        // Find modified terms
        for id in old_ids.intersection(&new_ids) {
            let old_term = old.get_term(id).unwrap();
            let new_term = new.get_term(id).unwrap();

            let (details, is_breaking) = Self::diff_terms(old_term, new_term);

            // Check for deprecation status change (even if no other details)
            let obsolete_changed = old_term.obsolete != new_term.obsolete;

            if !details.is_empty() || obsolete_changed {
                // Check for deprecation change
                let kind = if !old_term.obsolete && new_term.obsolete {
                    stats.deprecated += 1;
                    ChangeKind::Deprecated
                } else if old_term.obsolete && !new_term.obsolete {
                    ChangeKind::Undeprecated
                } else {
                    stats.modified += 1;
                    ChangeKind::Modified
                };

                if is_breaking {
                    stats.breaking += 1;
                }

                let description = if kind == ChangeKind::Deprecated {
                    if let Some(replacement) = &new_term.replaced_by {
                        format!("Deprecated: {} (use {} instead)", id, replacement)
                    } else {
                        format!("Deprecated: {}", id)
                    }
                } else {
                    format!("Modified: {} ({} changes)", id, details.len())
                };

                changes.push(OntologyChange {
                    term_id: id.to_string(),
                    kind,
                    details,
                    breaking: is_breaking,
                    description,
                });
            }
        }

        // Sort changes: breaking first, then by kind, then by ID
        changes.sort_by(|a, b| {
            b.breaking
                .cmp(&a.breaking)
                .then_with(|| (a.kind as u8).cmp(&(b.kind as u8)))
                .then_with(|| a.term_id.cmp(&b.term_id))
        });

        OntologyDiff {
            ontology_id: new.id.clone(),
            old_version: old.version.clone(),
            new_version: new.version.clone(),
            changes,
            stats,
        }
    }

    /// Diff two terms, returning (changes, is_breaking)
    fn diff_terms(old: &SnapshotTerm, new: &SnapshotTerm) -> (Vec<TermChange>, bool) {
        let mut changes = Vec::new();
        let mut breaking = false;

        // Label change (not breaking)
        if old.label != new.label {
            changes.push(TermChange::LabelChanged {
                old: old.label.clone(),
                new: new.label.clone(),
            });
        }

        // Definition change (not breaking)
        if old.definition != new.definition {
            changes.push(TermChange::DefinitionChanged {
                old: old.definition.clone(),
                new: new.definition.clone(),
            });
        }

        // Superclass changes
        let old_supers: HashSet<&str> = old.superclasses.iter().map(|s| s.as_str()).collect();
        let new_supers: HashSet<&str> = new.superclasses.iter().map(|s| s.as_str()).collect();

        for added in new_supers.difference(&old_supers) {
            changes.push(TermChange::SuperclassAdded(added.to_string()));
        }

        for removed in old_supers.difference(&new_supers) {
            changes.push(TermChange::SuperclassRemoved(removed.to_string()));
            breaking = true; // Removing superclass is breaking
        }

        // Synonym changes (not breaking)
        let old_syns: HashSet<&str> = old.synonyms.iter().map(|s| s.as_str()).collect();
        let new_syns: HashSet<&str> = new.synonyms.iter().map(|s| s.as_str()).collect();

        for added in new_syns.difference(&old_syns) {
            changes.push(TermChange::SynonymAdded(added.to_string()));
        }

        for removed in old_syns.difference(&new_syns) {
            changes.push(TermChange::SynonymRemoved(removed.to_string()));
        }

        // Deprecation - always record as a change when obsolete status changes
        if !old.obsolete && new.obsolete {
            if let Some(replacement) = &new.replaced_by {
                changes.push(TermChange::ReplacedBy(replacement.clone()));
            } else {
                // No replacement - this is a breaking deprecation
                // We still need to record something to indicate the change
                breaking = true;
            }
        }

        (changes, breaking)
    }

    /// Check if there are any breaking changes
    pub fn has_breaking_changes(&self) -> bool {
        self.stats.breaking > 0
    }

    /// Get number of breaking changes
    pub fn breaking_count(&self) -> usize {
        self.stats.breaking
    }

    /// Get iterator over breaking changes
    pub fn breaking_changes(&self) -> impl Iterator<Item = &OntologyChange> {
        self.changes.iter().filter(|c| c.breaking)
    }

    /// Get iterator over all changes
    pub fn all_changes(&self) -> impl Iterator<Item = &OntologyChange> {
        self.changes.iter()
    }

    /// Get added terms
    pub fn added(&self) -> impl Iterator<Item = &OntologyChange> {
        self.changes.iter().filter(|c| c.kind == ChangeKind::Added)
    }

    /// Get removed terms
    pub fn removed(&self) -> impl Iterator<Item = &OntologyChange> {
        self.changes
            .iter()
            .filter(|c| c.kind == ChangeKind::Removed)
    }

    /// Get deprecated terms
    pub fn deprecated(&self) -> impl Iterator<Item = &OntologyChange> {
        self.changes
            .iter()
            .filter(|c| c.kind == ChangeKind::Deprecated)
    }

    /// Generate a summary report
    pub fn summary(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "Ontology Diff: {} {} -> {}\n",
            self.ontology_id, self.old_version, self.new_version
        ));
        output.push_str(&format!(
            "  Added: {}, Removed: {}, Modified: {}, Deprecated: {}\n",
            self.stats.added, self.stats.removed, self.stats.modified, self.stats.deprecated
        ));

        if self.has_breaking_changes() {
            output.push_str(&format!("  ⚠️  {} BREAKING CHANGES\n", self.stats.breaking));
        }

        output
    }

    /// Generate a detailed report
    pub fn detailed_report(&self) -> String {
        let mut output = self.summary();
        output.push('\n');

        if self.has_breaking_changes() {
            output.push_str("Breaking Changes:\n");
            for change in self.breaking_changes() {
                output.push_str(&format!("  - {}\n", change.description));
                for detail in &change.details {
                    output.push_str(&format!("      {}\n", detail));
                }
            }
            output.push('\n');
        }

        if self.stats.deprecated > 0 {
            output.push_str("Deprecations:\n");
            for change in self.deprecated() {
                output.push_str(&format!("  - {}\n", change.description));
            }
            output.push('\n');
        }

        if self.stats.added > 0 {
            output.push_str(&format!("Added ({}):\n", self.stats.added));
            for change in self.added().take(10) {
                output.push_str(&format!("  + {}\n", change.term_id));
            }
            if self.stats.added > 10 {
                output.push_str(&format!("  ... and {} more\n", self.stats.added - 10));
            }
            output.push('\n');
        }

        output
    }

    /// Check if the diff is empty (no changes)
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    /// Get terms that can be safely migrated (deprecated with replacement)
    pub fn migratable(&self) -> Vec<(&str, &str)> {
        self.changes
            .iter()
            .filter(|c| c.kind == ChangeKind::Deprecated)
            .filter_map(|c| {
                c.details.iter().find_map(|d| {
                    if let TermChange::ReplacedBy(replacement) = d {
                        Some((c.term_id.as_str(), replacement.as_str()))
                    } else {
                        None
                    }
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_term(id: &str, label: &str) -> SnapshotTerm {
        SnapshotTerm {
            id: id.to_string(),
            label: Some(label.to_string()),
            definition: None,
            superclasses: vec![],
            synonyms: vec![],
            obsolete: false,
            replaced_by: None,
        }
    }

    #[test]
    fn test_diff_added_term() {
        let old = OntologySnapshot::new("test", "1.0.0");
        let mut new = OntologySnapshot::new("test", "2.0.0");
        new.add_term(make_term("TEST:001", "New Term"));

        let diff = OntologyDiff::compute(&old, &new);

        assert_eq!(diff.stats.added, 1);
        assert!(!diff.has_breaking_changes());
    }

    #[test]
    fn test_diff_removed_term() {
        let mut old = OntologySnapshot::new("test", "1.0.0");
        old.add_term(make_term("TEST:001", "Old Term"));
        let new = OntologySnapshot::new("test", "2.0.0");

        let diff = OntologyDiff::compute(&old, &new);

        assert_eq!(diff.stats.removed, 1);
        assert!(diff.has_breaking_changes());
    }

    #[test]
    fn test_diff_modified_label() {
        let mut old = OntologySnapshot::new("test", "1.0.0");
        old.add_term(make_term("TEST:001", "Old Label"));

        let mut new = OntologySnapshot::new("test", "2.0.0");
        new.add_term(make_term("TEST:001", "New Label"));

        let diff = OntologyDiff::compute(&old, &new);

        assert_eq!(diff.stats.modified, 1);
        assert!(!diff.has_breaking_changes()); // Label change is not breaking
    }

    #[test]
    fn test_diff_superclass_removed() {
        let mut old = OntologySnapshot::new("test", "1.0.0");
        let mut old_term = make_term("TEST:001", "Term");
        old_term.superclasses = vec!["TEST:000".to_string()];
        old.add_term(old_term);

        let mut new = OntologySnapshot::new("test", "2.0.0");
        new.add_term(make_term("TEST:001", "Term"));

        let diff = OntologyDiff::compute(&old, &new);

        assert!(diff.has_breaking_changes()); // Removing superclass is breaking
    }

    #[test]
    fn test_diff_deprecated_with_replacement() {
        let mut old = OntologySnapshot::new("test", "1.0.0");
        old.add_term(make_term("TEST:001", "Old Term"));
        old.add_term(make_term("TEST:002", "New Term"));

        let mut new = OntologySnapshot::new("test", "2.0.0");
        let mut deprecated = make_term("TEST:001", "Old Term");
        deprecated.obsolete = true;
        deprecated.replaced_by = Some("TEST:002".to_string());
        new.add_term(deprecated);
        new.add_term(make_term("TEST:002", "New Term"));

        let diff = OntologyDiff::compute(&old, &new);

        assert_eq!(diff.stats.deprecated, 1);
        // Deprecation with replacement is not hard-breaking
        assert!(!diff.has_breaking_changes());

        let migratable = diff.migratable();
        assert_eq!(migratable.len(), 1);
        assert_eq!(migratable[0], ("TEST:001", "TEST:002"));
    }

    #[test]
    fn test_diff_deprecated_without_replacement() {
        let mut old = OntologySnapshot::new("test", "1.0.0");
        old.add_term(make_term("TEST:001", "Old Term"));

        let mut new = OntologySnapshot::new("test", "2.0.0");
        let mut deprecated = make_term("TEST:001", "Old Term");
        deprecated.obsolete = true;
        new.add_term(deprecated);

        let diff = OntologyDiff::compute(&old, &new);

        assert_eq!(diff.stats.deprecated, 1);
        // Deprecation without replacement IS breaking
        assert!(diff.has_breaking_changes());
    }

    #[test]
    fn test_diff_summary() {
        let mut old = OntologySnapshot::new("CHEBI", "2024-01-01");
        old.add_term(make_term("CHEBI:001", "Removed"));
        old.add_term(make_term("CHEBI:002", "Modified"));

        let mut new = OntologySnapshot::new("CHEBI", "2024-06-01");
        new.add_term(make_term("CHEBI:002", "Modified (new label)"));
        new.add_term(make_term("CHEBI:003", "Added"));

        let diff = OntologyDiff::compute(&old, &new);
        let summary = diff.summary();

        assert!(summary.contains("CHEBI"));
        assert!(summary.contains("2024-01-01"));
        assert!(summary.contains("2024-06-01"));
        assert!(summary.contains("BREAKING"));
    }

    #[test]
    fn test_empty_diff() {
        let mut old = OntologySnapshot::new("test", "1.0.0");
        old.add_term(make_term("TEST:001", "Same"));

        let mut new = OntologySnapshot::new("test", "1.0.1");
        new.add_term(make_term("TEST:001", "Same"));

        let diff = OntologyDiff::compute(&old, &new);

        assert!(diff.is_empty());
    }
}
