//! Deprecation Tracking
//!
//! This module tracks deprecated ontology terms and provides warnings
//! when they are used in code. It supports:
//!
//! - Tracking deprecated terms across ontology versions
//! - Providing replacement suggestions
//! - Generating compiler warnings
//! - Automatic migration suggestions
//!
//! # Deprecation Levels
//!
//! - `Warning`: Term is deprecated but still works
//! - `Error`: Term is removed and will cause compilation to fail
//! - `Silent`: Term is deprecated but don't warn (for internal use)
//!
//! # Usage
//!
//! ```rust,ignore
//! use sounio::ontology::version::{DeprecationTracker, DeprecationLevel};
//!
//! let mut tracker = DeprecationTracker::new();
//!
//! // Register a deprecated term
//! tracker.add_deprecated(DeprecatedTerm {
//!     id: "CHEBI:12345".to_string(),
//!     replacement: Some(Replacement::single("CHEBI:67890")),
//!     level: DeprecationLevel::Warning,
//!     message: Some("Use CHEBI:67890 instead".to_string()),
//!     since_version: Some("2024-01-01".to_string()),
//! });
//!
//! // Check if a term is deprecated
//! if let Some(warning) = tracker.check("CHEBI:12345") {
//!     eprintln!("{}", warning);
//! }
//! ```

use std::collections::HashMap;

use crate::common::Span;

/// Level of deprecation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DeprecationLevel {
    /// Just a warning, code still works
    #[default]
    Warning,
    /// Error, code won't compile
    Error,
    /// Silent deprecation (internal)
    Silent,
}

impl std::fmt::Display for DeprecationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeprecationLevel::Warning => write!(f, "warning"),
            DeprecationLevel::Error => write!(f, "error"),
            DeprecationLevel::Silent => write!(f, "silent"),
        }
    }
}

/// A deprecated term entry
#[derive(Debug, Clone)]
pub struct DeprecatedTerm {
    /// The deprecated term ID (CURIE)
    pub id: String,
    /// Replacement suggestion(s)
    pub replacement: Option<Replacement>,
    /// Deprecation level
    pub level: DeprecationLevel,
    /// Custom deprecation message
    pub message: Option<String>,
    /// Version when deprecated
    pub since_version: Option<String>,
    /// Expected removal version
    pub removal_version: Option<String>,
    /// Reason for deprecation
    pub reason: Option<String>,
}

/// Replacement suggestion for a deprecated term
#[derive(Debug, Clone)]
pub enum Replacement {
    /// Single replacement term
    Single(String),
    /// Multiple alternative replacements (user chooses)
    Alternatives(Vec<String>),
    /// No direct replacement exists
    None { reason: String },
    /// Migration requires code changes
    Migration { description: String },
}

impl Replacement {
    /// Create a single replacement
    pub fn single(id: impl Into<String>) -> Self {
        Replacement::Single(id.into())
    }

    /// Create alternatives
    pub fn alternatives(ids: Vec<String>) -> Self {
        Replacement::Alternatives(ids)
    }

    /// No replacement available
    pub fn none(reason: impl Into<String>) -> Self {
        Replacement::None {
            reason: reason.into(),
        }
    }

    /// Get the primary replacement if available
    pub fn primary(&self) -> Option<&str> {
        match self {
            Replacement::Single(id) => Some(id),
            Replacement::Alternatives(ids) => ids.first().map(|s| s.as_str()),
            _ => None,
        }
    }

    /// Get all replacement options
    pub fn all(&self) -> Vec<&str> {
        match self {
            Replacement::Single(id) => vec![id.as_str()],
            Replacement::Alternatives(ids) => ids.iter().map(|s| s.as_str()).collect(),
            _ => vec![],
        }
    }
}

/// A deprecation warning generated during compilation
#[derive(Debug, Clone)]
pub struct DeprecationWarning {
    /// The deprecated term ID
    pub term_id: String,
    /// Where it was used
    pub span: Option<Span>,
    /// The deprecation level
    pub level: DeprecationLevel,
    /// Human-readable message
    pub message: String,
    /// Suggested replacement
    pub suggestion: Option<String>,
    /// Help text
    pub help: Option<String>,
}

impl DeprecationWarning {
    /// Format as a compiler diagnostic
    pub fn format(&self) -> String {
        let mut output = String::new();

        let level_str = match self.level {
            DeprecationLevel::Warning => "warning",
            DeprecationLevel::Error => "error",
            DeprecationLevel::Silent => return String::new(),
        };

        output.push_str(&format!(
            "{}[{}]: {}\n",
            level_str, self.term_id, self.message
        ));

        if let Some(span) = self.span {
            output.push_str(&format!(
                "  --> at byte offset {}..{}\n",
                span.start, span.end
            ));
        }

        if let Some(suggestion) = &self.suggestion {
            output.push_str(&format!("  = suggestion: {}\n", suggestion));
        }

        if let Some(help) = &self.help {
            output.push_str(&format!("  = help: {}\n", help));
        }

        output
    }

    /// Format for machine-readable output (JSON-like)
    pub fn to_json(&self) -> String {
        let level = match self.level {
            DeprecationLevel::Warning => "warning",
            DeprecationLevel::Error => "error",
            DeprecationLevel::Silent => "silent",
        };

        format!(
            r#"{{"level":"{}","term":"{}","message":"{}","suggestion":{}}}"#,
            level,
            self.term_id,
            escape_json(&self.message),
            self.suggestion
                .as_ref()
                .map(|s| format!("\"{}\"", escape_json(s)))
                .unwrap_or_else(|| "null".to_string())
        )
    }
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// Tracks deprecated terms and generates warnings
#[derive(Debug, Clone)]
pub struct DeprecationTracker {
    /// Deprecated terms by ID
    deprecated: HashMap<String, DeprecatedTerm>,
    /// Generated warnings
    warnings: Vec<DeprecationWarning>,
    /// Whether to treat warnings as errors
    warnings_as_errors: bool,
    /// Suppressed term IDs
    suppressed: std::collections::HashSet<String>,
}

impl DeprecationTracker {
    /// Create a new tracker
    pub fn new() -> Self {
        Self {
            deprecated: HashMap::new(),
            warnings: Vec::new(),
            warnings_as_errors: false,
            suppressed: std::collections::HashSet::new(),
        }
    }

    /// Treat all deprecation warnings as errors
    pub fn warnings_as_errors(mut self, enable: bool) -> Self {
        self.warnings_as_errors = enable;
        self
    }

    /// Add a deprecated term
    pub fn add_deprecated(&mut self, term: DeprecatedTerm) {
        self.deprecated.insert(term.id.clone(), term);
    }

    /// Add multiple deprecated terms
    pub fn add_all(&mut self, terms: impl IntoIterator<Item = DeprecatedTerm>) {
        for term in terms {
            self.add_deprecated(term);
        }
    }

    /// Suppress warnings for a specific term
    pub fn suppress(&mut self, id: impl Into<String>) {
        self.suppressed.insert(id.into());
    }

    /// Check if a term is deprecated
    pub fn is_deprecated(&self, id: &str) -> bool {
        self.deprecated.contains_key(id)
    }

    /// Get deprecation info for a term
    pub fn get_deprecated(&self, id: &str) -> Option<&DeprecatedTerm> {
        self.deprecated.get(id)
    }

    /// Check a term and generate a warning if deprecated
    pub fn check(&mut self, id: &str, span: Option<Span>) -> Option<DeprecationWarning> {
        // Check if suppressed
        if self.suppressed.contains(id) {
            return None;
        }

        let term = self.deprecated.get(id)?;

        // Skip silent deprecations
        if term.level == DeprecationLevel::Silent {
            return None;
        }

        let level = if self.warnings_as_errors {
            DeprecationLevel::Error
        } else {
            term.level
        };

        let message = term
            .message
            .clone()
            .unwrap_or_else(|| format!("Term '{}' is deprecated", id));

        let suggestion = term.replacement.as_ref().map(|r| match r {
            Replacement::Single(replacement) => format!("use '{}' instead", replacement),
            Replacement::Alternatives(alts) => format!("use one of: {}", alts.join(", ")),
            Replacement::None { reason } => format!("no replacement: {}", reason),
            Replacement::Migration { description } => description.clone(),
        });

        let help = term.since_version.as_ref().map(|v| {
            if let Some(removal) = &term.removal_version {
                format!("deprecated since {}, will be removed in {}", v, removal)
            } else {
                format!("deprecated since {}", v)
            }
        });

        let warning = DeprecationWarning {
            term_id: id.to_string(),
            span,
            level,
            message,
            suggestion,
            help,
        };

        self.warnings.push(warning.clone());
        Some(warning)
    }

    /// Check multiple terms
    pub fn check_all<'a>(
        &mut self,
        terms: impl IntoIterator<Item = (&'a str, Option<Span>)>,
    ) -> Vec<DeprecationWarning> {
        terms
            .into_iter()
            .filter_map(|(id, span)| self.check(id, span))
            .collect()
    }

    /// Get all generated warnings
    pub fn warnings(&self) -> &[DeprecationWarning] {
        &self.warnings
    }

    /// Get warnings as errors (for build failure)
    pub fn errors(&self) -> impl Iterator<Item = &DeprecationWarning> {
        self.warnings
            .iter()
            .filter(|w| w.level == DeprecationLevel::Error)
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        self.warnings
            .iter()
            .any(|w| w.level == DeprecationLevel::Error)
    }

    /// Clear all warnings
    pub fn clear_warnings(&mut self) {
        self.warnings.clear();
    }

    /// Get all deprecated terms
    pub fn all_deprecated(&self) -> impl Iterator<Item = &DeprecatedTerm> {
        self.deprecated.values()
    }

    /// Get number of deprecated terms tracked
    pub fn deprecated_count(&self) -> usize {
        self.deprecated.len()
    }

    /// Generate a migration guide
    pub fn migration_guide(&self) -> String {
        let mut output = String::new();
        output.push_str("# Migration Guide\n\n");

        let mut by_replacement: HashMap<Option<&str>, Vec<&DeprecatedTerm>> = HashMap::new();

        for term in self.deprecated.values() {
            let key = term.replacement.as_ref().and_then(|r| r.primary());
            by_replacement.entry(key).or_default().push(term);
        }

        // Terms with simple replacements
        output.push_str("## Simple Replacements\n\n");
        output.push_str("| Deprecated | Replacement | Notes |\n");
        output.push_str("|------------|-------------|-------|\n");

        for (replacement, terms) in &by_replacement {
            if let Some(repl) = replacement {
                for term in terms {
                    let notes = term.reason.as_deref().unwrap_or("-");
                    output.push_str(&format!("| {} | {} | {} |\n", term.id, repl, notes));
                }
            }
        }
        output.push('\n');

        // Terms without replacements
        if let Some(no_replacement) = by_replacement.get(&None)
            && !no_replacement.is_empty()
        {
            output.push_str("## No Direct Replacement\n\n");
            for term in no_replacement {
                output.push_str(&format!("- **{}**", term.id));
                if let Some(reason) = &term.reason {
                    output.push_str(&format!(": {}", reason));
                }
                output.push('\n');
            }
        }

        output
    }

    /// Load deprecations from ontology diff
    pub fn from_diff(diff: &super::diff::OntologyDiff) -> Self {
        let mut tracker = Self::new();

        for change in diff.deprecated() {
            let replacement = change.details.iter().find_map(|d| {
                if let super::diff::TermChange::ReplacedBy(id) = d {
                    Some(Replacement::single(id.clone()))
                } else {
                    None
                }
            });

            tracker.add_deprecated(DeprecatedTerm {
                id: change.term_id.clone(),
                replacement,
                level: if change.breaking {
                    DeprecationLevel::Error
                } else {
                    DeprecationLevel::Warning
                },
                message: Some(change.description.clone()),
                since_version: Some(diff.new_version.clone()),
                removal_version: None,
                reason: None,
            });
        }

        // Also add removed terms as errors
        for change in diff.removed() {
            tracker.add_deprecated(DeprecatedTerm {
                id: change.term_id.clone(),
                replacement: None,
                level: DeprecationLevel::Error,
                message: Some(format!("Term '{}' has been removed", change.term_id)),
                since_version: Some(diff.new_version.clone()),
                removal_version: Some(diff.new_version.clone()),
                reason: Some("removed from ontology".to_string()),
            });
        }

        tracker
    }
}

impl Default for DeprecationTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_deprecated() {
        let mut tracker = DeprecationTracker::new();
        tracker.add_deprecated(DeprecatedTerm {
            id: "TEST:001".to_string(),
            replacement: Some(Replacement::single("TEST:002")),
            level: DeprecationLevel::Warning,
            message: Some("Use TEST:002 instead".to_string()),
            since_version: Some("2024-01-01".to_string()),
            removal_version: None,
            reason: None,
        });

        let warning = tracker.check("TEST:001", None);
        assert!(warning.is_some());
        let w = warning.unwrap();
        assert_eq!(w.level, DeprecationLevel::Warning);
        assert!(w.suggestion.is_some());
    }

    #[test]
    fn test_check_not_deprecated() {
        let mut tracker = DeprecationTracker::new();
        let warning = tracker.check("TEST:999", None);
        assert!(warning.is_none());
    }

    #[test]
    fn test_suppressed() {
        let mut tracker = DeprecationTracker::new();
        tracker.add_deprecated(DeprecatedTerm {
            id: "TEST:001".to_string(),
            replacement: None,
            level: DeprecationLevel::Warning,
            message: None,
            since_version: None,
            removal_version: None,
            reason: None,
        });

        tracker.suppress("TEST:001");
        let warning = tracker.check("TEST:001", None);
        assert!(warning.is_none());
    }

    #[test]
    fn test_warnings_as_errors() {
        let mut tracker = DeprecationTracker::new().warnings_as_errors(true);
        tracker.add_deprecated(DeprecatedTerm {
            id: "TEST:001".to_string(),
            replacement: None,
            level: DeprecationLevel::Warning,
            message: None,
            since_version: None,
            removal_version: None,
            reason: None,
        });

        let warning = tracker.check("TEST:001", None).unwrap();
        assert_eq!(warning.level, DeprecationLevel::Error);
    }

    #[test]
    fn test_replacement_alternatives() {
        let repl = Replacement::alternatives(vec!["A".to_string(), "B".to_string()]);
        assert_eq!(repl.primary(), Some("A"));
        assert_eq!(repl.all(), vec!["A", "B"]);
    }

    #[test]
    fn test_warning_format() {
        let warning = DeprecationWarning {
            term_id: "TEST:001".to_string(),
            span: Some(Span { start: 10, end: 20 }),
            level: DeprecationLevel::Warning,
            message: "Term is deprecated".to_string(),
            suggestion: Some("use TEST:002".to_string()),
            help: Some("deprecated since 2024-01-01".to_string()),
        };

        let formatted = warning.format();
        assert!(formatted.contains("warning"));
        assert!(formatted.contains("TEST:001"));
        assert!(formatted.contains("suggestion"));
    }

    #[test]
    fn test_migration_guide() {
        let mut tracker = DeprecationTracker::new();
        tracker.add_deprecated(DeprecatedTerm {
            id: "OLD:001".to_string(),
            replacement: Some(Replacement::single("NEW:001")),
            level: DeprecationLevel::Warning,
            message: None,
            since_version: None,
            removal_version: None,
            reason: Some("renamed".to_string()),
        });

        let guide = tracker.migration_guide();
        assert!(guide.contains("OLD:001"));
        assert!(guide.contains("NEW:001"));
        assert!(guide.contains("renamed"));
    }

    #[test]
    fn test_from_diff() {
        use super::super::diff::{OntologySnapshot, SnapshotTerm};

        let mut old = OntologySnapshot::new("test", "1.0.0");
        old.add_term(SnapshotTerm {
            id: "TEST:001".to_string(),
            label: Some("Old".to_string()),
            definition: None,
            superclasses: vec![],
            synonyms: vec![],
            obsolete: false,
            replaced_by: None,
        });

        let mut new = OntologySnapshot::new("test", "2.0.0");
        new.add_term(SnapshotTerm {
            id: "TEST:001".to_string(),
            label: Some("Old".to_string()),
            definition: None,
            superclasses: vec![],
            synonyms: vec![],
            obsolete: true,
            replaced_by: Some("TEST:002".to_string()),
        });

        let diff = super::super::diff::OntologyDiff::compute(&old, &new);
        let tracker = DeprecationTracker::from_diff(&diff);

        assert!(tracker.is_deprecated("TEST:001"));
    }

    #[test]
    fn test_has_errors() {
        let mut tracker = DeprecationTracker::new();
        tracker.add_deprecated(DeprecatedTerm {
            id: "TEST:001".to_string(),
            replacement: None,
            level: DeprecationLevel::Error,
            message: None,
            since_version: None,
            removal_version: None,
            reason: None,
        });

        tracker.check("TEST:001", None);
        assert!(tracker.has_errors());
    }
}
