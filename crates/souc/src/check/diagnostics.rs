//! Rich Diagnostics for Type Compatibility Errors
//!
//! When types are incompatible, we need to explain WHY in a way that's
//! actually helpful. This module generates detailed diagnostics that:
//!
//! 1. Show the semantic distance between types
//! 2. Visualize the path through the ontology
//! 3. Suggest alternative types that would work
//! 4. Explain what threshold was violated
//!
//! # Example Output
//!
//! ```text
//! error[E0501]: semantic type mismatch
//!   --> drug_protocol.dm:42:15
//!    |
//! 42 |     administer(ibuprofen);
//!    |                ^^^^^^^^^ expected `ChEBI:Aspirin`, found `ChEBI:Ibuprofen`
//!    |
//!    = note: semantic distance 0.342 exceeds threshold 0.15
//!    = note: both are NSAIDs, but different active ingredients
//!    |
//!    |  ChEBI:Aspirin
//!    |      └── is-a: ChEBI:NSAID
//!    |                    └── is-a: ChEBI:AntiInflammatory
//!    |  ChEBI:Ibuprofen
//!    |      └── is-a: ChEBI:NSAID
//!    |                    └── is-a: ChEBI:AntiInflammatory
//!    |
//!    = help: use #[compat(0.35)] to allow this coercion
//!    = help: or convert explicitly: ibuprofen.as_compatible::<Aspirin>()
//! ```

use std::fmt::Write;

use crate::common::Span;
use crate::ontology::distance::{SemanticDistance, SemanticDistanceIndex};
use crate::ontology::loader::IRI;

use super::compatibility::{Compatibility, CompatibilityChecker, IncompatibilityReason};

/// Diagnostic severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Info,
    Hint,
}

impl DiagnosticSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            DiagnosticSeverity::Error => "error",
            DiagnosticSeverity::Warning => "warning",
            DiagnosticSeverity::Info => "info",
            DiagnosticSeverity::Hint => "hint",
        }
    }

    pub fn color_code(&self) -> &'static str {
        match self {
            DiagnosticSeverity::Error => "\x1b[31m",   // Red
            DiagnosticSeverity::Warning => "\x1b[33m", // Yellow
            DiagnosticSeverity::Info => "\x1b[34m",    // Blue
            DiagnosticSeverity::Hint => "\x1b[32m",    // Green
        }
    }
}

/// A rich diagnostic message for type compatibility issues
#[derive(Debug, Clone)]
pub struct CompatibilityDiagnostic {
    /// Severity of the diagnostic
    pub severity: DiagnosticSeverity,
    /// Error code (e.g., "E0501")
    pub code: String,
    /// Primary message
    pub message: String,
    /// Source location
    pub span: Option<Span>,
    /// Source type
    pub source_type: IRI,
    /// Target type
    pub target_type: IRI,
    /// Computed distance
    pub distance: SemanticDistance,
    /// Threshold that was exceeded
    pub threshold: f64,
    /// Additional notes explaining the issue
    pub notes: Vec<String>,
    /// Suggestions for fixing
    pub suggestions: Vec<DiagnosticSuggestion>,
    /// Ontology path visualization
    pub path_visualization: Option<String>,
}

/// A suggestion for fixing a compatibility issue
#[derive(Debug, Clone)]
pub struct DiagnosticSuggestion {
    /// Description of what to do
    pub message: String,
    /// Code to insert/replace (if applicable)
    pub replacement: Option<String>,
    /// Span to replace (if applicable)
    pub span: Option<Span>,
}

impl CompatibilityDiagnostic {
    /// Create a new diagnostic for a type mismatch
    pub fn type_mismatch(
        source: &IRI,
        target: &IRI,
        distance: SemanticDistance,
        threshold: f64,
        span: Option<Span>,
    ) -> Self {
        let message = format!(
            "semantic type mismatch: expected `{}`, found `{}`",
            target, source
        );

        Self {
            severity: DiagnosticSeverity::Error,
            code: "E0501".to_string(),
            message,
            span,
            source_type: source.clone(),
            target_type: target.clone(),
            distance,
            threshold,
            notes: Vec::new(),
            suggestions: Vec::new(),
            path_visualization: None,
        }
    }

    /// Add a note to the diagnostic
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Add distance note
    pub fn with_distance_note(mut self) -> Self {
        self.notes.push(format!(
            "semantic distance {:.3} exceeds threshold {:.3}",
            self.distance.conceptual, self.threshold
        ));
        self
    }

    /// Add a suggestion
    pub fn with_suggestion(mut self, suggestion: DiagnosticSuggestion) -> Self {
        self.suggestions.push(suggestion);
        self
    }

    /// Add annotation suggestion
    pub fn with_annotation_suggestion(mut self) -> Self {
        let threshold = (self.distance.conceptual * 1.1).min(1.0); // Suggest slightly above actual
        self.suggestions.push(DiagnosticSuggestion {
            message: format!("use #[compat({:.2})] to allow this coercion", threshold),
            replacement: Some(format!("#[compat({:.2})]", threshold)),
            span: None,
        });
        self
    }

    /// Add explicit conversion suggestion
    pub fn with_conversion_suggestion(mut self, var_name: &str) -> Self {
        self.suggestions.push(DiagnosticSuggestion {
            message: format!(
                "or convert explicitly: {}.as_compatible::<{}>()",
                var_name, self.target_type
            ),
            replacement: Some(format!(
                "{}.as_compatible::<{}>()",
                var_name, self.target_type
            )),
            span: self.span,
        });
        self
    }

    /// Set path visualization
    pub fn with_path_visualization(mut self, viz: String) -> Self {
        self.path_visualization = Some(viz);
        self
    }

    /// Format as a full diagnostic message
    pub fn format(&self, colored: bool) -> String {
        let mut output = String::new();

        // Header line
        let severity_color = if colored {
            self.severity.color_code()
        } else {
            ""
        };
        let reset = if colored { "\x1b[0m" } else { "" };
        let bold = if colored { "\x1b[1m" } else { "" };

        writeln!(
            &mut output,
            "{}{}{}[{}]{}: {}",
            severity_color,
            bold,
            self.severity.as_str(),
            self.code,
            reset,
            self.message
        )
        .ok();

        // Location
        if let Some(ref span) = self.span {
            writeln!(&mut output, "  --> byte {}..{}", span.start, span.end).ok();
        }

        // Notes
        for note in &self.notes {
            writeln!(&mut output, "   = note: {}", note).ok();
        }

        // Path visualization
        if let Some(ref viz) = self.path_visualization {
            writeln!(&mut output, "   |").ok();
            for line in viz.lines() {
                writeln!(&mut output, "   |  {}", line).ok();
            }
            writeln!(&mut output, "   |").ok();
        }

        // Suggestions
        for suggestion in &self.suggestions {
            let hint_color = if colored { "\x1b[32m" } else { "" };
            writeln!(
                &mut output,
                "   {}= help{}: {}",
                hint_color, reset, suggestion.message
            )
            .ok();
        }

        output
    }
}

/// Diagnostic builder for creating rich diagnostics
pub struct DiagnosticBuilder<'a> {
    checker: &'a CompatibilityChecker,
    distance_index: &'a SemanticDistanceIndex,
}

impl<'a> DiagnosticBuilder<'a> {
    pub fn new(
        checker: &'a CompatibilityChecker,
        distance_index: &'a SemanticDistanceIndex,
    ) -> Self {
        Self {
            checker,
            distance_index,
        }
    }

    /// Build a diagnostic from a compatibility check result
    pub fn build_diagnostic(
        &self,
        source: &IRI,
        target: &IRI,
        compatibility: &Compatibility,
        span: Option<Span>,
        var_name: Option<&str>,
    ) -> Option<CompatibilityDiagnostic> {
        match compatibility {
            Compatibility::Identical | Compatibility::Compatible { .. } => None,

            Compatibility::Incompatible { distance, reason } => {
                let threshold = match reason {
                    IncompatibilityReason::DistanceExceedsThreshold { threshold, .. } => *threshold,
                    _ => self.checker.context().default_threshold,
                };

                let mut diag = CompatibilityDiagnostic::type_mismatch(
                    source, target, *distance, threshold, span,
                )
                .with_distance_note();

                // Add reason-specific notes
                match reason {
                    IncompatibilityReason::CrossOntologyNoMapping {
                        source_ontology,
                        target_ontology,
                    } => {
                        diag = diag.with_note(format!(
                            "no SSSOM mapping between {} and {}",
                            source_ontology, target_ontology
                        ));
                    }
                    IncompatibilityReason::Unrelated => {
                        diag = diag.with_note("types have no common ancestor in the ontology");
                    }
                    IncompatibilityReason::StructuralMismatch(msg) => {
                        diag = diag.with_note(format!("structural issue: {}", msg));
                    }
                    _ => {}
                }

                // Add path visualization
                let viz = self.build_path_visualization(source, target);
                if !viz.is_empty() {
                    diag = diag.with_path_visualization(viz);
                }

                // Add suggestions
                diag = diag.with_annotation_suggestion();
                if let Some(name) = var_name {
                    diag = diag.with_conversion_suggestion(name);
                }

                // Add alternative suggestions
                if let Some(alternatives) = self.find_alternatives(source, target, threshold) {
                    for alt in alternatives {
                        diag = diag.with_suggestion(DiagnosticSuggestion {
                            message: format!(
                                "consider using `{}` instead (distance: {:.3})",
                                alt.0, alt.1.conceptual
                            ),
                            replacement: None,
                            span: None,
                        });
                    }
                }

                Some(diag)
            }

            Compatibility::Unknown { reason } => {
                let mut diag = CompatibilityDiagnostic::type_mismatch(
                    source,
                    target,
                    SemanticDistance::MAX,
                    self.checker.context().default_threshold,
                    span,
                );
                diag.severity = DiagnosticSeverity::Warning;
                diag.code = "W0501".to_string();
                diag.message = format!(
                    "cannot determine compatibility between `{}` and `{}`",
                    source, target
                );
                diag = diag.with_note(reason.clone());
                Some(diag)
            }
        }
    }

    /// Build a path visualization showing the relationship between types
    fn build_path_visualization(&self, source: &IRI, target: &IRI) -> String {
        let mut output = String::new();

        // Get ancestors for source
        let source_path = self.get_ancestor_path(source, 3);
        let target_path = self.get_ancestor_path(target, 3);

        // Find common ancestor
        let common = self.find_common_ancestor(&source_path, &target_path);

        // Format source path
        writeln!(&mut output, "{}", source).ok();
        for (i, ancestor) in source_path.iter().enumerate() {
            let prefix = "    ".repeat(i + 1);
            let marker = if Some(ancestor) == common.as_ref() {
                "└── [LCA] is-a:"
            } else {
                "└── is-a:"
            };
            writeln!(&mut output, "{}{} {}", prefix, marker, ancestor).ok();
        }

        // Format target path
        writeln!(&mut output, "{}", target).ok();
        for (i, ancestor) in target_path.iter().enumerate() {
            let prefix = "    ".repeat(i + 1);
            let marker = if Some(ancestor) == common.as_ref() {
                "└── [LCA] is-a:"
            } else {
                "└── is-a:"
            };
            writeln!(&mut output, "{}{} {}", prefix, marker, ancestor).ok();
        }

        output
    }

    /// Get ancestor path up to max_depth
    fn get_ancestor_path(&self, iri: &IRI, max_depth: usize) -> Vec<IRI> {
        let mut path = Vec::new();
        let mut current = iri.clone();

        for _ in 0..max_depth {
            if let Some(parent) = self.get_direct_parent(&current) {
                path.push(parent.clone());
                current = parent;
            } else {
                break;
            }
        }

        path
    }

    /// Get direct parent (first superclass)
    fn get_direct_parent(&self, _iri: &IRI) -> Option<IRI> {
        // This would query the distance index's hierarchy graph
        // For now, return None as placeholder
        None
    }

    /// Find common ancestor between two paths
    fn find_common_ancestor(&self, path1: &[IRI], path2: &[IRI]) -> Option<IRI> {
        for iri in path1 {
            if path2.contains(iri) {
                return Some(iri.clone());
            }
        }
        None
    }

    /// Find alternative types that would be compatible
    fn find_alternatives(
        &self,
        _source: &IRI,
        _target: &IRI,
        _threshold: f64,
    ) -> Option<Vec<(IRI, SemanticDistance)>> {
        // This would search for types in the same ontology that are
        // within the threshold distance to the target
        // For now, return None as placeholder
        None
    }
}

/// Collect diagnostics during type checking
#[derive(Default)]
pub struct DiagnosticCollector {
    diagnostics: Vec<CompatibilityDiagnostic>,
}

impl DiagnosticCollector {
    pub fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
        }
    }

    pub fn add(&mut self, diagnostic: CompatibilityDiagnostic) {
        self.diagnostics.push(diagnostic);
    }

    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity == DiagnosticSeverity::Error)
    }

    pub fn error_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == DiagnosticSeverity::Error)
            .count()
    }

    pub fn warning_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == DiagnosticSeverity::Warning)
            .count()
    }

    pub fn iter(&self) -> impl Iterator<Item = &CompatibilityDiagnostic> {
        self.diagnostics.iter()
    }

    pub fn format_all(&self, colored: bool) -> String {
        let mut output = String::new();
        for diag in &self.diagnostics {
            output.push_str(&diag.format(colored));
            output.push('\n');
        }

        // Summary
        let errors = self.error_count();
        let warnings = self.warning_count();
        if errors > 0 || warnings > 0 {
            write!(
                &mut output,
                "{} error(s), {} warning(s) emitted",
                errors, warnings
            )
            .ok();
        }

        output
    }

    pub fn take(self) -> Vec<CompatibilityDiagnostic> {
        self.diagnostics
    }
}

/// Error reporter that integrates with the compiler's error system
pub struct CompatibilityReporter {
    collector: DiagnosticCollector,
    colored: bool,
}

impl CompatibilityReporter {
    pub fn new(colored: bool) -> Self {
        Self {
            collector: DiagnosticCollector::new(),
            colored,
        }
    }

    pub fn report(&mut self, diagnostic: CompatibilityDiagnostic) {
        // Print immediately
        eprintln!("{}", diagnostic.format(self.colored));
        self.collector.add(diagnostic);
    }

    pub fn has_errors(&self) -> bool {
        self.collector.has_errors()
    }

    pub fn summary(&self) -> String {
        format!(
            "{} error(s), {} warning(s)",
            self.collector.error_count(),
            self.collector.warning_count()
        )
    }

    pub fn into_collector(self) -> DiagnosticCollector {
        self.collector
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_format() {
        let diag = CompatibilityDiagnostic::type_mismatch(
            &IRI::from_curie("ChEBI", "15365"),
            &IRI::from_curie("ChEBI", "15687"),
            SemanticDistance::new(0.35),
            0.15,
            None,
        )
        .with_distance_note()
        .with_annotation_suggestion();

        let formatted = diag.format(false);

        assert!(formatted.contains("E0501"));
        assert!(formatted.contains("semantic type mismatch"));
        assert!(formatted.contains("0.350"));
        assert!(formatted.contains("#[compat"));
    }

    #[test]
    fn test_diagnostic_collector() {
        let mut collector = DiagnosticCollector::new();

        collector.add(
            CompatibilityDiagnostic::type_mismatch(
                &IRI::new("test:A"),
                &IRI::new("test:B"),
                SemanticDistance::new(0.5),
                0.15,
                None,
            )
            .with_distance_note(),
        );

        assert!(collector.has_errors());
        assert_eq!(collector.error_count(), 1);
        assert_eq!(collector.warning_count(), 0);
    }

    #[test]
    fn test_severity_colors() {
        assert_eq!(DiagnosticSeverity::Error.as_str(), "error");
        assert_eq!(DiagnosticSeverity::Warning.as_str(), "warning");
        assert!(DiagnosticSeverity::Error.color_code().contains("31")); // Red
    }

    #[test]
    fn test_suggestion() {
        let suggestion = DiagnosticSuggestion {
            message: "try this".to_string(),
            replacement: Some("new_code".to_string()),
            span: None,
        };

        assert_eq!(suggestion.message, "try this");
        assert_eq!(suggestion.replacement, Some("new_code".to_string()));
    }

    #[test]
    fn test_path_visualization() {
        let diag = CompatibilityDiagnostic::type_mismatch(
            &IRI::from_curie("HP", "0001"),
            &IRI::from_curie("HP", "0002"),
            SemanticDistance::new(0.2),
            0.15,
            None,
        )
        .with_path_visualization(
            "HP:0001\n    └── is-a: HP:0000\nHP:0002\n    └── is-a: HP:0000".to_string(),
        );

        let formatted = diag.format(false);
        assert!(formatted.contains("HP:0001"));
        assert!(formatted.contains("is-a"));
    }
}
