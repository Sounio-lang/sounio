//! Layout Diagnostics - Day 39
//!
//! Rich diagnostic output for layout decisions:
//! - Constraint satisfaction/conflict messages
//! - Layout explanations
//! - Performance predictions

use super::constraint::{ConstraintSource, ForcedRegion, LayoutConstraint};
use super::plan::{LayoutPlan, MemoryRegion};
use super::solver::{ConstraintConflict, ConstraintWarning, SatisfiedConstraint, SolverResult};

/// Diagnostic severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticLevel {
    Error,
    Warning,
    Info,
    Help,
}

impl std::fmt::Display for DiagnosticLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiagnosticLevel::Error => write!(f, "error"),
            DiagnosticLevel::Warning => write!(f, "warning"),
            DiagnosticLevel::Info => write!(f, "info"),
            DiagnosticLevel::Help => write!(f, "help"),
        }
    }
}

/// A layout diagnostic message
#[derive(Debug, Clone)]
pub struct LayoutDiagnostic {
    pub level: DiagnosticLevel,
    pub code: String,
    pub message: String,
    pub source: Option<ConstraintSource>,
    pub notes: Vec<String>,
    pub help: Option<String>,
}

impl LayoutDiagnostic {
    pub fn error(code: &str, message: impl Into<String>) -> Self {
        Self {
            level: DiagnosticLevel::Error,
            code: code.to_string(),
            message: message.into(),
            source: None,
            notes: Vec::new(),
            help: None,
        }
    }

    pub fn warning(code: &str, message: impl Into<String>) -> Self {
        Self {
            level: DiagnosticLevel::Warning,
            code: code.to_string(),
            message: message.into(),
            source: None,
            notes: Vec::new(),
            help: None,
        }
    }

    pub fn info(code: &str, message: impl Into<String>) -> Self {
        Self {
            level: DiagnosticLevel::Info,
            code: code.to_string(),
            message: message.into(),
            source: None,
            notes: Vec::new(),
            help: None,
        }
    }

    pub fn with_source(mut self, source: ConstraintSource) -> Self {
        self.source = Some(source);
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    /// Format as a human-readable string
    pub fn to_string_human(&self) -> String {
        let mut output = String::new();

        // Header line
        let level_color = match self.level {
            DiagnosticLevel::Error => "\x1b[31m",   // Red
            DiagnosticLevel::Warning => "\x1b[33m", // Yellow
            DiagnosticLevel::Info => "\x1b[36m",    // Cyan
            DiagnosticLevel::Help => "\x1b[32m",    // Green
        };
        let reset = "\x1b[0m";

        output.push_str(&format!(
            "{}{}[{}]{}: {}\n",
            level_color, self.level, self.code, reset, self.message
        ));

        // Source location
        if let Some(source) = &self.source {
            output.push_str(&format!(
                "  --> {}:{}:{}\n",
                source.file, source.line, source.column
            ));
            output.push_str("   |\n");
            output.push_str(&format!("{:>3} | #[{}]\n", source.line, source.attribute));
            output.push_str("   |\n");
        }

        // Notes
        for note in &self.notes {
            output.push_str(&format!("  = note: {}\n", note));
        }

        // Help
        if let Some(help) = &self.help {
            output.push_str(&format!("  = help: {}\n", help));
        }

        output
    }

    /// Format without ANSI colors
    pub fn to_string_plain(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "{}[{}]: {}\n",
            self.level, self.code, self.message
        ));

        if let Some(source) = &self.source {
            output.push_str(&format!(
                "  --> {}:{}:{}\n",
                source.file, source.line, source.column
            ));
        }

        for note in &self.notes {
            output.push_str(&format!("  = note: {}\n", note));
        }

        if let Some(help) = &self.help {
            output.push_str(&format!("  = help: {}\n", help));
        }

        output
    }
}

/// Generate diagnostics from solver result
pub fn generate_solver_diagnostics(result: &SolverResult) -> Vec<LayoutDiagnostic> {
    let mut diagnostics = Vec::new();

    // Conflicts are errors
    for conflict in &result.conflicts {
        diagnostics.push(conflict_diagnostic(conflict));
    }

    // Warnings
    for warning in &result.warnings {
        diagnostics.push(warning_diagnostic(warning));
    }

    // Info for satisfied constraints (verbose mode)
    for satisfied in &result.satisfied {
        diagnostics.push(satisfied_diagnostic(satisfied));
    }

    diagnostics
}

fn conflict_diagnostic(conflict: &ConstraintConflict) -> LayoutDiagnostic {
    let source = extract_source(&conflict.constraint_a);

    LayoutDiagnostic::error(
        "L0801",
        format!("Conflicting layout constraints: {}", conflict.reason),
    )
    .with_source(source.unwrap_or_else(|| ConstraintSource::new("<unknown>", 0, 0, "unknown")))
    .with_note(format!(
        "First constraint: {}",
        describe_constraint(&conflict.constraint_a)
    ))
    .with_note(format!(
        "Second constraint: {}",
        describe_constraint(&conflict.constraint_b)
    ))
    .with_help("Remove one of the conflicting constraints")
}

fn warning_diagnostic(warning: &ConstraintWarning) -> LayoutDiagnostic {
    let source = extract_source(&warning.constraint);

    LayoutDiagnostic::warning("L0802", format!("Layout constraint: {}", warning.message))
        .with_source(source.unwrap_or_else(|| ConstraintSource::new("<unknown>", 0, 0, "unknown")))
}

fn satisfied_diagnostic(satisfied: &SatisfiedConstraint) -> LayoutDiagnostic {
    let source = extract_source(&satisfied.constraint);

    LayoutDiagnostic::info("L0803", format!("Constraint applied: {}", satisfied.impact))
        .with_source(source.unwrap_or_else(|| ConstraintSource::new("<unknown>", 0, 0, "unknown")))
}

/// Generate a layout summary diagnostic
pub fn layout_summary_diagnostic(layout: &LayoutPlan) -> LayoutDiagnostic {
    let summary = layout.summary();

    LayoutDiagnostic::info(
        "L0804",
        format!(
            "Layout synthesis complete: {} concepts → Hot: {}, Warm: {}, Cold: {}",
            summary.total, summary.hot, summary.warm, summary.cold
        ),
    )
    .with_help("Use `dc layout visualize` for detailed analysis")
}

/// Generate explanation diagnostic for #[explain_layout]
pub fn explain_layout_diagnostic(
    layout: &LayoutPlan,
    concepts: &[String],
    scope_name: &str,
) -> LayoutDiagnostic {
    let mut explanation = format!("Layout explanation for '{}':\n", scope_name);

    for concept in concepts {
        if let Some(concept_layout) = layout.get(concept) {
            let region_desc = match concept_layout.region {
                MemoryRegion::Hot => "Hot (stack/L1-L2 cache)",
                MemoryRegion::Warm => "Warm (arena/L2-L3 cache)",
                MemoryRegion::Cold => "Cold (heap/RAM)",
            };

            explanation.push_str(&format!(
                "  {} → {} (cluster {})\n",
                concept, region_desc, concept_layout.cluster_id
            ));
        }
    }

    LayoutDiagnostic::info("L0805", explanation)
}

/// Extract source from a constraint
fn extract_source(constraint: &LayoutConstraint) -> Option<ConstraintSource> {
    match constraint {
        LayoutConstraint::Colocate { source, .. }
        | LayoutConstraint::Separate { source, .. }
        | LayoutConstraint::ForceRegion { source, .. } => Some(source.clone()),
        LayoutConstraint::Explain { .. } => None,
    }
}

/// Describe a constraint for diagnostic output
fn describe_constraint(constraint: &LayoutConstraint) -> String {
    match constraint {
        LayoutConstraint::Colocate { concepts, .. } => {
            format!("#[colocate({})]", concepts.join(", "))
        }
        LayoutConstraint::Separate { concepts, .. } => {
            format!("#[separate({})]", concepts.join(", "))
        }
        LayoutConstraint::ForceRegion {
            concept, region, ..
        } => {
            let attr = match region {
                ForcedRegion::Hot => "hot",
                ForcedRegion::Warm => "warm",
                ForcedRegion::Cold => "cold",
            };
            format!("#[{}] on {}", attr, concept)
        }
        LayoutConstraint::Explain { scope_name, .. } => {
            format!("#[explain_layout] for {}", scope_name)
        }
    }
}

/// Generate diagnostics for constraint validation
pub fn validate_constraints_diagnostic(
    result: &SolverResult,
    verbose: bool,
) -> Vec<LayoutDiagnostic> {
    let mut diagnostics = Vec::new();

    if result.conflicts.is_empty() && result.warnings.is_empty() {
        diagnostics.push(
            LayoutDiagnostic::info("L0806", "All layout constraints satisfied")
                .with_note(format!("{} constraints processed", result.satisfied.len())),
        );

        if verbose {
            for satisfied in &result.satisfied {
                diagnostics.push(satisfied_diagnostic(satisfied));
            }
        }
    } else {
        for conflict in &result.conflicts {
            diagnostics.push(conflict_diagnostic(conflict));
        }

        for warning in &result.warnings {
            diagnostics.push(warning_diagnostic(warning));
        }
    }

    diagnostics
}

/// Format diagnostics for terminal output
pub fn format_diagnostics(diagnostics: &[LayoutDiagnostic], use_color: bool) -> String {
    let mut output = String::new();

    for diag in diagnostics {
        if use_color {
            output.push_str(&diag.to_string_human());
        } else {
            output.push_str(&diag.to_string_plain());
        }
        output.push('\n');
    }

    // Summary line
    let errors = diagnostics
        .iter()
        .filter(|d| d.level == DiagnosticLevel::Error)
        .count();
    let warnings = diagnostics
        .iter()
        .filter(|d| d.level == DiagnosticLevel::Warning)
        .count();

    if errors > 0 || warnings > 0 {
        output.push_str(&format!(
            "{} error(s), {} warning(s) emitted\n",
            errors, warnings
        ));
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_error() {
        let diag = LayoutDiagnostic::error("L0001", "Test error")
            .with_note("This is a note")
            .with_help("Try this fix");

        let output = diag.to_string_plain();
        assert!(output.contains("error[L0001]"));
        assert!(output.contains("Test error"));
        assert!(output.contains("This is a note"));
        assert!(output.contains("Try this fix"));
    }

    #[test]
    fn test_diagnostic_with_source() {
        let source = ConstraintSource::new("test.sio", 10, 5, "colocate");
        let diag = LayoutDiagnostic::warning("L0002", "Test warning").with_source(source);

        let output = diag.to_string_plain();
        assert!(output.contains("warning[L0002]"));
        assert!(output.contains("test.sio:10:5"));
    }

    #[test]
    fn test_describe_constraint() {
        let colocate = LayoutConstraint::Colocate {
            concepts: vec!["A".to_string(), "B".to_string()],
            source: ConstraintSource::new("test.sio", 1, 1, "colocate"),
        };

        assert_eq!(describe_constraint(&colocate), "#[colocate(A, B)]");
    }
}
