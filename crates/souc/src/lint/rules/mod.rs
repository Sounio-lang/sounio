//! Lint rules organized by category
//!
//! This module provides a structured organization of lint rules:
//!
//! - `linear`: Linear/affine type lints (resource safety)
//! - `effects`: Effect system lints (algebraic effects)
//! - `epistemic`: Epistemic type lints (Knowledge<T>, confidence, provenance)
//! - `style`: Code style lints (naming, formatting)
//! - `perf`: Performance lints (allocations, cloning)

pub mod complexity;
pub mod correctness;
pub mod effects;
pub mod epistemic;
pub mod linear;
pub mod perf;
pub mod safety;
pub mod style;

// Re-export all lints for convenience
pub use complexity::*;
pub use correctness::*;
pub use effects::*;
pub use epistemic::*;
pub use linear::*;
pub use perf::*;
pub use safety::*;
pub use style::*;

use super::{Lint, LintCategory};

/// All built-in lints organized by category
pub fn all_lints() -> Vec<Box<dyn Lint>> {
    let mut lints: Vec<Box<dyn Lint>> = Vec::new();

    // Correctness lints
    lints.push(Box::new(correctness::UnusedFunction));
    lints.push(Box::new(correctness::UnreachableCode));
    lints.push(Box::new(correctness::DivisionByZero));
    lints.push(Box::new(correctness::InfiniteLoop));

    // Linear type lints
    lints.push(Box::new(linear::UnusedLinearValue));
    lints.push(Box::new(linear::LinearValueCopied));
    lints.push(Box::new(linear::LinearEscape));
    lints.push(Box::new(linear::AffineNotConsumed));

    // Effect lints
    lints.push(Box::new(effects::UnnecessaryEffect));
    lints.push(Box::new(effects::MissingEffectAnnotation));
    lints.push(Box::new(effects::EffectInPureContext));
    lints.push(Box::new(effects::IoInHotLoop));

    // Epistemic lints
    lints.push(Box::new(epistemic::LowConfidencePropagation));
    lints.push(Box::new(epistemic::UntraceableProvenance));
    lints.push(Box::new(epistemic::ConfidenceNotChecked));
    lints.push(Box::new(epistemic::StaleKnowledge));

    // Style lints
    lints.push(Box::new(style::UnusedVariable));
    lints.push(Box::new(style::UnusedImport));
    lints.push(Box::new(style::NamingConvention));
    lints.push(Box::new(style::RedundantSemicolon));

    // Performance lints
    lints.push(Box::new(perf::ExpensiveCloneInLoop));
    lints.push(Box::new(perf::UnnecessaryAllocation));
    lints.push(Box::new(perf::LargeStackValue));

    // Complexity lints
    lints.push(Box::new(complexity::TooManyArguments));
    lints.push(Box::new(complexity::TooLongFunction));
    lints.push(Box::new(complexity::DeepNesting));
    lints.push(Box::new(complexity::CyclomaticComplexity));

    // Safety lints
    lints.push(Box::new(safety::UnsafeBlock));

    lints
}

/// Get lints by category
pub fn lints_by_category(category: LintCategory) -> Vec<Box<dyn Lint>> {
    all_lints()
        .into_iter()
        .filter(|l| l.category() == category)
        .collect()
}

/// Lint group definitions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LintGroup {
    /// All correctness-related lints
    Correctness,
    /// All style-related lints
    Style,
    /// All performance-related lints
    Performance,
    /// All effect-related lints
    Effects,
    /// All linear/affine type lints
    Linear,
    /// All epistemic type lints
    Epistemic,
    /// Pedantic lints (very strict)
    Pedantic,
    /// Nursery lints (experimental)
    Nursery,
}

impl LintGroup {
    /// Get all lint IDs in this group
    pub fn lint_ids(&self) -> &'static [&'static str] {
        match self {
            LintGroup::Correctness => &[
                "unused_variable",
                "unused_import",
                "division_by_zero",
                "infinite_loop",
                "unreachable_code",
            ],
            LintGroup::Style => &["naming_convention", "redundant_semicolon", "missing_docs"],
            LintGroup::Performance => &[
                "expensive_clone_in_loop",
                "unnecessary_allocation",
                "large_stack_value",
            ],
            LintGroup::Effects => &[
                "unnecessary_effect",
                "missing_effect_annotation",
                "effect_in_pure_context",
                "io_in_hot_loop",
            ],
            LintGroup::Linear => &[
                "unused_linear_value",
                "linear_value_copied",
                "linear_escape",
                "affine_not_consumed",
            ],
            LintGroup::Epistemic => &[
                "low_confidence_propagation",
                "untraceable_provenance",
                "confidence_not_checked",
                "stale_knowledge",
            ],
            LintGroup::Pedantic => &["naming_convention", "missing_docs", "redundant_semicolon"],
            LintGroup::Nursery => &["stale_knowledge", "io_in_hot_loop"],
        }
    }

    /// Get group name
    pub fn name(&self) -> &'static str {
        match self {
            LintGroup::Correctness => "correctness",
            LintGroup::Style => "style",
            LintGroup::Performance => "performance",
            LintGroup::Effects => "effects",
            LintGroup::Linear => "linear",
            LintGroup::Epistemic => "epistemic",
            LintGroup::Pedantic => "pedantic",
            LintGroup::Nursery => "nursery",
        }
    }

    /// Parse group name
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "correctness" => Some(LintGroup::Correctness),
            "style" => Some(LintGroup::Style),
            "performance" | "perf" => Some(LintGroup::Performance),
            "effects" => Some(LintGroup::Effects),
            "linear" => Some(LintGroup::Linear),
            "epistemic" => Some(LintGroup::Epistemic),
            "pedantic" => Some(LintGroup::Pedantic),
            "nursery" => Some(LintGroup::Nursery),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_lints_count() {
        let lints = all_lints();
        assert!(lints.len() >= 15, "Expected at least 15 lints");
    }

    #[test]
    fn test_lint_groups() {
        assert!(
            LintGroup::Linear
                .lint_ids()
                .contains(&"unused_linear_value")
        );
        assert!(
            LintGroup::Epistemic
                .lint_ids()
                .contains(&"low_confidence_propagation")
        );
    }

    #[test]
    fn test_group_from_name() {
        assert_eq!(LintGroup::from_name("linear"), Some(LintGroup::Linear));
        assert_eq!(
            LintGroup::from_name("EPISTEMIC"),
            Some(LintGroup::Epistemic)
        );
        assert_eq!(LintGroup::from_name("unknown"), None);
    }
}
