//! Safety lints
//!
//! Lints for safety-related issues.

use crate::lint::{Lint, LintCategory, LintLevel};

/// Unsafe block usage warning
///
/// Note: Sounio doesn't currently have unsafe blocks like Rust,
/// but this lint is placeholder for future safety features.
pub struct UnsafeBlock;

impl Lint for UnsafeBlock {
    fn id(&self) -> &'static str {
        "unsafe_block"
    }

    fn name(&self) -> &'static str {
        "Unsafe Block"
    }

    fn category(&self) -> LintCategory {
        LintCategory::Safety
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn description(&self) -> &'static str {
        "Warn about usage of unsafe blocks (placeholder for future safety features)"
    }

    // No check implementation yet - Sounio doesn't have unsafe blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lint_id() {
        assert_eq!(UnsafeBlock.id(), "unsafe_block");
    }

    #[test]
    fn test_lint_category() {
        assert_eq!(UnsafeBlock.category(), LintCategory::Safety);
    }
}
