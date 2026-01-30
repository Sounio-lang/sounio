//! Effect system lints
//!
//! Lints for Sounio's algebraic effect system.

use crate::ast::{Block, Expr, Stmt};
use crate::common::Span;
use crate::lint::{Lint, LintCategory, LintContext, LintLevel};

// =============================================================================
// Unnecessary Effect
// =============================================================================

/// Detects effect declarations that are not used in the function body
pub struct UnnecessaryEffect;

impl Lint for UnnecessaryEffect {
    fn id(&self) -> &'static str {
        "unnecessary_effect"
    }

    fn name(&self) -> &'static str {
        "Unnecessary Effect"
    }

    fn description(&self) -> &'static str {
        "Detects effect annotations that are declared but not used."
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::Effects
    }
}

// =============================================================================
// Missing Effect Annotation
// =============================================================================

/// Detects effectful operations without proper effect annotations
pub struct MissingEffectAnnotation;

impl Lint for MissingEffectAnnotation {
    fn id(&self) -> &'static str {
        "missing_effect_annotation"
    }

    fn name(&self) -> &'static str {
        "Missing Effect Annotation"
    }

    fn description(&self) -> &'static str {
        "Warns when effectful operations are used without effect annotations."
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::Effects
    }
}

// =============================================================================
// Effect In Pure Context
// =============================================================================

/// Detects effect usage in pure contexts
pub struct EffectInPureContext;

impl Lint for EffectInPureContext {
    fn id(&self) -> &'static str {
        "effect_in_pure_context"
    }

    fn name(&self) -> &'static str {
        "Effect In Pure Context"
    }

    fn description(&self) -> &'static str {
        "Warns when effects are performed in a context marked as pure."
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::Effects
    }
}

// =============================================================================
// IO In Hot Loop
// =============================================================================

/// Detects IO operations inside hot loops
pub struct IoInHotLoop;

impl Lint for IoInHotLoop {
    fn id(&self) -> &'static str {
        "io_in_hot_loop"
    }

    fn name(&self) -> &'static str {
        "IO In Hot Loop"
    }

    fn description(&self) -> &'static str {
        "Warns when IO operations are performed inside tight loops."
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::Effects
    }

    fn check_expr(&self, expr: &Expr, ctx: &mut LintContext) {
        match expr {
            Expr::Loop { body, .. } | Expr::While { body, .. } | Expr::For { body, .. } => {
                check_io_in_block(body, ctx, self);
            }
            _ => {}
        }
    }
}

fn check_io_in_block(block: &Block, ctx: &mut LintContext, lint: &IoInHotLoop) {
    for stmt in &block.stmts {
        if let Stmt::Expr { expr, .. } = stmt {
            check_io_in_expr(expr, ctx, lint);
        }
    }
}

fn check_io_in_expr(expr: &Expr, ctx: &mut LintContext, lint: &IoInHotLoop) {
    match expr {
        Expr::Perform { effect, .. } => {
            // Check if effect is IO (effect is a Path)
            if effect.segments.iter().any(|s| s == "IO" || s == "io") {
                ctx.report_with_help(
                    lint,
                    "IO operation inside loop",
                    Span::dummy(),
                    "consider batching IO operations outside the loop",
                );
            }
        }
        Expr::Call { callee, args, .. } => {
            check_io_in_expr(callee, ctx, lint);
            for arg in args {
                check_io_in_expr(arg, ctx, lint);
            }
        }
        Expr::MethodCall { receiver, args, .. } => {
            check_io_in_expr(receiver, ctx, lint);
            for arg in args {
                check_io_in_expr(arg, ctx, lint);
            }
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            check_io_in_expr(condition, ctx, lint);
            check_io_in_block(then_branch, ctx, lint);
            if let Some(else_expr) = else_branch {
                check_io_in_expr(else_expr, ctx, lint);
            }
        }
        Expr::Block { block, .. } => {
            check_io_in_block(block, ctx, lint);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lint_ids() {
        assert_eq!(UnnecessaryEffect.id(), "unnecessary_effect");
        assert_eq!(MissingEffectAnnotation.id(), "missing_effect_annotation");
        assert_eq!(EffectInPureContext.id(), "effect_in_pure_context");
        assert_eq!(IoInHotLoop.id(), "io_in_hot_loop");
    }

    #[test]
    fn test_lint_categories() {
        assert_eq!(UnnecessaryEffect.category(), LintCategory::Effects);
        assert_eq!(MissingEffectAnnotation.category(), LintCategory::Effects);
        assert_eq!(EffectInPureContext.category(), LintCategory::Effects);
        assert_eq!(IoInHotLoop.category(), LintCategory::Effects);
    }
}
