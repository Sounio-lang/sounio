//! Complexity lints
//!
//! Lints that detect overly complex code patterns.

use crate::ast::{Block, Expr, Item, Stmt};
use crate::common::Span;
use crate::lint::{Lint, LintCategory, LintContext, LintLevel};

/// Maximum function arguments threshold
const MAX_ARGS: usize = 7;

/// Maximum function length threshold (statements)
const MAX_STATEMENTS: usize = 100;

/// Maximum nesting depth
const MAX_NESTING: usize = 5;

/// Maximum cyclomatic complexity
const MAX_COMPLEXITY: usize = 10;

// =============================================================================
// Too Many Arguments
// =============================================================================

/// Too many function arguments
pub struct TooManyArguments;

impl Lint for TooManyArguments {
    fn id(&self) -> &'static str {
        "too_many_arguments"
    }

    fn name(&self) -> &'static str {
        "Too Many Arguments"
    }

    fn category(&self) -> LintCategory {
        LintCategory::Complexity
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn description(&self) -> &'static str {
        "Warn when a function has too many parameters (default threshold: 7)"
    }

    fn check_item(&self, item: &Item, ctx: &mut LintContext) {
        if let Item::Function(f) = item {
            let arg_count = f.params.len();
            if arg_count > MAX_ARGS {
                ctx.report_with_help(
                    self,
                    format!(
                        "function `{}` has {} arguments (maximum allowed: {})",
                        f.name, arg_count, MAX_ARGS
                    ),
                    f.span,
                    "consider grouping related parameters into a struct",
                );
            }
        }
    }
}

// =============================================================================
// Too Long Function
// =============================================================================

/// Too long function body
pub struct TooLongFunction;

impl Lint for TooLongFunction {
    fn id(&self) -> &'static str {
        "too_long_function"
    }

    fn name(&self) -> &'static str {
        "Too Long Function"
    }

    fn category(&self) -> LintCategory {
        LintCategory::Complexity
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn description(&self) -> &'static str {
        "Warn when a function is too long (default threshold: 100 statements)"
    }

    fn check_item(&self, item: &Item, ctx: &mut LintContext) {
        if let Item::Function(f) = item {
            let stmt_count = count_statements(&f.body);
            if stmt_count > MAX_STATEMENTS {
                ctx.report_with_help(
                    self,
                    format!(
                        "function `{}` has {} statements (maximum allowed: {})",
                        f.name, stmt_count, MAX_STATEMENTS
                    ),
                    f.span,
                    "consider breaking this function into smaller functions",
                );
            }
        }
    }
}

fn count_statements(block: &Block) -> usize {
    let mut count = 0;
    for stmt in &block.stmts {
        count += 1;
        match stmt {
            Stmt::Expr { expr, .. } => {
                count += count_expr_statements(expr);
            }
            Stmt::Let {
                value: Some(expr), ..
            } => {
                count += count_expr_statements(expr);
            }
            _ => {}
        }
    }
    count
}

fn count_expr_statements(expr: &Expr) -> usize {
    match expr {
        Expr::If {
            then_branch,
            else_branch,
            ..
        } => {
            count_statements(then_branch)
                + else_branch.as_ref().map_or(0, |e| count_expr_statements(e))
        }
        Expr::While { body, .. } | Expr::Loop { body, .. } | Expr::For { body, .. } => {
            count_statements(body)
        }
        Expr::Block { block, .. } => count_statements(block),
        Expr::Match { arms, .. } => arms
            .iter()
            .map(|arm| count_expr_statements(&arm.body))
            .sum(),
        _ => 0,
    }
}

// =============================================================================
// Deep Nesting
// =============================================================================

/// Deep nesting detection
pub struct DeepNesting;

impl Lint for DeepNesting {
    fn id(&self) -> &'static str {
        "deep_nesting"
    }

    fn name(&self) -> &'static str {
        "Deep Nesting"
    }

    fn category(&self) -> LintCategory {
        LintCategory::Complexity
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn description(&self) -> &'static str {
        "Warn when code has too much nesting (default threshold: 5 levels)"
    }

    fn check_item(&self, item: &Item, ctx: &mut LintContext) {
        if let Item::Function(f) = item {
            check_nesting_block(&f.body, 0, ctx, self, f.span);
        }
    }
}

fn check_nesting_block(
    block: &Block,
    depth: usize,
    ctx: &mut LintContext,
    lint: &DeepNesting,
    fn_span: Span,
) {
    for stmt in &block.stmts {
        match stmt {
            Stmt::Expr { expr, .. } => {
                check_nesting_expr(expr, depth, ctx, lint, fn_span);
            }
            Stmt::Let {
                value: Some(expr), ..
            } => {
                check_nesting_expr(expr, depth, ctx, lint, fn_span);
            }
            _ => {}
        }
    }
}

fn check_nesting_expr(
    expr: &Expr,
    depth: usize,
    ctx: &mut LintContext,
    lint: &DeepNesting,
    fn_span: Span,
) {
    let new_depth = depth + 1;

    if new_depth > MAX_NESTING {
        ctx.report_with_help(
            lint,
            format!(
                "code is nested {} levels deep (maximum allowed: {})",
                new_depth, MAX_NESTING
            ),
            fn_span,
            "consider extracting inner logic into separate functions",
        );
        return; // Don't emit multiple warnings for same deep block
    }

    match expr {
        Expr::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            check_nesting_expr(condition, depth, ctx, lint, fn_span);
            check_nesting_block(then_branch, new_depth, ctx, lint, fn_span);
            if let Some(else_expr) = else_branch {
                check_nesting_expr(else_expr, depth, ctx, lint, fn_span);
            }
        }
        Expr::While {
            condition, body, ..
        } => {
            check_nesting_expr(condition, depth, ctx, lint, fn_span);
            check_nesting_block(body, new_depth, ctx, lint, fn_span);
        }
        Expr::Loop { body, .. } | Expr::For { body, .. } => {
            check_nesting_block(body, new_depth, ctx, lint, fn_span);
        }
        Expr::Match {
            scrutinee, arms, ..
        } => {
            check_nesting_expr(scrutinee, depth, ctx, lint, fn_span);
            for arm in arms {
                check_nesting_expr(&arm.body, new_depth, ctx, lint, fn_span);
            }
        }
        Expr::Block { block, .. } => {
            check_nesting_block(block, new_depth, ctx, lint, fn_span);
        }
        _ => {}
    }
}

// =============================================================================
// Cyclomatic Complexity
// =============================================================================

/// Cyclomatic complexity measurement
pub struct CyclomaticComplexity;

impl Lint for CyclomaticComplexity {
    fn id(&self) -> &'static str {
        "cyclomatic_complexity"
    }

    fn name(&self) -> &'static str {
        "Cyclomatic Complexity"
    }

    fn category(&self) -> LintCategory {
        LintCategory::Complexity
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn description(&self) -> &'static str {
        "Warn when function cyclomatic complexity exceeds threshold (default: 10)"
    }

    fn check_item(&self, item: &Item, ctx: &mut LintContext) {
        if let Item::Function(f) = item {
            // Cyclomatic complexity = edges - nodes + 2
            // Simplified: 1 + number of decision points
            let complexity = 1 + count_decision_points(&f.body);

            if complexity > MAX_COMPLEXITY {
                ctx.report_with_help(
                    self,
                    format!(
                        "function `{}` has cyclomatic complexity of {} (maximum allowed: {})",
                        f.name, complexity, MAX_COMPLEXITY
                    ),
                    f.span,
                    "consider breaking this function into smaller, simpler functions",
                );
            }
        }
    }
}

fn count_decision_points(block: &Block) -> usize {
    let mut count = 0;
    for stmt in &block.stmts {
        match stmt {
            Stmt::Expr { expr, .. } => {
                count += count_expr_decision_points(expr);
            }
            Stmt::Let {
                value: Some(expr), ..
            } => {
                count += count_expr_decision_points(expr);
            }
            _ => {}
        }
    }
    count
}

fn count_expr_decision_points(expr: &Expr) -> usize {
    match expr {
        Expr::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            1 // if counts as one decision point
                + count_expr_decision_points(condition)
                + count_decision_points(then_branch)
                + else_branch.as_ref().map_or(0, |e| count_expr_decision_points(e))
        }
        Expr::While {
            condition, body, ..
        } => 1 + count_expr_decision_points(condition) + count_decision_points(body),
        Expr::Loop { body, .. } => 1 + count_decision_points(body),
        Expr::For { body, .. } => 1 + count_decision_points(body),
        Expr::Match {
            scrutinee, arms, ..
        } => {
            // Each match arm (except first) adds to complexity
            let arm_count = arms.len().saturating_sub(1);
            arm_count
                + count_expr_decision_points(scrutinee)
                + arms
                    .iter()
                    .map(|a| count_expr_decision_points(&a.body))
                    .sum::<usize>()
        }
        Expr::Binary {
            left, op, right, ..
        } => {
            let op_complexity = match op {
                crate::ast::BinaryOp::And | crate::ast::BinaryOp::Or => 1, // Short-circuit operators add complexity
                _ => 0,
            };
            op_complexity + count_expr_decision_points(left) + count_expr_decision_points(right)
        }
        Expr::Block { block, .. } => count_decision_points(block),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lint_ids() {
        assert_eq!(TooManyArguments.id(), "too_many_arguments");
        assert_eq!(TooLongFunction.id(), "too_long_function");
        assert_eq!(DeepNesting.id(), "deep_nesting");
        assert_eq!(CyclomaticComplexity.id(), "cyclomatic_complexity");
    }

    #[test]
    fn test_lint_categories() {
        assert_eq!(TooManyArguments.category(), LintCategory::Complexity);
        assert_eq!(TooLongFunction.category(), LintCategory::Complexity);
        assert_eq!(DeepNesting.category(), LintCategory::Complexity);
        assert_eq!(CyclomaticComplexity.category(), LintCategory::Complexity);
    }
}
