//! Correctness lints
//!
//! Lints that detect bugs and logical errors in code.

use crate::ast::{Block, Expr, Item, Literal, Stmt};
use crate::common::Span;
use crate::lint::{Lint, LintCategory, LintContext, LintLevel};

// =============================================================================
// Unused Function
// =============================================================================

/// Unused function declaration
pub struct UnusedFunction;

impl Lint for UnusedFunction {
    fn id(&self) -> &'static str {
        "unused_function"
    }

    fn name(&self) -> &'static str {
        "Unused Function"
    }

    fn category(&self) -> LintCategory {
        LintCategory::Correctness
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn description(&self) -> &'static str {
        "Warn about functions that are declared but never called"
    }
}

// =============================================================================
// Unreachable Code
// =============================================================================

/// Unreachable code detection
pub struct UnreachableCode;

impl Lint for UnreachableCode {
    fn id(&self) -> &'static str {
        "unreachable_code"
    }

    fn name(&self) -> &'static str {
        "Unreachable Code"
    }

    fn category(&self) -> LintCategory {
        LintCategory::Correctness
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn description(&self) -> &'static str {
        "Warn about code that can never be executed"
    }

    fn check_item(&self, item: &Item, ctx: &mut LintContext) {
        if let Item::Function(f) = item {
            check_unreachable_block(&f.body, ctx, self);
        }
    }
}

fn check_unreachable_block(block: &Block, ctx: &mut LintContext, lint: &UnreachableCode) {
    let mut seen_return = false;

    for stmt in &block.stmts {
        if seen_return {
            ctx.report(lint, "unreachable code", Span::dummy());
            return;
        }

        match stmt {
            Stmt::Expr { expr, .. } => {
                if is_return_expr(expr) {
                    seen_return = true;
                } else if let Expr::Block { block, .. } = expr {
                    check_unreachable_block(block, ctx, lint);
                }
            }
            _ => {}
        }
    }
}

fn is_return_expr(expr: &Expr) -> bool {
    matches!(expr, Expr::Return { .. })
}

// =============================================================================
// Division by Zero
// =============================================================================

/// Division by zero detection
pub struct DivisionByZero;

impl Lint for DivisionByZero {
    fn id(&self) -> &'static str {
        "division_by_zero"
    }

    fn name(&self) -> &'static str {
        "Division by Zero"
    }

    fn category(&self) -> LintCategory {
        LintCategory::Correctness
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Deny
    }

    fn description(&self) -> &'static str {
        "Error on division by literal zero"
    }

    fn check_expr(&self, expr: &Expr, ctx: &mut LintContext) {
        if let Expr::Binary { op, right, .. } = expr {
            if matches!(op, crate::ast::BinaryOp::Div | crate::ast::BinaryOp::Rem) {
                if is_zero_literal(right) {
                    ctx.report_with_help(
                        self,
                        "division by zero",
                        Span::dummy(),
                        "this will cause a runtime panic",
                    );
                }
            }
        }
    }
}

fn is_zero_literal(expr: &Expr) -> bool {
    match expr {
        Expr::Literal {
            value: Literal::Int(0),
            ..
        } => true,
        Expr::Literal {
            value: Literal::Float(f),
            ..
        } => *f == 0.0,
        Expr::Literal {
            value: Literal::TypedInt(0, _),
            ..
        } => true,
        Expr::Literal {
            value: Literal::TypedFloat(f, _),
            ..
        } => *f == 0.0,
        _ => false,
    }
}

// =============================================================================
// Infinite Loop
// =============================================================================

/// Infinite loop detection
pub struct InfiniteLoop;

impl Lint for InfiniteLoop {
    fn id(&self) -> &'static str {
        "infinite_loop"
    }

    fn name(&self) -> &'static str {
        "Infinite Loop"
    }

    fn category(&self) -> LintCategory {
        LintCategory::Correctness
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn description(&self) -> &'static str {
        "Warn about obvious infinite loops without break conditions"
    }

    fn check_expr(&self, expr: &Expr, ctx: &mut LintContext) {
        match expr {
            Expr::While {
                condition, body, ..
            } => {
                if is_literal_true(condition) && !has_break_or_return(body) {
                    ctx.report_with_help(
                        self,
                        "this loop will never terminate",
                        Span::dummy(),
                        "add a break condition or return statement",
                    );
                }
            }
            Expr::Loop { body, .. } => {
                if !has_break_or_return(body) {
                    ctx.report_with_help(
                        self,
                        "this loop will never terminate",
                        Span::dummy(),
                        "add a break or return statement",
                    );
                }
            }
            _ => {}
        }
    }
}

fn is_literal_true(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Literal {
            value: Literal::Bool(true),
            ..
        }
    )
}

fn has_break_or_return(block: &Block) -> bool {
    for stmt in &block.stmts {
        if let Stmt::Expr { expr, .. } = stmt {
            if has_break_or_return_expr(expr) {
                return true;
            }
        }
    }
    false
}

fn has_break_or_return_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Break { .. } | Expr::Return { .. } => true,
        Expr::If {
            then_branch,
            else_branch,
            ..
        } => {
            has_break_or_return(then_branch)
                || else_branch
                    .as_ref()
                    .map_or(false, |e| has_break_or_return_expr(e))
        }
        Expr::Block { block, .. } => has_break_or_return(block),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lint_ids() {
        assert_eq!(UnusedFunction.id(), "unused_function");
        assert_eq!(UnreachableCode.id(), "unreachable_code");
        assert_eq!(DivisionByZero.id(), "division_by_zero");
        assert_eq!(InfiniteLoop.id(), "infinite_loop");
    }

    #[test]
    fn test_lint_categories() {
        assert_eq!(UnusedFunction.category(), LintCategory::Correctness);
        assert_eq!(UnreachableCode.category(), LintCategory::Correctness);
        assert_eq!(DivisionByZero.category(), LintCategory::Correctness);
        assert_eq!(InfiniteLoop.category(), LintCategory::Correctness);
    }

    #[test]
    fn test_default_levels() {
        assert_eq!(UnusedFunction.default_level(), LintLevel::Warn);
        assert_eq!(DivisionByZero.default_level(), LintLevel::Deny);
    }
}
