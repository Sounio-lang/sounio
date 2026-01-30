//! Linear and affine type lints
//!
//! These lints detect issues with linear and affine types, ensuring proper
//! resource management and consumption semantics.
//!
//! # Linear Types in Sounio
//!
//! Linear types must be used exactly once:
//! - `LINEAR`: Must be consumed exactly once (file handles, unique resources)
//! - `AFFINE`: Can be consumed at most once (optional resources, tokens)
//! - `RELEVANT`: Must be used at least once (mandatory evidence, audit data)
//!
//! # Example
//!
//! ```sio
//! linear struct FileHandle { fd: i32 }
//!
//! fn process_file(handle: FileHandle) {
//!     // handle must be consumed before function returns
//!     close(handle)  // OK: consumed
//! }
//!
//! fn bad_process(handle: FileHandle) {
//!     // ERROR: linear value not consumed
//! }
//! ```

// Compiler internals often require deep nesting for pattern matching and control flow.
#![allow(clippy::excessive_nesting)]

use crate::ast::{Expr, Item, Pattern, Stmt, TypeExpr};
use crate::common::Span;
use crate::lint::{Applicability, Edit, Lint, LintCategory, LintContext, LintLevel, Suggestion};

// =============================================================================
// Unused Linear Value
// =============================================================================

/// Detects linear values that are not consumed
///
/// Linear types must be used exactly once. This lint fires when a linear
/// value goes out of scope without being consumed.
///
/// # Example
///
/// ```sio
/// linear struct Token { id: i32 }
///
/// fn example() {
///     let token = Token { id: 42 };  // WARNING: linear value not consumed
/// }  // token goes out of scope unconsumed
/// ```
pub struct UnusedLinearValue;

impl Lint for UnusedLinearValue {
    fn id(&self) -> &'static str {
        "unused_linear_value"
    }

    fn name(&self) -> &'static str {
        "Unused Linear Value"
    }

    fn description(&self) -> &'static str {
        "Detects linear values that are declared but not consumed. Linear types must be used exactly once."
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Deny
    }

    fn category(&self) -> LintCategory {
        LintCategory::Safety
    }

    fn check_item(&self, item: &Item, ctx: &mut LintContext) {
        if let Item::Function(f) = item {
            // Track linear bindings in function scope
            let mut linear_bindings: Vec<(String, Span, bool)> = Vec::new();

            // Check function body for linear variable declarations and usage
            check_block_for_linear_usage(&f.body, &mut linear_bindings);

            // Report unused linear values
            for (name, span, consumed) in linear_bindings {
                if !consumed && !name.starts_with('_') {
                    ctx.report_with_help(
                        self,
                        format!("linear value `{}` is never consumed", name),
                        span,
                        "linear values must be used exactly once; consider consuming it or marking with `_` prefix if intentionally dropped",
                    );
                }
            }
        }
    }

    fn check_stmt(&self, stmt: &Stmt, ctx: &mut LintContext) {
        // Check for let bindings of linear types
        if let Stmt::Let {
            pattern,
            ty: Some(ty),
            ..
        } = stmt
        {
            if is_linear_type(ty) {
                if let Pattern::Binding { name, .. } = pattern {
                    // Register this as a linear binding to track
                    ctx.define_name(
                        name,
                        Span { start: 0, end: 0 },
                        crate::lint::NameKind::Variable,
                    );
                }
            }
        }
    }
}

/// Check a block for linear variable usage
fn check_block_for_linear_usage(
    block: &crate::ast::Block,
    linear_bindings: &mut Vec<(String, Span, bool)>,
) {
    for stmt in &block.stmts {
        match stmt {
            Stmt::Let {
                pattern,
                ty: Some(ty),
                value,
                ..
            } if is_linear_type(ty) => {
                if let Pattern::Binding { name, .. } = pattern {
                    linear_bindings.push((name.clone(), Span { start: 0, end: 0 }, false));
                }

                // Check if value is a linear variable being moved
                if let Some(expr) = value {
                    mark_linear_consumed(expr, linear_bindings);
                }
            }
            Stmt::Expr { expr, .. } => {
                mark_linear_consumed(expr, linear_bindings);
            }
            Stmt::Assign { value, .. } => {
                mark_linear_consumed(value, linear_bindings);
            }
            _ => {}
        }
    }
}

/// Mark linear variables as consumed when used
fn mark_linear_consumed(expr: &Expr, linear_bindings: &mut Vec<(String, Span, bool)>) {
    match expr {
        Expr::Path { path, .. } => {
            if !path.segments.is_empty() {
                let name = &path.segments[0];
                for (binding_name, _, consumed) in linear_bindings.iter_mut() {
                    if binding_name == name {
                        *consumed = true;
                    }
                }
            }
        }
        Expr::Call { callee, args, .. } => {
            mark_linear_consumed(callee, linear_bindings);
            for arg in args {
                mark_linear_consumed(arg, linear_bindings);
            }
        }
        Expr::MethodCall { receiver, args, .. } => {
            mark_linear_consumed(receiver, linear_bindings);
            for arg in args {
                mark_linear_consumed(arg, linear_bindings);
            }
        }
        Expr::Binary { left, right, .. } => {
            mark_linear_consumed(left, linear_bindings);
            mark_linear_consumed(right, linear_bindings);
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            mark_linear_consumed(condition, linear_bindings);
            check_block_for_linear_usage(then_branch, linear_bindings);
            if let Some(else_expr) = else_branch {
                mark_linear_consumed(else_expr, linear_bindings);
            }
        }
        Expr::Block { block, .. } => {
            check_block_for_linear_usage(block, linear_bindings);
        }
        _ => {}
    }
}

/// Check if a type is linear
fn is_linear_type(ty: &TypeExpr) -> bool {
    match ty {
        TypeExpr::Named { path, .. } => {
            let name = path.segments.last().map(|s| s.as_str()).unwrap_or("");
            matches!(
                name,
                "Linear" | "FileHandle" | "Socket" | "Connection" | "Transaction" | "Lock"
            )
        }
        _ => false,
    }
}

// =============================================================================
// Linear Value Copied
// =============================================================================

/// Detects attempts to copy linear types
///
/// Linear types cannot be copied; they can only be moved. This lint detects
/// when a linear value is used in a context that would require copying.
pub struct LinearValueCopied;

impl Lint for LinearValueCopied {
    fn id(&self) -> &'static str {
        "linear_value_copied"
    }

    fn name(&self) -> &'static str {
        "Linear Value Copied"
    }

    fn description(&self) -> &'static str {
        "Detects attempts to copy linear types. Linear values can only be moved, not copied."
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Deny
    }

    fn category(&self) -> LintCategory {
        LintCategory::Safety
    }

    fn check_expr(&self, expr: &Expr, ctx: &mut LintContext) {
        // Check for clone() calls on linear types
        if let Expr::MethodCall {
            method, receiver, ..
        } = expr
        {
            if method == "clone" || method == "copy" {
                // Check if receiver might be a linear type
                if let Expr::Path { path, .. } = receiver.as_ref() {
                    if !path.segments.is_empty() {
                        ctx.report_with_help(
                            self,
                            format!(
                                "cannot clone linear value `{}`",
                                path.segments.join("::")
                            ),
                            Span::dummy(),
                            "linear types must be moved, not copied; consider restructuring to avoid the need for copying",
                        );
                    }
                }
            }
        }

        // Check for assignment that would copy (when variable is used again after)
        // This requires more context from type checking, so we do a basic check
    }
}

// =============================================================================
// Linear Escape
// =============================================================================

/// Detects linear values escaping scope without consumption
///
/// A linear value must be consumed within the scope where it is valid.
/// This lint detects when a linear value might escape (e.g., stored in
/// a non-linear container) without proper consumption.
pub struct LinearEscape;

impl Lint for LinearEscape {
    fn id(&self) -> &'static str {
        "linear_escape"
    }

    fn name(&self) -> &'static str {
        "Linear Escape"
    }

    fn description(&self) -> &'static str {
        "Detects linear values escaping their scope without being properly consumed."
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Deny
    }

    fn category(&self) -> LintCategory {
        LintCategory::Safety
    }

    fn check_expr(&self, expr: &Expr, ctx: &mut LintContext) {
        // Check for linear values being stored in containers
        if let Expr::MethodCall { method, args, .. } = expr {
            // Methods that store values in containers
            let storing_methods = ["push", "insert", "add", "append", "extend"];

            if storing_methods.contains(&method.as_str()) {
                // Check if any argument might be a linear value being escaped
                for arg in args {
                    if let Expr::Path { path, .. } = arg {
                        // This is a simplified check; full implementation would
                        // need type information to determine if path refers to linear type
                        ctx.report_with_help(
                            self,
                            "potential linear value escaping into container",
                            Span::dummy(),
                            "if this value is linear, ensure it is properly consumed; linear values cannot be stored in non-linear containers",
                        );
                    }
                }
            }
        }

        // Check for closures capturing linear values
        if let Expr::Closure { body, .. } = expr {
            // Closures that capture linear values need special handling
            check_closure_captures_linear(body, ctx, Span::dummy());
        }
    }

    fn check_item(&self, item: &Item, ctx: &mut LintContext) {
        // Check function return types for linear values
        if let Item::Function(f) = item {
            if let Some(ret_ty) = &f.return_type {
                if is_linear_type(ret_ty) {
                    // Linear return types are OK, but warn if the value might not be consumed
                    // by the caller
                }
            }
        }
    }
}

/// Check if a closure captures linear values
fn check_closure_captures_linear(body: &Expr, _ctx: &mut LintContext, _span: Span) {
    fn check_expr_for_captures(expr: &Expr, captures: &mut Vec<String>) {
        match expr {
            Expr::Path { path, .. } => {
                if !path.segments.is_empty() {
                    captures.push(path.segments[0].clone());
                }
            }
            Expr::Binary { left, right, .. } => {
                check_expr_for_captures(left, captures);
                check_expr_for_captures(right, captures);
            }
            Expr::Call { callee, args, .. } => {
                check_expr_for_captures(callee, captures);
                for arg in args {
                    check_expr_for_captures(arg, captures);
                }
            }
            _ => {}
        }
    }

    let mut captures = Vec::new();
    check_expr_for_captures(body, &mut captures);

    // Note: Full implementation would check if captured variables are linear
    // For now, we just note that captures exist
}

// =============================================================================
// Affine Not Consumed
// =============================================================================

/// Warns when affine values are never used
///
/// Affine types can be used at most once, but it's often a code smell if
/// they are never used at all. This lint provides a warning (not error)
/// for affine values that are declared but never consumed.
pub struct AffineNotConsumed;

impl Lint for AffineNotConsumed {
    fn id(&self) -> &'static str {
        "affine_not_consumed"
    }

    fn name(&self) -> &'static str {
        "Affine Not Consumed"
    }

    fn description(&self) -> &'static str {
        "Warns when affine values are declared but never used. While valid, this may indicate dead code."
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::Correctness
    }

    fn check_item(&self, item: &Item, ctx: &mut LintContext) {
        if let Item::Function(f) = item {
            let mut affine_bindings: Vec<(String, Span, bool)> = Vec::new();

            // Check function body for affine variable declarations and usage
            check_block_for_affine_usage(&f.body, &mut affine_bindings);

            // Report unused affine values (warning, not error)
            for (name, span, consumed) in affine_bindings {
                if !consumed && !name.starts_with('_') {
                    ctx.report_with_suggestion(
                        self,
                        format!("affine value `{}` is declared but never used", name),
                        span,
                        Suggestion {
                            message: "prefix with underscore if intentionally unused".into(),
                            edits: vec![Edit {
                                span: Span {
                                    start: span.start,
                                    end: span.start,
                                },
                                replacement: "_".into(),
                            }],
                            applicability: Applicability::MachineApplicable,
                        },
                    );
                }
            }
        }
    }

    fn check_stmt(&self, _stmt: &Stmt, _ctx: &mut LintContext) {
        // Check for let bindings of affine types
        // Note: actual tracking happens in check_item
    }
}

/// Check a block for affine variable usage
fn check_block_for_affine_usage(
    block: &crate::ast::Block,
    affine_bindings: &mut Vec<(String, Span, bool)>,
) {
    for stmt in &block.stmts {
        match stmt {
            Stmt::Let {
                pattern,
                ty: Some(ty),
                value,
                ..
            } if is_affine_type(ty) => {
                if let Pattern::Binding { name, .. } = pattern {
                    affine_bindings.push((name.clone(), Span { start: 0, end: 0 }, false));
                }

                if let Some(expr) = value {
                    mark_affine_consumed(expr, affine_bindings);
                }
            }
            Stmt::Expr { expr, .. } => {
                mark_affine_consumed(expr, affine_bindings);
            }
            Stmt::Assign { value, .. } => {
                mark_affine_consumed(value, affine_bindings);
            }
            _ => {}
        }
    }
}

/// Mark affine variables as consumed when used
fn mark_affine_consumed(expr: &Expr, affine_bindings: &mut Vec<(String, Span, bool)>) {
    match expr {
        Expr::Path { path, .. } => {
            if !path.segments.is_empty() {
                let name = &path.segments[0];
                for (binding_name, _, consumed) in affine_bindings.iter_mut() {
                    if binding_name == name {
                        *consumed = true;
                    }
                }
            }
        }
        Expr::Call { callee, args, .. } => {
            mark_affine_consumed(callee, affine_bindings);
            for arg in args {
                mark_affine_consumed(arg, affine_bindings);
            }
        }
        Expr::MethodCall { receiver, args, .. } => {
            mark_affine_consumed(receiver, affine_bindings);
            for arg in args {
                mark_affine_consumed(arg, affine_bindings);
            }
        }
        Expr::Binary { left, right, .. } => {
            mark_affine_consumed(left, affine_bindings);
            mark_affine_consumed(right, affine_bindings);
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            mark_affine_consumed(condition, affine_bindings);
            check_block_for_affine_usage(then_branch, affine_bindings);
            if let Some(else_expr) = else_branch {
                mark_affine_consumed(else_expr, affine_bindings);
            }
        }
        Expr::Block { block, .. } => {
            check_block_for_affine_usage(block, affine_bindings);
        }
        _ => {}
    }
}

/// Check if a type is affine
fn is_affine_type(ty: &TypeExpr) -> bool {
    match ty {
        TypeExpr::Named { path, .. } => {
            let name = path.segments.last().map(|s| s.as_str()).unwrap_or("");
            matches!(
                name,
                "Affine" | "Credential" | "Token" | "Ticket" | "Permit" | "Authorization"
            )
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Path;

    #[test]
    fn test_unused_linear_value_lint() {
        let lint = UnusedLinearValue;
        assert_eq!(lint.id(), "unused_linear_value");
        assert_eq!(lint.default_level(), LintLevel::Deny);
        assert_eq!(lint.category(), LintCategory::Safety);
    }

    #[test]
    fn test_linear_value_copied_lint() {
        let lint = LinearValueCopied;
        assert_eq!(lint.id(), "linear_value_copied");
        assert_eq!(lint.default_level(), LintLevel::Deny);
    }

    #[test]
    fn test_linear_escape_lint() {
        let lint = LinearEscape;
        assert_eq!(lint.id(), "linear_escape");
        assert_eq!(lint.default_level(), LintLevel::Deny);
    }

    #[test]
    fn test_affine_not_consumed_lint() {
        let lint = AffineNotConsumed;
        assert_eq!(lint.id(), "affine_not_consumed");
        assert_eq!(lint.default_level(), LintLevel::Warn);
    }

    #[test]
    fn test_is_linear_type() {
        assert!(is_linear_type(&TypeExpr::Named {
            path: Path::simple("FileHandle"),
            args: vec![],
            unit: None,
        }));

        assert!(!is_linear_type(&TypeExpr::Named {
            path: Path::simple("String"),
            args: vec![],
            unit: None,
        }));
    }

    #[test]
    fn test_is_affine_type() {
        assert!(is_affine_type(&TypeExpr::Named {
            path: Path::simple("Credential"),
            args: vec![],
            unit: None,
        }));

        assert!(!is_affine_type(&TypeExpr::Named {
            path: Path::simple("i32"),
            args: vec![],
            unit: None,
        }));
    }
}
