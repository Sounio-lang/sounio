//! Performance lints
//!
//! These lints detect potential performance issues in Sounio code, including
//! expensive operations in loops, unnecessary allocations, and large stack
//! values.
//!
//! # Performance Considerations in Sounio
//!
//! Sounio is designed for scientific computing where performance matters.
//! These lints help identify common performance anti-patterns:
//!
//! - Cloning large values inside loops
//! - Heap allocations when stack would suffice
//! - Passing large values by value instead of reference
//!
//! # Example
//!
//! ```sio
//! // Avoid this:
//! for i in 0..1000 {
//!     let copy = large_matrix.clone()  // WARNING: expensive clone in loop
//!     process(copy)
//! }
//!
//! // Better:
//! for i in 0..1000 {
//!     process(&large_matrix)  // Pass by reference
//! }
//! ```

// Compiler internals often require deep nesting for pattern matching and control flow.
#![allow(clippy::excessive_nesting)]

use crate::ast::{Block, Expr, Item, Pattern, Stmt, TypeExpr};
use crate::common::Span;
use crate::lint::{Lint, LintCategory, LintContext, LintLevel};

/// Size threshold in bytes above which values are considered "large"
const LARGE_VALUE_THRESHOLD: usize = 128;

/// Known expensive types to clone
const EXPENSIVE_TYPES: &[&str] = &[
    "Vec",
    "String",
    "HashMap",
    "BTreeMap",
    "HashSet",
    "BTreeSet",
    "Matrix",
    "Tensor",
    "Image",
    "DataFrame",
    "Knowledge",
    "Box",
    "Rc",
    "Arc",
];

// =============================================================================
// Expensive Clone In Loop
// =============================================================================

/// Detects expensive clone operations inside loops
///
/// Cloning large or complex data structures inside a loop can severely
/// impact performance. This lint identifies such cases and suggests
/// alternatives like moving the clone outside the loop or using references.
pub struct ExpensiveCloneInLoop;

impl Lint for ExpensiveCloneInLoop {
    fn id(&self) -> &'static str {
        "expensive_clone_in_loop"
    }

    fn name(&self) -> &'static str {
        "Expensive Clone In Loop"
    }

    fn description(&self) -> &'static str {
        "Warns when cloning expensive types inside loops."
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::Performance
    }

    fn check_expr(&self, expr: &Expr, ctx: &mut LintContext) {
        match expr {
            Expr::Loop { body, .. } => {
                check_block_for_clones(body, ctx, self, "loop");
            }
            Expr::While { body, .. } => {
                check_block_for_clones(body, ctx, self, "while loop");
            }
            Expr::For { body, .. } => {
                check_block_for_clones(body, ctx, self, "for loop");
            }
            _ => {}
        }
    }
}

/// Check a block for expensive clone operations
fn check_block_for_clones(
    block: &Block,
    ctx: &mut LintContext,
    lint: &ExpensiveCloneInLoop,
    loop_type: &str,
) {
    let mut clone_calls: Vec<String> = Vec::new();

    fn find_clones_in_block(block: &Block, clones: &mut Vec<String>) {
        for stmt in &block.stmts {
            match stmt {
                Stmt::Expr { expr, .. } => {
                    find_clones_in_expr(expr, clones);
                }
                Stmt::Let {
                    value: Some(expr), ..
                } => {
                    find_clones_in_expr(expr, clones);
                }
                Stmt::Assign { value, .. } => {
                    find_clones_in_expr(value, clones);
                }
                _ => {}
            }
        }
    }

    fn find_clones_in_expr(expr: &Expr, clones: &mut Vec<String>) {
        match expr {
            Expr::MethodCall {
                method, receiver, ..
            } => {
                if method == "clone" || method == "to_owned" || method == "to_vec" {
                    // Check if receiver is an expensive type
                    if let Expr::Path { path, .. } = receiver.as_ref() {
                        if !path.segments.is_empty() {
                            let name = path.segments.join("::");
                            clones.push(name);
                        }
                    } else {
                        // Any clone in a loop is potentially expensive
                        clones.push("unknown".to_string());
                    }
                }
                find_clones_in_expr(receiver, clones);
            }
            Expr::Call { callee, args, .. } => {
                find_clones_in_expr(callee, clones);
                for arg in args {
                    find_clones_in_expr(arg, clones);
                }
            }
            Expr::Binary { left, right, .. } => {
                find_clones_in_expr(left, clones);
                find_clones_in_expr(right, clones);
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                find_clones_in_expr(condition, clones);
                find_clones_in_block(then_branch, clones);
                if let Some(else_expr) = else_branch {
                    find_clones_in_expr(else_expr, clones);
                }
            }
            Expr::Block { block, .. } => {
                find_clones_in_block(block, clones);
            }
            // Don't recurse into nested loops - they have their own check
            _ => {}
        }
    }

    find_clones_in_block(block, &mut clone_calls);

    for name in clone_calls {
        ctx.report_with_help(
            lint,
            format!("potentially expensive clone of `{}` inside {}", name, loop_type),
            Span::dummy(),
            "consider moving the clone outside the loop, using a reference, or using Cow for conditionally-owned data",
        );
    }
}

// =============================================================================
// Unnecessary Allocation
// =============================================================================

/// Detects heap allocations that could potentially use stack allocation
///
/// Some heap allocations (like small Vec or Box) could be replaced with
/// stack-based alternatives for better performance.
pub struct UnnecessaryAllocation;

impl Lint for UnnecessaryAllocation {
    fn id(&self) -> &'static str {
        "unnecessary_allocation"
    }

    fn name(&self) -> &'static str {
        "Unnecessary Allocation"
    }

    fn description(&self) -> &'static str {
        "Warns when heap allocation could potentially be replaced with stack allocation."
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::Performance
    }

    fn check_expr(&self, expr: &Expr, ctx: &mut LintContext) {
        // Check for Box::new with small types
        if let Expr::Call { callee, args, .. } = expr {
            if let Expr::Path { path, .. } = callee.as_ref() {
                let name = path.segments.join("::");

                // Check for Box::new(small_value)
                if name == "Box::new" || name == "box" {
                    if args.len() == 1 {
                        if is_small_type_expr(&args[0]) {
                            ctx.report_with_help(
                                self,
                                "unnecessary heap allocation for small value",
                                Span::dummy(),
                                "consider using the value directly on the stack instead of boxing it",
                            );
                        }
                    }
                }

                // Check for Vec::new() when a fixed array could work
                if name == "Vec::new" || name == "vec!" {
                    // This is often fine, but we can suggest alternatives
                    // when the size is known at compile time
                }

                // Check for String::new() when &str could work
                if name == "String::new" {
                    ctx.report_with_help(
                        self,
                        "empty String allocation",
                        Span::dummy(),
                        "if you just need an empty string literal, use \"\" instead of String::new()",
                    );
                }

                // Check for String::from("literal")
                if name == "String::from" || name == "to_string" {
                    if args.len() == 1 {
                        if is_string_literal(&args[0]) {
                            ctx.report_with_help(
                                self,
                                "String allocation from literal",
                                Span::dummy(),
                                "if the string doesn't need to be mutated, consider using &str instead",
                            );
                        }
                    }
                }
            }
        }

        // Check for to_vec() on slices when slice would work
        if let Expr::MethodCall { method, .. } = expr {
            if method == "to_vec" {
                ctx.report_with_help(
                    self,
                    "potential unnecessary allocation with to_vec()",
                    Span::dummy(),
                    "if you don't need to modify the data, consider using a slice reference instead",
                );
            }
        }
    }
}

/// Check if an expression is likely a small type (fits in register)
fn is_small_type_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Literal { .. } => true,
        Expr::Path { .. } => {
            // Could be small, would need type info to be sure
            false
        }
        Expr::Tuple { elements, .. } => {
            elements.len() <= 2 && elements.iter().all(is_small_type_expr)
        }
        _ => false,
    }
}

/// Check if an expression is a string literal
fn is_string_literal(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Literal {
            value: crate::ast::Literal::String(_),
            ..
        }
    )
}

// =============================================================================
// Large Stack Value
// =============================================================================

/// Detects large values being passed by value instead of by reference
///
/// Passing large structs or arrays by value incurs a copy. For large types,
/// it's usually better to pass by reference.
pub struct LargeStackValue;

impl Lint for LargeStackValue {
    fn id(&self) -> &'static str {
        "large_stack_value"
    }

    fn name(&self) -> &'static str {
        "Large Stack Value"
    }

    fn description(&self) -> &'static str {
        "Warns when large values are passed by value instead of by reference."
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::Performance
    }

    fn check_item(&self, item: &Item, ctx: &mut LintContext) {
        // Check function parameters for large types passed by value
        if let Item::Function(f) = item {
            for param in &f.params {
                let ty = &param.ty;
                if is_large_type(ty) && !is_reference_type(ty) {
                    let param_name = match &param.pattern {
                        Pattern::Binding { name, .. } => name.as_str(),
                        _ => "?",
                    };
                    ctx.report_with_help(
                        self,
                        format!("parameter `{}` has large type passed by value", param_name),
                        Span::dummy(),
                        "consider passing by reference (&T) to avoid copying",
                    );
                }
            }
        }

        // Check struct definitions for very large inline fields
        if let Item::Struct(s) = item {
            for field in &s.fields {
                if is_very_large_type(&field.ty) {
                    ctx.report_with_help(
                        self,
                        format!("field `{}` has very large type", field.name),
                        Span::dummy(),
                        "consider using Box<T> to heap-allocate the large value",
                    );
                }
            }
        }
    }

    fn check_expr(&self, expr: &Expr, ctx: &mut LintContext) {
        // Check function calls with large value arguments
        if let Expr::Call { args, .. } = expr {
            for arg in args {
                // Note: Would need type information to properly check this
                // For now, check for known large type constructors
                if is_large_value_expr(arg) {
                    ctx.report_with_help(
                        self,
                        "large value passed by value to function",
                        Span::dummy(),
                        "if the function doesn't need ownership, consider passing a reference",
                    );
                }
            }
        }

        // Check for large arrays on stack
        if let Expr::Array { elements, .. } = expr {
            if elements.len() > 1024 {
                ctx.report_with_help(
                    self,
                    format!("large array with {} elements on stack", elements.len()),
                    Span::dummy(),
                    "consider using Vec<T> or Box<[T]> for large arrays",
                );
            }
        }

        // Check for large struct literals
        if let Expr::StructLit { fields, .. } = expr {
            if fields.len() > 20 {
                ctx.report_with_help(
                    self,
                    format!("struct literal with {} fields", fields.len()),
                    Span::dummy(),
                    "structs with many fields may benefit from being boxed or using a builder pattern",
                );
            }
        }
    }
}

/// Check if a type is considered "large"
fn is_large_type(ty: &TypeExpr) -> bool {
    match ty {
        TypeExpr::Array { size: Some(n), .. } => {
            // Check if size expression is a literal we can evaluate
            if let Expr::Literal {
                value: crate::ast::Literal::Int(size),
                ..
            } = n.as_ref()
            {
                *size > 16
            } else {
                false
            }
        }
        TypeExpr::Named { path, .. } => {
            // Known large types
            let large_types = ["Matrix", "Tensor", "Image", "DataFrame", "Buffer", "Chunk"];
            path.segments
                .last()
                .map_or(false, |name| large_types.contains(&name.as_str()))
        }
        TypeExpr::Tuple(elements) => elements.len() > 4,
        _ => false,
    }
}

/// Check if a type is "very large" (should definitely be boxed)
fn is_very_large_type(ty: &TypeExpr) -> bool {
    match ty {
        TypeExpr::Array { size: Some(n), .. } => {
            if let Expr::Literal {
                value: crate::ast::Literal::Int(size),
                ..
            } = n.as_ref()
            {
                *size > 1024
            } else {
                false
            }
        }
        TypeExpr::Named { path, .. } => {
            // Known very large types
            let very_large = ["LargeBuffer", "ImageBuffer", "VoxelGrid"];
            path.segments
                .last()
                .map_or(false, |name| very_large.contains(&name.as_str()))
        }
        TypeExpr::Tuple(elements) => elements.len() > 10,
        _ => false,
    }
}

/// Check if a type is a reference type
fn is_reference_type(ty: &TypeExpr) -> bool {
    matches!(ty, TypeExpr::Reference { .. })
}

/// Check if an expression creates a large value
fn is_large_value_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Array { elements, .. } => elements.len() > 32,
        Expr::StructLit { fields, .. } => fields.len() > 10,
        Expr::Call { callee, .. } => {
            // Check for known large type constructors
            if let Expr::Path { path, .. } = callee.as_ref() {
                let name = path.segments.join("::");
                let large_constructors = ["Matrix::new", "Tensor::new", "Image::new"];
                large_constructors.iter().any(|c| name.contains(c))
            } else {
                false
            }
        }
        _ => false,
    }
}

// =============================================================================
// Unnecessary Clone
// =============================================================================

/// Warns about unnecessary clone operations
///
/// Clone operations on types that implement Copy are unnecessary since
/// they are automatically copied.
pub struct UnnecessaryClone;

impl Lint for UnnecessaryClone {
    fn id(&self) -> &'static str {
        "unnecessary_clone"
    }

    fn name(&self) -> &'static str {
        "Unnecessary Clone"
    }

    fn description(&self) -> &'static str {
        "Warns about clone operations that may be unnecessary."
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::Performance
    }

    fn check_expr(&self, expr: &Expr, ctx: &mut LintContext) {
        // Check for .clone() on literals or simple values
        if let Expr::MethodCall {
            method, receiver, ..
        } = expr
        {
            if method == "clone" {
                // Check if receiver is a simple value that doesn't need cloning
                if is_trivially_copyable(receiver) {
                    ctx.report_with_help(
                        self,
                        "calling clone() on a value that implements Copy",
                        Span::dummy(),
                        "remove the .clone() call; Copy types are cloned automatically",
                    );
                }
            }
        }
    }
}

/// Check if an expression is trivially copyable (Copy type)
fn is_trivially_copyable(expr: &Expr) -> bool {
    match expr {
        Expr::Literal { value, .. } => {
            // Numeric and boolean literals are Copy
            matches!(
                value,
                crate::ast::Literal::Int(_)
                    | crate::ast::Literal::Float(_)
                    | crate::ast::Literal::Bool(_)
                    | crate::ast::Literal::Char(_)
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
    fn test_lint_ids() {
        assert_eq!(ExpensiveCloneInLoop.id(), "expensive_clone_in_loop");
        assert_eq!(UnnecessaryAllocation.id(), "unnecessary_allocation");
        assert_eq!(LargeStackValue.id(), "large_stack_value");
    }

    #[test]
    fn test_lint_categories() {
        assert_eq!(ExpensiveCloneInLoop.category(), LintCategory::Performance);
        assert_eq!(UnnecessaryAllocation.category(), LintCategory::Performance);
        assert_eq!(LargeStackValue.category(), LintCategory::Performance);
    }

    #[test]
    fn test_lint_levels() {
        assert_eq!(ExpensiveCloneInLoop.default_level(), LintLevel::Warn);
        assert_eq!(UnnecessaryAllocation.default_level(), LintLevel::Warn);
        assert_eq!(LargeStackValue.default_level(), LintLevel::Warn);
    }

    #[test]
    fn test_is_large_type() {
        // Tuples with more than 4 elements are considered large
        assert!(is_large_type(&TypeExpr::Tuple(vec![
            TypeExpr::Named {
                path: Path::simple("i32"),
                args: vec![],
                unit: None
            },
            TypeExpr::Named {
                path: Path::simple("i32"),
                args: vec![],
                unit: None
            },
            TypeExpr::Named {
                path: Path::simple("i32"),
                args: vec![],
                unit: None
            },
            TypeExpr::Named {
                path: Path::simple("i32"),
                args: vec![],
                unit: None
            },
            TypeExpr::Named {
                path: Path::simple("i32"),
                args: vec![],
                unit: None
            },
        ])));

        assert!(!is_large_type(&TypeExpr::Tuple(vec![
            TypeExpr::Named {
                path: Path::simple("i32"),
                args: vec![],
                unit: None
            },
            TypeExpr::Named {
                path: Path::simple("i32"),
                args: vec![],
                unit: None
            },
        ])));
    }

    #[test]
    fn test_is_reference_type() {
        assert!(is_reference_type(&TypeExpr::Reference {
            mutable: false,
            inner: Box::new(TypeExpr::Named {
                path: Path::simple("i32"),
                args: vec![],
                unit: None,
            }),
        }));

        assert!(!is_reference_type(&TypeExpr::Named {
            path: Path::simple("i32"),
            args: vec![],
            unit: None,
        }));
    }
}
