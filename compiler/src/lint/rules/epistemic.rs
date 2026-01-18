//! Epistemic type lints
//!
//! These lints detect issues with Sounio's epistemic type system, ensuring
//! proper handling of Knowledge<T> values, confidence levels, and provenance
//! tracking.
//!
//! # Epistemic Types in Sounio
//!
//! Sounio treats knowledge as a first-class type with:
//! - Confidence (epsilon): How certain are we about this value?
//! - Provenance: Where did this knowledge come from?
//! - Temporal validity: When is this knowledge valid?
//!
//! # Example
//!
//! ```sio
//! // Knowledge with confidence and provenance
//! let measurement: Knowledge<mg> = measure(sample, confidence: 0.95)
//!
//! // Epistemic operations
//! if measurement.confidence >= 0.9 {
//!     // Safe to use high-confidence value
//!     process(measurement.unwrap())
//! }
//! ```

use crate::ast::{Block, Expr, Item, Pattern, Stmt, TypeExpr};
use crate::common::Span;
use crate::lint::{Lint, LintCategory, LintContext, LintLevel};

/// Confidence threshold below which values should be treated with caution
const LOW_CONFIDENCE_THRESHOLD: f64 = 0.7;

// =============================================================================
// Low Confidence Propagation
// =============================================================================

/// Detects propagation of values below confidence threshold
///
/// When Knowledge<T> values with low confidence are used in computations,
/// the resulting uncertainty can compound. This lint warns when low-confidence
/// values are propagated without explicit acknowledgment.
pub struct LowConfidencePropagation;

impl Lint for LowConfidencePropagation {
    fn id(&self) -> &'static str {
        "low_confidence_propagation"
    }

    fn name(&self) -> &'static str {
        "Low Confidence Propagation"
    }

    fn description(&self) -> &'static str {
        "Warns when Knowledge<T> values with low confidence are propagated without explicit handling."
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::Epistemic
    }

    fn check_expr(&self, expr: &Expr, ctx: &mut LintContext) {
        // Check for Knowledge literal with low confidence
        if let Expr::StructLit { path, fields, .. } = expr {
            let name = path.segments.join("::");
            if name == "Knowledge" || name.ends_with("::Knowledge") {
                // Look for confidence field
                for (field_name, field_value) in fields {
                    if field_name == "confidence" || field_name == "epsilon" {
                        if let Some(conf_value) = extract_float_literal(field_value) {
                            if conf_value < LOW_CONFIDENCE_THRESHOLD {
                                ctx.report_with_help(
                                    self,
                                    format!(
                                        "Knowledge value has low confidence ({:.2}), below threshold {:.2}",
                                        conf_value, LOW_CONFIDENCE_THRESHOLD
                                    ),
                                    Span::dummy(),
                                    "consider checking confidence before propagating this value, or use epistemic composition operators",
                                );
                            }
                        }
                    }
                }
            }
        }

        // Check for binary operations involving Knowledge types without confidence check
        if let Expr::Binary {
            left, right, op, ..
        } = expr
        {
            let left_is_knowledge = is_knowledge_expr(left);
            let right_is_knowledge = is_knowledge_expr(right);

            if left_is_knowledge || right_is_knowledge {
                // Arithmetic on Knowledge values should use epistemic operators
                if matches!(
                    op,
                    crate::ast::BinaryOp::Add
                        | crate::ast::BinaryOp::Sub
                        | crate::ast::BinaryOp::Mul
                        | crate::ast::BinaryOp::Div
                ) {
                    ctx.report_with_help(
                        self,
                        "arithmetic operation on Knowledge values may propagate uncertainty",
                        Span::dummy(),
                        "use epistemic composition operators (compose, merge) to properly track uncertainty propagation",
                    );
                }
            }
        }

        // Check for method calls that might extract values without confidence check
        if let Expr::MethodCall {
            method, receiver, ..
        } = expr
        {
            if method == "unwrap" || method == "value" {
                if is_knowledge_expr(receiver) {
                    ctx.report_with_help(
                        self,
                        "extracting value from Knowledge without checking confidence",
                        Span::dummy(),
                        "consider using `unwrap_checked(threshold)` or checking `confidence >= threshold` first",
                    );
                }
            }
        }
    }
}

/// Check if an expression is a Knowledge type
fn is_knowledge_expr(expr: &Expr) -> bool {
    match expr {
        Expr::StructLit { path, .. } => {
            let name = path.segments.join("::");
            name == "Knowledge" || name.ends_with("::Knowledge")
        }
        Expr::MethodCall {
            method, receiver, ..
        } => {
            // Methods that return Knowledge
            let knowledge_methods = ["measure", "sample", "observe", "infer", "estimate"];
            knowledge_methods.contains(&method.as_str()) || is_knowledge_expr(receiver)
        }
        Expr::Call { callee, .. } => {
            if let Expr::Path { path, .. } = callee.as_ref() {
                let name = path.segments.join("::");
                name.contains("Knowledge") || name.contains("measure") || name.contains("observe")
            } else {
                false
            }
        }
        Expr::Path { path, .. } => {
            // Variable reference - check naming convention
            !path.segments.is_empty()
                && (path.segments[0].starts_with("knowledge")
                    || path.segments[0].starts_with("measurement")
                    || path.segments[0].starts_with("observation"))
        }
        _ => false,
    }
}

/// Check if a type is Knowledge<T>
fn is_knowledge_type(ty: &TypeExpr) -> bool {
    match ty {
        TypeExpr::Named { path, .. } => {
            let name = path.segments.join("::");
            name == "Knowledge" || name.starts_with("Knowledge")
        }
        TypeExpr::Knowledge { .. } => true,
        _ => false,
    }
}

/// Extract float literal value from an expression
fn extract_float_literal(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Literal {
            value: crate::ast::Literal::Float(f),
            ..
        } => Some(*f),
        Expr::Literal {
            value: crate::ast::Literal::Int(i),
            ..
        } => Some(*i as f64),
        _ => None,
    }
}

// =============================================================================
// Untraceable Provenance
// =============================================================================

/// Detects Knowledge<T> values losing provenance chain
///
/// Provenance tracking is essential for epistemic computing. This lint
/// warns when Knowledge values are transformed in ways that break the
/// provenance chain.
pub struct UntraceableProvenance;

impl Lint for UntraceableProvenance {
    fn id(&self) -> &'static str {
        "untraceable_provenance"
    }

    fn name(&self) -> &'static str {
        "Untraceable Provenance"
    }

    fn description(&self) -> &'static str {
        "Warns when Knowledge<T> values lose their provenance tracking chain."
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::Epistemic
    }

    fn check_expr(&self, expr: &Expr, ctx: &mut LintContext) {
        // Check for Knowledge construction without provenance
        if let Expr::StructLit { path, fields, .. } = expr {
            let name = path.segments.join("::");
            if name == "Knowledge" || name.ends_with("::Knowledge") {
                let has_provenance = fields.iter().any(|(field_name, _)| {
                    field_name == "provenance" || field_name == "source" || field_name == "origin"
                });

                if !has_provenance {
                    ctx.report_with_help(
                        self,
                        "Knowledge value created without provenance information",
                        Span::dummy(),
                        "add a provenance field to track the origin of this knowledge (e.g., provenance: Measured { source: \"sensor_1\" })",
                    );
                }
            }
        }

        // Check for operations that might lose provenance
        if let Expr::Call { callee, args, .. } = expr {
            if let Expr::Path { path, .. } = callee.as_ref() {
                let name = path.segments.join("::");

                // Functions that should preserve provenance
                let provenance_breaking = ["clone", "copy", "cast", "transmute"];
                if provenance_breaking.iter().any(|pb| name.contains(pb)) {
                    // Check if any argument is a Knowledge type
                    for arg in args {
                        if is_knowledge_expr(arg) {
                            ctx.report_with_help(
                                self,
                                format!(
                                    "operation `{}` may break provenance chain of Knowledge value",
                                    name
                                ),
                                Span::dummy(),
                                "use provenance-preserving operations like `derive_from` to maintain the chain",
                            );
                        }
                    }
                }
            }
        }

        // Check for direct value extraction without provenance propagation
        if let Expr::MethodCall {
            method, receiver, ..
        } = expr
        {
            if method == "unwrap" || method == "value" || method == "inner" {
                if is_knowledge_expr(receiver) {
                    ctx.report_with_help(
                        self,
                        "extracting raw value from Knowledge loses provenance",
                        Span::dummy(),
                        "if provenance must be preserved, use `map` or `and_then` to transform while maintaining provenance",
                    );
                }
            }
        }
    }
}

// =============================================================================
// Confidence Not Checked
// =============================================================================

/// Warns when uncertain values are used without checking confidence
///
/// Using Knowledge<T> values without first checking their confidence level
/// can lead to making decisions based on unreliable data.
pub struct ConfidenceNotChecked;

impl Lint for ConfidenceNotChecked {
    fn id(&self) -> &'static str {
        "confidence_not_checked"
    }

    fn name(&self) -> &'static str {
        "Confidence Not Checked"
    }

    fn description(&self) -> &'static str {
        "Warns when Knowledge<T> values are used without first checking their confidence level."
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::Epistemic
    }

    fn check_expr(&self, expr: &Expr, ctx: &mut LintContext) {
        // Check for direct use of Knowledge in critical operations
        if let Expr::Call { callee, args, .. } = expr {
            if let Expr::Path { path, .. } = callee.as_ref() {
                let name = path.segments.join("::");

                // Critical operations that should require confidence check
                let critical_ops = [
                    "commit", "save", "persist", "submit", "execute", "apply", "deploy", "publish",
                ];

                if critical_ops.iter().any(|op| name.contains(op)) {
                    for arg in args {
                        if is_knowledge_expr(arg) {
                            ctx.report_with_help(
                                self,
                                format!(
                                    "Knowledge value passed to critical operation `{}` without confidence check",
                                    name
                                ),
                                Span::dummy(),
                                "check confidence level before passing to critical operations: `if knowledge.confidence >= threshold { ... }`",
                            );
                        }
                    }
                }
            }
        }

        // Check for unwrap without prior confidence check
        if let Expr::MethodCall {
            method, receiver, ..
        } = expr
        {
            if method == "unwrap" && is_knowledge_expr(receiver) {
                ctx.report_with_help(
                    self,
                    "calling `unwrap()` on Knowledge without checking confidence",
                    Span::dummy(),
                    "use `unwrap_if(threshold)` or check confidence first with an if statement",
                );
            }
        }

        // Check for Knowledge in conditionals that could branch on uncertain values
        if let Expr::If { condition, .. } = expr {
            if is_knowledge_expr(condition) && !contains_confidence_check(condition) {
                ctx.report_with_help(
                    self,
                    "conditional on Knowledge value without confidence check",
                    Span::dummy(),
                    "include confidence check in condition: `if knowledge.confidence >= threshold && knowledge.value > x`",
                );
            }
        }
    }
}

/// Check if an expression contains a confidence check
fn contains_confidence_check(expr: &Expr) -> bool {
    match expr {
        Expr::Binary {
            left, right, op, ..
        } => {
            let left_is_confidence = is_confidence_access(left);
            let right_is_confidence = is_confidence_access(right);

            if (left_is_confidence || right_is_confidence)
                && matches!(
                    op,
                    crate::ast::BinaryOp::Ge
                        | crate::ast::BinaryOp::Gt
                        | crate::ast::BinaryOp::Le
                        | crate::ast::BinaryOp::Lt
                        | crate::ast::BinaryOp::Eq
                )
            {
                return true;
            }

            // Recurse for compound conditions
            if matches!(op, crate::ast::BinaryOp::And | crate::ast::BinaryOp::Or) {
                return contains_confidence_check(left) || contains_confidence_check(right);
            }

            false
        }
        Expr::MethodCall { method, .. } => {
            let confidence_methods = [
                "is_confident",
                "is_reliable",
                "has_sufficient_confidence",
                "confidence_above",
            ];
            confidence_methods.contains(&method.as_str())
        }
        _ => false,
    }
}

/// Check if an expression is a confidence field access
fn is_confidence_access(expr: &Expr) -> bool {
    match expr {
        Expr::Field { field, .. } => {
            field == "confidence" || field == "epsilon" || field == "certainty"
        }
        Expr::MethodCall { method, .. } => {
            method == "confidence" || method == "get_confidence" || method == "certainty"
        }
        _ => false,
    }
}

// =============================================================================
// Stale Knowledge
// =============================================================================

/// Warns when Knowledge<T> values may be stale (temporal decay)
///
/// Knowledge has temporal validity. This lint warns when Knowledge values
/// are used that may have become stale due to time passing.
pub struct StaleKnowledge;

impl Lint for StaleKnowledge {
    fn id(&self) -> &'static str {
        "stale_knowledge"
    }

    fn name(&self) -> &'static str {
        "Stale Knowledge"
    }

    fn description(&self) -> &'static str {
        "Warns when Knowledge<T> values may be stale due to temporal decay."
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::Epistemic
    }

    fn check_expr(&self, expr: &Expr, ctx: &mut LintContext) {
        // Check for Knowledge usage without validity check
        if let Expr::MethodCall {
            method, receiver, ..
        } = expr
        {
            if (method == "unwrap" || method == "value" || method == "get")
                && is_knowledge_expr(receiver)
            {
                ctx.report_with_help(
                    self,
                    "accessing Knowledge value without checking temporal validity",
                    Span::dummy(),
                    "check `is_valid()` or `validity_remaining()` before accessing potentially stale knowledge",
                );
            }
        }

        // Check for Knowledge in loops without refresh
        if let Expr::Loop { body, .. } | Expr::While { body, .. } | Expr::For { body, .. } = expr {
            let mut knowledge_usages: Vec<String> = Vec::new();
            find_knowledge_in_block(body, &mut knowledge_usages);

            if !knowledge_usages.is_empty() && !has_refresh_in_block(body) {
                ctx.report_with_help(
                    self,
                    format!(
                        "Knowledge value(s) {:?} used in loop without refresh",
                        knowledge_usages
                    ),
                    Span::dummy(),
                    "long-running loops should periodically refresh Knowledge values to ensure freshness",
                );
            }
        }
    }

    fn check_item(&self, item: &Item, ctx: &mut LintContext) {
        // Check for global/static Knowledge values (likely to become stale)
        if let Item::Global(g) = item {
            if let Pattern::Binding { name, .. } = &g.pattern {
                if let Some(ty) = &g.ty {
                    if is_knowledge_type(ty) {
                        ctx.report_with_help(
                            self,
                            format!(
                                "global Knowledge value `{}` may become stale over time",
                                name
                            ),
                            g.span,
                            "consider using a function to fetch fresh Knowledge, or implement a refresh mechanism",
                        );
                    }
                }
            }
        }
    }
}

/// Find Knowledge usages in a block
fn find_knowledge_in_block(block: &Block, usages: &mut Vec<String>) {
    for stmt in &block.stmts {
        match stmt {
            Stmt::Expr { expr, .. } => {
                find_knowledge_in_expr(expr, usages);
            }
            Stmt::Let {
                value: Some(expr), ..
            } => {
                find_knowledge_in_expr(expr, usages);
            }
            Stmt::Assign { value, target, .. } => {
                find_knowledge_in_expr(value, usages);
                find_knowledge_in_expr(target, usages);
            }
            _ => {}
        }
    }
}

/// Find Knowledge usages in an expression
fn find_knowledge_in_expr(expr: &Expr, usages: &mut Vec<String>) {
    match expr {
        Expr::Path { path, .. } => {
            if !path.segments.is_empty() {
                let name = &path.segments[0];
                if name.starts_with("knowledge")
                    || name.starts_with("measurement")
                    || name.ends_with("_knowledge")
                {
                    usages.push(name.clone());
                }
            }
        }
        Expr::MethodCall {
            receiver, args, ..
        } => {
            find_knowledge_in_expr(receiver, usages);
            for arg in args {
                find_knowledge_in_expr(arg, usages);
            }
        }
        Expr::Call { callee, args, .. } => {
            find_knowledge_in_expr(callee, usages);
            for arg in args {
                find_knowledge_in_expr(arg, usages);
            }
        }
        Expr::Binary { left, right, .. } => {
            find_knowledge_in_expr(left, usages);
            find_knowledge_in_expr(right, usages);
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            find_knowledge_in_expr(condition, usages);
            find_knowledge_in_block(then_branch, usages);
            if let Some(else_expr) = else_branch {
                find_knowledge_in_expr(else_expr, usages);
            }
        }
        Expr::Block { block, .. } => {
            find_knowledge_in_block(block, usages);
        }
        _ => {}
    }
}

/// Check if a block contains a refresh operation
fn has_refresh_in_block(block: &Block) -> bool {
    for stmt in &block.stmts {
        match stmt {
            Stmt::Expr { expr, .. } => {
                if has_refresh_in_expr(expr) {
                    return true;
                }
            }
            Stmt::Let {
                value: Some(expr), ..
            } => {
                if has_refresh_in_expr(expr) {
                    return true;
                }
            }
            Stmt::Assign { value, .. } => {
                if has_refresh_in_expr(value) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Check if an expression contains a refresh operation
fn has_refresh_in_expr(expr: &Expr) -> bool {
    match expr {
        Expr::MethodCall { method, .. } => {
            let refresh_methods = ["refresh", "update", "renew", "fetch", "reload", "revalidate"];
            refresh_methods.contains(&method.as_str())
        }
        Expr::Call { callee, .. } => {
            if let Expr::Path { path, .. } = callee.as_ref() {
                let name = path.segments.join("::");
                name.contains("refresh") || name.contains("fetch") || name.contains("reload")
            } else {
                false
            }
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_low_confidence_propagation_lint() {
        let lint = LowConfidencePropagation;
        assert_eq!(lint.id(), "low_confidence_propagation");
        assert_eq!(lint.default_level(), LintLevel::Warn);
        assert_eq!(lint.category(), LintCategory::Epistemic);
    }

    #[test]
    fn test_untraceable_provenance_lint() {
        let lint = UntraceableProvenance;
        assert_eq!(lint.id(), "untraceable_provenance");
        assert_eq!(lint.default_level(), LintLevel::Warn);
    }

    #[test]
    fn test_confidence_not_checked_lint() {
        let lint = ConfidenceNotChecked;
        assert_eq!(lint.id(), "confidence_not_checked");
        assert_eq!(lint.default_level(), LintLevel::Warn);
    }

    #[test]
    fn test_stale_knowledge_lint() {
        let lint = StaleKnowledge;
        assert_eq!(lint.id(), "stale_knowledge");
        assert_eq!(lint.default_level(), LintLevel::Warn);
    }

    #[test]
    fn test_is_knowledge_type() {
        use crate::ast::Path;

        assert!(is_knowledge_type(&TypeExpr::Named {
            path: Path::simple("Knowledge"),
            args: vec![],
            unit: None,
        }));

        assert!(!is_knowledge_type(&TypeExpr::Named {
            path: Path::simple("String"),
            args: vec![],
            unit: None,
        }));
    }

    #[test]
    fn test_extract_float_literal() {
        use crate::ast::Literal;
        use crate::common::NodeId;

        let float_expr = Expr::Literal {
            id: NodeId::dummy(),
            value: Literal::Float(0.95),
        };
        assert_eq!(extract_float_literal(&float_expr), Some(0.95));

        let int_expr = Expr::Literal {
            id: NodeId::dummy(),
            value: Literal::Int(1),
        };
        assert_eq!(extract_float_literal(&int_expr), Some(1.0));
    }
}
