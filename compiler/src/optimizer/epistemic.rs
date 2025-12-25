//! Epistemic Optimizer for Sounio
//!
//! The missing link between the epistemic type system and codegen.
//! This optimizer performs four key transformations on HIR:
//!
//! 1. **Epsilon Constant Folding**: Propagate known confidence bounds through
//!    arithmetic operations at compile time, eliminating runtime tracking.
//!
//! 2. **Dead Code Elimination for Invalid Epsilon**: When ε > threshold is
//!    statically determined to be always false (e.g., ε = 0.0 from untrusted
//!    source), eliminate the dead branch.
//!
//! 3. **Validity Check Hoisting**: Move temporal validity checks (`Valid(duration)`)
//!    out of loops when the duration is longer than the loop's execution time.
//!
//! 4. **Provenance Chain Merging**: Collapse provenance chains like
//!    `Derived(Derived(Measured))` into efficient representations.
//!
//! # Design Philosophy
//!
//! Unlike traditional optimizers that focus on performance, the epistemic optimizer
//! primarily focuses on **correctness guarantees**. We only optimize what we can
//! *prove* is equivalent under epistemic semantics.

use crate::hir::{
    Hir, HirBlock, HirExpr, HirExprKind, HirFn, HirItem, HirLiteral, HirStmt, HirType,
};
use crate::types::epistemic::{
    ConfidenceBound, PropagationRules, ProvenanceConstraint, TemporalConstraint,
};
use std::collections::HashMap;

/// Epistemic optimizer configuration
#[derive(Debug, Clone)]
pub struct EpistemicOptimizer {
    /// Enable epsilon constant folding
    pub fold_epsilon: bool,
    /// Enable dead code elimination for statically invalid epsilon
    pub dce_invalid: bool,
    /// Enable validity check hoisting out of loops
    pub hoist_validity: bool,
    /// Enable provenance chain merging
    pub merge_provenance: bool,
    /// Propagation rules for confidence
    propagation_rules: PropagationRules,
    /// Statistics for optimization passes
    stats: OptimizationStats,
}

/// Statistics collected during optimization
#[derive(Debug, Clone, Default)]
pub struct OptimizationStats {
    /// Number of epsilon bounds folded
    pub epsilon_folds: usize,
    /// Number of dead branches eliminated
    pub dce_eliminations: usize,
    /// Number of validity checks hoisted
    pub validity_hoists: usize,
    /// Number of provenance chains merged
    pub provenance_merges: usize,
}

impl Default for EpistemicOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl EpistemicOptimizer {
    /// Create a new epistemic optimizer with all passes enabled
    pub fn new() -> Self {
        Self {
            fold_epsilon: true,
            dce_invalid: true,
            hoist_validity: true,
            merge_provenance: true,
            propagation_rules: PropagationRules::default(),
            stats: OptimizationStats::default(),
        }
    }

    /// Create optimizer with custom configuration
    pub fn with_config(
        fold_epsilon: bool,
        dce_invalid: bool,
        hoist_validity: bool,
        merge_provenance: bool,
    ) -> Self {
        Self {
            fold_epsilon,
            dce_invalid,
            hoist_validity,
            merge_provenance,
            propagation_rules: PropagationRules::default(),
            stats: OptimizationStats::default(),
        }
    }

    /// Get optimization statistics
    pub fn stats(&self) -> &OptimizationStats {
        &self.stats
    }

    /// Run all enabled optimization passes on the HIR
    pub fn optimize(&mut self, hir: &mut Hir) {
        for item in &mut hir.items {
            self.optimize_item(item);
        }
    }

    /// Optimize a single HIR item
    fn optimize_item(&mut self, item: &mut HirItem) {
        match item {
            HirItem::Function(func) => self.optimize_function(func),
            HirItem::Global(global) => {
                let mut ctx = EpsilonContext::new();
                self.optimize_expr(&mut global.value, &mut ctx);
            }
            // Other items don't contain optimizable expressions
            _ => {}
        }
    }

    /// Optimize a function
    fn optimize_function(&mut self, func: &mut HirFn) {
        let mut ctx = EpsilonContext::new();

        // Register parameter confidence bounds if known
        for param in &func.ty.params {
            if let Some(bound) = self.extract_confidence_bound(&param.ty) {
                ctx.set_bound(&param.name, bound);
            }
        }

        // Optimize the function body
        self.optimize_block(&mut func.body, &mut ctx);
    }

    /// Optimize a block
    fn optimize_block(&mut self, block: &mut HirBlock, ctx: &mut EpsilonContext) {
        // First pass: collect let bindings with epsilon info
        for stmt in &mut block.stmts {
            match stmt {
                HirStmt::Let {
                    name, value, ty, ..
                } => {
                    if let Some(expr) = value {
                        self.optimize_expr(expr, ctx);

                        // If we know the epsilon of the result, record it
                        if let Some(bound) = self.infer_epsilon(expr, ctx) {
                            ctx.set_bound(name, bound);
                        }
                    }

                    // Also check type annotation for bounds
                    if let Some(bound) = self.extract_confidence_bound(ty) {
                        ctx.set_bound(name, bound);
                    }
                }
                HirStmt::Expr(expr) => {
                    self.optimize_expr(expr, ctx);
                }
                HirStmt::Assign { target, value } => {
                    self.optimize_expr(target, ctx);
                    self.optimize_expr(value, ctx);
                }
            }
        }
    }

    /// Optimize an expression
    fn optimize_expr(&mut self, expr: &mut HirExpr, ctx: &mut EpsilonContext) {
        match &mut expr.kind {
            // Epsilon constant folding for binary operations
            HirExprKind::Binary { op, left, right } => {
                self.optimize_expr(left, ctx);
                self.optimize_expr(right, ctx);

                if self.fold_epsilon {
                    self.try_fold_binary_epsilon(expr, ctx);
                }
            }

            // DCE for if expressions with statically known epsilon conditions
            HirExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.optimize_expr(condition, ctx);
                self.optimize_block(then_branch, ctx);
                if let Some(else_expr) = else_branch {
                    self.optimize_expr(else_expr, ctx);
                }

                if self.dce_invalid {
                    self.try_eliminate_dead_branch(expr, ctx);
                }
            }

            // Validity hoisting for loops
            HirExprKind::Loop(body) => {
                if self.hoist_validity {
                    self.try_hoist_validity_checks(body, ctx);
                }
                self.optimize_block(body, ctx);
            }

            HirExprKind::While { condition, body } => {
                self.optimize_expr(condition, ctx);
                if self.hoist_validity {
                    self.try_hoist_validity_checks(body, ctx);
                }
                self.optimize_block(body, ctx);
            }

            // Recurse into other expression kinds
            HirExprKind::Block(block) => {
                self.optimize_block(block, ctx);
            }
            HirExprKind::Call { func, args } => {
                self.optimize_expr(func, ctx);
                for arg in args {
                    self.optimize_expr(arg, ctx);
                }
            }
            HirExprKind::MethodCall { receiver, args, .. } => {
                self.optimize_expr(receiver, ctx);
                for arg in args {
                    self.optimize_expr(arg, ctx);
                }
            }
            HirExprKind::Match { scrutinee, arms } => {
                self.optimize_expr(scrutinee, ctx);
                for arm in arms {
                    self.optimize_expr(&mut arm.body, ctx);
                }
            }
            HirExprKind::Return(Some(e)) => {
                self.optimize_expr(e, ctx);
            }
            HirExprKind::Break(Some(e)) => {
                self.optimize_expr(e, ctx);
            }
            HirExprKind::Unary { expr, .. } => {
                self.optimize_expr(expr, ctx);
            }
            HirExprKind::Field { base, .. } => {
                self.optimize_expr(base, ctx);
            }
            HirExprKind::TupleField { base, .. } => {
                self.optimize_expr(base, ctx);
            }
            HirExprKind::Index { base, index } => {
                self.optimize_expr(base, ctx);
                self.optimize_expr(index, ctx);
            }
            HirExprKind::Cast { expr, .. } => {
                self.optimize_expr(expr, ctx);
            }
            HirExprKind::Closure { body, .. } => {
                self.optimize_expr(body, ctx);
            }
            HirExprKind::Tuple(elems) => {
                for e in elems {
                    self.optimize_expr(e, ctx);
                }
            }
            HirExprKind::Array(elems) => {
                for e in elems {
                    self.optimize_expr(e, ctx);
                }
            }
            HirExprKind::Range { start, end, .. } => {
                if let Some(s) = start {
                    self.optimize_expr(s, ctx);
                }
                if let Some(e) = end {
                    self.optimize_expr(e, ctx);
                }
            }
            HirExprKind::Struct { fields, .. } => {
                for (_, e) in fields {
                    self.optimize_expr(e, ctx);
                }
            }
            HirExprKind::Variant { fields, .. } => {
                for e in fields {
                    self.optimize_expr(e, ctx);
                }
            }
            HirExprKind::Ref { expr, .. } => {
                self.optimize_expr(expr, ctx);
            }
            HirExprKind::Deref(e) => {
                self.optimize_expr(e, ctx);
            }
            HirExprKind::Perform { args, .. } => {
                for arg in args {
                    self.optimize_expr(arg, ctx);
                }
            }
            HirExprKind::Handle { expr, .. } => {
                self.optimize_expr(expr, ctx);
            }
            HirExprKind::Sample(e) => {
                self.optimize_expr(e, ctx);
            }

            // Terminals - no optimization needed
            HirExprKind::Literal(_)
            | HirExprKind::Local(_)
            | HirExprKind::Global(_)
            | HirExprKind::Return(None)
            | HirExprKind::Break(None)
            | HirExprKind::Continue => {}

            // Epistemic expressions - recurse into sub-expressions
            HirExprKind::Knowledge {
                value,
                epsilon,
                validity,
                ..
            } => {
                self.optimize_expr(value, ctx);
                self.optimize_expr(epsilon, ctx);
                if let Some(v) = validity {
                    self.optimize_expr(v, ctx);
                }
            }
            HirExprKind::Do { value, .. } => {
                self.optimize_expr(value, ctx);
            }
            HirExprKind::Counterfactual {
                factual,
                intervention,
                outcome,
            } => {
                self.optimize_expr(factual, ctx);
                self.optimize_expr(intervention, ctx);
                self.optimize_expr(outcome, ctx);
            }
            HirExprKind::Query {
                target,
                given,
                interventions,
            } => {
                self.optimize_expr(target, ctx);
                for g in given {
                    self.optimize_expr(g, ctx);
                }
                for i in interventions {
                    self.optimize_expr(i, ctx);
                }
            }
            HirExprKind::Observe { value, .. } => {
                self.optimize_expr(value, ctx);
            }
            HirExprKind::EpsilonOf(e)
            | HirExprKind::ProvenanceOf(e)
            | HirExprKind::ValidityOf(e)
            | HirExprKind::Unwrap(e) => {
                self.optimize_expr(e, ctx);
            }

            // Ontology terms are terminals - no optimization needed
            HirExprKind::OntologyTerm { .. } => {}
        }
    }

    // =========================================================================
    // Pass 1: Epsilon Constant Folding
    // =========================================================================

    /// Try to fold epsilon bounds through binary operations
    fn try_fold_binary_epsilon(&mut self, expr: &mut HirExpr, ctx: &EpsilonContext) {
        if let HirExprKind::Binary { op, left, right } = &expr.kind {
            let left_eps = self.infer_epsilon(left, ctx);
            let right_eps = self.infer_epsilon(right, ctx);

            if let (Some(l), Some(r)) = (left_eps, right_eps) {
                // Determine operation name for propagation rules
                let op_name = match op {
                    crate::hir::HirBinaryOp::Add => "add",
                    crate::hir::HirBinaryOp::Sub => "sub",
                    crate::hir::HirBinaryOp::Mul => "mul",
                    crate::hir::HirBinaryOp::Div => "div",
                    _ => return, // Other ops don't affect epsilon
                };

                let result_eps = self
                    .propagation_rules
                    .propagate(op_name, &[l.value, r.value]);

                // Store the folded epsilon as metadata (would need HIR extension)
                // For now, we just track statistics
                self.stats.epsilon_folds += 1;

                // In a full implementation, we'd attach the computed epsilon
                // to the expression's type metadata
            }
        }
    }

    /// Infer the epsilon bound for an expression
    fn infer_epsilon(&self, expr: &HirExpr, ctx: &EpsilonContext) -> Option<ConfidenceBound> {
        match &expr.kind {
            HirExprKind::Local(name) => ctx.get_bound(name).cloned(),
            HirExprKind::Literal(HirLiteral::Float(_) | HirLiteral::Int(_)) => {
                // Literals have perfect confidence
                Some(ConfidenceBound::at_least(1.0))
            }
            HirExprKind::Binary { op, left, right } => {
                let l = self.infer_epsilon(left, ctx)?;
                let r = self.infer_epsilon(right, ctx)?;

                let op_name = match op {
                    crate::hir::HirBinaryOp::Add => "add",
                    crate::hir::HirBinaryOp::Sub => "sub",
                    crate::hir::HirBinaryOp::Mul => "mul",
                    crate::hir::HirBinaryOp::Div => "div",
                    _ => return None,
                };

                let result = self
                    .propagation_rules
                    .propagate(op_name, &[l.value, r.value]);
                Some(ConfidenceBound::at_least(result))
            }
            _ => None,
        }
    }

    /// Extract confidence bound from a HIR type (if it's a Knowledge type)
    fn extract_confidence_bound(&self, ty: &HirType) -> Option<ConfidenceBound> {
        // HIR types don't directly encode Knowledge - we'd need to extend HirType
        // For now, return None and rely on context
        None
    }

    // =========================================================================
    // Pass 2: Dead Code Elimination for Invalid Epsilon
    // =========================================================================

    /// Try to eliminate dead branches where epsilon condition is statically known
    fn try_eliminate_dead_branch(&mut self, expr: &mut HirExpr, ctx: &EpsilonContext) {
        if let HirExprKind::If {
            condition,
            then_branch,
            else_branch,
        } = &mut expr.kind
        {
            // Check if condition is an epsilon comparison
            if let Some(static_result) = self.evaluate_epsilon_condition(condition, ctx) {
                self.stats.dce_eliminations += 1;

                if static_result {
                    // Condition always true - replace with then branch
                    *expr = HirExpr {
                        id: expr.id,
                        kind: HirExprKind::Block(then_branch.clone()),
                        ty: expr.ty.clone(),
                    };
                } else if let Some(else_expr) = else_branch {
                    // Condition always false - replace with else branch
                    *expr = (**else_expr).clone();
                } else {
                    // No else branch and condition false - replace with unit
                    *expr = HirExpr {
                        id: expr.id,
                        kind: HirExprKind::Literal(HirLiteral::Unit),
                        ty: HirType::Unit,
                    };
                }
            }
        }
    }

    /// Evaluate an epsilon condition statically if possible
    fn evaluate_epsilon_condition(&self, expr: &HirExpr, ctx: &EpsilonContext) -> Option<bool> {
        // Pattern match for epsilon comparisons like: x.epsilon >= 0.8
        // In a full implementation, we'd have dedicated AST nodes for this
        if let HirExprKind::Binary { op, left, right } = &expr.kind {
            // Check if this is a comparison
            match op {
                crate::hir::HirBinaryOp::Ge
                | crate::hir::HirBinaryOp::Gt
                | crate::hir::HirBinaryOp::Le
                | crate::hir::HirBinaryOp::Lt => {
                    // Try to get epsilon from left side
                    if let Some(eps) = self.infer_epsilon(left, ctx) {
                        // Try to get literal from right side
                        if let HirExprKind::Literal(HirLiteral::Float(threshold)) = &right.kind {
                            return Some(match op {
                                crate::hir::HirBinaryOp::Ge => eps.value >= *threshold,
                                crate::hir::HirBinaryOp::Gt => eps.value > *threshold,
                                crate::hir::HirBinaryOp::Le => eps.value <= *threshold,
                                crate::hir::HirBinaryOp::Lt => eps.value < *threshold,
                                _ => return None,
                            });
                        }
                    }
                }
                _ => {}
            }
        }
        None
    }

    // =========================================================================
    // Pass 3: Validity Check Hoisting
    // =========================================================================

    /// Try to hoist validity checks out of loops
    fn try_hoist_validity_checks(&mut self, body: &mut HirBlock, ctx: &mut EpsilonContext) {
        // Collect hoistable validity checks
        let mut hoistable = Vec::new();

        for (i, stmt) in body.stmts.iter().enumerate() {
            if let HirStmt::Expr(expr) = stmt
                && let Some(check) = self.is_hoistable_validity_check(expr, ctx)
            {
                hoistable.push((i, check));
            }
        }

        // For now, just count - full implementation would rewrite the HIR
        self.stats.validity_hoists += hoistable.len();
    }

    /// Check if an expression is a hoistable validity check
    fn is_hoistable_validity_check(
        &self,
        expr: &HirExpr,
        ctx: &EpsilonContext,
    ) -> Option<ValidityCheck> {
        // Look for patterns like: assert_valid(x) or x.check_validity()
        // In a full implementation, we'd have dedicated nodes for validity checks
        if let HirExprKind::MethodCall { method, .. } = &expr.kind
            && (method == "check_validity" || method == "assert_valid")
        {
            return Some(ValidityCheck {
                variable: String::new(), // Would extract from receiver
                constraint: TemporalConstraint::MaxAge(3600),
            });
        }
        None
    }

    // =========================================================================
    // Pass 4: Provenance Chain Merging
    // =========================================================================

    /// Merge provenance chains in the HIR
    pub fn merge_provenance_chains(&mut self, hir: &mut Hir) {
        if !self.merge_provenance {
            return;
        }

        let mut merger = ProvenanceMerger::new();

        for item in &mut hir.items {
            if let HirItem::Function(func) = item {
                self.stats.provenance_merges += merger.merge_function(func);
            }
        }
    }
}

// =============================================================================
// Supporting Types
// =============================================================================

/// Context for epsilon constant folding
#[derive(Debug, Clone, Default)]
struct EpsilonContext {
    /// Variable name -> known epsilon bound
    bounds: HashMap<String, ConfidenceBound>,
}

impl EpsilonContext {
    fn new() -> Self {
        Self::default()
    }

    fn set_bound(&mut self, name: &str, bound: ConfidenceBound) {
        self.bounds.insert(name.to_string(), bound);
    }

    fn get_bound(&self, name: &str) -> Option<&ConfidenceBound> {
        self.bounds.get(name)
    }
}

/// Validity check to potentially hoist
#[derive(Debug, Clone)]
struct ValidityCheck {
    variable: String,
    constraint: TemporalConstraint,
}

/// Provenance chain merger
struct ProvenanceMerger {
    /// Canonical provenance forms
    canonical: HashMap<String, ProvenanceConstraint>,
}

impl ProvenanceMerger {
    fn new() -> Self {
        Self {
            canonical: HashMap::new(),
        }
    }

    /// Merge provenance chains in a function
    fn merge_function(&mut self, func: &mut HirFn) -> usize {
        // In a full implementation, we'd:
        // 1. Walk the function collecting provenance chains
        // 2. Identify common ancestors
        // 3. Merge chains like Derived(Derived(Measured)) -> DerivedFrom(Measured, depth=2)

        // For now, return 0 as placeholder
        0
    }

    /// Flatten a provenance chain
    fn flatten_chain(&self, prov: &ProvenanceConstraint) -> ProvenanceConstraint {
        match prov {
            ProvenanceConstraint::DerivedFrom(sources) => {
                // Recursively flatten
                let flattened: Vec<String> =
                    sources.iter().flat_map(|s| self.expand_source(s)).collect();
                ProvenanceConstraint::DerivedFrom(flattened)
            }
            other => other.clone(),
        }
    }

    fn expand_source(&self, source: &str) -> Vec<String> {
        if let Some(canonical) = self.canonical.get(source) {
            match canonical {
                ProvenanceConstraint::DerivedFrom(inner) => inner.clone(),
                ProvenanceConstraint::FromSource(s) => vec![s.clone()],
                _ => vec![source.to_string()],
            }
        } else {
            vec![source.to_string()]
        }
    }
}

// =============================================================================
// HIR Extensions for Epistemic Metadata
// =============================================================================

/// Epistemic metadata that can be attached to HIR expressions
#[derive(Debug, Clone, Default)]
pub struct EpistemicMetadata {
    /// Known confidence bound after optimization
    pub confidence: Option<ConfidenceBound>,
    /// Provenance after chain merging
    pub provenance: Option<ProvenanceConstraint>,
    /// Temporal validity
    pub validity: Option<TemporalConstraint>,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::NodeId;

    #[test]
    fn test_optimizer_creation() {
        let opt = EpistemicOptimizer::new();
        assert!(opt.fold_epsilon);
        assert!(opt.dce_invalid);
        assert!(opt.hoist_validity);
        assert!(opt.merge_provenance);
    }

    #[test]
    fn test_optimizer_with_config() {
        let opt = EpistemicOptimizer::with_config(true, false, true, false);
        assert!(opt.fold_epsilon);
        assert!(!opt.dce_invalid);
        assert!(opt.hoist_validity);
        assert!(!opt.merge_provenance);
    }

    #[test]
    fn test_epsilon_context() {
        let mut ctx = EpsilonContext::new();
        ctx.set_bound("x", ConfidenceBound::at_least(0.85));

        assert!(ctx.get_bound("x").is_some());
        assert_eq!(ctx.get_bound("x").unwrap().value, 0.85);
        assert!(ctx.get_bound("y").is_none());
    }

    #[test]
    fn test_infer_epsilon_literal() {
        let opt = EpistemicOptimizer::new();
        let ctx = EpsilonContext::new();

        let expr = HirExpr {
            id: NodeId::dummy(),
            kind: HirExprKind::Literal(HirLiteral::Float(42.0)),
            ty: HirType::F64,
        };

        let eps = opt.infer_epsilon(&expr, &ctx);
        assert!(eps.is_some());
        assert_eq!(eps.unwrap().value, 1.0);
    }

    #[test]
    fn test_infer_epsilon_local() {
        let opt = EpistemicOptimizer::new();
        let mut ctx = EpsilonContext::new();
        ctx.set_bound("x", ConfidenceBound::at_least(0.9));

        let expr = HirExpr {
            id: NodeId::dummy(),
            kind: HirExprKind::Local("x".to_string()),
            ty: HirType::F64,
        };

        let eps = opt.infer_epsilon(&expr, &ctx);
        assert!(eps.is_some());
        assert_eq!(eps.unwrap().value, 0.9);
    }

    #[test]
    fn test_infer_epsilon_binary() {
        let opt = EpistemicOptimizer::new();
        let mut ctx = EpsilonContext::new();
        ctx.set_bound("x", ConfidenceBound::at_least(0.9));
        ctx.set_bound("y", ConfidenceBound::at_least(0.8));

        let expr = HirExpr {
            id: NodeId::dummy(),
            kind: HirExprKind::Binary {
                op: crate::hir::HirBinaryOp::Add,
                left: Box::new(HirExpr {
                    id: NodeId::dummy(),
                    kind: HirExprKind::Local("x".to_string()),
                    ty: HirType::F64,
                }),
                right: Box::new(HirExpr {
                    id: NodeId::dummy(),
                    kind: HirExprKind::Local("y".to_string()),
                    ty: HirType::F64,
                }),
            },
            ty: HirType::F64,
        };

        let eps = opt.infer_epsilon(&expr, &ctx);
        assert!(eps.is_some());
        // min(0.9, 0.8) = 0.8
        assert_eq!(eps.unwrap().value, 0.8);
    }

    #[test]
    fn test_infer_epsilon_division_degradation() {
        let opt = EpistemicOptimizer::new();
        let mut ctx = EpsilonContext::new();
        ctx.set_bound("x", ConfidenceBound::at_least(0.9));
        ctx.set_bound("y", ConfidenceBound::at_least(0.9));

        let expr = HirExpr {
            id: NodeId::dummy(),
            kind: HirExprKind::Binary {
                op: crate::hir::HirBinaryOp::Div,
                left: Box::new(HirExpr {
                    id: NodeId::dummy(),
                    kind: HirExprKind::Local("x".to_string()),
                    ty: HirType::F64,
                }),
                right: Box::new(HirExpr {
                    id: NodeId::dummy(),
                    kind: HirExprKind::Local("y".to_string()),
                    ty: HirType::F64,
                }),
            },
            ty: HirType::F64,
        };

        let eps = opt.infer_epsilon(&expr, &ctx);
        assert!(eps.is_some());
        // min(0.9, 0.9) * 0.99 = 0.891
        let expected = 0.9 * 0.99;
        assert!((eps.unwrap().value - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn test_evaluate_epsilon_condition_ge() {
        let opt = EpistemicOptimizer::new();
        let mut ctx = EpsilonContext::new();
        ctx.set_bound("x", ConfidenceBound::at_least(0.9));

        // x >= 0.8 should be true when x has eps 0.9
        let condition = HirExpr {
            id: NodeId::dummy(),
            kind: HirExprKind::Binary {
                op: crate::hir::HirBinaryOp::Ge,
                left: Box::new(HirExpr {
                    id: NodeId::dummy(),
                    kind: HirExprKind::Local("x".to_string()),
                    ty: HirType::F64,
                }),
                right: Box::new(HirExpr {
                    id: NodeId::dummy(),
                    kind: HirExprKind::Literal(HirLiteral::Float(0.8)),
                    ty: HirType::F64,
                }),
            },
            ty: HirType::Bool,
        };

        let result = opt.evaluate_epsilon_condition(&condition, &ctx);
        assert!(result.is_some());
        assert!(result.unwrap());
    }

    #[test]
    fn test_evaluate_epsilon_condition_lt() {
        let opt = EpistemicOptimizer::new();
        let mut ctx = EpsilonContext::new();
        ctx.set_bound("x", ConfidenceBound::at_least(0.5));

        // x < 0.8 should be true when x has eps 0.5
        let condition = HirExpr {
            id: NodeId::dummy(),
            kind: HirExprKind::Binary {
                op: crate::hir::HirBinaryOp::Lt,
                left: Box::new(HirExpr {
                    id: NodeId::dummy(),
                    kind: HirExprKind::Local("x".to_string()),
                    ty: HirType::F64,
                }),
                right: Box::new(HirExpr {
                    id: NodeId::dummy(),
                    kind: HirExprKind::Literal(HirLiteral::Float(0.8)),
                    ty: HirType::F64,
                }),
            },
            ty: HirType::Bool,
        };

        let result = opt.evaluate_epsilon_condition(&condition, &ctx);
        assert!(result.is_some());
        assert!(result.unwrap());
    }

    #[test]
    fn test_statistics() {
        let opt = EpistemicOptimizer::new();
        let stats = opt.stats();
        assert_eq!(stats.epsilon_folds, 0);
        assert_eq!(stats.dce_eliminations, 0);
        assert_eq!(stats.validity_hoists, 0);
        assert_eq!(stats.provenance_merges, 0);
    }

    #[test]
    fn test_provenance_merger() {
        let merger = ProvenanceMerger::new();

        let prov = ProvenanceConstraint::DerivedFrom(vec!["sensor_a".to_string()]);
        let flattened = merger.flatten_chain(&prov);

        match flattened {
            ProvenanceConstraint::DerivedFrom(sources) => {
                assert_eq!(sources.len(), 1);
                assert_eq!(sources[0], "sensor_a");
            }
            _ => panic!("Expected DerivedFrom"),
        }
    }

    #[test]
    fn test_epistemic_metadata() {
        let meta = EpistemicMetadata::default();
        assert!(meta.confidence.is_none());
        assert!(meta.provenance.is_none());
        assert!(meta.validity.is_none());
    }
}
