//! Coercion Insertion
//!
//! Inserts runtime coercion calls where semantic distance allows
//! implicit conversion between ontological types.

use std::collections::HashMap;

use crate::common::Span;
use crate::hir::{HirExpr, HirExprKind, HirStmt, HirType};

use super::unify_distance::{CoercionKind, CoercionSite};

/// Inserts coercions into the HIR
pub struct CoercionInserter {
    /// Pending coercions to insert (keyed by span for lookup)
    coercions: Vec<CoercionSite>,
    /// Explicit casts recorded (don't need insertion, just tracking)
    explicit_casts: Vec<ExplicitCast>,
    /// Coercion counter for generating unique names
    counter: u32,
}

/// An explicit cast recorded from source
#[derive(Debug, Clone)]
pub struct ExplicitCast {
    pub span: Span,
    pub from_type: HirType,
    pub to_type: HirType,
}

impl CoercionInserter {
    pub fn new() -> Self {
        Self {
            coercions: Vec::new(),
            explicit_casts: Vec::new(),
            counter: 0,
        }
    }

    /// Record a coercion to be inserted
    pub fn record_coercion(&mut self, coercion: CoercionSite) {
        self.coercions.push(coercion);
    }

    /// Record an explicit cast (no insertion needed)
    pub fn record_explicit_cast(&mut self, span: Span, from: &HirType, to: &HirType) {
        self.explicit_casts.push(ExplicitCast {
            span,
            from_type: from.clone(),
            to_type: to.clone(),
        });
    }

    /// Take all recorded coercions
    pub fn take_coercions(&mut self) -> Vec<CoercionSite> {
        std::mem::take(&mut self.coercions)
    }

    /// Get coercions (non-consuming)
    pub fn coercions(&self) -> &[CoercionSite] {
        &self.coercions
    }

    /// Get explicit casts
    pub fn explicit_casts(&self) -> &[ExplicitCast] {
        &self.explicit_casts
    }

    /// Apply all coercions to an expression tree
    /// Note: This modifies the HIR in-place by wrapping expressions with coercion calls
    pub fn apply_to_expr(&mut self, expr: &mut HirExpr) {
        // Clone coercions to avoid borrow conflict
        let coercions = self.coercions.clone();

        // Build a map of spans to coercions for quick lookup
        let coercion_map: HashMap<(usize, usize), &CoercionSite> = coercions
            .iter()
            .map(|c| ((c.span.start, c.span.end), c))
            .collect();

        Self::apply_to_expr_inner_static(expr, &coercion_map);
    }

    fn apply_to_expr_inner_static(
        expr: &mut HirExpr,
        coercion_map: &HashMap<(usize, usize), &CoercionSite>,
    ) {
        // Recursively process sub-expressions first
        match &mut expr.kind {
            HirExprKind::Call { func, args } => {
                Self::apply_to_expr_inner_static(func, coercion_map);
                for arg in args {
                    Self::apply_to_expr_inner_static(arg, coercion_map);
                }
            }

            HirExprKind::MethodCall { receiver, args, .. } => {
                Self::apply_to_expr_inner_static(receiver, coercion_map);
                for arg in args {
                    Self::apply_to_expr_inner_static(arg, coercion_map);
                }
            }

            HirExprKind::Field { base, .. } => {
                Self::apply_to_expr_inner_static(base, coercion_map);
            }

            HirExprKind::TupleField { base, .. } => {
                Self::apply_to_expr_inner_static(base, coercion_map);
            }

            HirExprKind::Index { base, index } => {
                Self::apply_to_expr_inner_static(base, coercion_map);
                Self::apply_to_expr_inner_static(index, coercion_map);
            }

            HirExprKind::Binary { left, right, .. } => {
                Self::apply_to_expr_inner_static(left, coercion_map);
                Self::apply_to_expr_inner_static(right, coercion_map);
            }

            HirExprKind::Unary { expr: operand, .. } => {
                Self::apply_to_expr_inner_static(operand, coercion_map);
            }

            HirExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                Self::apply_to_expr_inner_static(condition, coercion_map);
                // then_branch is a HirBlock, process its statements
                for stmt in &mut then_branch.stmts {
                    Self::apply_to_stmt_static(stmt, coercion_map);
                }
                if let Some(else_expr) = else_branch {
                    Self::apply_to_expr_inner_static(else_expr, coercion_map);
                }
            }

            HirExprKind::Match { scrutinee, arms } => {
                Self::apply_to_expr_inner_static(scrutinee, coercion_map);
                for arm in arms {
                    Self::apply_to_expr_inner_static(&mut arm.body, coercion_map);
                }
            }

            HirExprKind::Loop(block) => {
                for stmt in &mut block.stmts {
                    Self::apply_to_stmt_static(stmt, coercion_map);
                }
            }

            HirExprKind::While { condition, body } => {
                Self::apply_to_expr_inner_static(condition, coercion_map);
                for stmt in &mut body.stmts {
                    Self::apply_to_stmt_static(stmt, coercion_map);
                }
            }

            HirExprKind::Block(block) => {
                for stmt in &mut block.stmts {
                    Self::apply_to_stmt_static(stmt, coercion_map);
                }
            }

            HirExprKind::Return(ret_expr) => {
                if let Some(expr) = ret_expr {
                    Self::apply_to_expr_inner_static(expr, coercion_map);
                }
            }

            HirExprKind::Array(elements) => {
                for elem in elements {
                    Self::apply_to_expr_inner_static(elem, coercion_map);
                }
            }

            HirExprKind::Range { start, end, .. } => {
                if let Some(s) = start {
                    Self::apply_to_expr_inner_static(s, coercion_map);
                }
                if let Some(e) = end {
                    Self::apply_to_expr_inner_static(e, coercion_map);
                }
            }

            HirExprKind::Tuple(elements) => {
                for elem in elements {
                    Self::apply_to_expr_inner_static(elem, coercion_map);
                }
            }

            HirExprKind::Struct { fields, .. } => {
                for (_, field_expr) in fields {
                    Self::apply_to_expr_inner_static(field_expr, coercion_map);
                }
            }

            HirExprKind::Variant { fields, .. } => {
                for field_expr in fields {
                    Self::apply_to_expr_inner_static(field_expr, coercion_map);
                }
            }

            HirExprKind::Closure { body, .. } => {
                Self::apply_to_expr_inner_static(body, coercion_map);
            }

            HirExprKind::Cast { expr: inner, .. } => {
                Self::apply_to_expr_inner_static(inner, coercion_map);
            }

            HirExprKind::Ref { expr: inner, .. } => {
                Self::apply_to_expr_inner_static(inner, coercion_map);
            }

            HirExprKind::Deref(inner) => {
                Self::apply_to_expr_inner_static(inner, coercion_map);
            }

            // Leaf expressions - nothing to recurse into
            HirExprKind::Literal(_)
            | HirExprKind::Local(_)
            | HirExprKind::Global(_)
            | HirExprKind::Break(_)
            | HirExprKind::Continue => {}

            // Effect expressions
            HirExprKind::Perform { args, .. } => {
                for arg in args {
                    Self::apply_to_expr_inner_static(arg, coercion_map);
                }
            }

            HirExprKind::Handle { expr: inner, .. } => {
                Self::apply_to_expr_inner_static(inner, coercion_map);
            }

            HirExprKind::Sample(inner) => {
                Self::apply_to_expr_inner_static(inner, coercion_map);
            }

            // Epistemic expressions
            HirExprKind::Knowledge {
                value,
                epsilon,
                validity,
                ..
            } => {
                Self::apply_to_expr_inner_static(value, coercion_map);
                Self::apply_to_expr_inner_static(epsilon, coercion_map);
                if let Some(v) = validity {
                    Self::apply_to_expr_inner_static(v, coercion_map);
                }
            }

            HirExprKind::Do { value, .. } => {
                Self::apply_to_expr_inner_static(value, coercion_map);
            }

            HirExprKind::Counterfactual {
                factual,
                intervention,
                outcome,
            } => {
                Self::apply_to_expr_inner_static(factual, coercion_map);
                Self::apply_to_expr_inner_static(intervention, coercion_map);
                Self::apply_to_expr_inner_static(outcome, coercion_map);
            }

            HirExprKind::Query {
                target,
                given,
                interventions,
            } => {
                Self::apply_to_expr_inner_static(target, coercion_map);
                for g in given {
                    Self::apply_to_expr_inner_static(g, coercion_map);
                }
                for i in interventions {
                    Self::apply_to_expr_inner_static(i, coercion_map);
                }
            }

            HirExprKind::Observe { value, .. } => {
                Self::apply_to_expr_inner_static(value, coercion_map);
            }

            HirExprKind::EpsilonOf(inner)
            | HirExprKind::ProvenanceOf(inner)
            | HirExprKind::ValidityOf(inner)
            | HirExprKind::Unwrap(inner) => {
                Self::apply_to_expr_inner_static(inner, coercion_map);
            }

            // Ontology terms are leaf expressions - no sub-expressions
            HirExprKind::OntologyTerm { .. } => {}
        }
    }

    fn apply_to_stmt_static(
        stmt: &mut HirStmt,
        coercion_map: &HashMap<(usize, usize), &CoercionSite>,
    ) {
        match stmt {
            HirStmt::Let { value, .. } => {
                if let Some(init_expr) = value {
                    Self::apply_to_expr_inner_static(init_expr, coercion_map);
                }
            }

            HirStmt::Expr(expr) => {
                Self::apply_to_expr_inner_static(expr, coercion_map);
            }

            HirStmt::Assign { target, value } => {
                Self::apply_to_expr_inner_static(target, coercion_map);
                Self::apply_to_expr_inner_static(value, coercion_map);
            }
        }
    }

    /// Generate next coercion ID
    fn next_coercion_id(&mut self) -> u32 {
        let id = self.counter;
        self.counter += 1;
        id
    }

    /// Clear all recorded coercions
    pub fn clear(&mut self) {
        self.coercions.clear();
        self.explicit_casts.clear();
    }

    /// Check if there are any coercions pending
    pub fn has_coercions(&self) -> bool {
        !self.coercions.is_empty()
    }

    /// Count of coercions by kind
    pub fn coercion_stats(&self) -> CoercionStats {
        let mut stats = CoercionStats::default();
        for coercion in &self.coercions {
            match coercion.kind {
                CoercionKind::Subtype => stats.subtype += 1,
                CoercionKind::SemanticProximity => stats.semantic += 1,
                CoercionKind::CrossOntology => stats.cross_ontology += 1,
                CoercionKind::ExplicitCast => stats.explicit += 1,
            }
        }
        stats
    }
}

impl Default for CoercionInserter {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata attached to a coercion
#[derive(Debug, Clone)]
pub struct CoercionMetadata {
    /// Unique ID
    pub id: u32,
    /// Source type
    pub from_type: HirType,
    /// Target type
    pub to_type: HirType,
    /// Semantic distance
    pub distance: f32,
    /// Kind of coercion
    pub kind: CoercionKind,
}

/// Statistics about coercions
#[derive(Debug, Clone, Default)]
pub struct CoercionStats {
    pub subtype: usize,
    pub semantic: usize,
    pub cross_ontology: usize,
    pub explicit: usize,
}

impl CoercionStats {
    pub fn total(&self) -> usize {
        self.subtype + self.semantic + self.cross_ontology + self.explicit
    }
}

/// Coercion validation at compile time
pub struct CoercionValidator {
    /// Maximum allowed distance for automatic coercion
    max_auto_distance: f32,
    /// Maximum allowed distance for cross-ontology coercion
    max_cross_ontology_distance: f32,
}

impl CoercionValidator {
    pub fn new() -> Self {
        Self {
            max_auto_distance: 0.25,
            max_cross_ontology_distance: 0.15,
        }
    }

    /// Check if a coercion is valid
    pub fn is_valid(&self, coercion: &CoercionSite) -> bool {
        match coercion.kind {
            CoercionKind::Subtype => true, // Always valid
            CoercionKind::SemanticProximity => coercion.distance <= self.max_auto_distance,
            CoercionKind::CrossOntology => coercion.distance <= self.max_cross_ontology_distance,
            CoercionKind::ExplicitCast => true, // User requested, always valid
        }
    }

    /// Validate all coercions, returning invalid ones
    pub fn validate_all<'a>(&self, coercions: &'a [CoercionSite]) -> Vec<&'a CoercionSite> {
        coercions.iter().filter(|c| !self.is_valid(c)).collect()
    }
}

impl Default for CoercionValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_span() -> Span {
        Span::new(0, 10)
    }

    #[test]
    fn test_coercion_inserter_new() {
        let inserter = CoercionInserter::new();
        assert!(!inserter.has_coercions());
        assert_eq!(inserter.coercions().len(), 0);
    }

    #[test]
    fn test_record_coercion() {
        let mut inserter = CoercionInserter::new();

        inserter.record_coercion(CoercionSite {
            span: test_span(),
            from_type: HirType::I32,
            to_type: HirType::I64,
            distance: 0.0,
            kind: CoercionKind::Subtype,
        });

        assert!(inserter.has_coercions());
        assert_eq!(inserter.coercions().len(), 1);
    }

    #[test]
    fn test_coercion_stats() {
        let mut inserter = CoercionInserter::new();

        inserter.record_coercion(CoercionSite {
            span: test_span(),
            from_type: HirType::I32,
            to_type: HirType::I64,
            distance: 0.0,
            kind: CoercionKind::Subtype,
        });

        inserter.record_coercion(CoercionSite {
            span: test_span(),
            from_type: HirType::Ontology {
                namespace: "pbpk".to_string(),
                term: "Concentration".to_string(),
            },
            to_type: HirType::Ontology {
                namespace: "pbpk".to_string(),
                term: "PlasmaConcentration".to_string(),
            },
            distance: 0.12,
            kind: CoercionKind::SemanticProximity,
        });

        let stats = inserter.coercion_stats();
        assert_eq!(stats.subtype, 1);
        assert_eq!(stats.semantic, 1);
        assert_eq!(stats.total(), 2);
    }

    #[test]
    fn test_coercion_validator() {
        let validator = CoercionValidator::new();

        // Subtype always valid
        assert!(validator.is_valid(&CoercionSite {
            span: test_span(),
            from_type: HirType::I32,
            to_type: HirType::I64,
            distance: 0.0,
            kind: CoercionKind::Subtype,
        }));

        // Semantic within limit
        assert!(validator.is_valid(&CoercionSite {
            span: test_span(),
            from_type: HirType::I32,
            to_type: HirType::I64,
            distance: 0.20,
            kind: CoercionKind::SemanticProximity,
        }));

        // Semantic exceeds limit
        assert!(!validator.is_valid(&CoercionSite {
            span: test_span(),
            from_type: HirType::I32,
            to_type: HirType::I64,
            distance: 0.30,
            kind: CoercionKind::SemanticProximity,
        }));
    }

    #[test]
    fn test_take_coercions() {
        let mut inserter = CoercionInserter::new();

        inserter.record_coercion(CoercionSite {
            span: test_span(),
            from_type: HirType::I32,
            to_type: HirType::I64,
            distance: 0.0,
            kind: CoercionKind::Subtype,
        });

        let coercions = inserter.take_coercions();
        assert_eq!(coercions.len(), 1);
        assert!(!inserter.has_coercions()); // Now empty
    }
}
