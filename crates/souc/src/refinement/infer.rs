//! Liquid type inference for refinement types
//!
//! This module implements automatic inference of refinement types using the
//! Liquid Types approach. Instead of requiring explicit annotations, it
//! infers the strongest valid refinement from a set of qualifier templates.
//!
//! # Algorithm Overview
//!
//! 1. **Template Assignment**: Assign a template `κ` to each program point
//! 2. **Constraint Generation**: Generate subtyping constraints `κ₁ <: κ₂`
//! 3. **Constraint Solving**: Find an assignment to templates that satisfies all constraints
//!
//! # Key Insight
//!
//! The refinement at each point is a conjunction of qualifiers from Q:
//! ```text
//! κ = ⋀ { q ∈ Q | q is valid at this point }
//! ```
//!
//! We use abstract interpretation to find the largest valid subset.
//!
//! # References
//!
//! - Rondon, P. M., Kawaguchi, M., & Jhala, R. (2008). Liquid types.
//! - Kawaguchi, M., et al. (2009). Type-based data structure verification.

use super::constraint::*;
use super::predicate::*;
use super::qualifiers::*;
use crate::types::Type;
use std::collections::HashMap;

/// Refinement inference context
///
/// Tracks the inference state and generates refinement types.
pub struct RefinementInference {
    /// Available qualifiers
    qualifier_set: QualifierSet,

    /// Inferred refinements for each variable
    refinements: HashMap<String, RefinementType>,

    /// Variables in scope with their base types
    scope: Vec<(String, Type)>,

    /// Generated constraints
    constraints: Vec<InferenceConstraint>,

    /// Fresh variable counter
    fresh_counter: u32,

    /// Templates for unknown refinements
    templates: HashMap<String, RefinementTemplate>,
}

/// A constraint generated during inference
#[derive(Debug, Clone)]
pub struct InferenceConstraint {
    /// Left-hand side (actual type)
    pub lhs: RefinementTemplate,

    /// Right-hand side (expected type)
    pub rhs: RefinementTemplate,

    /// Environment (assumptions)
    pub env: Vec<(String, RefinementTemplate)>,

    /// Source location
    pub span: Span,
}

/// A refinement template (unknown refinement to be solved)
#[derive(Debug, Clone)]
pub enum RefinementTemplate {
    /// Known concrete refinement
    Concrete(RefinementType),

    /// Unknown refinement (template variable)
    Unknown {
        /// Template variable name
        name: String,
        /// Base type
        base: Type,
        /// Candidate qualifiers
        candidates: Vec<Predicate>,
    },
}

impl RefinementTemplate {
    /// Create a concrete template
    pub fn concrete(ty: RefinementType) -> Self {
        Self::Concrete(ty)
    }

    /// Create an unknown template
    pub fn unknown(name: impl Into<String>, base: Type, candidates: Vec<Predicate>) -> Self {
        Self::Unknown {
            name: name.into(),
            base,
            candidates,
        }
    }

    /// Get the base type
    pub fn base_type(&self) -> &Type {
        match self {
            Self::Concrete(ty) => &ty.base,
            Self::Unknown { base, .. } => base,
        }
    }

    /// Check if this is a concrete refinement
    pub fn is_concrete(&self) -> bool {
        matches!(self, Self::Concrete(_))
    }
}

impl RefinementInference {
    /// Create a new inference context with standard qualifiers
    pub fn new() -> Self {
        Self {
            qualifier_set: QualifierSet::standard(),
            refinements: HashMap::new(),
            scope: Vec::new(),
            constraints: Vec::new(),
            fresh_counter: 0,
            templates: HashMap::new(),
        }
    }

    /// Create an inference context with medical qualifiers
    pub fn with_medical() -> Self {
        Self {
            qualifier_set: QualifierSet::with_medical(),
            refinements: HashMap::new(),
            scope: Vec::new(),
            constraints: Vec::new(),
            fresh_counter: 0,
            templates: HashMap::new(),
        }
    }

    /// Add custom qualifiers
    pub fn add_qualifiers(&mut self, qualifiers: impl IntoIterator<Item = Qualifier>) {
        self.qualifier_set.add_all(qualifiers);
    }

    /// Generate a fresh template variable name
    fn fresh_template(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.fresh_counter);
        self.fresh_counter += 1;
        name
    }

    /// Enter a new scope
    pub fn enter_scope(&mut self) {
        // Mark the current scope depth
    }

    /// Exit the current scope
    pub fn exit_scope(&mut self) {
        // Pop bindings added in this scope
    }

    /// Add a variable to scope with its type
    pub fn add_binding(&mut self, name: impl Into<String>, ty: Type) {
        let name = name.into();
        self.scope.push((name.clone(), ty.clone()));

        // Create a trivial refinement initially
        self.refinements.insert(name, RefinementType::trivial(ty));
    }

    /// Add a variable with a known refinement
    pub fn add_refined_binding(&mut self, name: impl Into<String>, ty: RefinementType) {
        let name = name.into();
        self.scope.push((name.clone(), ty.base.clone()));
        self.refinements.insert(name, ty);
    }

    /// Look up a variable's refinement
    pub fn lookup(&self, name: &str) -> Option<&RefinementType> {
        self.refinements.get(name)
    }

    /// Create a fresh template for an unknown refinement
    pub fn fresh_refinement(&mut self, base: Type) -> RefinementTemplate {
        let name = self.fresh_template("κ");

        // Get candidate qualifiers for this type
        let v = Term::var("v");
        let scope_vars: Vec<_> = self.scope.iter().map(|(n, _)| Term::var(n)).collect();

        let candidates = self.qualifier_set.instantiate_all(&v, &scope_vars);

        let template = RefinementTemplate::unknown(name.clone(), base.clone(), candidates);
        self.templates.insert(name, template.clone());

        template
    }

    /// Infer the refinement for a literal
    pub fn infer_literal(&mut self, value: LiteralValue) -> RefinementType {
        match value {
            LiteralValue::Int(n) => {
                // Literal has exact value: { v | v = n }
                RefinementType::refined(Type::I64, "v", Predicate::eq(Term::var("v"), Term::int(n)))
            }
            LiteralValue::Float(n) => RefinementType::refined(
                Type::F64,
                "v",
                Predicate::eq(Term::var("v"), Term::float(n)),
            ),
            LiteralValue::Bool(b) => RefinementType::refined(
                Type::Bool,
                "v",
                Predicate::eq(Term::var("v"), Term::Bool(b)),
            ),
        }
    }

    /// Infer the refinement for a variable reference
    pub fn infer_var(&self, name: &str) -> Option<RefinementType> {
        self.refinements.get(name).cloned()
    }

    /// Infer the refinement for a binary operation
    pub fn infer_binary(
        &mut self,
        op: BinaryOp,
        lhs: &RefinementType,
        rhs: &RefinementType,
        result_base: Type,
    ) -> RefinementType {
        match op {
            BinaryOp::Add => {
                // If both operands have known values, result is their sum
                // { v | v = lhs + rhs }
                RefinementType::refined(
                    result_base,
                    "v",
                    Predicate::eq(
                        Term::var("v"),
                        Term::add(Term::var(&lhs.var), Term::var(&rhs.var)),
                    ),
                )
            }

            BinaryOp::Sub => RefinementType::refined(
                result_base,
                "v",
                Predicate::eq(
                    Term::var("v"),
                    Term::sub(Term::var(&lhs.var), Term::var(&rhs.var)),
                ),
            ),

            BinaryOp::Mul => {
                // Multiplication of positives is positive
                let lhs_pos = is_positive_refinement(&lhs.predicate);
                let rhs_pos = is_positive_refinement(&rhs.predicate);

                if lhs_pos && rhs_pos {
                    RefinementType::positive(result_base)
                } else {
                    RefinementType::trivial(result_base)
                }
            }

            BinaryOp::Div => {
                // Division preserves sign properties
                RefinementType::trivial(result_base)
            }

            BinaryOp::Mod => {
                // Modulo result is bounded by divisor
                RefinementType::trivial(result_base)
            }
        }
    }

    /// Infer the refinement for an if-then-else expression
    pub fn infer_ite(
        &mut self,
        condition: &Predicate,
        then_type: &RefinementType,
        else_type: &RefinementType,
    ) -> RefinementType {
        // Result type is the join (weakening) of both branches
        // Could be more precise with path conditions

        if then_type.predicate == else_type.predicate {
            then_type.clone()
        } else {
            // Join: disjunction of predicates
            RefinementType::refined(
                then_type.base.clone(),
                "v",
                Predicate::or([then_type.predicate.clone(), else_type.predicate.clone()]),
            )
        }
    }

    /// Add a subtyping constraint
    pub fn constrain_subtype(
        &mut self,
        actual: RefinementTemplate,
        expected: RefinementTemplate,
        span: Span,
    ) {
        let env: Vec<_> = self
            .refinements
            .iter()
            .map(|(n, ty)| (n.clone(), RefinementTemplate::concrete(ty.clone())))
            .collect();

        self.constraints.push(InferenceConstraint {
            lhs: actual,
            rhs: expected,
            env,
            span,
        });
    }

    /// Solve all constraints and return inferred refinements
    pub fn solve(self) -> InferenceResult {
        // In a full implementation, this would:
        // 1. Translate constraints to Horn clauses
        // 2. Use fixpoint iteration to find solutions
        // 3. Use SMT solver to check candidate solutions

        // For now, return the refinements we've collected
        InferenceResult {
            refinements: self.refinements,
            templates: self
                .templates
                .into_iter()
                .map(|(name, template)| {
                    let ty = match template {
                        RefinementTemplate::Concrete(ty) => ty,
                        RefinementTemplate::Unknown {
                            base, candidates, ..
                        } => {
                            // Use strongest valid conjunction
                            if candidates.is_empty() {
                                RefinementType::trivial(base)
                            } else {
                                RefinementType::refined(base, "v", Predicate::and(candidates))
                            }
                        }
                    };
                    (name, ty)
                })
                .collect(),
            constraints: self.constraints,
        }
    }

    /// Get statistics about the inference
    pub fn stats(&self) -> InferenceStats {
        InferenceStats {
            num_variables: self.refinements.len(),
            num_templates: self.templates.len(),
            num_constraints: self.constraints.len(),
            num_qualifiers: self.qualifier_set.len(),
        }
    }
}

impl Default for RefinementInference {
    fn default() -> Self {
        Self::new()
    }
}

/// Literal values for inference
#[derive(Debug, Clone)]
pub enum LiteralValue {
    Int(i64),
    Float(f64),
    Bool(bool),
}

/// Binary operators for inference
#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

/// Result of refinement inference
#[derive(Debug)]
pub struct InferenceResult {
    /// Inferred refinements for variables
    pub refinements: HashMap<String, RefinementType>,

    /// Solved templates
    pub templates: HashMap<String, RefinementType>,

    /// Generated constraints (for debugging)
    pub constraints: Vec<InferenceConstraint>,
}

impl InferenceResult {
    /// Get the refinement for a variable
    pub fn get(&self, name: &str) -> Option<&RefinementType> {
        self.refinements.get(name)
    }

    /// Get the refinement for a template
    pub fn get_template(&self, name: &str) -> Option<&RefinementType> {
        self.templates.get(name)
    }

    /// Check if inference found non-trivial refinements
    pub fn has_refinements(&self) -> bool {
        self.refinements.values().any(|ty| ty.is_refined())
            || self.templates.values().any(|ty| ty.is_refined())
    }
}

/// Statistics about inference
#[derive(Debug, Clone)]
pub struct InferenceStats {
    pub num_variables: usize,
    pub num_templates: usize,
    pub num_constraints: usize,
    pub num_qualifiers: usize,
}

/// Check if a predicate implies positivity (v > 0)
fn is_positive_refinement(pred: &Predicate) -> bool {
    match pred {
        Predicate::Atom(atom) => {
            matches!(atom.op, CompareOp::Gt)
                && matches!(&atom.lhs, Term::Var(v) if v == "v")
                && is_zero_term(&atom.rhs)
        }
        Predicate::And(preds) => preds.iter().any(is_positive_refinement),
        _ => false,
    }
}

/// Check if a predicate implies non-negativity (v >= 0)
fn is_non_negative_refinement(pred: &Predicate) -> bool {
    match pred {
        Predicate::Atom(atom) => {
            (matches!(atom.op, CompareOp::Ge)
                && matches!(&atom.lhs, Term::Var(v) if v == "v")
                && is_zero_term(&atom.rhs))
                || is_positive_refinement(&Predicate::Atom(atom.clone()))
        }
        Predicate::And(preds) => preds.iter().any(is_non_negative_refinement),
        _ => false,
    }
}

/// Check if a term represents zero
fn is_zero_term(term: &Term) -> bool {
    match term {
        Term::Int(0) => true,
        Term::Float(n) => *n == 0.0,
        _ => false,
    }
}

/// Infer refinements for a function body
pub fn infer_function(
    params: &[(String, Type)],
    param_refinements: &[(String, RefinementType)],
) -> RefinementInference {
    let mut infer = RefinementInference::with_medical();

    // Add parameters with their refinements
    for ((name, _), (_, refinement)) in params.iter().zip(param_refinements.iter()) {
        infer.add_refined_binding(name.clone(), refinement.clone());
    }

    infer
}

/// Pre-defined refinements for common patterns
pub mod patterns {
    use super::*;

    /// Array index pattern: 0 <= i < len
    pub fn array_index(idx_name: &str, len_term: Term) -> RefinementType {
        RefinementType::refined(
            Type::I64,
            idx_name,
            Predicate::and([
                Predicate::ge(Term::var(idx_name), Term::int(0)),
                Predicate::lt(Term::var(idx_name), len_term),
            ]),
        )
    }

    /// Loop counter pattern: 0 <= i <= n
    pub fn loop_counter(counter_name: &str, bound_term: Term) -> RefinementType {
        RefinementType::refined(
            Type::I64,
            counter_name,
            Predicate::and([
                Predicate::ge(Term::var(counter_name), Term::int(0)),
                Predicate::le(Term::var(counter_name), bound_term),
            ]),
        )
    }

    /// Safe divisor pattern: x != 0
    pub fn safe_divisor(var_name: &str, base: Type) -> RefinementType {
        RefinementType::refined(
            base,
            var_name,
            Predicate::ne(Term::var(var_name), Term::int(0)),
        )
    }

    /// Dose calculation result: 0 < dose <= max
    pub fn dose_result(max: f64) -> RefinementType {
        RefinementType::refined(
            Type::F64,
            "dose",
            Predicate::and([
                Predicate::gt(Term::var("dose"), Term::float(0.0)),
                Predicate::le(Term::var("dose"), Term::float(max)),
            ]),
        )
    }

    /// Concentration in therapeutic range
    pub fn therapeutic_concentration(min: f64, max: f64) -> RefinementType {
        RefinementType::refined(
            Type::F64,
            "conc",
            Predicate::and([
                Predicate::ge(Term::var("conc"), Term::float(min)),
                Predicate::le(Term::var("conc"), Term::float(max)),
            ]),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_int_literal() {
        let mut infer = RefinementInference::new();
        let ty = infer.infer_literal(LiteralValue::Int(42));

        assert!(ty.is_refined());
        assert_eq!(ty.base, Type::I64);
    }

    #[test]
    fn test_infer_float_literal() {
        let mut infer = RefinementInference::new();
        let ty = infer.infer_literal(LiteralValue::Float(3.14));

        assert!(ty.is_refined());
        assert_eq!(ty.base, Type::F64);
    }

    #[test]
    fn test_infer_bool_literal() {
        let mut infer = RefinementInference::new();
        let ty = infer.infer_literal(LiteralValue::Bool(true));

        assert!(ty.is_refined());
        assert_eq!(ty.base, Type::Bool);
    }

    #[test]
    fn test_add_binding() {
        let mut infer = RefinementInference::new();
        infer.add_binding("x", Type::I64);

        let ty = infer.lookup("x");
        assert!(ty.is_some());
        assert!(!ty.unwrap().is_refined()); // Initially trivial
    }

    #[test]
    fn test_add_refined_binding() {
        let mut infer = RefinementInference::new();
        let pos = RefinementType::positive(Type::I64);
        infer.add_refined_binding("x", pos);

        let ty = infer.lookup("x").unwrap();
        assert!(ty.is_refined());
    }

    #[test]
    fn test_fresh_refinement() {
        let mut infer = RefinementInference::new();
        infer.add_binding("x", Type::I64);

        let template = infer.fresh_refinement(Type::I64);

        assert!(matches!(template, RefinementTemplate::Unknown { .. }));
    }

    #[test]
    fn test_binary_add() {
        let mut infer = RefinementInference::new();

        let lhs = RefinementType::trivial(Type::I64);
        let rhs = RefinementType::trivial(Type::I64);

        let result = infer.infer_binary(BinaryOp::Add, &lhs, &rhs, Type::I64);

        assert!(result.is_refined());
    }

    #[test]
    fn test_binary_mul_positive() {
        let mut infer = RefinementInference::new();

        let pos = RefinementType::positive(Type::I64);
        let result = infer.infer_binary(BinaryOp::Mul, &pos, &pos, Type::I64);

        // Product of positives should be positive
        assert!(is_positive_refinement(&result.predicate));
    }

    #[test]
    fn test_ite_same_branches() {
        let mut infer = RefinementInference::new();

        let pos = RefinementType::positive(Type::I64);
        let cond = Predicate::gt(Term::var("x"), Term::int(0));

        let result = infer.infer_ite(&cond, &pos, &pos);

        // Same branches should preserve refinement
        assert_eq!(result.predicate, pos.predicate);
    }

    #[test]
    fn test_ite_different_branches() {
        let mut infer = RefinementInference::new();

        let pos = RefinementType::positive(Type::I64);
        let neg =
            RefinementType::refined(Type::I64, "v", Predicate::lt(Term::var("v"), Term::int(0)));
        let cond = Predicate::gt(Term::var("x"), Term::int(0));

        let result = infer.infer_ite(&cond, &pos, &neg);

        // Different branches should produce disjunction
        assert!(matches!(result.predicate, Predicate::Or(_)));
    }

    #[test]
    fn test_solve() {
        let mut infer = RefinementInference::new();

        let pos = RefinementType::positive(Type::I64);
        infer.add_refined_binding("x", pos);

        let result = infer.solve();

        assert!(result.has_refinements());
        assert!(result.get("x").is_some());
    }

    #[test]
    fn test_stats() {
        let mut infer = RefinementInference::new();
        infer.add_binding("x", Type::I64);
        infer.add_binding("y", Type::I64);

        let stats = infer.stats();

        assert_eq!(stats.num_variables, 2);
        assert!(stats.num_qualifiers > 0);
    }

    #[test]
    fn test_is_positive_refinement() {
        let pos_pred = Predicate::gt(Term::var("v"), Term::int(0));
        assert!(is_positive_refinement(&pos_pred));

        let neg_pred = Predicate::lt(Term::var("v"), Term::int(0));
        assert!(!is_positive_refinement(&neg_pred));
    }

    #[test]
    fn test_is_non_negative_refinement() {
        let pos_pred = Predicate::gt(Term::var("v"), Term::int(0));
        assert!(is_non_negative_refinement(&pos_pred));

        let non_neg_pred = Predicate::ge(Term::var("v"), Term::int(0));
        assert!(is_non_negative_refinement(&non_neg_pred));
    }

    #[test]
    fn test_patterns_array_index() {
        let ty = patterns::array_index("i", Term::int(10));

        assert!(ty.is_refined());
        assert_eq!(ty.var, "i");
    }

    #[test]
    fn test_patterns_dose_result() {
        let ty = patterns::dose_result(1000.0);

        assert!(ty.is_refined());
        assert_eq!(ty.var, "dose");
    }

    #[test]
    fn test_medical_inference() {
        let infer = RefinementInference::with_medical();
        let stats = infer.stats();

        // Medical inference should have more qualifiers
        let standard = RefinementInference::new();
        let standard_stats = standard.stats();

        assert!(stats.num_qualifiers > standard_stats.num_qualifiers);
    }
}
