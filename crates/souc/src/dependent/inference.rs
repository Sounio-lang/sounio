//! Constraint-based type inference for dependent epistemic types
//!
//! This module implements type inference by:
//! 1. Generating constraints from the program
//! 2. Solving constraints to find type variable substitutions
//! 3. Checking that predicates are satisfiable
//!
//! # Constraint Kinds
//!
//! - **Equality**: τ₁ = τ₂
//! - **Subtyping**: τ₁ <: τ₂
//! - **Confidence**: ε₁ ≥ ε₂
//! - **Predicate**: P must be provable

use super::TypeContext;
use super::predicates::Predicate;
use super::proof_search::{ProofResult, ProofSearchConfig, ProofSearcher};
use super::types::{ConfidenceType, EpistemicType, OntologyType};
use std::collections::HashMap;

/// A type variable
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeVar {
    /// Variable name
    pub name: String,
    /// Variable kind
    pub kind: TypeVarKind,
    /// Unique ID
    pub id: usize,
}

impl TypeVar {
    /// Create a new type variable
    pub fn new(name: impl Into<String>, kind: TypeVarKind, id: usize) -> Self {
        Self {
            name: name.into(),
            kind,
            id,
        }
    }

    /// Create a confidence type variable
    pub fn confidence(name: impl Into<String>, id: usize) -> Self {
        Self::new(name, TypeVarKind::Confidence, id)
    }

    /// Create an ontology type variable
    pub fn ontology(name: impl Into<String>, id: usize) -> Self {
        Self::new(name, TypeVarKind::Ontology, id)
    }

    /// Create an epistemic type variable
    pub fn epistemic(name: impl Into<String>, id: usize) -> Self {
        Self::new(name, TypeVarKind::Epistemic, id)
    }
}

impl std::fmt::Display for TypeVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "?{}_{}", self.name, self.id)
    }
}

/// Kind of type variable
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeVarKind {
    /// Confidence value
    Confidence,
    /// Ontology
    Ontology,
    /// Full epistemic type
    Epistemic,
    /// Base type (non-epistemic)
    Base,
}

/// A constraint to be solved
#[derive(Debug, Clone)]
pub struct Constraint {
    /// The kind of constraint
    pub kind: ConstraintKind,
    /// Source location (for error reporting)
    pub location: Option<super::gradual::SourceLocation>,
    /// Context/reason for constraint
    pub reason: String,
}

impl Constraint {
    /// Create a new constraint
    pub fn new(kind: ConstraintKind, reason: impl Into<String>) -> Self {
        Self {
            kind,
            location: None,
            reason: reason.into(),
        }
    }

    /// Add location
    pub fn with_location(mut self, location: super::gradual::SourceLocation) -> Self {
        self.location = Some(location);
        self
    }
}

impl std::fmt::Display for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.kind, self.reason)
    }
}

/// Kind of constraint
#[derive(Debug, Clone)]
pub enum ConstraintKind {
    /// Type equality: τ₁ = τ₂
    Eq(EpistemicType, EpistemicType),

    /// Subtyping: τ₁ <: τ₂
    Subtype(EpistemicType, EpistemicType),

    /// Confidence constraint: ε₁ ≥ ε₂
    ConfidenceGeq(ConfidenceType, ConfidenceType),

    /// Confidence equality: ε₁ = ε₂
    ConfidenceEq(ConfidenceType, ConfidenceType),

    /// Ontology constraint: δ₁ ⊇ δ₂
    OntologySuperset(OntologyType, OntologyType),

    /// Predicate must be provable
    Predicate(Predicate),

    /// Variable instantiation: ?X = τ
    Instantiate(TypeVar, EpistemicType),

    /// Confidence variable instantiation: ?ε = confidence
    ConfidenceInstantiate(TypeVar, ConfidenceType),
}

impl std::fmt::Display for ConstraintKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Eq(t1, t2) => write!(f, "{} = {}", t1, t2),
            Self::Subtype(t1, t2) => write!(f, "{} <: {}", t1, t2),
            Self::ConfidenceGeq(c1, c2) => write!(f, "{} ≥ {}", c1, c2),
            Self::ConfidenceEq(c1, c2) => write!(f, "{} = {}", c1, c2),
            Self::OntologySuperset(o1, o2) => write!(f, "{} ⊇ {}", o1, o2),
            Self::Predicate(p) => write!(f, "prove {}", p),
            Self::Instantiate(v, t) => write!(f, "{} := {}", v, t),
            Self::ConfidenceInstantiate(v, c) => write!(f, "{} := {}", v, c),
        }
    }
}

/// Result of type inference
pub type InferenceResult<T> = Result<T, InferenceError>;

/// Error during type inference
#[derive(Debug, Clone, thiserror::Error)]
pub enum InferenceError {
    #[error("Unification failed: {0}")]
    UnificationFailed(String),

    #[error("Constraint unsatisfiable: {0}")]
    UnsatisfiableConstraint(String),

    #[error("Occurs check failed: {var} occurs in {ty}")]
    OccursCheck { var: String, ty: String },

    #[error("Unbound variable: {0}")]
    UnboundVariable(String),

    #[error("Kind mismatch: expected {expected}, found {found}")]
    KindMismatch { expected: String, found: String },

    #[error("Cannot prove predicate: {0}")]
    ProofFailed(String),
}

/// Type inference context
#[derive(Debug)]
pub struct InferenceContext {
    /// Generated constraints
    constraints: Vec<Constraint>,
    /// Type variable counter
    var_counter: usize,
    /// Type variable substitution
    substitution: HashMap<String, TypeVarValue>,
    /// Confidence variable substitution
    confidence_subst: HashMap<String, ConfidenceType>,
    /// Type context
    type_ctx: TypeContext,
    /// Whether to allow gradual typing
    gradual: bool,
}

/// Value assigned to a type variable
#[derive(Debug, Clone)]
pub enum TypeVarValue {
    /// Epistemic type
    Epistemic(EpistemicType),
    /// Confidence value
    Confidence(ConfidenceType),
    /// Ontology value
    Ontology(OntologyType),
    /// Still unresolved
    Unresolved,
}

impl Default for InferenceContext {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceContext {
    /// Create a new inference context
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            var_counter: 0,
            substitution: HashMap::new(),
            confidence_subst: HashMap::new(),
            type_ctx: TypeContext::new(),
            gradual: false,
        }
    }

    /// Enable gradual typing
    pub fn with_gradual(mut self, gradual: bool) -> Self {
        self.gradual = gradual;
        self
    }

    /// Generate a fresh type variable
    pub fn fresh_var(&mut self, prefix: &str, kind: TypeVarKind) -> TypeVar {
        let id = self.var_counter;
        self.var_counter += 1;
        TypeVar::new(prefix, kind, id)
    }

    /// Generate a fresh confidence type variable
    pub fn fresh_confidence(&mut self, prefix: &str) -> ConfidenceType {
        let var = self.fresh_var(prefix, TypeVarKind::Confidence);
        ConfidenceType::var(var.to_string())
    }

    /// Generate a fresh ontology type variable
    pub fn fresh_ontology(&mut self, prefix: &str) -> OntologyType {
        let var = self.fresh_var(prefix, TypeVarKind::Ontology);
        OntologyType::Var(var.to_string())
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Add an equality constraint
    pub fn eq(&mut self, t1: EpistemicType, t2: EpistemicType, reason: &str) {
        self.add_constraint(Constraint::new(ConstraintKind::Eq(t1, t2), reason));
    }

    /// Add a subtyping constraint
    pub fn subtype(&mut self, sub: EpistemicType, sup: EpistemicType, reason: &str) {
        self.add_constraint(Constraint::new(ConstraintKind::Subtype(sub, sup), reason));
    }

    /// Add a confidence constraint
    pub fn confidence_geq(&mut self, c1: ConfidenceType, c2: ConfidenceType, reason: &str) {
        self.add_constraint(Constraint::new(
            ConstraintKind::ConfidenceGeq(c1, c2),
            reason,
        ));
    }

    /// Add a predicate constraint
    pub fn must_prove(&mut self, pred: Predicate, reason: &str) {
        self.add_constraint(Constraint::new(ConstraintKind::Predicate(pred), reason));
    }

    /// Bind a type variable to the type context
    pub fn bind_type(&mut self, name: impl Into<String>, ty: EpistemicType) {
        let name = name.into();
        self.substitution.insert(name, TypeVarValue::Epistemic(ty));
    }

    /// Bind a confidence variable
    pub fn bind_confidence(&mut self, name: impl Into<String>, conf: ConfidenceType) {
        let name = name.into();
        self.confidence_subst.insert(name.clone(), conf.clone());
        self.type_ctx.bind_confidence(name, conf);
    }

    /// Get all constraints
    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    /// Get substitution
    pub fn substitution(&self) -> &HashMap<String, TypeVarValue> {
        &self.substitution
    }
}

/// Constraint solver
pub struct ConstraintSolver {
    /// Inference context
    ctx: InferenceContext,
    /// Proof search configuration
    proof_config: ProofSearchConfig,
    /// Errors encountered
    errors: Vec<InferenceError>,
}

impl ConstraintSolver {
    /// Create a new solver
    pub fn new(ctx: InferenceContext) -> Self {
        Self {
            ctx,
            proof_config: ProofSearchConfig::default(),
            errors: Vec::new(),
        }
    }

    /// Enable gradual typing for proof search
    pub fn with_gradual(mut self, gradual: bool) -> Self {
        self.proof_config.allow_gradual = gradual;
        self
    }

    /// Solve all constraints
    pub fn solve(mut self) -> InferenceResult<InferenceContext> {
        // Process constraints iteratively until fixed point
        let mut iterations = 0;
        let max_iterations = 100;

        // Track which constraints are solved
        let mut solved: std::collections::HashSet<usize> = std::collections::HashSet::new();

        loop {
            iterations += 1;
            if iterations > max_iterations {
                return Err(InferenceError::UnificationFailed(
                    "Maximum iterations exceeded".to_string(),
                ));
            }

            // Clone constraints to avoid borrow issues
            let constraints: Vec<_> = self.ctx.constraints.clone();
            let mut made_progress = false;

            for (i, constraint) in constraints.iter().enumerate() {
                if solved.contains(&i) {
                    continue;
                }

                let result = self.solve_constraint(constraint);
                if result {
                    solved.insert(i);
                    made_progress = true;
                }
            }

            // If all constraints are solved or no progress was made, stop
            if solved.len() == constraints.len() || !made_progress {
                break;
            }
        }

        // Check for remaining unsolved constraints
        self.check_unsolved()?;

        if !self.errors.is_empty() {
            return Err(self.errors.remove(0));
        }

        Ok(self.ctx)
    }

    /// Solve a single constraint
    fn solve_constraint(&mut self, constraint: &Constraint) -> bool {
        match &constraint.kind {
            ConstraintKind::Eq(t1, t2) => self.unify_types(t1, t2),
            ConstraintKind::Subtype(sub, sup) => self.check_subtype(sub, sup),
            ConstraintKind::ConfidenceGeq(c1, c2) => self.check_confidence_geq(c1, c2),
            ConstraintKind::ConfidenceEq(c1, c2) => self.unify_confidence(c1, c2),
            ConstraintKind::OntologySuperset(o1, o2) => self.check_ontology_superset(o1, o2),
            ConstraintKind::Predicate(pred) => self.prove_predicate(pred),
            ConstraintKind::Instantiate(var, ty) => {
                self.ctx
                    .substitution
                    .insert(var.to_string(), TypeVarValue::Epistemic(ty.clone()));
                true
            }
            ConstraintKind::ConfidenceInstantiate(var, conf) => {
                self.ctx
                    .confidence_subst
                    .insert(var.to_string(), conf.clone());
                true
            }
        }
    }

    /// Unify two epistemic types
    fn unify_types(&mut self, t1: &EpistemicType, t2: &EpistemicType) -> bool {
        match (t1, t2) {
            // Variable unification
            (EpistemicType::Var(v), t) | (t, EpistemicType::Var(v)) => {
                // Occurs check
                if self.occurs_in(v, t) {
                    self.errors.push(InferenceError::OccursCheck {
                        var: v.clone(),
                        ty: format!("{}", t),
                    });
                    return false;
                }
                self.ctx
                    .substitution
                    .insert(v.clone(), TypeVarValue::Epistemic(t.clone()));
                true
            }

            // Knowledge unification
            (
                EpistemicType::Knowledge {
                    inner: i1,
                    confidence: c1,
                    ontology: o1,
                    ..
                },
                EpistemicType::Knowledge {
                    inner: i2,
                    confidence: c2,
                    ontology: o2,
                    ..
                },
            ) => {
                if i1 != i2 {
                    self.errors.push(InferenceError::UnificationFailed(format!(
                        "Inner types differ: {:?} vs {:?}",
                        i1, i2
                    )));
                    return false;
                }
                self.unify_confidence(c1, c2) && self.unify_ontology(o1, o2)
            }

            // Unknown compatible with anything (gradual)
            (EpistemicType::Unknown, _) | (_, EpistemicType::Unknown) => {
                self.proof_config.allow_gradual
            }

            // Refinement unification
            (
                EpistemicType::Refinement {
                    base: b1,
                    predicate: p1,
                },
                EpistemicType::Refinement {
                    base: b2,
                    predicate: p2,
                },
            ) => self.unify_types(b1, b2) && p1 == p2,

            _ => {
                self.errors.push(InferenceError::UnificationFailed(format!(
                    "Cannot unify {} with {}",
                    t1, t2
                )));
                false
            }
        }
    }

    /// Unify two confidence types
    fn unify_confidence(&mut self, c1: &ConfidenceType, c2: &ConfidenceType) -> bool {
        match (c1, c2) {
            (ConfidenceType::Var(v), c) | (c, ConfidenceType::Var(v)) => {
                // Check if already bound
                if let Some(bound) = self.ctx.confidence_subst.get(v) {
                    return bound.definitionally_equal(c);
                }
                self.ctx.confidence_subst.insert(v.clone(), c.clone());
                self.ctx.type_ctx.bind_confidence(v.clone(), c.clone());
                true
            }
            (ConfidenceType::Literal(l1), ConfidenceType::Literal(l2)) => (l1 - l2).abs() < 1e-10,
            (ConfidenceType::Unknown, _) | (_, ConfidenceType::Unknown) => {
                self.proof_config.allow_gradual
            }
            _ => c1.definitionally_equal(c2),
        }
    }

    /// Unify two ontology types
    fn unify_ontology(&mut self, o1: &OntologyType, o2: &OntologyType) -> bool {
        match (o1, o2) {
            (OntologyType::Var(v), o) | (o, OntologyType::Var(v)) => {
                // Would need ontology substitution tracking
                true
            }
            (OntologyType::Unknown, _) | (_, OntologyType::Unknown) => {
                self.proof_config.allow_gradual
            }
            _ => o1.definitionally_equal(o2),
        }
    }

    /// Check subtype relation
    fn check_subtype(&mut self, sub: &EpistemicType, sup: &EpistemicType) -> bool {
        let checker = super::subtyping::SubtypeChecker::new(&self.ctx.type_ctx)
            .with_gradual(self.proof_config.allow_gradual);
        let result = checker.check(sub, sup);
        result.is_subtype()
    }

    /// Check confidence constraint
    fn check_confidence_geq(&mut self, c1: &ConfidenceType, c2: &ConfidenceType) -> bool {
        // Apply current substitution
        let c1_subst = self.apply_confidence_subst(c1);
        let c2_subst = self.apply_confidence_subst(c2);

        // Try to evaluate
        if let (Some(v1), Some(v2)) = (
            c1_subst.evaluate(&self.ctx.type_ctx),
            c2_subst.evaluate(&self.ctx.type_ctx),
        ) {
            return v1 >= v2;
        }

        // Try lower bounds
        if let (Some(lb), Some(v2)) = (
            c1_subst.lower_bound(&self.ctx.type_ctx),
            c2_subst.evaluate(&self.ctx.type_ctx),
        ) && lb >= v2
        {
            return true;
        }

        // Gradual fallback
        self.proof_config.allow_gradual
    }

    /// Check ontology superset
    fn check_ontology_superset(&mut self, o1: &OntologyType, o2: &OntologyType) -> bool {
        o1.contains(o2)
    }

    /// Try to prove a predicate
    fn prove_predicate(&mut self, pred: &Predicate) -> bool {
        let mut searcher =
            ProofSearcher::with_config(&self.ctx.type_ctx, self.proof_config.clone());
        let result = searcher.search(pred);

        match result {
            ProofResult::Proven(_) => true,
            ProofResult::Disproven { reason } => {
                self.errors.push(InferenceError::ProofFailed(reason));
                false
            }
            ProofResult::Unknown { reason } => {
                if self.proof_config.allow_gradual {
                    true
                } else {
                    self.errors.push(InferenceError::ProofFailed(reason));
                    false
                }
            }
        }
    }

    /// Apply confidence substitution
    fn apply_confidence_subst(&self, conf: &ConfidenceType) -> ConfidenceType {
        match conf {
            ConfidenceType::Var(v) => self
                .ctx
                .confidence_subst
                .get(v)
                .cloned()
                .unwrap_or_else(|| conf.clone()),
            ConfidenceType::Product(a, b) => ConfidenceType::product(
                self.apply_confidence_subst(a),
                self.apply_confidence_subst(b),
            ),
            ConfidenceType::DempsterShafer(a, b) => ConfidenceType::dempster_shafer(
                self.apply_confidence_subst(a),
                self.apply_confidence_subst(b),
            ),
            ConfidenceType::Min(a, b) => ConfidenceType::min(
                self.apply_confidence_subst(a),
                self.apply_confidence_subst(b),
            ),
            ConfidenceType::Max(a, b) => ConfidenceType::max(
                self.apply_confidence_subst(a),
                self.apply_confidence_subst(b),
            ),
            ConfidenceType::Decay {
                base,
                lambda,
                elapsed,
            } => ConfidenceType::decay(self.apply_confidence_subst(base), *lambda, *elapsed),
            _ => conf.clone(),
        }
    }

    /// Check for variable occurrences (occurs check)
    fn occurs_in(&self, var: &str, ty: &EpistemicType) -> bool {
        match ty {
            EpistemicType::Var(v) => v == var,
            EpistemicType::Knowledge { .. } => false,
            EpistemicType::CausalKnowledge { .. } => false,
            EpistemicType::StructuralKnowledge { .. } => false,
            EpistemicType::Refinement { base, .. } => self.occurs_in(var, base),
            EpistemicType::Pi { body, .. } => self.occurs_in(var, body),
            EpistemicType::Sigma { snd_type, .. } => self.occurs_in(var, snd_type),
            EpistemicType::Proof(_) => false,
            EpistemicType::Unknown => false,
        }
    }

    /// Check for remaining unsolved constraints
    fn check_unsolved(&mut self) -> InferenceResult<()> {
        // Clone constraints to avoid borrow issues
        let constraints: Vec<_> = self.ctx.constraints.clone();

        for constraint in constraints {
            match &constraint.kind {
                ConstraintKind::Eq(t1, t2) => {
                    if !self.types_equal_after_subst(t1, t2) {
                        return Err(InferenceError::UnsatisfiableConstraint(format!(
                            "Could not unify {} = {}",
                            t1, t2
                        )));
                    }
                }
                ConstraintKind::Subtype(sub, sup) => {
                    if !self.check_subtype(sub, sup) && !self.proof_config.allow_gradual {
                        return Err(InferenceError::UnsatisfiableConstraint(format!(
                            "Could not prove {} <: {}",
                            sub, sup
                        )));
                    }
                }
                ConstraintKind::Predicate(pred) => {
                    if !self.prove_predicate(pred) && !self.proof_config.allow_gradual {
                        return Err(InferenceError::UnsatisfiableConstraint(format!(
                            "Could not prove {}",
                            pred
                        )));
                    }
                }
                ConstraintKind::ConfidenceGeq(c1, c2) => {
                    if !self.check_confidence_geq(c1, c2) && !self.proof_config.allow_gradual {
                        return Err(InferenceError::UnsatisfiableConstraint(format!(
                            "Could not prove {} >= {}",
                            c1, c2
                        )));
                    }
                }
                ConstraintKind::ConfidenceEq(c1, c2) => {
                    if !self.unify_confidence(c1, c2) && !self.proof_config.allow_gradual {
                        return Err(InferenceError::UnsatisfiableConstraint(format!(
                            "Could not unify {} = {}",
                            c1, c2
                        )));
                    }
                }
                ConstraintKind::OntologySuperset(o1, o2) => {
                    if !self.check_ontology_superset(o1, o2) && !self.proof_config.allow_gradual {
                        return Err(InferenceError::UnsatisfiableConstraint(format!(
                            "Could not prove {} ⊇ {}",
                            o1, o2
                        )));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Check if two types are equal after applying substitution
    fn types_equal_after_subst(&self, t1: &EpistemicType, t2: &EpistemicType) -> bool {
        let t1_subst = self.apply_type_subst(t1);
        let t2_subst = self.apply_type_subst(t2);

        match (&t1_subst, &t2_subst) {
            (EpistemicType::Var(v1), EpistemicType::Var(v2)) => v1 == v2,
            (
                EpistemicType::Knowledge {
                    inner: i1,
                    confidence: c1,
                    ontology: o1,
                    ..
                },
                EpistemicType::Knowledge {
                    inner: i2,
                    confidence: c2,
                    ontology: o2,
                    ..
                },
            ) => i1 == i2 && c1.definitionally_equal(c2) && o1.definitionally_equal(o2),
            _ => false,
        }
    }

    /// Apply type substitution
    fn apply_type_subst(&self, ty: &EpistemicType) -> EpistemicType {
        match ty {
            EpistemicType::Var(v) => {
                if let Some(TypeVarValue::Epistemic(t)) = self.ctx.substitution.get(v) {
                    self.apply_type_subst(t)
                } else {
                    ty.clone()
                }
            }
            _ => ty.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Type;

    #[test]
    fn test_fresh_var() {
        let mut ctx = InferenceContext::new();
        let v1 = ctx.fresh_var("T", TypeVarKind::Epistemic);
        let v2 = ctx.fresh_var("T", TypeVarKind::Epistemic);
        assert_ne!(v1.id, v2.id);
    }

    #[test]
    fn test_confidence_unification() {
        let mut ctx = InferenceContext::new();
        let c1 = ctx.fresh_confidence("ε");
        ctx.confidence_geq(c1.clone(), ConfidenceType::literal(0.95), "test");

        let solver = ConstraintSolver::new(ctx).with_gradual(true);
        let result = solver.solve();
        assert!(result.is_ok());
    }

    #[test]
    fn test_type_var_instantiation() {
        let mut ctx = InferenceContext::new();
        let var = ctx.fresh_var("T", TypeVarKind::Epistemic);

        let ty = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
        );

        ctx.add_constraint(Constraint::new(
            ConstraintKind::Instantiate(var.clone(), ty.clone()),
            "test",
        ));

        let solver = ConstraintSolver::new(ctx);
        let result = solver.solve().unwrap();

        assert!(result.substitution.contains_key(&var.to_string()));
    }

    #[test]
    fn test_confidence_binding() {
        let mut ctx = InferenceContext::new();
        ctx.bind_confidence("ε", ConfidenceType::literal(0.97));

        ctx.confidence_geq(
            ConfidenceType::var("ε"),
            ConfidenceType::literal(0.95),
            "test",
        );

        let solver = ConstraintSolver::new(ctx);
        let result = solver.solve();
        assert!(result.is_ok());
    }

    #[test]
    fn test_unsatisfiable_constraint() {
        let mut ctx = InferenceContext::new();
        ctx.bind_confidence("ε", ConfidenceType::literal(0.80));

        ctx.confidence_geq(
            ConfidenceType::var("ε"),
            ConfidenceType::literal(0.95),
            "test",
        );

        let solver = ConstraintSolver::new(ctx);
        let result = solver.solve();
        // Without gradual, this should fail
        // With gradual (default false), should also fail
        assert!(result.is_err() || result.unwrap().gradual);
    }

    #[test]
    fn test_gradual_allows_unknown() {
        let mut ctx = InferenceContext::new().with_gradual(true);

        ctx.confidence_geq(
            ConfidenceType::var("unknown"),
            ConfidenceType::literal(0.95),
            "test",
        );

        let solver = ConstraintSolver::new(ctx).with_gradual(true);
        let result = solver.solve();
        assert!(result.is_ok());
    }

    #[test]
    fn test_subtype_constraint() {
        let mut ctx = InferenceContext::new();

        let high = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.97),
            OntologyType::concrete("PKPD"),
        );

        let low = EpistemicType::knowledge(
            Type::F64,
            ConfidenceType::literal(0.95),
            OntologyType::concrete("PKPD"),
        );

        ctx.subtype(high, low, "test");

        let solver = ConstraintSolver::new(ctx);
        let result = solver.solve();
        assert!(result.is_ok());
    }

    #[test]
    fn test_predicate_constraint() {
        let mut ctx = InferenceContext::new();

        ctx.must_prove(
            Predicate::confidence_geq(ConfidenceType::literal(0.97), ConfidenceType::literal(0.95)),
            "test",
        );

        let solver = ConstraintSolver::new(ctx);
        let result = solver.solve();
        assert!(result.is_ok());
    }
}
