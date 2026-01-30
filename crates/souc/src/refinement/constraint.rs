//! Constraint generation for refinement type checking
//!
//! This module generates verification constraints from the program that must
//! be checked by the SMT solver. Constraints arise from:
//! - Subtyping checks (refinement implication)
//! - Array bounds checks
//! - Division by zero checks
//! - Custom assertions
//! - Medical safety constraints

use super::predicate::*;
use crate::types::Type;
use std::collections::HashMap;

/// A span in source code for error reporting
#[derive(Debug, Clone, Copy, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    /// Create a new span
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Create a dummy span for testing
    pub fn dummy() -> Self {
        Self { start: 0, end: 0 }
    }
}

/// A verification constraint to be checked by the SMT solver
#[derive(Debug, Clone)]
pub struct Constraint {
    /// Environment (assumptions about variables in scope)
    pub env: Vec<Binding>,

    /// The predicate to prove
    pub goal: Predicate,

    /// Source location for error reporting
    pub span: Span,

    /// Description of what this constraint checks
    pub reason: ConstraintReason,
}

impl Constraint {
    /// Create a new constraint
    pub fn new(env: Vec<Binding>, goal: Predicate, span: Span, reason: ConstraintReason) -> Self {
        Self {
            env,
            goal,
            span,
            reason,
        }
    }

    /// Get all free variables in this constraint
    pub fn free_vars(&self) -> std::collections::HashSet<String> {
        let mut vars = self.goal.free_vars();
        for binding in &self.env {
            vars.extend(binding.ty.predicate.free_vars());
        }
        vars
    }

    /// Build the full constraint as: env ⊢ goal
    /// Returns: (env_pred => goal)
    pub fn as_implication(&self) -> Predicate {
        let env_pred = self.env_predicate();
        Predicate::implies(env_pred, self.goal.clone())
    }

    /// Build the environment predicate from bindings
    fn env_predicate(&self) -> Predicate {
        let preds: Vec<_> = self
            .env
            .iter()
            .filter_map(|b| {
                if b.ty.predicate == Predicate::True {
                    None
                } else {
                    Some(b.ty.predicate.substitute(&b.ty.var, &Term::var(&b.name)))
                }
            })
            .collect();

        Predicate::and(preds)
    }
}

/// Reason for a constraint (for error messages)
#[derive(Debug, Clone)]
pub enum ConstraintReason {
    /// Subtyping check: sub <: sup
    Subtype {
        sub: RefinementType,
        sup: RefinementType,
    },

    /// Function precondition check
    Precondition { func: String, param: String },

    /// Function postcondition check
    Postcondition { func: String },

    /// Array bounds check
    BoundsCheck { index: Term, length: Term },

    /// Division by zero check
    DivisionCheck { divisor: Term },

    /// Custom assertion
    Assert { message: String },

    /// Medical safety constraint
    SafetyConstraint { description: String },

    /// Type annotation check
    TypeAnnotation {
        expected: RefinementType,
        actual: RefinementType,
    },

    /// Return type check
    ReturnType {
        func: String,
        expected: RefinementType,
    },
}

impl std::fmt::Display for ConstraintReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstraintReason::Subtype { sub, sup } => {
                write!(f, "subtype check: {} <: {}", sub, sup)
            }
            ConstraintReason::Precondition { func, param } => {
                write!(
                    f,
                    "precondition of parameter '{}' in function '{}'",
                    param, func
                )
            }
            ConstraintReason::Postcondition { func } => {
                write!(f, "postcondition of function '{}'", func)
            }
            ConstraintReason::BoundsCheck { index, length } => {
                write!(f, "array bounds: 0 <= {} < {}", index, length)
            }
            ConstraintReason::DivisionCheck { divisor } => {
                write!(f, "division by zero check: {} != 0", divisor)
            }
            ConstraintReason::Assert { message } => {
                write!(f, "assertion: {}", message)
            }
            ConstraintReason::SafetyConstraint { description } => {
                write!(f, "safety: {}", description)
            }
            ConstraintReason::TypeAnnotation { expected, actual } => {
                write!(f, "type annotation: expected {}, got {}", expected, actual)
            }
            ConstraintReason::ReturnType { func, expected } => {
                write!(f, "return type of '{}': expected {}", func, expected)
            }
        }
    }
}

/// A variable binding in the environment
#[derive(Debug, Clone)]
pub struct Binding {
    /// Variable name
    pub name: String,
    /// Refined type
    pub ty: RefinementType,
}

impl Binding {
    /// Create a new binding
    pub fn new(name: impl Into<String>, ty: RefinementType) -> Self {
        Self {
            name: name.into(),
            ty,
        }
    }
}

/// Constraint generator
///
/// Collects verification constraints as the type checker traverses the program.
pub struct ConstraintGenerator {
    /// Generated constraints
    constraints: Vec<Constraint>,

    /// Current environment (variable bindings)
    env: Vec<Binding>,

    /// Fresh variable counter
    fresh_counter: u32,

    /// Known refinement types for named types
    known_types: HashMap<String, RefinementType>,

    /// Path conditions (from if-then-else branches)
    path_conditions: Vec<Predicate>,
}

impl ConstraintGenerator {
    /// Create a new constraint generator
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            env: Vec::new(),
            fresh_counter: 0,
            known_types: HashMap::new(),
            path_conditions: Vec::new(),
        }
    }

    /// Generate a fresh variable name
    pub fn fresh(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.fresh_counter);
        self.fresh_counter += 1;
        name
    }

    /// Push a binding onto the environment
    pub fn push_binding(&mut self, name: impl Into<String>, ty: RefinementType) {
        self.env.push(Binding::new(name, ty));
    }

    /// Pop a binding from the environment
    pub fn pop_binding(&mut self) -> Option<Binding> {
        self.env.pop()
    }

    /// Push a path condition (for if-then-else)
    pub fn push_path_condition(&mut self, cond: Predicate) {
        self.path_conditions.push(cond);
    }

    /// Pop a path condition
    pub fn pop_path_condition(&mut self) -> Option<Predicate> {
        self.path_conditions.pop()
    }

    /// Register a known refinement type for a named type
    pub fn register_type(&mut self, name: impl Into<String>, ty: RefinementType) {
        self.known_types.insert(name.into(), ty);
    }

    /// Look up a known refinement type
    pub fn lookup_type(&self, name: &str) -> Option<&RefinementType> {
        self.known_types.get(name)
    }

    /// Look up a binding in the environment
    pub fn lookup_binding(&self, name: &str) -> Option<&Binding> {
        self.env.iter().rev().find(|b| b.name == name)
    }

    /// Get the current environment
    pub fn current_env(&self) -> Vec<Binding> {
        let mut env = self.env.clone();

        // Add path conditions as anonymous bindings
        for (i, cond) in self.path_conditions.iter().enumerate() {
            env.push(Binding::new(
                format!("__path_{}", i),
                RefinementType::refined(Type::Bool, "v", cond.clone()),
            ));
        }

        env
    }

    /// Add a subtyping constraint: sub <: sup
    pub fn add_subtype(&mut self, sub: &RefinementType, sup: &RefinementType, span: Span) {
        // { v: T | P } <: { v: T | Q }
        // iff ∀v. P(v) ⟹ Q(v)

        // Substitute the refinement variable in sub to match sup
        let sub_pred = sub.predicate.substitute(&sub.var, &Term::var(&sup.var));

        // Build: env ⊢ P ⟹ Q
        let goal = Predicate::implies(sub_pred, sup.predicate.clone());

        // Skip trivial constraints
        if goal == Predicate::True {
            return;
        }

        self.constraints.push(Constraint::new(
            self.current_env(),
            goal,
            span,
            ConstraintReason::Subtype {
                sub: sub.clone(),
                sup: sup.clone(),
            },
        ));
    }

    /// Add a precondition constraint
    pub fn add_precondition(
        &mut self,
        func: &str,
        param: &str,
        arg: &RefinementType,
        param_ty: &RefinementType,
        span: Span,
    ) {
        let sub_pred = arg
            .predicate
            .substitute(&arg.var, &Term::var(&param_ty.var));
        let goal = Predicate::implies(sub_pred, param_ty.predicate.clone());

        if goal == Predicate::True {
            return;
        }

        self.constraints.push(Constraint::new(
            self.current_env(),
            goal,
            span,
            ConstraintReason::Precondition {
                func: func.to_string(),
                param: param.to_string(),
            },
        ));
    }

    /// Add a postcondition constraint
    pub fn add_postcondition(
        &mut self,
        func: &str,
        result: &RefinementType,
        expected: &RefinementType,
        span: Span,
    ) {
        let sub_pred = result
            .predicate
            .substitute(&result.var, &Term::var(&expected.var));
        let goal = Predicate::implies(sub_pred, expected.predicate.clone());

        if goal == Predicate::True {
            return;
        }

        self.constraints.push(Constraint::new(
            self.current_env(),
            goal,
            span,
            ConstraintReason::Postcondition {
                func: func.to_string(),
            },
        ));
    }

    /// Add an array bounds check constraint
    pub fn add_bounds_check(&mut self, index: Term, length: Term, span: Span) {
        // 0 <= index < length
        let lower = Predicate::Atom(Atom::new(CompareOp::Ge, index.clone(), Term::int(0)));

        let upper = Predicate::Atom(Atom::new(CompareOp::Lt, index.clone(), length.clone()));

        let goal = Predicate::and([lower, upper]);

        self.constraints.push(Constraint::new(
            self.current_env(),
            goal,
            span,
            ConstraintReason::BoundsCheck { index, length },
        ));
    }

    /// Add a division by zero check constraint
    pub fn add_division_check(&mut self, divisor: Term, span: Span) {
        // divisor != 0
        let goal = Predicate::Atom(Atom::new(CompareOp::Ne, divisor.clone(), Term::int(0)));

        self.constraints.push(Constraint::new(
            self.current_env(),
            goal,
            span,
            ConstraintReason::DivisionCheck { divisor },
        ));
    }

    /// Add a custom assertion constraint
    pub fn add_assertion(&mut self, pred: Predicate, message: impl Into<String>, span: Span) {
        if pred == Predicate::True {
            return;
        }

        self.constraints.push(Constraint::new(
            self.current_env(),
            pred,
            span,
            ConstraintReason::Assert {
                message: message.into(),
            },
        ));
    }

    /// Add a medical safety constraint
    pub fn add_safety_constraint(
        &mut self,
        pred: Predicate,
        description: impl Into<String>,
        span: Span,
    ) {
        if pred == Predicate::True {
            return;
        }

        self.constraints.push(Constraint::new(
            self.current_env(),
            pred,
            span,
            ConstraintReason::SafetyConstraint {
                description: description.into(),
            },
        ));
    }

    /// Add a type annotation check constraint
    pub fn add_type_annotation(
        &mut self,
        actual: &RefinementType,
        expected: &RefinementType,
        span: Span,
    ) {
        let sub_pred = actual
            .predicate
            .substitute(&actual.var, &Term::var(&expected.var));
        let goal = Predicate::implies(sub_pred, expected.predicate.clone());

        if goal == Predicate::True {
            return;
        }

        self.constraints.push(Constraint::new(
            self.current_env(),
            goal,
            span,
            ConstraintReason::TypeAnnotation {
                expected: expected.clone(),
                actual: actual.clone(),
            },
        ));
    }

    /// Get all generated constraints
    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    /// Consume the generator and return all constraints
    pub fn into_constraints(self) -> Vec<Constraint> {
        self.constraints
    }

    /// Get the number of constraints
    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    /// Check if there are no constraints
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Clear all constraints (for testing)
    pub fn clear(&mut self) {
        self.constraints.clear();
        self.env.clear();
        self.path_conditions.clear();
    }
}

impl Default for ConstraintGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Pre-built medical safety constraints
pub mod medical_constraints {
    use super::*;

    /// Create a max dose constraint
    pub fn max_dose_constraint(dose_var: &str, max: f64) -> Predicate {
        Predicate::and([
            Predicate::gt(Term::var(dose_var), Term::float(0.0)),
            Predicate::le(Term::var(dose_var), Term::float(max)),
        ])
    }

    /// Create a therapeutic range constraint
    pub fn therapeutic_range_constraint(conc_var: &str, min: f64, max: f64) -> Predicate {
        Predicate::and([
            Predicate::ge(Term::var(conc_var), Term::float(min)),
            Predicate::le(Term::var(conc_var), Term::float(max)),
        ])
    }

    /// Create a valid CrCl constraint
    pub fn valid_crcl_constraint(crcl_var: &str) -> Predicate {
        Predicate::and([
            Predicate::gt(Term::var(crcl_var), Term::float(0.0)),
            Predicate::lt(Term::var(crcl_var), Term::float(200.0)),
        ])
    }

    /// Create a positive weight constraint
    pub fn positive_weight_constraint(weight_var: &str) -> Predicate {
        Predicate::and([
            Predicate::gt(Term::var(weight_var), Term::float(0.0)),
            Predicate::le(Term::var(weight_var), Term::float(500.0)),
        ])
    }

    /// Create a valid age constraint
    pub fn valid_age_constraint(age_var: &str) -> Predicate {
        Predicate::and([
            Predicate::ge(Term::var(age_var), Term::float(0.0)),
            Predicate::le(Term::var(age_var), Term::float(150.0)),
        ])
    }

    /// Create a non-zero divisor constraint (for CrCl calculation)
    pub fn safe_division_constraint(divisor: Term) -> Predicate {
        Predicate::ne(divisor, Term::float(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_generator_basic() {
        let mut cg = ConstraintGenerator::new();

        let pos_int = RefinementType::positive(Type::I64);
        cg.push_binding("x", pos_int.clone());

        assert_eq!(cg.current_env().len(), 1);

        let binding = cg.lookup_binding("x").unwrap();
        assert_eq!(binding.name, "x");
    }

    #[test]
    fn test_subtype_constraint() {
        let mut cg = ConstraintGenerator::new();

        let pos = RefinementType::positive(Type::I64);
        let non_neg = RefinementType::non_negative(Type::I64);

        cg.add_subtype(&pos, &non_neg, Span::dummy());

        // positive <: non_negative should generate one constraint
        assert_eq!(cg.len(), 1);
    }

    #[test]
    fn test_bounds_check_constraint() {
        let mut cg = ConstraintGenerator::new();

        cg.add_bounds_check(Term::var("i"), Term::var("len"), Span::dummy());

        assert_eq!(cg.len(), 1);

        let constraint = &cg.constraints()[0];
        assert!(matches!(
            constraint.reason,
            ConstraintReason::BoundsCheck { .. }
        ));
    }

    #[test]
    fn test_division_check_constraint() {
        let mut cg = ConstraintGenerator::new();

        cg.add_division_check(Term::var("x"), Span::dummy());

        assert_eq!(cg.len(), 1);

        let constraint = &cg.constraints()[0];
        assert!(matches!(
            constraint.reason,
            ConstraintReason::DivisionCheck { .. }
        ));
    }

    #[test]
    fn test_path_conditions() {
        let mut cg = ConstraintGenerator::new();

        // Simulate entering an if branch where x > 0
        let cond = Predicate::gt(Term::var("x"), Term::int(0));
        cg.push_path_condition(cond.clone());

        let env = cg.current_env();
        assert_eq!(env.len(), 1); // Path condition as binding

        cg.pop_path_condition();
        let env = cg.current_env();
        assert_eq!(env.len(), 0);
    }

    #[test]
    fn test_fresh_variable() {
        let mut cg = ConstraintGenerator::new();

        let v1 = cg.fresh("temp");
        let v2 = cg.fresh("temp");

        assert_ne!(v1, v2);
        assert!(v1.starts_with("temp_"));
        assert!(v2.starts_with("temp_"));
    }

    #[test]
    fn test_trivial_subtype_not_generated() {
        let mut cg = ConstraintGenerator::new();

        let trivial = RefinementType::trivial(Type::I64);

        // trivial <: trivial should not generate a constraint
        cg.add_subtype(&trivial, &trivial, Span::dummy());

        assert!(cg.is_empty());
    }

    #[test]
    fn test_medical_max_dose() {
        let constraint = medical_constraints::max_dose_constraint("dose", 1000.0);

        let vars = constraint.free_vars();
        assert!(vars.contains("dose"));
        assert_eq!(vars.len(), 1);
    }

    #[test]
    fn test_constraint_as_implication() {
        let mut cg = ConstraintGenerator::new();

        // Add binding: x > 0
        let pos = RefinementType::positive(Type::I64);
        cg.push_binding("x", pos);

        // Add constraint: x >= 0 (should be implied by x > 0)
        cg.add_assertion(
            Predicate::ge(Term::var("x"), Term::int(0)),
            "x is non-negative",
            Span::dummy(),
        );

        assert_eq!(cg.len(), 1);

        let constraint = &cg.constraints()[0];
        let impl_pred = constraint.as_implication();

        // Should be: (x > 0) => (x >= 0)
        assert!(matches!(impl_pred, Predicate::Implies(_, _)));
    }
}
