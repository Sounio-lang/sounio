//! Refinement subtype checking
//!
//! Determines when one refinement type is a subtype of another using
//! the SMT solver to verify the implication between predicates.
//!
//! # Subtyping Rule
//!
//! A refinement type `{x: T | P}` is a subtype of `{x: T | Q}` if:
//! ```text
//! ∀x. P(x) ⟹ Q(x)
//! ```
//!
//! This is equivalent to checking that the negation `P(x) ∧ ¬Q(x)` is unsatisfiable.

use super::constraint::*;
use super::predicate::*;
use super::solver::*;

#[allow(unused_imports)]
use crate::types::Type;

#[cfg(feature = "smt")]
use z3::Context;

/// Subtype checker for refinement types
///
/// Collects subtyping constraints and verifies them with the SMT solver.
pub struct SubtypeChecker {
    /// Constraint generator
    cg: ConstraintGenerator,

    /// Whether to perform strict checking (fail on Unknown results)
    strict: bool,
}

impl SubtypeChecker {
    /// Create a new subtype checker
    pub fn new() -> Self {
        Self {
            cg: ConstraintGenerator::new(),
            strict: false,
        }
    }

    /// Create a strict subtype checker that fails on Unknown results
    pub fn strict() -> Self {
        Self {
            cg: ConstraintGenerator::new(),
            strict: true,
        }
    }

    /// Check if `sub` is a subtype of `sup`
    ///
    /// Returns true if the base types match and the constraint was added.
    /// The actual verification happens in `verify()`.
    pub fn is_subtype(&mut self, sub: &RefinementType, sup: &RefinementType, span: Span) -> bool {
        // Base types must match
        if sub.base != sup.base {
            return false;
        }

        self.cg.add_subtype(sub, sup, span);
        true
    }

    /// Check if a value satisfies a refinement type
    pub fn check_value(&mut self, value: &Term, ty: &RefinementType, span: Span) {
        // Substitute the value for the refinement variable
        let pred = ty.predicate.substitute(&ty.var, value);

        self.cg
            .add_assertion(pred, format!("value satisfies {}", ty), span);
    }

    /// Check a precondition: argument type <: parameter type
    pub fn check_precondition(
        &mut self,
        func: &str,
        param: &str,
        arg: &RefinementType,
        param_ty: &RefinementType,
        span: Span,
    ) {
        self.cg.add_precondition(func, param, arg, param_ty, span);
    }

    /// Check a postcondition: result type <: declared return type
    pub fn check_postcondition(
        &mut self,
        func: &str,
        result: &RefinementType,
        expected: &RefinementType,
        span: Span,
    ) {
        self.cg.add_postcondition(func, result, expected, span);
    }

    /// Add an assumption to the environment
    pub fn assume(&mut self, name: impl Into<String>, ty: RefinementType) {
        self.cg.push_binding(name, ty);
    }

    /// Remove the most recent assumption
    pub fn unassume(&mut self) {
        self.cg.pop_binding();
    }

    /// Enter a branch with a path condition
    pub fn enter_branch(&mut self, condition: Predicate) {
        self.cg.push_path_condition(condition);
    }

    /// Exit a branch
    pub fn exit_branch(&mut self) {
        self.cg.pop_path_condition();
    }

    /// Add a bounds check
    pub fn check_bounds(&mut self, index: Term, length: Term, span: Span) {
        self.cg.add_bounds_check(index, length, span);
    }

    /// Add a division check
    pub fn check_division(&mut self, divisor: Term, span: Span) {
        self.cg.add_division_check(divisor, span);
    }

    /// Add a safety constraint
    pub fn check_safety(&mut self, pred: Predicate, desc: impl Into<String>, span: Span) {
        self.cg.add_safety_constraint(pred, desc, span);
    }

    /// Get the number of pending constraints
    pub fn pending_constraints(&self) -> usize {
        self.cg.len()
    }

    /// Verify all collected constraints using the SMT solver
    #[cfg(feature = "smt")]
    pub fn verify(self) -> SubtypeResult {
        let constraints = self.cg.into_constraints();

        if constraints.is_empty() {
            return SubtypeResult {
                valid: true,
                constraints: Vec::new(),
                results: Vec::new(),
                errors: Vec::new(),
            };
        }

        let cfg = z3::Config::new();
        let ctx = Context::new(&cfg);
        let mut solver = Z3Solver::new(&ctx);

        let results = solver.verify(&constraints);

        let mut errors = Vec::new();
        let mut all_valid = true;

        for (i, result) in results.iter().enumerate() {
            match result {
                VerifyResult::Valid => {}
                VerifyResult::Invalid { counterexample, .. } => {
                    all_valid = false;
                    errors.push(SubtypeError {
                        constraint_idx: i,
                        reason: constraints[i].reason.clone(),
                        span: constraints[i].span,
                        counterexample: counterexample.clone(),
                    });
                }
                VerifyResult::Unknown { reason, .. } => {
                    if self.strict {
                        all_valid = false;
                        errors.push(SubtypeError {
                            constraint_idx: i,
                            reason: constraints[i].reason.clone(),
                            span: constraints[i].span,
                            counterexample: None,
                        });
                    }
                }
            }
        }

        SubtypeResult {
            valid: all_valid,
            constraints,
            results,
            errors,
        }
    }

    /// Verify constraints without SMT (stub version)
    #[cfg(not(feature = "smt"))]
    pub fn verify(self) -> SubtypeResult {
        let constraints = self.cg.into_constraints();

        let results: Vec<_> = constraints
            .iter()
            .enumerate()
            .map(|(i, _)| VerifyResult::Unknown {
                constraint_idx: i,
                reason: "Z3 not available".to_string(),
            })
            .collect();

        // In non-SMT mode, try simple checking
        let mut errors = Vec::new();
        let mut all_valid = true;

        for (i, constraint) in constraints.iter().enumerate() {
            match SimpleChecker::check(&constraint.goal) {
                Some(true) => {}
                Some(false) => {
                    all_valid = false;
                    errors.push(SubtypeError {
                        constraint_idx: i,
                        reason: constraint.reason.clone(),
                        span: constraint.span,
                        counterexample: None,
                    });
                }
                None => {
                    if self.strict {
                        all_valid = false;
                        errors.push(SubtypeError {
                            constraint_idx: i,
                            reason: constraint.reason.clone(),
                            span: constraint.span,
                            counterexample: None,
                        });
                    }
                }
            }
        }

        SubtypeResult {
            valid: all_valid,
            constraints,
            results,
            errors,
        }
    }
}

impl Default for SubtypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of subtype verification
#[derive(Debug)]
pub struct SubtypeResult {
    /// Whether all constraints were verified
    pub valid: bool,

    /// All generated constraints
    pub constraints: Vec<Constraint>,

    /// Verification results for each constraint
    pub results: Vec<VerifyResult>,

    /// Errors from failed constraints
    pub errors: Vec<SubtypeError>,
}

impl SubtypeResult {
    /// Check if verification succeeded
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Get the first error, if any
    pub fn first_error(&self) -> Option<&SubtypeError> {
        self.errors.first()
    }

    /// Get all errors
    pub fn all_errors(&self) -> &[SubtypeError] {
        &self.errors
    }

    /// Get the number of constraints
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Get the number of valid constraints
    pub fn num_valid(&self) -> usize {
        self.results.iter().filter(|r| r.is_valid()).count()
    }

    /// Get the number of invalid constraints
    pub fn num_invalid(&self) -> usize {
        self.results.iter().filter(|r| r.is_invalid()).count()
    }

    /// Get the number of unknown constraints
    pub fn num_unknown(&self) -> usize {
        self.results.iter().filter(|r| r.is_unknown()).count()
    }
}

/// An error from a failed subtype check
#[derive(Debug)]
pub struct SubtypeError {
    /// Index of the failed constraint
    pub constraint_idx: usize,

    /// Reason for the constraint
    pub reason: ConstraintReason,

    /// Source location
    pub span: Span,

    /// Counterexample (if available)
    pub counterexample: Option<Counterexample>,
}

impl std::fmt::Display for SubtypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Refinement check failed: {}", self.reason)?;

        if let Some(ref ce) = self.counterexample {
            write!(f, "\n{}", ce)?;
        }

        Ok(())
    }
}

impl std::error::Error for SubtypeError {}

/// Check function signature refinements
///
/// Verifies that the function body's result type is a subtype of the declared return type.
pub fn check_function_signature(
    params: &[(String, RefinementType)],
    return_type: &RefinementType,
    body_type: &RefinementType,
    func_name: &str,
) -> SubtypeResult {
    let mut checker = SubtypeChecker::new();

    // Add parameters to environment
    for (name, ty) in params {
        checker.assume(name.clone(), ty.clone());
    }

    // Check return type
    checker.check_postcondition(func_name, body_type, return_type, Span::dummy());

    checker.verify()
}

/// Check function call refinements
///
/// Verifies that each argument type is a subtype of the corresponding parameter type.
pub fn check_function_call(
    func_name: &str,
    params: &[(String, RefinementType)],
    args: &[RefinementType],
    env: &[(String, RefinementType)],
) -> SubtypeResult {
    let mut checker = SubtypeChecker::new();

    // Add environment bindings
    for (name, ty) in env {
        checker.assume(name.clone(), ty.clone());
    }

    // Check each argument against parameter
    for ((param_name, param_ty), arg_ty) in params.iter().zip(args.iter()) {
        checker.check_precondition(func_name, param_name, arg_ty, param_ty, Span::dummy());
    }

    checker.verify()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trivial_subtype() {
        let mut checker = SubtypeChecker::new();

        let trivial = RefinementType::trivial(Type::I64);

        // trivial <: trivial should succeed
        assert!(checker.is_subtype(&trivial, &trivial, Span::dummy()));

        let result = checker.verify();
        assert!(result.is_valid());
    }

    #[test]
    fn test_positive_subtype_non_negative() {
        let mut checker = SubtypeChecker::new();

        let pos = RefinementType::positive(Type::I64);
        let non_neg = RefinementType::non_negative(Type::I64);

        // positive <: non_negative should succeed
        assert!(checker.is_subtype(&pos, &non_neg, Span::dummy()));

        // Without SMT, we can't verify this, but the constraint should be added
        assert_eq!(checker.pending_constraints(), 1);
    }

    #[test]
    fn test_base_type_mismatch() {
        let mut checker = SubtypeChecker::new();

        let int_pos = RefinementType::positive(Type::I64);
        let float_pos = RefinementType::positive(Type::F64);

        // Different base types should fail
        assert!(!checker.is_subtype(&int_pos, &float_pos, Span::dummy()));
    }

    #[test]
    fn test_check_value() {
        let mut checker = SubtypeChecker::new();

        let pos = RefinementType::positive(Type::I64);

        // Check that 42 satisfies positive
        checker.check_value(&Term::int(42), &pos, Span::dummy());

        // Simple checker should verify this
        let result = checker.verify();
        assert!(result.is_valid());
    }

    #[test]
    fn test_check_value_negative() {
        let mut checker = SubtypeChecker::new();

        let pos = RefinementType::positive(Type::I64);

        // Check that -5 does NOT satisfy positive
        checker.check_value(&Term::int(-5), &pos, Span::dummy());

        // Simple checker should catch this
        let result = checker.verify();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_with_assumptions() {
        let mut checker = SubtypeChecker::new();

        // Assume x > 0
        let pos = RefinementType::positive(Type::I64);
        checker.assume("x", pos);

        // Check that x >= 0 (should follow from x > 0)
        let non_neg = RefinementType::non_negative(Type::I64);
        checker.is_subtype(
            &RefinementType::refined(
                Type::I64,
                "v",
                Predicate::eq(Term::var("v"), Term::var("x")),
            ),
            &non_neg,
            Span::dummy(),
        );

        assert_eq!(checker.pending_constraints(), 1);
    }

    #[test]
    fn test_bounds_check() {
        let mut checker = SubtypeChecker::new();

        // Assume i >= 0 and i < 10
        let bounded = RefinementType::refined(
            Type::I64,
            "i",
            Predicate::and([
                Predicate::ge(Term::var("i"), Term::int(0)),
                Predicate::lt(Term::var("i"), Term::int(10)),
            ]),
        );
        checker.assume("i", bounded);

        // Check bounds against length 10
        checker.check_bounds(Term::var("i"), Term::int(10), Span::dummy());

        assert_eq!(checker.pending_constraints(), 1);
    }

    #[test]
    fn test_division_check() {
        let mut checker = SubtypeChecker::new();

        // Assume x != 0
        let non_zero =
            RefinementType::refined(Type::I64, "x", Predicate::ne(Term::var("x"), Term::int(0)));
        checker.assume("x", non_zero);

        // Check division safety
        checker.check_division(Term::var("x"), Span::dummy());

        assert_eq!(checker.pending_constraints(), 1);
    }

    #[test]
    fn test_function_signature_check() {
        let params = vec![("x".to_string(), RefinementType::positive(Type::I64))];
        let return_type = RefinementType::non_negative(Type::I64);
        let body_type = RefinementType::positive(Type::I64);

        let result = check_function_signature(&params, &return_type, &body_type, "test_fn");

        // positive <: non_negative should hold
        // Without SMT we can't verify, but structure should be correct
        assert_eq!(result.num_constraints(), 1);
    }

    #[test]
    fn test_subtype_result_counts() {
        let result = SubtypeResult {
            valid: true,
            constraints: vec![],
            results: vec![
                VerifyResult::Valid,
                VerifyResult::Valid,
                VerifyResult::Invalid {
                    constraint_idx: 2,
                    counterexample: None,
                },
                VerifyResult::Unknown {
                    constraint_idx: 3,
                    reason: "timeout".to_string(),
                },
            ],
            errors: vec![],
        };

        assert_eq!(result.num_valid(), 2);
        assert_eq!(result.num_invalid(), 1);
        assert_eq!(result.num_unknown(), 1);
    }

    #[test]
    fn test_path_conditions() {
        let mut checker = SubtypeChecker::new();

        // Assume x: Int
        checker.assume("x", RefinementType::trivial(Type::I64));

        // Enter branch where x > 0
        checker.enter_branch(Predicate::gt(Term::var("x"), Term::int(0)));

        // In this branch, x >= 0 should hold
        checker.check_safety(
            Predicate::ge(Term::var("x"), Term::int(0)),
            "x is non-negative in positive branch",
            Span::dummy(),
        );

        checker.exit_branch();

        assert_eq!(checker.pending_constraints(), 1);
    }
}
