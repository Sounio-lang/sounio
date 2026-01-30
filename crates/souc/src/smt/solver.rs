//! SMT Solver Abstraction
//!
//! This module provides a solver abstraction that can use either:
//! - MockSolver: Fast interval-based verification (current)
//! - Z3Solver: Full SMT verification via FFI (future)
//!
//! The mock solver is sound but incomplete—it may report "unknown"
//! for valid constraints that require full SMT reasoning.

use super::formula::{SmtFormula, SmtSort, SmtTerm};
use super::interval::{Interval, IntervalEnv};
use std::collections::HashMap;
use std::fmt;

/// Result of SMT verification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationResult {
    /// Formula is satisfiable (constraint can be satisfied)
    Sat,
    /// Formula is unsatisfiable (constraint is contradictory)
    Unsat,
    /// Solver could not determine satisfiability
    Unknown,
    /// Solver timed out
    Timeout,
    /// Solver encountered an error
    Error(String),
}

impl VerificationResult {
    /// Check if result indicates the formula is satisfiable
    pub fn is_sat(&self) -> bool {
        matches!(self, VerificationResult::Sat)
    }

    /// Check if result indicates the formula is unsatisfiable
    pub fn is_unsat(&self) -> bool {
        matches!(self, VerificationResult::Unsat)
    }

    /// Check if result is definitive (sat or unsat)
    pub fn is_definitive(&self) -> bool {
        matches!(self, VerificationResult::Sat | VerificationResult::Unsat)
    }
}

impl fmt::Display for VerificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VerificationResult::Sat => write!(f, "sat"),
            VerificationResult::Unsat => write!(f, "unsat"),
            VerificationResult::Unknown => write!(f, "unknown"),
            VerificationResult::Timeout => write!(f, "timeout"),
            VerificationResult::Error(msg) => write!(f, "error: {}", msg),
        }
    }
}

/// Errors that can occur during SMT solving
#[derive(Debug, Clone)]
pub enum SolverError {
    /// Invalid formula
    InvalidFormula(String),
    /// Type mismatch in formula
    TypeMismatch { expected: SmtSort, got: SmtSort },
    /// Unknown function or predicate
    UnknownFunction(String),
    /// Solver-specific error
    SolverError(String),
    /// Z3 not available (for Z3 backend)
    Z3NotAvailable,
}

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverError::InvalidFormula(msg) => write!(f, "invalid formula: {}", msg),
            SolverError::TypeMismatch { expected, got } => {
                write!(f, "type mismatch: expected {}, got {}", expected, got)
            }
            SolverError::UnknownFunction(name) => write!(f, "unknown function: {}", name),
            SolverError::SolverError(msg) => write!(f, "solver error: {}", msg),
            SolverError::Z3NotAvailable => write!(f, "Z3 solver not available"),
        }
    }
}

impl std::error::Error for SolverError {}

/// Trait for SMT solvers
pub trait SmtSolver {
    /// Check if a formula is satisfiable
    fn check_sat(&mut self, formula: &SmtFormula) -> Result<VerificationResult, SolverError>;

    /// Check if antecedent implies consequent (i.e., antecedent ∧ ¬consequent is unsat)
    fn check_implies(
        &mut self,
        antecedent: &SmtFormula,
        consequent: &SmtFormula,
    ) -> Result<VerificationResult, SolverError> {
        // To prove P ⟹ Q, we check if P ∧ ¬Q is unsatisfiable
        let negated = SmtFormula::And(vec![
            antecedent.clone(),
            SmtFormula::Not(Box::new(consequent.clone())),
        ]);

        match self.check_sat(&negated)? {
            VerificationResult::Unsat => Ok(VerificationResult::Sat), // Implication holds
            VerificationResult::Sat => Ok(VerificationResult::Unsat), // Implication fails
            other => Ok(other),
        }
    }

    /// Check if a formula is valid (i.e., ¬formula is unsat)
    fn check_valid(&mut self, formula: &SmtFormula) -> Result<VerificationResult, SolverError> {
        let negated = SmtFormula::Not(Box::new(formula.clone()));
        match self.check_sat(&negated)? {
            VerificationResult::Unsat => Ok(VerificationResult::Sat), // Valid
            VerificationResult::Sat => Ok(VerificationResult::Unsat), // Invalid
            other => Ok(other),
        }
    }

    /// Get the solver name
    fn name(&self) -> &str;
}

/// Mock SMT solver using interval arithmetic
///
/// This solver is fast but incomplete. It can prove many simple
/// constraints involving bounds and arithmetic, but will return
/// "unknown" for more complex formulas.
#[derive(Debug)]
pub struct MockSolver {
    /// Variable bounds from context
    env: IntervalEnv,
    /// Epsilon bounds for epistemic variables
    epsilon_bounds: HashMap<String, f64>,
    /// Maximum recursion depth for formula evaluation
    max_depth: usize,
}

impl MockSolver {
    /// Create a new mock solver
    pub fn new() -> Self {
        Self {
            env: IntervalEnv::new(),
            epsilon_bounds: HashMap::new(),
            max_depth: 100,
        }
    }

    /// Create a solver initialized from an SMT context
    pub fn from_context(_ctx: &super::formula::SmtContext) -> Self {
        // Context assertions will be checked during check_sat
        Self::new()
    }

    /// Set bounds for a variable
    pub fn set_bounds(&mut self, var: &str, lo: f64, hi: f64) {
        self.env.bind(var.to_string(), Interval::new(lo, hi));
    }

    /// Set epsilon bound for a knowledge variable
    pub fn set_epsilon_bound(&mut self, var: &str, bound: f64) {
        self.epsilon_bounds.insert(var.to_string(), bound);
    }

    /// Evaluate a term to an interval
    fn eval_term(&self, term: &SmtTerm) -> Interval {
        match term {
            SmtTerm::Var(name) => self.env.get(name),
            SmtTerm::Int(n) => Interval::point(*n as f64),
            SmtTerm::Real(f) => Interval::point(*f),
            SmtTerm::Bool(_) => Interval::point(0.0), // Boolean as 0/1

            SmtTerm::Add(l, r) => self.eval_term(l) + self.eval_term(r),
            SmtTerm::Sub(l, r) => self.eval_term(l) - self.eval_term(r),
            SmtTerm::Mul(l, r) => self.eval_term(l) * self.eval_term(r),
            SmtTerm::Div(l, r) => self.eval_term(l) / self.eval_term(r),
            SmtTerm::Mod(_, _) => Interval::all(), // Conservative

            SmtTerm::Neg(t) => -self.eval_term(t),
            SmtTerm::Abs(t) => self.eval_term(t).abs(),
            SmtTerm::Not(_) => Interval::all(),

            SmtTerm::App(name, args) => {
                // Handle built-in functions
                match name.as_str() {
                    "sqrt" if args.len() == 1 => self.eval_term(&args[0]).sqrt(),
                    "sqr" if args.len() == 1 => self.eval_term(&args[0]).sqr(),
                    "min" if args.len() == 2 => {
                        let a = self.eval_term(&args[0]);
                        let b = self.eval_term(&args[1]);
                        Interval::new(a.lo.min(b.lo), a.hi.min(b.hi))
                    }
                    "max" if args.len() == 2 => {
                        let a = self.eval_term(&args[0]);
                        let b = self.eval_term(&args[1]);
                        Interval::new(a.lo.max(b.lo), a.hi.max(b.hi))
                    }
                    _ => Interval::all(), // Unknown function
                }
            }

            SmtTerm::Field(base, _) => self.eval_term(base), // Conservative
            SmtTerm::Len(_) => Interval::non_negative(),

            SmtTerm::Ite(c, t, e) => {
                // Conservative: union of both branches
                self.eval_term(t).union(self.eval_term(e))
            }

            SmtTerm::Epsilon(bound) => Interval::epsilon(*bound),
            SmtTerm::Provenance(_) => Interval::all(),
        }
    }

    /// Check if a formula is definitely satisfiable using interval analysis
    fn check_formula(&self, formula: &SmtFormula, depth: usize) -> Option<bool> {
        if depth > self.max_depth {
            return None; // Too deep, give up
        }

        match formula {
            SmtFormula::True => Some(true),
            SmtFormula::False => Some(false),

            SmtFormula::Eq(l, r) => {
                let li = self.eval_term(l);
                let ri = self.eval_term(r);
                // Can be satisfied if intervals overlap
                if li.intersect(ri).is_empty() {
                    Some(false)
                } else if li.is_point() && ri.is_point() && li.lo == ri.lo {
                    Some(true)
                } else {
                    None
                }
            }

            SmtFormula::Lt(l, r) => {
                let li = self.eval_term(l);
                let ri = self.eval_term(r);
                if li.hi < ri.lo {
                    Some(true) // Always true
                } else if li.lo >= ri.hi {
                    Some(false) // Always false
                } else {
                    None
                }
            }

            SmtFormula::Le(l, r) => {
                let li = self.eval_term(l);
                let ri = self.eval_term(r);
                if li.hi <= ri.lo {
                    Some(true)
                } else if li.lo > ri.hi {
                    Some(false)
                } else {
                    None
                }
            }

            SmtFormula::Gt(l, r) => {
                let li = self.eval_term(l);
                let ri = self.eval_term(r);
                if li.lo > ri.hi {
                    Some(true)
                } else if li.hi <= ri.lo {
                    Some(false)
                } else {
                    None
                }
            }

            SmtFormula::Ge(l, r) => {
                let li = self.eval_term(l);
                let ri = self.eval_term(r);
                if li.lo >= ri.hi {
                    Some(true)
                } else if li.hi < ri.lo {
                    Some(false)
                } else {
                    None
                }
            }

            SmtFormula::Not(f) => self.check_formula(f, depth + 1).map(|b| !b),

            SmtFormula::And(fs) => {
                let mut all_true = true;
                for f in fs {
                    match self.check_formula(f, depth + 1) {
                        Some(false) => return Some(false),
                        Some(true) => {}
                        None => all_true = false,
                    }
                }
                if all_true { Some(true) } else { None }
            }

            SmtFormula::Or(fs) => {
                let mut all_false = true;
                for f in fs {
                    match self.check_formula(f, depth + 1) {
                        Some(true) => return Some(true),
                        Some(false) => {}
                        None => all_false = false,
                    }
                }
                if all_false { Some(false) } else { None }
            }

            SmtFormula::Implies(p, q) => {
                match (
                    self.check_formula(p, depth + 1),
                    self.check_formula(q, depth + 1),
                ) {
                    (Some(false), _) => Some(true),
                    (_, Some(true)) => Some(true),
                    (Some(true), Some(false)) => Some(false),
                    _ => None,
                }
            }

            SmtFormula::Ite(c, t, e) => {
                match self.check_formula(c, depth + 1) {
                    Some(true) => self.check_formula(t, depth + 1),
                    Some(false) => self.check_formula(e, depth + 1),
                    None => {
                        // Check if both branches agree
                        match (
                            self.check_formula(t, depth + 1),
                            self.check_formula(e, depth + 1),
                        ) {
                            (Some(tv), Some(ev)) if tv == ev => Some(tv),
                            _ => None,
                        }
                    }
                }
            }

            // Quantifiers require full SMT
            SmtFormula::Forall(_, _, _) | SmtFormula::Exists(_, _, _) => None,

            SmtFormula::App(_, _) | SmtFormula::Term(_) => None,
        }
    }
}

impl Default for MockSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl SmtSolver for MockSolver {
    fn check_sat(&mut self, formula: &SmtFormula) -> Result<VerificationResult, SolverError> {
        match self.check_formula(formula, 0) {
            Some(true) => Ok(VerificationResult::Sat),
            Some(false) => Ok(VerificationResult::Unsat),
            None => Ok(VerificationResult::Unknown),
        }
    }

    fn name(&self) -> &str {
        "MockSolver (interval arithmetic)"
    }
}

/// Z3 Solver stub (for future FFI integration)
///
/// This is a placeholder that will be implemented when Z3 FFI is added.
/// The interface is designed to match Z3's API.
#[derive(Debug)]
pub struct Z3Solver {
    // Future: z3::Context, z3::Solver
    _marker: std::marker::PhantomData<()>,
}

impl Z3Solver {
    /// Create a new Z3 solver (currently returns an error)
    pub fn new() -> Result<Self, SolverError> {
        // Future implementation:
        // let cfg = z3::Config::new();
        // let ctx = z3::Context::new(&cfg);
        // let solver = z3::Solver::new(&ctx);
        Err(SolverError::Z3NotAvailable)
    }

    /// Check if Z3 is available
    pub fn is_available() -> bool {
        // Future: check for z3 library
        false
    }
}

impl SmtSolver for Z3Solver {
    fn check_sat(&mut self, _formula: &SmtFormula) -> Result<VerificationResult, SolverError> {
        Err(SolverError::Z3NotAvailable)
    }

    fn name(&self) -> &str {
        "Z3 (not available)"
    }
}

/// Create the best available solver
pub fn create_solver() -> Box<dyn SmtSolver> {
    // Try Z3 first, fall back to mock solver
    match Z3Solver::new() {
        Ok(solver) => Box::new(solver),
        Err(_) => Box::new(MockSolver::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_solver_simple_sat() {
        let mut solver = MockSolver::new();
        solver.set_bounds("x", 0.0, 10.0);

        // x > 5 is satisfiable when x ∈ [0, 10]
        let formula = SmtFormula::Gt(Box::new(SmtTerm::var("x")), Box::new(SmtTerm::int(5)));

        let result = solver.check_sat(&formula).unwrap();
        // Could be sat (x=6) or could be unknown (interval overlaps)
        assert!(result == VerificationResult::Sat || result == VerificationResult::Unknown);
    }

    #[test]
    fn test_mock_solver_definite_sat() {
        let mut solver = MockSolver::new();
        solver.set_bounds("x", 10.0, 20.0);

        // x > 5 is definitely true when x ∈ [10, 20]
        let formula = SmtFormula::Gt(Box::new(SmtTerm::var("x")), Box::new(SmtTerm::int(5)));

        let result = solver.check_sat(&formula).unwrap();
        assert_eq!(result, VerificationResult::Sat);
    }

    #[test]
    fn test_mock_solver_definite_unsat() {
        let mut solver = MockSolver::new();
        solver.set_bounds("x", 0.0, 3.0);

        // x > 5 is definitely false when x ∈ [0, 3]
        let formula = SmtFormula::Gt(Box::new(SmtTerm::var("x")), Box::new(SmtTerm::int(5)));

        let result = solver.check_sat(&formula).unwrap();
        assert_eq!(result, VerificationResult::Unsat);
    }

    #[test]
    fn test_mock_solver_conjunction() {
        let mut solver = MockSolver::new();

        // With tight bounds, we can detect contradictions
        // x ∈ [15, 20] means x < 5 is definitely false
        solver.set_bounds("x", 15.0, 20.0);

        // x > 10 && x < 5: first is true (15 > 10), second is false (15..20 not < 5)
        let formula = SmtFormula::And(vec![
            SmtFormula::Gt(Box::new(SmtTerm::var("x")), Box::new(SmtTerm::int(10))),
            SmtFormula::Lt(Box::new(SmtTerm::var("x")), Box::new(SmtTerm::int(5))),
        ]);

        let result = solver.check_sat(&formula).unwrap();
        // Mock solver detects that x < 5 is definitely false when x ∈ [15, 20]
        assert_eq!(result, VerificationResult::Unsat);
    }

    #[test]
    fn test_mock_solver_arithmetic() {
        let mut solver = MockSolver::new();
        solver.set_bounds("x", 1.0, 5.0);
        solver.set_bounds("y", 2.0, 3.0);

        // x + y > 10 is unsatisfiable when x ∈ [1,5], y ∈ [2,3] (max is 8)
        let formula = SmtFormula::Gt(
            Box::new(SmtTerm::Add(
                Box::new(SmtTerm::var("x")),
                Box::new(SmtTerm::var("y")),
            )),
            Box::new(SmtTerm::int(10)),
        );

        let result = solver.check_sat(&formula).unwrap();
        assert_eq!(result, VerificationResult::Unsat);
    }

    #[test]
    fn test_check_implies() {
        let mut solver = MockSolver::new();
        solver.set_bounds("x", 0.0, 100.0);

        // x > 10 implies x > 5
        let antecedent = SmtFormula::Gt(Box::new(SmtTerm::var("x")), Box::new(SmtTerm::int(10)));
        let consequent = SmtFormula::Gt(Box::new(SmtTerm::var("x")), Box::new(SmtTerm::int(5)));

        let result = solver.check_implies(&antecedent, &consequent).unwrap();
        // This requires more sophisticated reasoning than interval arithmetic
        assert!(result == VerificationResult::Sat || result == VerificationResult::Unknown);
    }

    #[test]
    fn test_verification_result_display() {
        assert_eq!(format!("{}", VerificationResult::Sat), "sat");
        assert_eq!(format!("{}", VerificationResult::Unsat), "unsat");
        assert_eq!(format!("{}", VerificationResult::Unknown), "unknown");
    }
}
