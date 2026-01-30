//! Z3 SMT solver integration
//!
//! This module translates refinement constraints to Z3 queries and
//! interprets the results. When Z3 is not available (feature disabled),
//! it provides a fallback that reports constraints as unknown.
//!
//! # Theory
//!
//! To check if a refinement constraint holds, we:
//! 1. Translate the constraint to Z3's AST
//! 2. Assert the negation of the goal (we want to prove unsatisfiability)
//! 3. If UNSAT: the constraint is valid
//! 4. If SAT: we have a counterexample showing the constraint can fail
//! 5. If UNKNOWN: solver timeout or limitations

use super::constraint::*;
use super::predicate::*;

#[cfg(feature = "smt")]
use crate::types::Type;
#[cfg(feature = "smt")]
use std::collections::HashMap;

/// Result of constraint verification
#[derive(Debug, Clone)]
pub enum VerifyResult {
    /// The constraint is provably valid
    Valid,

    /// The constraint is invalid (can fail)
    Invalid {
        /// Index of the failed constraint
        constraint_idx: usize,
        /// Counterexample showing values that violate the constraint
        counterexample: Option<Counterexample>,
    },

    /// Verification inconclusive (solver timeout or limitations)
    Unknown {
        /// Index of the constraint
        constraint_idx: usize,
        /// Reason for unknown result
        reason: String,
    },
}

impl VerifyResult {
    /// Check if the result is valid
    pub fn is_valid(&self) -> bool {
        matches!(self, VerifyResult::Valid)
    }

    /// Check if the result is invalid
    pub fn is_invalid(&self) -> bool {
        matches!(self, VerifyResult::Invalid { .. })
    }

    /// Check if the result is unknown
    pub fn is_unknown(&self) -> bool {
        matches!(self, VerifyResult::Unknown { .. })
    }
}

/// A counterexample showing why a constraint failed
#[derive(Debug, Clone)]
pub struct Counterexample {
    /// Variable assignments that violate the constraint
    pub bindings: Vec<(String, CounterexampleValue)>,
}

impl Counterexample {
    /// Create a new counterexample
    pub fn new(bindings: Vec<(String, CounterexampleValue)>) -> Self {
        Self { bindings }
    }

    /// Get the value of a variable in the counterexample
    pub fn get(&self, name: &str) -> Option<&CounterexampleValue> {
        self.bindings
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v)
    }
}

impl std::fmt::Display for Counterexample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Counterexample:")?;
        for (name, value) in &self.bindings {
            writeln!(f, "  {} = {}", name, value)?;
        }
        Ok(())
    }
}

/// A value in a counterexample
#[derive(Debug, Clone)]
pub enum CounterexampleValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Unknown(String),
}

impl std::fmt::Display for CounterexampleValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CounterexampleValue::Int(n) => write!(f, "{}", n),
            CounterexampleValue::Float(n) => write!(f, "{:.6}", n),
            CounterexampleValue::Bool(b) => write!(f, "{}", b),
            CounterexampleValue::Unknown(s) => write!(f, "{}", s),
        }
    }
}

/// Z3 SMT solver wrapper
///
/// When compiled with the `smt` feature, this uses the actual Z3 solver.
/// Otherwise, it provides a stub that returns Unknown for all constraints.
#[cfg(feature = "smt")]
pub struct Z3Solver<'ctx> {
    ctx: &'ctx z3::Context,
    solver: z3::Solver<'ctx>,

    /// Variable declarations
    vars: HashMap<String, z3::ast::Dynamic<'ctx>>,

    /// Timeout in milliseconds
    timeout_ms: u32,
}

#[cfg(feature = "smt")]
impl<'ctx> Z3Solver<'ctx> {
    /// Create a new Z3 solver
    pub fn new(ctx: &'ctx z3::Context) -> Self {
        let solver = z3::Solver::new(ctx);

        Self {
            ctx,
            solver,
            vars: HashMap::new(),
            timeout_ms: 5000, // 5 second default timeout
        }
    }

    /// Set the solver timeout
    pub fn set_timeout(&mut self, ms: u32) {
        self.timeout_ms = ms;
        let params = z3::Params::new(self.ctx);
        params.set_u32("timeout", ms);
        self.solver.set_params(&params);
    }

    /// Verify a list of constraints
    pub fn verify(&mut self, constraints: &[Constraint]) -> Vec<VerifyResult> {
        constraints
            .iter()
            .enumerate()
            .map(|(i, c)| self.verify_single(i, c))
            .collect()
    }

    /// Verify a single constraint
    fn verify_single(&mut self, index: usize, constraint: &Constraint) -> VerifyResult {
        use z3::SatResult;

        self.solver.reset();
        self.vars.clear();

        // Add environment bindings as assumptions
        for binding in &constraint.env {
            self.declare_var(&binding.name, &binding.ty.base);

            // Add refinement predicate as assumption
            let pred = binding
                .ty
                .predicate
                .substitute(&binding.ty.var, &Term::var(&binding.name));

            if let Some(z3_pred) = self.translate_predicate(&pred) {
                self.solver.assert(&z3_pred);
            }
        }

        // Negate the goal (we want to prove it's unsatisfiable when negated)
        let negated_goal = Predicate::not(constraint.goal.clone());

        if let Some(z3_goal) = self.translate_predicate(&negated_goal) {
            self.solver.assert(&z3_goal);
        } else {
            return VerifyResult::Unknown {
                constraint_idx: index,
                reason: "Failed to translate goal to Z3".to_string(),
            };
        }

        // Check satisfiability
        match self.solver.check() {
            SatResult::Unsat => {
                // Negated goal is unsatisfiable, so original goal is valid
                VerifyResult::Valid
            }
            SatResult::Sat => {
                // Found a counterexample
                let counterexample = self.extract_counterexample();
                VerifyResult::Invalid {
                    constraint_idx: index,
                    counterexample,
                }
            }
            SatResult::Unknown => VerifyResult::Unknown {
                constraint_idx: index,
                reason: self
                    .solver
                    .get_reason_unknown()
                    .unwrap_or_else(|| "Unknown".to_string()),
            },
        }
    }

    /// Extract a counterexample from a SAT result
    fn extract_counterexample(&self) -> Option<Counterexample> {
        let model = self.solver.get_model()?;

        let bindings: Vec<_> = self
            .vars
            .iter()
            .filter_map(|(name, var)| {
                model.eval(var, true).map(|val| {
                    let value = self.extract_value(&val);
                    (name.clone(), value)
                })
            })
            .collect();

        Some(Counterexample::new(bindings))
    }

    /// Extract a value from a Z3 AST
    fn extract_value(&self, ast: &z3::ast::Dynamic<'ctx>) -> CounterexampleValue {
        if let Some(int_val) = ast.as_int() {
            if let Some(n) = int_val.as_i64() {
                return CounterexampleValue::Int(n);
            }
        }

        if let Some(real_val) = ast.as_real() {
            if let Some((num, den)) = real_val.as_real() {
                let f = num as f64 / den as f64;
                return CounterexampleValue::Float(f);
            }
        }

        if let Some(bool_val) = ast.as_bool() {
            if let Some(b) = bool_val.as_bool() {
                return CounterexampleValue::Bool(b);
            }
        }

        CounterexampleValue::Unknown(ast.to_string())
    }

    /// Declare a variable in Z3
    fn declare_var(&mut self, name: &str, ty: &Type) {
        use z3::ast::{Bool, Dynamic, Int, Real};

        let var: Dynamic = match ty {
            Type::I8
            | Type::I16
            | Type::I32
            | Type::I64
            | Type::I128
            | Type::Isize
            | Type::U8
            | Type::U16
            | Type::U32
            | Type::U64
            | Type::U128
            | Type::Usize => Int::new_const(self.ctx, name.to_string()).into(),
            Type::F32 | Type::F64 => Real::new_const(self.ctx, name.to_string()).into(),
            Type::Bool => Bool::new_const(self.ctx, name.to_string()).into(),
            _ => {
                // Default to integer for unknown types
                Int::new_const(self.ctx, name.to_string()).into()
            }
        };

        self.vars.insert(name.to_string(), var);
    }

    /// Translate a predicate to Z3
    fn translate_predicate(&mut self, pred: &Predicate) -> Option<z3::ast::Bool<'ctx>> {
        use z3::ast::Bool;

        match pred {
            Predicate::True => Some(Bool::from_bool(self.ctx, true)),
            Predicate::False => Some(Bool::from_bool(self.ctx, false)),

            Predicate::Atom(atom) => self.translate_atom(atom),

            Predicate::Not(p) => self.translate_predicate(p).map(|p| p.not()),

            Predicate::And(ps) => {
                let z3_preds: Option<Vec<_>> =
                    ps.iter().map(|p| self.translate_predicate(p)).collect();

                z3_preds.map(|preds| {
                    let refs: Vec<_> = preds.iter().collect();
                    Bool::and(self.ctx, &refs)
                })
            }

            Predicate::Or(ps) => {
                let z3_preds: Option<Vec<_>> =
                    ps.iter().map(|p| self.translate_predicate(p)).collect();

                z3_preds.map(|preds| {
                    let refs: Vec<_> = preds.iter().collect();
                    Bool::or(self.ctx, &refs)
                })
            }

            Predicate::Implies(p, q) => {
                let p_z3 = self.translate_predicate(p)?;
                let q_z3 = self.translate_predicate(q)?;
                Some(p_z3.implies(&q_z3))
            }

            Predicate::Ite(c, t, e) => {
                let c_z3 = self.translate_predicate(c)?;
                let t_z3 = self.translate_predicate(t)?;
                let e_z3 = self.translate_predicate(e)?;
                Some(c_z3.ite(&t_z3, &e_z3))
            }

            Predicate::Forall(var, ty, body) => {
                // Create a bound variable
                self.declare_var(var, ty);
                let var_ast = self.vars.get(var)?.clone();
                let body_z3 = self.translate_predicate(body)?;

                // Create forall quantifier
                let bound_vars: Vec<_> = vec![&var_ast];
                Some(z3::ast::forall_const(self.ctx, &bound_vars, &[], &body_z3))
            }

            Predicate::Exists(var, ty, body) => {
                // Create a bound variable
                self.declare_var(var, ty);
                let var_ast = self.vars.get(var)?.clone();
                let body_z3 = self.translate_predicate(body)?;

                // Create exists quantifier
                let bound_vars: Vec<_> = vec![&var_ast];
                Some(z3::ast::exists_const(self.ctx, &bound_vars, &[], &body_z3))
            }

            Predicate::App(name, args) => {
                // Handle special functions
                match name.as_str() {
                    "len" if args.len() == 1 => {
                        // len is modeled as an uninterpreted function
                        let arg = self.translate_term(&args[0])?;
                        // For simplicity, treat len(x) as a fresh integer variable
                        let len_name = format!("len_{}", args[0]);
                        if !self.vars.contains_key(&len_name) {
                            self.declare_var(&len_name, &Type::I64);
                        }
                        // len(x) >= 0 is always true
                        let len_var = self.vars.get(&len_name)?.as_int()?;
                        let zero = z3::ast::Int::from_i64(self.ctx, 0);
                        Some(len_var.ge(&zero))
                    }
                    _ => None, // Unknown predicate application
                }
            }
        }
    }

    /// Translate an atomic predicate
    fn translate_atom(&mut self, atom: &Atom) -> Option<z3::ast::Bool<'ctx>> {
        let lhs = self.translate_term(&atom.lhs)?;
        let rhs = self.translate_term(&atom.rhs)?;

        // Try integer comparison first
        if let (Some(l_int), Some(r_int)) = (lhs.as_int(), rhs.as_int()) {
            return Some(match atom.op {
                CompareOp::Eq => l_int._eq(&r_int),
                CompareOp::Ne => l_int._eq(&r_int).not(),
                CompareOp::Lt => l_int.lt(&r_int),
                CompareOp::Le => l_int.le(&r_int),
                CompareOp::Gt => l_int.gt(&r_int),
                CompareOp::Ge => l_int.ge(&r_int),
            });
        }

        // Try real comparison
        if let (Some(l_real), Some(r_real)) = (lhs.as_real(), rhs.as_real()) {
            return Some(match atom.op {
                CompareOp::Eq => l_real._eq(&r_real),
                CompareOp::Ne => l_real._eq(&r_real).not(),
                CompareOp::Lt => l_real.lt(&r_real),
                CompareOp::Le => l_real.le(&r_real),
                CompareOp::Gt => l_real.gt(&r_real),
                CompareOp::Ge => l_real.ge(&r_real),
            });
        }

        // Mixed: promote int to real
        if let (Some(l_int), Some(r_real)) = (lhs.as_int(), rhs.as_real()) {
            let l_real = z3::ast::Real::from_int(&l_int);
            return Some(match atom.op {
                CompareOp::Eq => l_real._eq(&r_real),
                CompareOp::Ne => l_real._eq(&r_real).not(),
                CompareOp::Lt => l_real.lt(&r_real),
                CompareOp::Le => l_real.le(&r_real),
                CompareOp::Gt => l_real.gt(&r_real),
                CompareOp::Ge => l_real.ge(&r_real),
            });
        }

        if let (Some(l_real), Some(r_int)) = (lhs.as_real(), rhs.as_int()) {
            let r_real = z3::ast::Real::from_int(&r_int);
            return Some(match atom.op {
                CompareOp::Eq => l_real._eq(&r_real),
                CompareOp::Ne => l_real._eq(&r_real).not(),
                CompareOp::Lt => l_real.lt(&r_real),
                CompareOp::Le => l_real.le(&r_real),
                CompareOp::Gt => l_real.gt(&r_real),
                CompareOp::Ge => l_real.ge(&r_real),
            });
        }

        None
    }

    /// Translate a term to Z3
    fn translate_term(&mut self, term: &Term) -> Option<z3::ast::Dynamic<'ctx>> {
        use z3::ast::{Bool, Dynamic, Int, Real};

        match term {
            Term::Var(name) => {
                self.vars.get(name).cloned().or_else(|| {
                    // Create new integer variable if not exists
                    let var = Int::new_const(self.ctx, name.to_string());
                    self.vars.insert(name.clone(), var.clone().into());
                    Some(var.into())
                })
            }

            Term::Int(n) => Some(Int::from_i64(self.ctx, *n).into()),

            Term::Float(n) => {
                // Approximate as rational
                let (num, den) = float_to_rational(*n);
                Some(Real::from_real(self.ctx, num as i32, den as i32).into())
            }

            Term::Bool(b) => Some(Bool::from_bool(self.ctx, *b).into()),

            Term::BinOp(op, l, r) => {
                let l_z3 = self.translate_term(l)?;
                let r_z3 = self.translate_term(r)?;

                // Integer operations
                if let (Some(l_int), Some(r_int)) = (l_z3.as_int(), r_z3.as_int()) {
                    return Some(match op {
                        BinOp::Add => (l_int + r_int).into(),
                        BinOp::Sub => (l_int - r_int).into(),
                        BinOp::Mul => (l_int * r_int).into(),
                        BinOp::Div => l_int.div(&r_int).into(),
                        BinOp::Mod => l_int.modulo(&r_int).into(),
                    });
                }

                // Real operations
                if let (Some(l_real), Some(r_real)) = (l_z3.as_real(), r_z3.as_real()) {
                    return Some(match op {
                        BinOp::Add => (l_real + r_real).into(),
                        BinOp::Sub => (l_real - r_real).into(),
                        BinOp::Mul => (l_real * r_real).into(),
                        BinOp::Div => (l_real / r_real).into(),
                        BinOp::Mod => return None, // No modulo for reals
                    });
                }

                // Mixed: promote int to real
                if let (Some(l_int), Some(r_real)) = (l_z3.as_int(), r_z3.as_real()) {
                    let l_real = Real::from_int(&l_int);
                    return Some(match op {
                        BinOp::Add => (l_real + r_real).into(),
                        BinOp::Sub => (l_real - r_real).into(),
                        BinOp::Mul => (l_real * r_real).into(),
                        BinOp::Div => (l_real / r_real).into(),
                        BinOp::Mod => return None,
                    });
                }

                if let (Some(l_real), Some(r_int)) = (l_z3.as_real(), r_z3.as_int()) {
                    let r_real = Real::from_int(&r_int);
                    return Some(match op {
                        BinOp::Add => (l_real + r_real).into(),
                        BinOp::Sub => (l_real - r_real).into(),
                        BinOp::Mul => (l_real * r_real).into(),
                        BinOp::Div => (l_real / r_real).into(),
                        BinOp::Mod => return None,
                    });
                }

                None
            }

            Term::UnOp(op, t) => {
                let t_z3 = self.translate_term(t)?;

                match op {
                    UnOp::Neg => {
                        if let Some(i) = t_z3.as_int() {
                            Some((-i).into())
                        } else if let Some(r) = t_z3.as_real() {
                            Some((-r).into())
                        } else {
                            None
                        }
                    }
                    UnOp::Not => t_z3.as_bool().map(|b| b.not().into()),
                    UnOp::Abs => {
                        // abs(x) = if x >= 0 then x else -x
                        if let Some(i) = t_z3.as_int() {
                            let zero = Int::from_i64(self.ctx, 0);
                            let neg = -i.clone();
                            Some(i.ge(&zero).ite(&i, &neg).into())
                        } else if let Some(r) = t_z3.as_real() {
                            let zero = Real::from_real(self.ctx, 0, 1);
                            let neg = -r.clone();
                            Some(r.ge(&zero).ite(&r, &neg).into())
                        } else {
                            None
                        }
                    }
                }
            }

            Term::Len(t) => {
                // Model len as an uninterpreted function returning non-negative int
                let t_z3 = self.translate_term(t)?;
                let len_name = format!("len_{}", t);

                if !self.vars.contains_key(&len_name) {
                    let len_var = Int::new_const(self.ctx, len_name.clone());
                    self.vars.insert(len_name.clone(), len_var.into());
                }

                self.vars.get(&len_name).cloned()
            }

            Term::Field(base, field) => {
                // Model field access as an uninterpreted function
                let base_z3 = self.translate_term(base)?;
                let field_name = format!("{}_{}", base, field);

                if !self.vars.contains_key(&field_name) {
                    let field_var = Int::new_const(self.ctx, field_name.clone());
                    self.vars.insert(field_name.clone(), field_var.into());
                }

                self.vars.get(&field_name).cloned()
            }

            Term::Ite(c, t, e) => {
                let c_z3 = self.translate_term(c)?;
                let t_z3 = self.translate_term(t)?;
                let e_z3 = self.translate_term(e)?;

                let c_bool = c_z3.as_bool()?;

                // Both branches should be same type
                if let (Some(t_int), Some(e_int)) = (t_z3.as_int(), e_z3.as_int()) {
                    Some(c_bool.ite(&t_int, &e_int).into())
                } else if let (Some(t_real), Some(e_real)) = (t_z3.as_real(), e_z3.as_real()) {
                    Some(c_bool.ite(&t_real, &e_real).into())
                } else if let (Some(t_bool), Some(e_bool)) = (t_z3.as_bool(), e_z3.as_bool()) {
                    Some(c_bool.ite(&t_bool, &e_bool).into())
                } else {
                    None
                }
            }

            Term::App(name, args) => {
                // Model as an uninterpreted function
                let app_name = format!("{}_{}", name, args.len());

                if !self.vars.contains_key(&app_name) {
                    let app_var = Int::new_const(self.ctx, app_name.clone());
                    self.vars.insert(app_name.clone(), app_var.into());
                }

                self.vars.get(&app_name).cloned()
            }
        }
    }
}

/// Convert float to approximate rational
fn float_to_rational(f: f64) -> (i64, i64) {
    const PRECISION: i64 = 1_000_000;
    let num = (f * PRECISION as f64).round() as i64;
    (num, PRECISION)
}

/// Stub solver when Z3 is not available
#[cfg(not(feature = "smt"))]
pub struct Z3Solver {
    timeout_ms: u32,
}

#[cfg(not(feature = "smt"))]
impl Z3Solver {
    /// Create a new stub solver
    pub fn new() -> Self {
        Self { timeout_ms: 5000 }
    }

    /// Set the solver timeout (no-op in stub)
    pub fn set_timeout(&mut self, ms: u32) {
        self.timeout_ms = ms;
    }

    /// Verify constraints (always returns Unknown in stub)
    pub fn verify(&mut self, constraints: &[Constraint]) -> Vec<VerifyResult> {
        constraints
            .iter()
            .enumerate()
            .map(|(i, _)| VerifyResult::Unknown {
                constraint_idx: i,
                reason: "Z3 not available (compile with --features smt)".to_string(),
            })
            .collect()
    }
}

#[cfg(not(feature = "smt"))]
impl Default for Z3Solver {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple constraint checker without SMT solver
///
/// Performs basic syntactic checks for simple constraints.
pub struct SimpleChecker;

impl SimpleChecker {
    /// Check a predicate using simple rules
    pub fn check(pred: &Predicate) -> Option<bool> {
        match pred {
            Predicate::True => Some(true),
            Predicate::False => Some(false),

            Predicate::Not(p) => Self::check(p).map(|b| !b),

            Predicate::And(ps) => {
                let results: Vec<_> = ps.iter().filter_map(Self::check).collect();
                if results.len() == ps.len() {
                    Some(results.iter().all(|&b| b))
                } else {
                    None
                }
            }

            Predicate::Or(ps) => {
                let results: Vec<_> = ps.iter().filter_map(Self::check).collect();
                if results.len() == ps.len() {
                    Some(results.iter().any(|&b| b))
                } else {
                    None
                }
            }

            Predicate::Implies(p, q) => match (Self::check(p), Self::check(q)) {
                (Some(false), _) => Some(true),
                (_, Some(true)) => Some(true),
                (Some(true), Some(false)) => Some(false),
                _ => None,
            },

            Predicate::Atom(atom) => Self::check_atom(atom),

            _ => None, // Can't handle quantifiers or applications simply
        }
    }

    /// Check an atomic predicate with constants
    fn check_atom(atom: &Atom) -> Option<bool> {
        match (&atom.lhs, &atom.rhs) {
            (Term::Int(l), Term::Int(r)) => Some(match atom.op {
                CompareOp::Eq => l == r,
                CompareOp::Ne => l != r,
                CompareOp::Lt => l < r,
                CompareOp::Le => l <= r,
                CompareOp::Gt => l > r,
                CompareOp::Ge => l >= r,
            }),

            (Term::Float(l), Term::Float(r)) => Some(match atom.op {
                CompareOp::Eq => (l - r).abs() < f64::EPSILON,
                CompareOp::Ne => (l - r).abs() >= f64::EPSILON,
                CompareOp::Lt => l < r,
                CompareOp::Le => l <= r,
                CompareOp::Gt => l > r,
                CompareOp::Ge => l >= r,
            }),

            (Term::Bool(l), Term::Bool(r)) => Some(match atom.op {
                CompareOp::Eq => l == r,
                CompareOp::Ne => l != r,
                _ => return None, // Can't compare booleans with <, >, etc.
            }),

            _ => None, // Can't check predicates with variables
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_checker_true() {
        assert_eq!(SimpleChecker::check(&Predicate::True), Some(true));
    }

    #[test]
    fn test_simple_checker_false() {
        assert_eq!(SimpleChecker::check(&Predicate::False), Some(false));
    }

    #[test]
    fn test_simple_checker_int_comparison() {
        let pred = Predicate::lt(Term::int(5), Term::int(10));
        assert_eq!(SimpleChecker::check(&pred), Some(true));

        let pred = Predicate::gt(Term::int(5), Term::int(10));
        assert_eq!(SimpleChecker::check(&pred), Some(false));
    }

    #[test]
    fn test_simple_checker_and() {
        let pred = Predicate::and([
            Predicate::lt(Term::int(5), Term::int(10)),
            Predicate::gt(Term::int(5), Term::int(0)),
        ]);
        assert_eq!(SimpleChecker::check(&pred), Some(true));

        let pred = Predicate::and([
            Predicate::lt(Term::int(5), Term::int(10)),
            Predicate::lt(Term::int(5), Term::int(0)),
        ]);
        assert_eq!(SimpleChecker::check(&pred), Some(false));
    }

    #[test]
    fn test_simple_checker_implies() {
        // false => anything = true
        let pred = Predicate::implies(Predicate::False, Predicate::False);
        assert_eq!(SimpleChecker::check(&pred), Some(true));

        // true => true = true
        let pred = Predicate::implies(Predicate::True, Predicate::True);
        assert_eq!(SimpleChecker::check(&pred), Some(true));

        // true => false = false
        let pred = Predicate::implies(Predicate::True, Predicate::False);
        assert_eq!(SimpleChecker::check(&pred), Some(false));
    }

    #[test]
    fn test_simple_checker_with_variables() {
        // Can't determine result with variables
        let pred = Predicate::gt(Term::var("x"), Term::int(0));
        assert_eq!(SimpleChecker::check(&pred), None);
    }

    #[test]
    fn test_verify_result_methods() {
        let valid = VerifyResult::Valid;
        assert!(valid.is_valid());
        assert!(!valid.is_invalid());
        assert!(!valid.is_unknown());

        let invalid = VerifyResult::Invalid {
            constraint_idx: 0,
            counterexample: None,
        };
        assert!(!invalid.is_valid());
        assert!(invalid.is_invalid());

        let unknown = VerifyResult::Unknown {
            constraint_idx: 0,
            reason: "test".to_string(),
        };
        assert!(unknown.is_unknown());
    }

    #[test]
    fn test_counterexample_display() {
        let ce = Counterexample::new(vec![
            ("x".to_string(), CounterexampleValue::Int(42)),
            ("y".to_string(), CounterexampleValue::Float(3.14)),
        ]);

        let display = format!("{}", ce);
        assert!(display.contains("x = 42"));
        assert!(display.contains("y = 3.14"));
    }

    #[test]
    fn test_float_to_rational() {
        let (num, den) = float_to_rational(0.5);
        assert_eq!(num as f64 / den as f64, 0.5);

        let (num, den) = float_to_rational(3.14159);
        let ratio = num as f64 / den as f64;
        assert!((ratio - 3.14159).abs() < 0.00001);
    }
}
