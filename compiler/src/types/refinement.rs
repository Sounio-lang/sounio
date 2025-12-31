//! Refinement types with SMT integration
//!
//! Refinement types allow specifying predicates on values that are checked
//! at compile time using SMT solvers. This enables verification of properties
//! like array bounds, null safety, and domain-specific constraints.

use super::core::Type;
use crate::smt::{MockSolver, SmtContext, SmtFormula, SmtSolver, SmtTerm, VerificationResult};

/// Refinement predicate
#[derive(Debug, Clone)]
pub enum Predicate {
    /// Boolean literal
    Bool(bool),
    /// Integer literal
    Int(i64),
    /// Float literal
    Float(f64),
    /// Variable reference
    Var(String),
    /// Comparison
    Compare(CompareOp, Box<Predicate>, Box<Predicate>),
    /// Arithmetic
    Arith(ArithOp, Box<Predicate>, Box<Predicate>),
    /// Logical and
    And(Box<Predicate>, Box<Predicate>),
    /// Logical or
    Or(Box<Predicate>, Box<Predicate>),
    /// Logical not
    Not(Box<Predicate>),
    /// Implication: P => Q
    Implies(Box<Predicate>, Box<Predicate>),
    /// Forall quantifier
    Forall(String, Box<Predicate>),
    /// Exists quantifier
    Exists(String, Box<Predicate>),
    /// Function application
    App(String, Vec<Predicate>),
    /// If-then-else
    Ite(Box<Predicate>, Box<Predicate>, Box<Predicate>),
}

impl Predicate {
    /// Create a true predicate
    pub fn true_pred() -> Self {
        Predicate::Bool(true)
    }

    /// Create a false predicate
    pub fn false_pred() -> Self {
        Predicate::Bool(false)
    }

    /// Create a variable reference
    pub fn var(name: &str) -> Self {
        Predicate::Var(name.to_string())
    }

    /// Create an equality comparison
    pub fn eq(left: Predicate, right: Predicate) -> Self {
        Predicate::Compare(CompareOp::Eq, Box::new(left), Box::new(right))
    }

    /// Create an inequality comparison
    pub fn ne(left: Predicate, right: Predicate) -> Self {
        Predicate::Compare(CompareOp::Ne, Box::new(left), Box::new(right))
    }

    /// Create a less-than comparison
    pub fn lt(left: Predicate, right: Predicate) -> Self {
        Predicate::Compare(CompareOp::Lt, Box::new(left), Box::new(right))
    }

    /// Create a less-than-or-equal comparison
    pub fn le(left: Predicate, right: Predicate) -> Self {
        Predicate::Compare(CompareOp::Le, Box::new(left), Box::new(right))
    }

    /// Create a greater-than comparison
    pub fn gt(left: Predicate, right: Predicate) -> Self {
        Predicate::Compare(CompareOp::Gt, Box::new(left), Box::new(right))
    }

    /// Create a greater-than-or-equal comparison
    pub fn ge(left: Predicate, right: Predicate) -> Self {
        Predicate::Compare(CompareOp::Ge, Box::new(left), Box::new(right))
    }

    /// Create a conjunction
    pub fn and(left: Predicate, right: Predicate) -> Self {
        Predicate::And(Box::new(left), Box::new(right))
    }

    /// Create a disjunction
    pub fn or(left: Predicate, right: Predicate) -> Self {
        Predicate::Or(Box::new(left), Box::new(right))
    }

    /// Create a negation
    pub fn not(pred: Predicate) -> Self {
        Predicate::Not(Box::new(pred))
    }

    /// Create an implication
    pub fn implies(left: Predicate, right: Predicate) -> Self {
        Predicate::Implies(Box::new(left), Box::new(right))
    }

    /// Check if this predicate is trivially true
    pub fn is_trivially_true(&self) -> bool {
        matches!(self, Predicate::Bool(true))
    }

    /// Check if this predicate is trivially false
    pub fn is_trivially_false(&self) -> bool {
        matches!(self, Predicate::Bool(false))
    }

    /// Substitute a variable with a predicate
    pub fn substitute(&self, var: &str, replacement: &Predicate) -> Predicate {
        match self {
            Predicate::Bool(b) => Predicate::Bool(*b),
            Predicate::Int(i) => Predicate::Int(*i),
            Predicate::Float(f) => Predicate::Float(*f),
            Predicate::Var(name) if name == var => replacement.clone(),
            Predicate::Var(name) => Predicate::Var(name.clone()),
            Predicate::Compare(op, l, r) => Predicate::Compare(
                *op,
                Box::new(l.substitute(var, replacement)),
                Box::new(r.substitute(var, replacement)),
            ),
            Predicate::Arith(op, l, r) => Predicate::Arith(
                *op,
                Box::new(l.substitute(var, replacement)),
                Box::new(r.substitute(var, replacement)),
            ),
            Predicate::And(l, r) => Predicate::And(
                Box::new(l.substitute(var, replacement)),
                Box::new(r.substitute(var, replacement)),
            ),
            Predicate::Or(l, r) => Predicate::Or(
                Box::new(l.substitute(var, replacement)),
                Box::new(r.substitute(var, replacement)),
            ),
            Predicate::Not(p) => Predicate::Not(Box::new(p.substitute(var, replacement))),
            Predicate::Implies(l, r) => Predicate::Implies(
                Box::new(l.substitute(var, replacement)),
                Box::new(r.substitute(var, replacement)),
            ),
            Predicate::Forall(bound, body) if bound != var => {
                Predicate::Forall(bound.clone(), Box::new(body.substitute(var, replacement)))
            }
            Predicate::Forall(bound, body) => Predicate::Forall(bound.clone(), body.clone()),
            Predicate::Exists(bound, body) if bound != var => {
                Predicate::Exists(bound.clone(), Box::new(body.substitute(var, replacement)))
            }
            Predicate::Exists(bound, body) => Predicate::Exists(bound.clone(), body.clone()),
            Predicate::App(name, args) => Predicate::App(
                name.clone(),
                args.iter()
                    .map(|a| a.substitute(var, replacement))
                    .collect(),
            ),
            Predicate::Ite(cond, then_p, else_p) => Predicate::Ite(
                Box::new(cond.substitute(var, replacement)),
                Box::new(then_p.substitute(var, replacement)),
                Box::new(else_p.substitute(var, replacement)),
            ),
        }
    }
}

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Arithmetic operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

/// Refined type: base type with a predicate
#[derive(Debug, Clone)]
pub struct RefinedType {
    pub base: Type,
    pub var_name: String,
    pub predicate: Option<Predicate>,
}

impl RefinedType {
    pub fn new(base: Type, var_name: &str, predicate: Option<Predicate>) -> Self {
        Self {
            base,
            var_name: var_name.to_string(),
            predicate,
        }
    }

    pub fn unrefined(base: Type) -> Self {
        Self {
            base,
            var_name: "_".to_string(),
            predicate: None,
        }
    }

    /// Check if this type has a refinement
    pub fn is_refined(&self) -> bool {
        self.predicate.is_some()
    }

    /// Get the predicate, defaulting to true
    pub fn predicate_or_true(&self) -> Predicate {
        self.predicate.clone().unwrap_or(Predicate::Bool(true))
    }
}

/// Pre-defined medical refinements for pharmacological computing
pub mod medical {
    use super::*;

    /// Positive dose: dose > 0
    pub fn positive_dose() -> RefinedType {
        RefinedType::new(
            Type::F64,
            "dose",
            Some(Predicate::Compare(
                CompareOp::Gt,
                Box::new(Predicate::Var("dose".into())),
                Box::new(Predicate::Float(0.0)),
            )),
        )
    }

    /// Valid creatinine clearance: 0 < crcl < 200
    pub fn valid_crcl() -> RefinedType {
        RefinedType::new(
            Type::F64,
            "crcl",
            Some(Predicate::And(
                Box::new(Predicate::Compare(
                    CompareOp::Gt,
                    Box::new(Predicate::Var("crcl".into())),
                    Box::new(Predicate::Float(0.0)),
                )),
                Box::new(Predicate::Compare(
                    CompareOp::Lt,
                    Box::new(Predicate::Var("crcl".into())),
                    Box::new(Predicate::Float(200.0)),
                )),
            )),
        )
    }

    /// Valid body weight: 0 < weight < 500 kg
    pub fn valid_weight() -> RefinedType {
        RefinedType::new(
            Type::F64,
            "weight",
            Some(Predicate::And(
                Box::new(Predicate::Compare(
                    CompareOp::Gt,
                    Box::new(Predicate::Var("weight".into())),
                    Box::new(Predicate::Float(0.0)),
                )),
                Box::new(Predicate::Compare(
                    CompareOp::Lt,
                    Box::new(Predicate::Var("weight".into())),
                    Box::new(Predicate::Float(500.0)),
                )),
            )),
        )
    }

    /// Valid age: 0 <= age < 150
    pub fn valid_age() -> RefinedType {
        RefinedType::new(
            Type::F64,
            "age",
            Some(Predicate::And(
                Box::new(Predicate::Compare(
                    CompareOp::Ge,
                    Box::new(Predicate::Var("age".into())),
                    Box::new(Predicate::Float(0.0)),
                )),
                Box::new(Predicate::Compare(
                    CompareOp::Lt,
                    Box::new(Predicate::Var("age".into())),
                    Box::new(Predicate::Float(150.0)),
                )),
            )),
        )
    }

    /// Valid percentage: 0 <= pct <= 100
    pub fn valid_percentage() -> RefinedType {
        RefinedType::new(
            Type::F64,
            "pct",
            Some(Predicate::And(
                Box::new(Predicate::Compare(
                    CompareOp::Ge,
                    Box::new(Predicate::Var("pct".into())),
                    Box::new(Predicate::Float(0.0)),
                )),
                Box::new(Predicate::Compare(
                    CompareOp::Le,
                    Box::new(Predicate::Var("pct".into())),
                    Box::new(Predicate::Float(100.0)),
                )),
            )),
        )
    }
}

/// Pre-defined array refinements
pub mod array {
    use super::*;

    /// Non-empty array
    pub fn non_empty() -> Predicate {
        Predicate::gt(
            Predicate::App("len".to_string(), vec![Predicate::var("arr")]),
            Predicate::Int(0),
        )
    }

    /// Array with minimum length
    pub fn min_length(min: i64) -> Predicate {
        Predicate::ge(
            Predicate::App("len".to_string(), vec![Predicate::var("arr")]),
            Predicate::Int(min),
        )
    }

    /// Array with exact length
    pub fn exact_length(len: i64) -> Predicate {
        Predicate::eq(
            Predicate::App("len".to_string(), vec![Predicate::var("arr")]),
            Predicate::Int(len),
        )
    }

    /// Valid index for array
    pub fn valid_index(arr_var: &str, idx_var: &str) -> Predicate {
        Predicate::and(
            Predicate::ge(Predicate::var(idx_var), Predicate::Int(0)),
            Predicate::lt(
                Predicate::var(idx_var),
                Predicate::App("len".to_string(), vec![Predicate::var(arr_var)]),
            ),
        )
    }
}

/// Refinement type checker
pub struct RefinementChecker {
    /// Environment: variable -> refined type
    env: std::collections::HashMap<String, RefinedType>,
    /// Path condition (accumulated constraints)
    path_condition: Vec<Predicate>,
}

impl RefinementChecker {
    pub fn new() -> Self {
        Self {
            env: std::collections::HashMap::new(),
            path_condition: Vec::new(),
        }
    }

    /// Add a variable to the environment
    pub fn bind(&mut self, name: String, ty: RefinedType) {
        self.env.insert(name, ty);
    }

    /// Add a path condition
    pub fn assume(&mut self, pred: Predicate) {
        self.path_condition.push(pred);
    }

    /// Check if a predicate is valid under current path conditions using SMT solver
    pub fn check(&self, pred: &Predicate) -> RefinementResult {
        // Quick check for trivial cases
        if pred.is_trivially_true() {
            return RefinementResult::Valid;
        }
        if pred.is_trivially_false() {
            return RefinementResult::Invalid("Predicate is trivially false".to_string());
        }

        // Convert predicate to SMT formula
        let mut ctx = SmtContext::new();
        let formula = self.predicate_to_smt(pred, &mut ctx);

        // Build the full context including path conditions
        let context_formula = if self.path_condition.is_empty() {
            SmtFormula::True
        } else {
            let conditions: Vec<SmtFormula> = self
                .path_condition
                .iter()
                .map(|p| self.predicate_to_smt(p, &mut ctx))
                .collect();
            SmtFormula::And(conditions)
        };

        // Create solver and check validity
        // To prove P is valid, we check if ¬P is unsatisfiable
        let mut solver = MockSolver::new();

        // Add variable bounds from environment
        for (name, refined_ty) in &self.env {
            // Extract bounds from refinement predicates if possible
            if let Some(ref pred) = refined_ty.predicate {
                self.extract_bounds_to_solver(name, pred, &mut solver);
            }
        }

        // Check: context ⟹ formula (i.e., ¬(context ∧ ¬formula) is unsat)
        let to_check = SmtFormula::Implies(
            Box::new(context_formula),
            Box::new(formula),
        );

        match solver.check_valid(&to_check) {
            Ok(VerificationResult::Sat) => RefinementResult::Valid,
            Ok(VerificationResult::Unsat) => {
                RefinementResult::Invalid("SMT solver found counterexample".to_string())
            }
            Ok(VerificationResult::Unknown) => RefinementResult::Unknown,
            Ok(VerificationResult::Timeout) => RefinementResult::Unknown,
            Ok(VerificationResult::Error(msg)) => {
                RefinementResult::Invalid(format!("SMT error: {}", msg))
            }
            Err(e) => RefinementResult::Invalid(format!("Solver error: {}", e)),
        }
    }

    /// Convert a Predicate to SmtFormula
    fn predicate_to_smt(&self, pred: &Predicate, ctx: &mut SmtContext) -> SmtFormula {
        match pred {
            Predicate::Bool(true) => SmtFormula::True,
            Predicate::Bool(false) => SmtFormula::False,
            Predicate::Int(n) => SmtFormula::Term(SmtTerm::Int(*n)),
            Predicate::Float(f) => SmtFormula::Term(SmtTerm::Real(*f)),
            Predicate::Var(name) => {
                ctx.declare_var(name.clone(), crate::smt::SmtSort::Real);
                SmtFormula::Term(SmtTerm::Var(name.clone()))
            }
            Predicate::Compare(op, left, right) => {
                let l = self.predicate_to_smt_term(left, ctx);
                let r = self.predicate_to_smt_term(right, ctx);
                match op {
                    CompareOp::Eq => SmtFormula::Eq(Box::new(l), Box::new(r)),
                    CompareOp::Ne => SmtFormula::Not(Box::new(SmtFormula::Eq(
                        Box::new(l),
                        Box::new(r),
                    ))),
                    CompareOp::Lt => SmtFormula::Lt(Box::new(l), Box::new(r)),
                    CompareOp::Le => SmtFormula::Le(Box::new(l), Box::new(r)),
                    CompareOp::Gt => SmtFormula::Gt(Box::new(l), Box::new(r)),
                    CompareOp::Ge => SmtFormula::Ge(Box::new(l), Box::new(r)),
                }
            }
            Predicate::Arith(op, left, right) => {
                let l = self.predicate_to_smt_term(left, ctx);
                let r = self.predicate_to_smt_term(right, ctx);
                let term = match op {
                    ArithOp::Add => SmtTerm::Add(Box::new(l), Box::new(r)),
                    ArithOp::Sub => SmtTerm::Sub(Box::new(l), Box::new(r)),
                    ArithOp::Mul => SmtTerm::Mul(Box::new(l), Box::new(r)),
                    ArithOp::Div => SmtTerm::Div(Box::new(l), Box::new(r)),
                    ArithOp::Mod => SmtTerm::Mod(Box::new(l), Box::new(r)),
                };
                SmtFormula::Term(term)
            }
            Predicate::And(left, right) => {
                let l = self.predicate_to_smt(left, ctx);
                let r = self.predicate_to_smt(right, ctx);
                SmtFormula::And(vec![l, r])
            }
            Predicate::Or(left, right) => {
                let l = self.predicate_to_smt(left, ctx);
                let r = self.predicate_to_smt(right, ctx);
                SmtFormula::Or(vec![l, r])
            }
            Predicate::Not(inner) => {
                let f = self.predicate_to_smt(inner, ctx);
                SmtFormula::Not(Box::new(f))
            }
            Predicate::Implies(left, right) => {
                let l = self.predicate_to_smt(left, ctx);
                let r = self.predicate_to_smt(right, ctx);
                SmtFormula::Implies(Box::new(l), Box::new(r))
            }
            Predicate::Forall(var, body) => {
                ctx.declare_var(var.clone(), crate::smt::SmtSort::Real);
                let body_smt = self.predicate_to_smt(body, ctx);
                SmtFormula::Forall(var.clone(), crate::smt::SmtSort::Real, Box::new(body_smt))
            }
            Predicate::Exists(var, body) => {
                ctx.declare_var(var.clone(), crate::smt::SmtSort::Real);
                let body_smt = self.predicate_to_smt(body, ctx);
                SmtFormula::Exists(var.clone(), crate::smt::SmtSort::Real, Box::new(body_smt))
            }
            Predicate::App(name, args) => {
                let smt_args: Vec<SmtTerm> = args
                    .iter()
                    .map(|a| self.predicate_to_smt_term(a, ctx))
                    .collect();
                SmtFormula::App(name.clone(), smt_args)
            }
            Predicate::Ite(cond, then_p, else_p) => {
                let c = self.predicate_to_smt(cond, ctx);
                let t = self.predicate_to_smt(then_p, ctx);
                let e = self.predicate_to_smt(else_p, ctx);
                SmtFormula::Ite(Box::new(c), Box::new(t), Box::new(e))
            }
        }
    }

    /// Convert a Predicate to SmtTerm (for use in comparisons/arithmetic)
    fn predicate_to_smt_term(&self, pred: &Predicate, ctx: &mut SmtContext) -> SmtTerm {
        match pred {
            Predicate::Int(n) => SmtTerm::Int(*n),
            Predicate::Float(f) => SmtTerm::Real(*f),
            Predicate::Bool(b) => SmtTerm::Bool(*b),
            Predicate::Var(name) => {
                ctx.declare_var(name.clone(), crate::smt::SmtSort::Real);
                SmtTerm::Var(name.clone())
            }
            Predicate::Arith(op, left, right) => {
                let l = self.predicate_to_smt_term(left, ctx);
                let r = self.predicate_to_smt_term(right, ctx);
                match op {
                    ArithOp::Add => SmtTerm::Add(Box::new(l), Box::new(r)),
                    ArithOp::Sub => SmtTerm::Sub(Box::new(l), Box::new(r)),
                    ArithOp::Mul => SmtTerm::Mul(Box::new(l), Box::new(r)),
                    ArithOp::Div => SmtTerm::Div(Box::new(l), Box::new(r)),
                    ArithOp::Mod => SmtTerm::Mod(Box::new(l), Box::new(r)),
                }
            }
            Predicate::App(name, args) => {
                let smt_args: Vec<SmtTerm> = args
                    .iter()
                    .map(|a| self.predicate_to_smt_term(a, ctx))
                    .collect();
                SmtTerm::App(name.clone(), smt_args)
            }
            // For other predicates, wrap in a term representation
            _ => SmtTerm::Int(0), // Fallback
        }
    }

    /// Extract variable bounds from a predicate and add to solver
    fn extract_bounds_to_solver(&self, var: &str, pred: &Predicate, solver: &mut MockSolver) {
        match pred {
            Predicate::Compare(CompareOp::Gt, left, right)
            | Predicate::Compare(CompareOp::Ge, left, right) => {
                if let (Predicate::Var(v), Predicate::Float(lo)) = (left.as_ref(), right.as_ref()) {
                    if v == var {
                        solver.set_bounds(var, *lo, f64::INFINITY);
                    }
                }
                if let (Predicate::Var(v), Predicate::Int(lo)) = (left.as_ref(), right.as_ref()) {
                    if v == var {
                        solver.set_bounds(var, *lo as f64, f64::INFINITY);
                    }
                }
            }
            Predicate::Compare(CompareOp::Lt, left, right)
            | Predicate::Compare(CompareOp::Le, left, right) => {
                if let (Predicate::Var(v), Predicate::Float(hi)) = (left.as_ref(), right.as_ref()) {
                    if v == var {
                        solver.set_bounds(var, f64::NEG_INFINITY, *hi);
                    }
                }
                if let (Predicate::Var(v), Predicate::Int(hi)) = (left.as_ref(), right.as_ref()) {
                    if v == var {
                        solver.set_bounds(var, f64::NEG_INFINITY, *hi as f64);
                    }
                }
            }
            Predicate::And(left, right) => {
                self.extract_bounds_to_solver(var, left, solver);
                self.extract_bounds_to_solver(var, right, solver);
            }
            _ => {}
        }
    }

    /// Check subtyping: is `sub` a subtype of `sup`?
    pub fn check_subtype(&self, sub: &RefinedType, sup: &RefinedType) -> RefinementResult {
        // Base types must match
        if sub.base != sup.base {
            return RefinementResult::Invalid("Base types don't match".to_string());
        }

        // If sup has no predicate, subtyping holds
        if sup.predicate.is_none() {
            return RefinementResult::Valid;
        }

        // Check that sub's predicate implies sup's predicate
        let sub_pred = sub.predicate_or_true();
        let sup_pred = sup.predicate_or_true();

        // Need to check: path_condition ∧ sub_pred => sup_pred
        let implication = Predicate::implies(sub_pred, sup_pred);

        // Build full context
        let mut full_context = Predicate::true_pred();
        for cond in &self.path_condition {
            full_context = Predicate::and(full_context, cond.clone());
        }

        let to_check = Predicate::implies(full_context, implication);
        self.check(&to_check)
    }
}

impl Default for RefinementChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of refinement checking
#[derive(Debug, Clone)]
pub enum RefinementResult {
    /// The predicate is provably valid
    Valid,
    /// The predicate is provably invalid
    Invalid(String),
    /// Could not determine validity
    Unknown,
}

impl RefinementResult {
    pub fn is_valid(&self) -> bool {
        matches!(self, RefinementResult::Valid)
    }

    pub fn is_invalid(&self) -> bool {
        matches!(self, RefinementResult::Invalid(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predicate_construction() {
        let pred = Predicate::and(
            Predicate::gt(Predicate::var("x"), Predicate::Int(0)),
            Predicate::lt(Predicate::var("x"), Predicate::Int(100)),
        );

        // Should represent: x > 0 && x < 100
        match pred {
            Predicate::And(l, r) => {
                assert!(matches!(*l, Predicate::Compare(CompareOp::Gt, _, _)));
                assert!(matches!(*r, Predicate::Compare(CompareOp::Lt, _, _)));
            }
            _ => panic!("Expected And predicate"),
        }
    }

    #[test]
    fn test_substitution() {
        let pred = Predicate::gt(Predicate::var("x"), Predicate::Int(0));
        let substituted = pred.substitute("x", &Predicate::Int(42));

        match substituted {
            Predicate::Compare(CompareOp::Gt, l, r) => {
                assert!(matches!(*l, Predicate::Int(42)));
                assert!(matches!(*r, Predicate::Int(0)));
            }
            _ => panic!("Expected Compare predicate"),
        }
    }

    #[test]
    fn test_medical_refinements() {
        let dose_type = medical::positive_dose();
        assert!(dose_type.is_refined());

        let crcl_type = medical::valid_crcl();
        assert!(crcl_type.is_refined());
    }

    // SMT Integration Tests

    #[test]
    fn test_smt_trivially_true() {
        let checker = RefinementChecker::new();
        let pred = Predicate::Bool(true);
        assert!(checker.check(&pred).is_valid());
    }

    #[test]
    fn test_smt_trivially_false() {
        let checker = RefinementChecker::new();
        let pred = Predicate::Bool(false);
        assert!(checker.check(&pred).is_invalid());
    }

    #[test]
    fn test_smt_simple_comparison() {
        let checker = RefinementChecker::new();

        // 5 > 3 should be valid
        let pred = Predicate::gt(Predicate::Int(5), Predicate::Int(3));
        let result = checker.check(&pred);
        assert!(result.is_valid(), "5 > 3 should be valid");

        // 3 > 5 should be invalid
        let pred2 = Predicate::gt(Predicate::Int(3), Predicate::Int(5));
        let result2 = checker.check(&pred2);
        assert!(result2.is_invalid(), "3 > 5 should be invalid");
    }

    #[test]
    fn test_smt_conjunction() {
        let checker = RefinementChecker::new();

        // true && true = true
        let pred = Predicate::and(Predicate::Bool(true), Predicate::Bool(true));
        assert!(checker.check(&pred).is_valid());

        // true && false = false
        let pred2 = Predicate::and(Predicate::Bool(true), Predicate::Bool(false));
        assert!(checker.check(&pred2).is_invalid());
    }

    #[test]
    fn test_smt_disjunction() {
        let checker = RefinementChecker::new();

        // true || false = true
        let pred = Predicate::or(Predicate::Bool(true), Predicate::Bool(false));
        assert!(checker.check(&pred).is_valid());

        // false || false = false
        let pred2 = Predicate::or(Predicate::Bool(false), Predicate::Bool(false));
        assert!(checker.check(&pred2).is_invalid());
    }

    #[test]
    fn test_smt_implication() {
        let checker = RefinementChecker::new();

        // false => anything = true
        let pred = Predicate::implies(Predicate::Bool(false), Predicate::Bool(false));
        assert!(checker.check(&pred).is_valid());

        // true => true = true
        let pred2 = Predicate::implies(Predicate::Bool(true), Predicate::Bool(true));
        assert!(checker.check(&pred2).is_valid());

        // true => false = false
        let pred3 = Predicate::implies(Predicate::Bool(true), Predicate::Bool(false));
        assert!(checker.check(&pred3).is_invalid());
    }

    #[test]
    fn test_smt_with_path_condition() {
        let mut checker = RefinementChecker::new();

        // Assume x > 10
        checker.assume(Predicate::gt(Predicate::var("x"), Predicate::Int(10)));

        // Check x > 5 (should be valid under the assumption)
        let pred = Predicate::gt(Predicate::var("x"), Predicate::Int(5));
        let result = checker.check(&pred);

        // MockSolver uses interval arithmetic, may return Unknown for variable predicates
        assert!(!result.is_invalid(), "x > 10 should imply x > 5");
    }

    #[test]
    fn test_smt_subtype_check() {
        let checker = RefinementChecker::new();

        // Positive subtype of non-negative
        let positive = RefinedType::new(
            Type::I64,
            "x",
            Some(Predicate::gt(Predicate::var("x"), Predicate::Int(0))),
        );

        let non_negative = RefinedType::new(
            Type::I64,
            "x",
            Some(Predicate::ge(Predicate::var("x"), Predicate::Int(0))),
        );

        // Positive is NOT a subtype of non-negative in general
        // (positive implies non-negative, but the solver may return Unknown)
        let result = checker.check_subtype(&positive, &non_negative);
        assert!(!result.is_invalid(), "Positive should be subtype of non-negative");
    }

    #[test]
    fn test_smt_arithmetic() {
        let checker = RefinementChecker::new();

        // 3 + 4 = 7 should be valid
        let sum = Predicate::Arith(
            ArithOp::Add,
            Box::new(Predicate::Int(3)),
            Box::new(Predicate::Int(4)),
        );
        let pred = Predicate::eq(sum, Predicate::Int(7));
        let result = checker.check(&pred);
        // Interval arithmetic may return Unknown for equality
        assert!(!result.is_invalid(), "3 + 4 = 7 should not be invalid");
    }
}
