//! Unit inference
//!
//! Infers units of measure from context using constraint solving.

use super::units::{Unit, UnitChecker, UnitOp};
use crate::common::Span;
use std::collections::{HashMap, VecDeque};

/// Unit variable for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitVar(pub u32);

/// A unit expression that may contain variables
#[derive(Debug, Clone, PartialEq)]
pub enum UnitExpr {
    /// Known concrete unit
    Concrete(Unit),
    /// Unit variable (to be inferred)
    Var(UnitVar),
    /// Product of units
    Product(Box<UnitExpr>, Box<UnitExpr>),
    /// Quotient of units
    Quotient(Box<UnitExpr>, Box<UnitExpr>),
    /// Power of a unit
    Power(Box<UnitExpr>, i32),
}

impl UnitExpr {
    pub fn dimensionless() -> Self {
        UnitExpr::Concrete(Unit::dimensionless())
    }

    pub fn var(id: u32) -> Self {
        UnitExpr::Var(UnitVar(id))
    }

    pub fn concrete(unit: Unit) -> Self {
        UnitExpr::Concrete(unit)
    }

    /// Check if this expression contains any variables
    pub fn has_vars(&self) -> bool {
        match self {
            UnitExpr::Concrete(_) => false,
            UnitExpr::Var(_) => true,
            UnitExpr::Product(a, b) | UnitExpr::Quotient(a, b) => a.has_vars() || b.has_vars(),
            UnitExpr::Power(e, _) => e.has_vars(),
        }
    }

    /// Collect all unit variables in this expression
    pub fn collect_vars(&self, vars: &mut Vec<UnitVar>) {
        match self {
            UnitExpr::Concrete(_) => {}
            UnitExpr::Var(v) => vars.push(*v),
            UnitExpr::Product(a, b) | UnitExpr::Quotient(a, b) => {
                a.collect_vars(vars);
                b.collect_vars(vars);
            }
            UnitExpr::Power(e, _) => e.collect_vars(vars),
        }
    }

    /// Substitute a unit variable with a concrete unit
    pub fn substitute(&self, var: UnitVar, unit: &Unit) -> UnitExpr {
        match self {
            UnitExpr::Concrete(u) => UnitExpr::Concrete(u.clone()),
            UnitExpr::Var(v) if *v == var => UnitExpr::Concrete(unit.clone()),
            UnitExpr::Var(v) => UnitExpr::Var(*v),
            UnitExpr::Product(a, b) => UnitExpr::Product(
                Box::new(a.substitute(var, unit)),
                Box::new(b.substitute(var, unit)),
            ),
            UnitExpr::Quotient(a, b) => UnitExpr::Quotient(
                Box::new(a.substitute(var, unit)),
                Box::new(b.substitute(var, unit)),
            ),
            UnitExpr::Power(e, n) => UnitExpr::Power(Box::new(e.substitute(var, unit)), *n),
        }
    }

    /// Try to evaluate to a concrete unit (if no variables remain)
    pub fn evaluate(&self) -> Option<Unit> {
        match self {
            UnitExpr::Concrete(u) => Some(u.clone()),
            UnitExpr::Var(_) => None,
            UnitExpr::Product(a, b) => {
                let a = a.evaluate()?;
                let b = b.evaluate()?;
                Some(a.multiply(&b))
            }
            UnitExpr::Quotient(a, b) => {
                let a = a.evaluate()?;
                let b = b.evaluate()?;
                Some(a.divide(&b))
            }
            UnitExpr::Power(e, n) => {
                let e = e.evaluate()?;
                Some(e.power(*n))
            }
        }
    }
}

/// A unit constraint
#[derive(Debug, Clone)]
pub enum UnitConstraint {
    /// Two unit expressions must be equal
    Equal(UnitExpr, UnitExpr, Span),
    /// A unit variable must have a specific unit
    Assign(UnitVar, Unit, Span),
}

/// Unit inference context
#[derive(Debug)]
pub struct UnitInference {
    /// Unit checker for parsing and validation
    checker: UnitChecker,
    /// Counter for fresh unit variables
    next_var: u32,
    /// Collected constraints
    constraints: Vec<UnitConstraint>,
    /// Current substitution (solved variables)
    substitution: HashMap<UnitVar, Unit>,
    /// Errors collected during inference
    errors: Vec<UnitInferenceError>,
}

/// Unit inference error
#[derive(Debug, Clone)]
pub struct UnitInferenceError {
    pub message: String,
    pub span: Span,
    pub expected: Option<String>,
    pub found: Option<String>,
}

impl UnitInferenceError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
            expected: None,
            found: None,
        }
    }

    pub fn mismatch(expected: &Unit, found: &Unit, span: Span) -> Self {
        Self {
            message: format!(
                "unit mismatch: expected `{}`, found `{}`",
                expected.format(),
                found.format()
            ),
            span,
            expected: Some(expected.format()),
            found: Some(found.format()),
        }
    }

    pub fn cannot_add(u1: &Unit, u2: &Unit, span: Span) -> Self {
        Self {
            message: format!(
                "cannot add values with different units: `{}` and `{}`",
                u1.format(),
                u2.format()
            ),
            span,
            expected: Some(u1.format()),
            found: Some(u2.format()),
        }
    }
}

impl UnitInference {
    pub fn new() -> Self {
        Self {
            checker: UnitChecker::new(),
            next_var: 0,
            constraints: Vec::new(),
            substitution: HashMap::new(),
            errors: Vec::new(),
        }
    }

    /// Create a fresh unit variable
    pub fn fresh_var(&mut self) -> UnitVar {
        let var = UnitVar(self.next_var);
        self.next_var += 1;
        var
    }

    /// Add an equality constraint
    pub fn constrain_equal(&mut self, u1: UnitExpr, u2: UnitExpr, span: Span) {
        self.constraints.push(UnitConstraint::Equal(u1, u2, span));
    }

    /// Add an assignment constraint
    pub fn constrain_assign(&mut self, var: UnitVar, unit: Unit, span: Span) {
        self.constraints
            .push(UnitConstraint::Assign(var, unit, span));
    }

    /// Get the unit for an expression, creating a fresh variable if unknown
    pub fn get_or_create(&mut self, known: Option<&str>) -> UnitExpr {
        match known {
            Some(s) => {
                if let Some(unit) = self.checker.parse(s) {
                    UnitExpr::Concrete(unit)
                } else {
                    UnitExpr::Var(self.fresh_var())
                }
            }
            None => UnitExpr::Var(self.fresh_var()),
        }
    }

    /// Infer the unit for a binary operation
    pub fn infer_binary(
        &mut self,
        op: UnitOp,
        left: &UnitExpr,
        right: &UnitExpr,
        span: Span,
    ) -> UnitExpr {
        match op {
            UnitOp::Add | UnitOp::Sub => {
                // Addition/subtraction requires same units
                self.constrain_equal(left.clone(), right.clone(), span);
                left.clone()
            }
            UnitOp::Mul => {
                // Multiplication combines units
                UnitExpr::Product(Box::new(left.clone()), Box::new(right.clone()))
            }
            UnitOp::Div => {
                // Division divides units
                UnitExpr::Quotient(Box::new(left.clone()), Box::new(right.clone()))
            }
        }
    }

    /// Solve all collected constraints
    pub fn solve(&mut self) -> Result<(), Vec<UnitInferenceError>> {
        let mut worklist: VecDeque<_> = self.constraints.drain(..).collect();
        let mut iterations = 0;
        let max_iterations = 1000;

        while let Some(constraint) = worklist.pop_front() {
            iterations += 1;
            if iterations > max_iterations {
                self.errors.push(UnitInferenceError::new(
                    "unit inference did not converge",
                    Span::dummy(),
                ));
                break;
            }

            match constraint {
                UnitConstraint::Assign(var, unit, span) => {
                    if let Some(existing) = self.substitution.get(&var) {
                        if !existing.is_compatible(&unit) {
                            self.errors
                                .push(UnitInferenceError::mismatch(existing, &unit, span));
                        }
                    } else {
                        self.substitution.insert(var, unit);
                    }
                }
                UnitConstraint::Equal(u1, u2, span) => {
                    let u1 = self.apply_substitution(&u1);
                    let u2 = self.apply_substitution(&u2);

                    match (u1.evaluate(), u2.evaluate()) {
                        (Some(c1), Some(c2)) => {
                            if !c1.is_compatible(&c2) {
                                self.errors
                                    .push(UnitInferenceError::mismatch(&c1, &c2, span));
                            }
                        }
                        (Some(c), None) => {
                            // u2 has variables, try to solve for them
                            if let UnitExpr::Var(v) = u2 {
                                self.substitution.insert(v, c);
                            }
                        }
                        (None, Some(c)) => {
                            // u1 has variables, try to solve for them
                            if let UnitExpr::Var(v) = u1 {
                                self.substitution.insert(v, c);
                            }
                        }
                        (None, None) => {
                            // Both have variables, defer
                            worklist.push_back(UnitConstraint::Equal(u1, u2, span));
                        }
                    }
                }
            }
        }

        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    /// Apply current substitution to a unit expression
    fn apply_substitution(&self, expr: &UnitExpr) -> UnitExpr {
        match expr {
            UnitExpr::Concrete(_) => expr.clone(),
            UnitExpr::Var(v) => {
                if let Some(unit) = self.substitution.get(v) {
                    UnitExpr::Concrete(unit.clone())
                } else {
                    expr.clone()
                }
            }
            UnitExpr::Product(a, b) => UnitExpr::Product(
                Box::new(self.apply_substitution(a)),
                Box::new(self.apply_substitution(b)),
            ),
            UnitExpr::Quotient(a, b) => UnitExpr::Quotient(
                Box::new(self.apply_substitution(a)),
                Box::new(self.apply_substitution(b)),
            ),
            UnitExpr::Power(e, n) => UnitExpr::Power(Box::new(self.apply_substitution(e)), *n),
        }
    }

    /// Look up the solved unit for a variable
    pub fn lookup(&self, var: UnitVar) -> Option<&Unit> {
        self.substitution.get(&var)
    }

    /// Get access to the unit checker
    pub fn checker(&self) -> &UnitChecker {
        &self.checker
    }

    /// Get mutable access to the unit checker
    pub fn checker_mut(&mut self) -> &mut UnitChecker {
        &mut self.checker
    }

    /// Get all errors
    pub fn errors(&self) -> &[UnitInferenceError] {
        &self.errors
    }

    /// Check if inference succeeded without errors
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }
}

impl Default for UnitInference {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_var_substitution() {
        let var = UnitVar(0);
        let expr = UnitExpr::Var(var);
        let mg = Unit::base("mg");

        let substituted = expr.substitute(var, &mg);
        assert_eq!(substituted.evaluate(), Some(mg));
    }

    #[test]
    fn test_unit_expr_evaluate() {
        let mg = Unit::base("mg");
        let ml = Unit::base("mL");

        let expr = UnitExpr::Quotient(
            Box::new(UnitExpr::Concrete(mg.clone())),
            Box::new(UnitExpr::Concrete(ml.clone())),
        );

        let result = expr.evaluate().unwrap();
        assert!(result.is_compatible(&mg.divide(&ml)));
    }

    #[test]
    fn test_inference_basic() {
        let mut inf = UnitInference::new();

        let var = inf.fresh_var();
        let mg = Unit::base("mg");

        inf.constrain_assign(var, mg.clone(), Span::dummy());

        assert!(inf.solve().is_ok());
        assert_eq!(inf.lookup(var), Some(&mg));
    }

    #[test]
    fn test_inference_binary_add() {
        let mut inf = UnitInference::new();

        let mg = Unit::base("mg");
        let u1 = UnitExpr::Concrete(mg.clone());
        let var = inf.fresh_var();
        let u2 = UnitExpr::Var(var);

        // Adding mg + unknown should constrain unknown to mg
        let _result = inf.infer_binary(UnitOp::Add, &u1, &u2, Span::dummy());

        assert!(inf.solve().is_ok());
        assert_eq!(inf.lookup(var), Some(&mg));
    }

    #[test]
    fn test_inference_binary_div() {
        let mut inf = UnitInference::new();

        let mg = Unit::base("mg");
        let ml = Unit::base("mL");

        let u1 = UnitExpr::Concrete(mg.clone());
        let u2 = UnitExpr::Concrete(ml.clone());

        let result = inf.infer_binary(UnitOp::Div, &u1, &u2, Span::dummy());

        let evaluated = result.evaluate().unwrap();
        assert!(evaluated.is_compatible(&mg.divide(&ml)));
    }

    #[test]
    fn test_inference_mismatch() {
        let mut inf = UnitInference::new();

        let mg = Unit::base("mg");
        let ml = Unit::base("mL");

        let u1 = UnitExpr::Concrete(mg);
        let u2 = UnitExpr::Concrete(ml);

        // Adding mg + mL should fail
        let _ = inf.infer_binary(UnitOp::Add, &u1, &u2, Span::dummy());

        let result = inf.solve();
        assert!(result.is_err());
    }
}
