//! SMT Solver Integration for Refinement Type Verification
//!
//! This module provides the interface between the refinement type system and
//! SMT solvers. Currently implements a mockup solver using interval arithmetic,
//! with infrastructure prepared for Z3 FFI integration.
//!
//! # Architecture
//!
//! ```text
//! Predicate → SmtFormula → Solver → VerificationResult
//!     ↓                       ↓
//! Epistemic Properties    Z3 (FFI) or MockSolver
//! ```
//!
//! # Epistemic Verification
//!
//! The SMT system verifies epistemic properties:
//! - **EpsilonNonWidening**: ε doesn't increase through operations
//! - **BoundedUncertainty**: ε stays within specified bounds
//! - **ProvenanceValid**: Data lineage is traceable
//! - **KnowledgeConsistent**: Knowledge values satisfy type constraints

mod epistemic;
mod formula;
mod interval;
mod solver;
pub mod z3_solver;

pub use epistemic::{EpistemicConstraint, EpistemicProperty, EpistemicVerifier};
pub use formula::{SmtContext, SmtFormula, SmtSort, SmtTerm};
pub use interval::{Interval, IntervalArithmetic};
pub use solver::{MockSolver, SmtSolver, SolverError, VerificationResult};

// Z3 solver exports
pub use z3_solver::{
    Counterexample, CounterexampleValue, EpistemicProperty as Z3EpistemicProperty,
    EpistemicVerifier as Z3EpistemicVerifier, EpistemicVerifyResult, MockEpistemicVerifier,
    create_epistemic_verifier,
};

#[cfg(feature = "smt")]
pub use z3_solver::{Z3Model, Z3Solver, Z3Stats, create_z3_context};

use crate::refinement::predicate::{Atom, BinOp, CompareOp, Predicate, Term, UnOp};

/// Translate a refinement predicate to SMT formula
pub fn predicate_to_smt(pred: &Predicate, ctx: &mut SmtContext) -> SmtFormula {
    match pred {
        Predicate::True => SmtFormula::True,
        Predicate::False => SmtFormula::False,

        Predicate::Atom(atom) => translate_atom(atom, ctx),

        Predicate::Not(p) => SmtFormula::Not(Box::new(predicate_to_smt(p, ctx))),

        Predicate::And(ps) => {
            let formulas: Vec<_> = ps.iter().map(|p| predicate_to_smt(p, ctx)).collect();
            SmtFormula::And(formulas)
        }

        Predicate::Or(ps) => {
            let formulas: Vec<_> = ps.iter().map(|p| predicate_to_smt(p, ctx)).collect();
            SmtFormula::Or(formulas)
        }

        Predicate::Implies(p, q) => SmtFormula::Implies(
            Box::new(predicate_to_smt(p, ctx)),
            Box::new(predicate_to_smt(q, ctx)),
        ),

        Predicate::Forall(var, _ty, body) => {
            let sort = SmtSort::Real; // Simplification for now
            ctx.declare_var(var.clone(), sort.clone());
            SmtFormula::Forall(var.clone(), sort, Box::new(predicate_to_smt(body, ctx)))
        }

        Predicate::Exists(var, _ty, body) => {
            let sort = SmtSort::Real;
            ctx.declare_var(var.clone(), sort.clone());
            SmtFormula::Exists(var.clone(), sort, Box::new(predicate_to_smt(body, ctx)))
        }

        Predicate::App(name, args) => {
            let smt_args: Vec<_> = args.iter().map(|t| term_to_smt(t, ctx)).collect();
            SmtFormula::App(name.clone(), smt_args)
        }

        Predicate::Ite(cond, then_p, else_p) => SmtFormula::Ite(
            Box::new(predicate_to_smt(cond, ctx)),
            Box::new(predicate_to_smt(then_p, ctx)),
            Box::new(predicate_to_smt(else_p, ctx)),
        ),
    }
}

/// Translate an atomic predicate to SMT formula
fn translate_atom(atom: &Atom, ctx: &mut SmtContext) -> SmtFormula {
    let lhs = term_to_smt(&atom.lhs, ctx);
    let rhs = term_to_smt(&atom.rhs, ctx);

    match atom.op {
        CompareOp::Eq => SmtFormula::Eq(Box::new(lhs), Box::new(rhs)),
        CompareOp::Ne => SmtFormula::Not(Box::new(SmtFormula::Eq(Box::new(lhs), Box::new(rhs)))),
        CompareOp::Lt => SmtFormula::Lt(Box::new(lhs), Box::new(rhs)),
        CompareOp::Le => SmtFormula::Le(Box::new(lhs), Box::new(rhs)),
        CompareOp::Gt => SmtFormula::Gt(Box::new(lhs), Box::new(rhs)),
        CompareOp::Ge => SmtFormula::Ge(Box::new(lhs), Box::new(rhs)),
    }
}

/// Translate a term to SMT term
fn term_to_smt(term: &Term, ctx: &mut SmtContext) -> SmtTerm {
    match term {
        Term::Var(name) => {
            if !ctx.has_var(name) {
                ctx.declare_var(name.clone(), SmtSort::Real);
            }
            SmtTerm::Var(name.clone())
        }

        Term::Int(n) => SmtTerm::Int(*n),
        Term::Float(f) => SmtTerm::Real(*f),
        Term::Bool(b) => SmtTerm::Bool(*b),

        Term::BinOp(op, lhs, rhs) => {
            let l = term_to_smt(lhs, ctx);
            let r = term_to_smt(rhs, ctx);
            match op {
                BinOp::Add => SmtTerm::Add(Box::new(l), Box::new(r)),
                BinOp::Sub => SmtTerm::Sub(Box::new(l), Box::new(r)),
                BinOp::Mul => SmtTerm::Mul(Box::new(l), Box::new(r)),
                BinOp::Div => SmtTerm::Div(Box::new(l), Box::new(r)),
                BinOp::Mod => SmtTerm::Mod(Box::new(l), Box::new(r)),
            }
        }

        Term::UnOp(op, t) => {
            let inner = term_to_smt(t, ctx);
            match op {
                UnOp::Neg => SmtTerm::Neg(Box::new(inner)),
                UnOp::Not => SmtTerm::Not(Box::new(inner)),
                UnOp::Abs => SmtTerm::Abs(Box::new(inner)),
            }
        }

        Term::App(name, args) => {
            let smt_args: Vec<_> = args.iter().map(|a| term_to_smt(a, ctx)).collect();
            SmtTerm::App(name.clone(), smt_args)
        }

        Term::Field(base, field) => {
            let base_smt = term_to_smt(base, ctx);
            SmtTerm::Field(Box::new(base_smt), field.clone())
        }

        Term::Len(t) => {
            let inner = term_to_smt(t, ctx);
            SmtTerm::Len(Box::new(inner))
        }

        Term::Ite(cond, then_t, else_t) => SmtTerm::Ite(
            Box::new(term_to_smt(cond, ctx)),
            Box::new(term_to_smt(then_t, ctx)),
            Box::new(term_to_smt(else_t, ctx)),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_predicate_translation() {
        let mut ctx = SmtContext::new();

        // x > 0
        let pred = Predicate::gt(Term::var("x"), Term::int(0));
        let formula = predicate_to_smt(&pred, &mut ctx);

        assert!(matches!(formula, SmtFormula::Gt(_, _)));
    }

    #[test]
    fn test_conjunction_translation() {
        let mut ctx = SmtContext::new();

        // x > 0 && x < 100
        let pred = Predicate::and([
            Predicate::gt(Term::var("x"), Term::int(0)),
            Predicate::lt(Term::var("x"), Term::int(100)),
        ]);
        let formula = predicate_to_smt(&pred, &mut ctx);

        assert!(matches!(formula, SmtFormula::And(_)));
    }

    #[test]
    fn test_arithmetic_term_translation() {
        let mut ctx = SmtContext::new();

        // x + y > 0
        let pred = Predicate::gt(Term::add(Term::var("x"), Term::var("y")), Term::int(0));
        let formula = predicate_to_smt(&pred, &mut ctx);

        assert!(matches!(formula, SmtFormula::Gt(_, _)));
    }
}
