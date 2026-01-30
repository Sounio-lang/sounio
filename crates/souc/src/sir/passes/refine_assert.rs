//! Refinement Assertion Pass
//!
//! This pass finds parameters and return values with refinement types and
//! inserts runtime assertions where static proof is not possible.
//!
//! # Strategy
//!
//! 1. **Identify Refinement Types**: Scan function signatures for refinement predicates
//! 2. **Attempt Static Proof**: Use Z3 SMT solver to prove predicates hold
//! 3. **Insert Assertions**: Where proof fails, insert runtime `Assert` instruction
//! 4. **Support Simple Bounds**: Handle common patterns like `{ x: i32 | x > 0 }`
//!
//! # Z3 Integration
//!
//! When compiled with `--features smt`, this pass uses Z3 for static verification.
//! The workflow is:
//! - Convert predicates to the refinement module's format
//! - Build constraints with environment bindings
//! - Invoke Z3Solver to prove or refute
//! - Insert runtime assertions only when proof fails
//!
//! # Limitations
//!
//! - Complex predicates with quantifiers may timeout
//! - Function calls in predicates use uninterpreted functions

use crate::sir::{
    blocks::{Instruction, SirFunction},
    module::SirModule,
    ops::{CmpOp, FailureMode, SirInst},
    values::ValueId,
};
use crate::types::refinement::Predicate;

// Z3 integration (feature-gated)
#[cfg(feature = "smt")]
use crate::refinement::{
    constraint::{Binding, Constraint, ConstraintReason, Span},
    predicate::{Atom, CompareOp as RefCompareOp, Predicate as RefPredicate, RefinementType, Term},
    solver::Z3Solver,
};
#[cfg(feature = "smt")]
use crate::types::Type;

/// Refinement assertion pass
pub struct RefinementAssertionPass {
    /// Enable Z3 solver for static verification
    pub use_smt: bool,
    /// Be strict (panic on violation vs degrade confidence)
    pub strict: bool,
}

impl RefinementAssertionPass {
    pub fn new() -> Self {
        Self {
            use_smt: true,
            strict: true,
        }
    }

    /// Run the pass on a module
    pub fn run(&mut self, module: &mut SirModule) -> RefinementAssertResult {
        let mut result = RefinementAssertResult::default();

        for func in &mut module.functions {
            let func_result = self.run_on_function(func);
            result.assertions_inserted += func_result.assertions_inserted;
            result.proofs_attempted += func_result.proofs_attempted;
            result.proofs_succeeded += func_result.proofs_succeeded;
            result.modified |= func_result.modified;
        }

        result
    }

    /// Run on a single function
    fn run_on_function(&mut self, func: &mut SirFunction) -> RefinementAssertResult {
        let mut result = RefinementAssertResult::default();

        // For now, we'll implement a simple version that looks for
        // metadata annotations on values and inserts assertions

        // In a full implementation, we would:
        // 1. Parse refinement predicates from HIR types
        // 2. Lower them to SIR-level constraints
        // 3. Use Z3 to attempt static proof
        // 4. Insert assertions where proof fails

        // Placeholder: Insert assertions for function parameters with range metadata
        if let Some(entry_block) = func.blocks.first_mut() {
            let mut new_instructions = Vec::new();

            // Check for parameters with range requirements
            for (i, _param) in entry_block.params.iter().enumerate() {
                // In a real implementation, we'd check for refinement metadata
                // For now, demonstrate the infrastructure

                // Example: If parameter has refinement type { x: i32 | x > 0 }
                // we would insert an assertion here

                // Placeholder for demonstration
                if i < func.params.len() {
                    // This would be where we check if param has refinement type
                    // and insert appropriate assertion
                }
            }

            if !new_instructions.is_empty() {
                // Prepend new instructions to the entry block
                new_instructions.extend(entry_block.instructions.drain(..));
                entry_block.instructions = new_instructions;
                result.modified = true;
            }
        }

        result
    }

    /// Attempt to prove a predicate using SMT solver
    fn try_prove_predicate(&self, predicate: &Predicate) -> ProofResult {
        if !self.use_smt {
            return ProofResult::Unknown;
        }

        // Check for trivially true/false predicates
        if predicate.is_trivially_true() {
            return ProofResult::Valid;
        }
        if predicate.is_trivially_false() {
            return ProofResult::Invalid;
        }

        // For simple bounds, we can do static analysis
        match predicate {
            Predicate::Compare(op, lhs, rhs) => {
                // Try to evaluate statically
                if let (Some(l_val), Some(r_val)) = (self.eval_const(lhs), self.eval_const(rhs)) {
                    let holds = match op {
                        crate::types::refinement::CompareOp::Eq => l_val == r_val,
                        crate::types::refinement::CompareOp::Ne => l_val != r_val,
                        crate::types::refinement::CompareOp::Lt => l_val < r_val,
                        crate::types::refinement::CompareOp::Le => l_val <= r_val,
                        crate::types::refinement::CompareOp::Gt => l_val > r_val,
                        crate::types::refinement::CompareOp::Ge => l_val >= r_val,
                    };

                    return if holds {
                        ProofResult::Valid
                    } else {
                        ProofResult::Invalid
                    };
                }
            }
            _ => {}
        }

        // Try Z3 SMT solver for complex predicates (feature-gated)
        #[cfg(feature = "smt")]
        {
            return self.prove_with_z3(predicate);
        }

        // Fallback: conservatively return Unknown
        #[cfg(not(feature = "smt"))]
        ProofResult::Unknown
    }

    /// Prove a predicate using Z3 SMT solver
    #[cfg(feature = "smt")]
    fn prove_with_z3(&self, predicate: &Predicate) -> ProofResult {
        // Convert to refinement module predicate format
        let ref_pred = match Self::convert_predicate(predicate) {
            Some(p) => p,
            None => return ProofResult::Unknown,
        };

        // Create a Z3 context and solver
        let cfg = z3::Config::new();
        let ctx = z3::Context::new(&cfg);
        let mut solver = Z3Solver::new(&ctx);
        solver.set_timeout(2000); // 2 second timeout for individual checks

        // Build a constraint to verify
        let constraint = Constraint::new(
            Vec::new(), // No environment bindings for standalone proof
            ref_pred,
            Span::dummy(),
            ConstraintReason::Assert {
                message: "refinement check".to_string(),
            },
        );

        // Verify the constraint
        let results = solver.verify(&[constraint]);

        match results.first() {
            Some(crate::refinement::VerifyResult::Valid) => ProofResult::Valid,
            Some(crate::refinement::VerifyResult::Invalid { .. }) => ProofResult::Invalid,
            Some(crate::refinement::VerifyResult::Unknown { .. }) | None => ProofResult::Unknown,
        }
    }

    /// Convert types::refinement::Predicate to refinement::predicate::Predicate
    #[cfg(feature = "smt")]
    fn convert_predicate(pred: &Predicate) -> Option<RefPredicate> {
        match pred {
            Predicate::Bool(true) => Some(RefPredicate::True),
            Predicate::Bool(false) => Some(RefPredicate::False),
            Predicate::Int(n) => {
                // For integer literals in boolean context, treat non-zero as true
                Some(if *n != 0 {
                    RefPredicate::True
                } else {
                    RefPredicate::False
                })
            }
            Predicate::Float(_) => None, // Float literals don't make sense as predicates
            Predicate::Var(name) => {
                // Variable in boolean context - create a comparison with true
                Some(RefPredicate::Atom(Atom::new(
                    RefCompareOp::Eq,
                    Term::var(name),
                    Term::bool(true),
                )))
            }
            Predicate::Compare(op, lhs, rhs) => {
                let l_term = Self::convert_to_term(lhs)?;
                let r_term = Self::convert_to_term(rhs)?;
                let ref_op = Self::convert_compare_op(*op);
                Some(RefPredicate::Atom(Atom::new(ref_op, l_term, r_term)))
            }
            Predicate::And(lhs, rhs) => {
                let l = Self::convert_predicate(lhs)?;
                let r = Self::convert_predicate(rhs)?;
                Some(RefPredicate::and([l, r]))
            }
            Predicate::Or(lhs, rhs) => {
                let l = Self::convert_predicate(lhs)?;
                let r = Self::convert_predicate(rhs)?;
                Some(RefPredicate::or([l, r]))
            }
            Predicate::Not(inner) => {
                let p = Self::convert_predicate(inner)?;
                Some(RefPredicate::not(p))
            }
            Predicate::Implies(lhs, rhs) => {
                let l = Self::convert_predicate(lhs)?;
                let r = Self::convert_predicate(rhs)?;
                Some(RefPredicate::implies(l, r))
            }
            Predicate::Forall(var, body) => {
                let body_pred = Self::convert_predicate(body)?;
                // Default to i64 type for quantified variables
                Some(RefPredicate::Forall(
                    var.clone(),
                    Type::I64,
                    Box::new(body_pred),
                ))
            }
            Predicate::Exists(var, body) => {
                let body_pred = Self::convert_predicate(body)?;
                Some(RefPredicate::Exists(
                    var.clone(),
                    Type::I64,
                    Box::new(body_pred),
                ))
            }
            Predicate::App(name, args) => {
                let ref_args: Option<Vec<_>> =
                    args.iter().map(|a| Self::convert_to_term(a)).collect();
                Some(RefPredicate::App(name.clone(), ref_args?))
            }
            Predicate::Ite(cond, then_p, else_p) => {
                let c = Self::convert_predicate(cond)?;
                let t = Self::convert_predicate(then_p)?;
                let e = Self::convert_predicate(else_p)?;
                Some(RefPredicate::Ite(Box::new(c), Box::new(t), Box::new(e)))
            }
            Predicate::Arith(_, _, _) => {
                // Arithmetic in predicate position - not directly a predicate
                None
            }
        }
    }

    /// Convert a Predicate (used as term) to Term
    #[cfg(feature = "smt")]
    fn convert_to_term(pred: &Predicate) -> Option<Term> {
        match pred {
            Predicate::Int(n) => Some(Term::int(*n)),
            Predicate::Float(f) => Some(Term::float(*f)),
            Predicate::Bool(b) => Some(Term::bool(*b)),
            Predicate::Var(name) => Some(Term::var(name)),
            Predicate::Arith(op, lhs, rhs) => {
                let l = Self::convert_to_term(lhs)?;
                let r = Self::convert_to_term(rhs)?;
                Some(match op {
                    crate::types::refinement::ArithOp::Add => Term::add(l, r),
                    crate::types::refinement::ArithOp::Sub => Term::sub(l, r),
                    crate::types::refinement::ArithOp::Mul => Term::mul(l, r),
                    crate::types::refinement::ArithOp::Div => Term::div(l, r),
                    crate::types::refinement::ArithOp::Mod => Term::modulo(l, r),
                })
            }
            Predicate::App(name, args) => {
                let ref_args: Option<Vec<_>> =
                    args.iter().map(|a| Self::convert_to_term(a)).collect();
                Some(Term::App(name.clone(), ref_args?))
            }
            // Other predicates don't convert to terms
            _ => None,
        }
    }

    /// Convert CompareOp from types::refinement to refinement::predicate
    #[cfg(feature = "smt")]
    fn convert_compare_op(op: crate::types::refinement::CompareOp) -> RefCompareOp {
        match op {
            crate::types::refinement::CompareOp::Eq => RefCompareOp::Eq,
            crate::types::refinement::CompareOp::Ne => RefCompareOp::Ne,
            crate::types::refinement::CompareOp::Lt => RefCompareOp::Lt,
            crate::types::refinement::CompareOp::Le => RefCompareOp::Le,
            crate::types::refinement::CompareOp::Gt => RefCompareOp::Gt,
            crate::types::refinement::CompareOp::Ge => RefCompareOp::Ge,
        }
    }

    /// Try to evaluate a predicate to a constant
    fn eval_const(&self, pred: &Predicate) -> Option<i64> {
        match pred {
            Predicate::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Insert an assertion for a refinement predicate
    fn insert_refinement_assert(
        &self,
        instructions: &mut Vec<Instruction>,
        condition: ValueId,
        predicate: &Predicate,
        var_name: &str,
    ) {
        let message = format!(
            "Refinement type violation for '{}': predicate {:?} does not hold",
            var_name, predicate
        );

        let failure_mode = if self.strict {
            FailureMode::Panic
        } else {
            FailureMode::DegradeConfidence(0.7)
        };

        let assert_inst = SirInst::Assert {
            cond: condition,
            message,
            failure_mode,
        };

        instructions.push(Instruction::void(assert_inst));
    }

    /// Convert a refinement predicate to a SIR comparison instruction
    fn predicate_to_sir_cmp(&self, predicate: &Predicate, _var: ValueId) -> Option<SirInst> {
        match predicate {
            Predicate::Compare(op, _lhs, _rhs) => {
                // Map refinement compare op to SIR compare op
                let _sir_op = match op {
                    crate::types::refinement::CompareOp::Eq => CmpOp::Eq,
                    crate::types::refinement::CompareOp::Ne => CmpOp::Ne,
                    crate::types::refinement::CompareOp::Lt => CmpOp::SLt,
                    crate::types::refinement::CompareOp::Le => CmpOp::SLe,
                    crate::types::refinement::CompareOp::Gt => CmpOp::SGt,
                    crate::types::refinement::CompareOp::Ge => CmpOp::SGe,
                };

                // Try to construct SIR comparison
                // This is a simplified version - full implementation would
                // handle more complex expressions

                None // Placeholder
            }
            _ => None,
        }
    }
}

impl Default for RefinementAssertionPass {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of SMT proof attempt
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProofResult {
    /// Predicate is provably valid
    Valid,
    /// Predicate is provably invalid
    Invalid,
    /// Could not determine (need runtime check)
    Unknown,
}

/// Result of running the refinement assertion pass
#[derive(Debug, Clone, Default)]
pub struct RefinementAssertResult {
    /// Was the module modified?
    pub modified: bool,
    /// Number of assertions inserted
    pub assertions_inserted: usize,
    /// Number of proof attempts
    pub proofs_attempted: usize,
    /// Number of successful proofs (no assertion needed)
    pub proofs_succeeded: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::refinement::CompareOp;

    #[test]
    fn test_proof_trivial_true() {
        let pass = RefinementAssertionPass::new();
        let pred = Predicate::Bool(true);
        let result = pass.try_prove_predicate(&pred);
        assert_eq!(result, ProofResult::Valid);
    }

    #[test]
    fn test_proof_trivial_false() {
        let pass = RefinementAssertionPass::new();
        let pred = Predicate::Bool(false);
        let result = pass.try_prove_predicate(&pred);
        assert_eq!(result, ProofResult::Invalid);
    }

    #[test]
    fn test_proof_simple_comparison() {
        let pass = RefinementAssertionPass::new();

        // 5 > 3 should be provably valid
        let pred = Predicate::Compare(
            CompareOp::Gt,
            Box::new(Predicate::Int(5)),
            Box::new(Predicate::Int(3)),
        );
        let result = pass.try_prove_predicate(&pred);
        assert_eq!(result, ProofResult::Valid);
    }

    #[test]
    fn test_proof_simple_comparison_false() {
        let pass = RefinementAssertionPass::new();

        // 3 > 5 should be provably invalid
        let pred = Predicate::Compare(
            CompareOp::Gt,
            Box::new(Predicate::Int(3)),
            Box::new(Predicate::Int(5)),
        );
        let result = pass.try_prove_predicate(&pred);
        assert_eq!(result, ProofResult::Invalid);
    }

    #[test]
    fn test_proof_equality() {
        let pass = RefinementAssertionPass::new();

        // 5 == 5 should be valid
        let pred = Predicate::Compare(
            CompareOp::Eq,
            Box::new(Predicate::Int(5)),
            Box::new(Predicate::Int(5)),
        );
        let result = pass.try_prove_predicate(&pred);
        assert_eq!(result, ProofResult::Valid);

        // 5 == 3 should be invalid
        let pred2 = Predicate::Compare(
            CompareOp::Eq,
            Box::new(Predicate::Int(5)),
            Box::new(Predicate::Int(3)),
        );
        let result2 = pass.try_prove_predicate(&pred2);
        assert_eq!(result2, ProofResult::Invalid);
    }

    #[test]
    fn test_proof_conjunction() {
        let pass = RefinementAssertionPass::new();

        // (5 > 3) && (10 > 5) should be valid
        let pred = Predicate::And(
            Box::new(Predicate::Compare(
                CompareOp::Gt,
                Box::new(Predicate::Int(5)),
                Box::new(Predicate::Int(3)),
            )),
            Box::new(Predicate::Compare(
                CompareOp::Gt,
                Box::new(Predicate::Int(10)),
                Box::new(Predicate::Int(5)),
            )),
        );
        // With Z3 this would be provably valid; without Z3, Unknown
        let result = pass.try_prove_predicate(&pred);
        assert!(result == ProofResult::Valid || result == ProofResult::Unknown);
    }

    #[test]
    fn test_proof_negation() {
        let pass = RefinementAssertionPass::new();

        // not(5 < 3) should be valid (since 5 >= 3)
        let pred = Predicate::Not(Box::new(Predicate::Compare(
            CompareOp::Lt,
            Box::new(Predicate::Int(5)),
            Box::new(Predicate::Int(3)),
        )));
        let result = pass.try_prove_predicate(&pred);
        assert!(result == ProofResult::Valid || result == ProofResult::Unknown);
    }

    #[test]
    fn test_proof_without_smt() {
        let mut pass = RefinementAssertionPass::new();
        pass.use_smt = false;

        // Even trivial predicates should return Unknown when SMT disabled
        let pred = Predicate::And(
            Box::new(Predicate::Bool(true)),
            Box::new(Predicate::Bool(true)),
        );
        let result = pass.try_prove_predicate(&pred);
        assert_eq!(result, ProofResult::Unknown);
    }

    #[cfg(feature = "smt")]
    mod z3_tests {
        use super::*;

        #[test]
        fn test_z3_predicate_conversion() {
            // Test converting various predicate types
            let pred = Predicate::Compare(
                CompareOp::Gt,
                Box::new(Predicate::Var("x".to_string())),
                Box::new(Predicate::Int(0)),
            );

            let converted = RefinementAssertionPass::convert_predicate(&pred);
            assert!(converted.is_some());
        }

        #[test]
        fn test_z3_term_conversion() {
            // Test arithmetic term conversion
            use crate::types::refinement::ArithOp;

            let term = Predicate::Arith(
                ArithOp::Add,
                Box::new(Predicate::Int(3)),
                Box::new(Predicate::Int(4)),
            );

            let converted = RefinementAssertionPass::convert_to_term(&term);
            assert!(converted.is_some());
        }

        #[test]
        fn test_z3_complex_predicate() {
            let pass = RefinementAssertionPass::new();

            // Test: (x > 0) && (x < 10) with concrete values substituted
            // This becomes (5 > 0) && (5 < 10) which is true
            let pred = Predicate::And(
                Box::new(Predicate::Compare(
                    CompareOp::Gt,
                    Box::new(Predicate::Int(5)),
                    Box::new(Predicate::Int(0)),
                )),
                Box::new(Predicate::Compare(
                    CompareOp::Lt,
                    Box::new(Predicate::Int(5)),
                    Box::new(Predicate::Int(10)),
                )),
            );

            let result = pass.try_prove_predicate(&pred);
            // Z3 should prove this valid
            assert_eq!(result, ProofResult::Valid);
        }

        #[test]
        fn test_z3_implication() {
            let pass = RefinementAssertionPass::new();

            // false => anything should be valid (ex falso quodlibet)
            let pred = Predicate::Implies(
                Box::new(Predicate::Bool(false)),
                Box::new(Predicate::Bool(false)),
            );

            let result = pass.try_prove_predicate(&pred);
            assert_eq!(result, ProofResult::Valid);
        }
    }
}
