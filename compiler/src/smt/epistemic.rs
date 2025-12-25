//! Epistemic Property Verification
//!
//! This module provides SMT-based verification for epistemic properties
//! in the Sounio type system. It ensures that:
//!
//! - Uncertainty (epsilon) doesn't widen unexpectedly
//! - Knowledge values maintain their provenance
//! - Epistemic operations preserve required invariants
//!
//! # Epistemic Invariants
//!
//! 1. **EpsilonNonWidening**: ε_output ≤ f(ε_inputs) for well-defined f
//! 2. **BoundedUncertainty**: ε ≤ declared_bound at all points
//! 3. **ProvenanceChain**: Every Knowledge value has traceable provenance
//! 4. **ValidityMaintained**: Validity predicates are preserved

use super::formula::{SmtContext, SmtFormula, SmtSort, SmtTerm};
use super::interval::{EpsilonPropagation, Interval};
use super::solver::{MockSolver, SmtSolver, SolverError, VerificationResult};
use std::collections::HashMap;
use std::fmt;

/// Epistemic properties that can be verified
#[derive(Debug, Clone, PartialEq)]
pub enum EpistemicProperty {
    /// Epsilon doesn't increase beyond expected propagation
    EpsilonNonWidening {
        /// Input epsilon bounds
        inputs: Vec<(String, f64)>,
        /// Output epsilon variable
        output: String,
        /// Maximum allowed output epsilon
        max_output: f64,
    },

    /// Uncertainty stays within declared bounds
    BoundedUncertainty {
        /// Variable name
        var: String,
        /// Upper bound on epsilon
        bound: f64,
    },

    /// Provenance is valid and traceable
    ProvenanceValid {
        /// Variable with provenance
        var: String,
        /// Required provenance kinds
        required_kinds: Vec<String>,
    },

    /// Knowledge value satisfies validity predicate
    ValidityMaintained {
        /// Variable name
        var: String,
        /// Validity predicate
        predicate: SmtFormula,
    },

    /// Causal intervention doesn't violate d-separation
    CausalConsistency {
        /// Intervention variable
        intervention: String,
        /// Outcome variable
        outcome: String,
        /// Confounders that must be controlled
        confounders: Vec<String>,
    },

    /// Counterfactual reasoning is consistent
    CounterfactualValid {
        /// Factual context
        factual: SmtFormula,
        /// Intervention
        intervention: SmtFormula,
        /// Expected relationship between factual and counterfactual
        relationship: SmtFormula,
    },

    /// Probability normalization: sum of probabilities = 1
    ProbabilityNormalized {
        /// Probability variables that should sum to 1
        vars: Vec<String>,
    },

    /// Monotonicity: if input increases, output increases/decreases
    Monotonic {
        /// Input variable
        input: String,
        /// Output variable
        output: String,
        /// Direction: true = increasing, false = decreasing
        increasing: bool,
    },
}

impl EpistemicProperty {
    /// Convert property to SMT formula for verification
    pub fn to_formula(&self, ctx: &mut SmtContext) -> SmtFormula {
        match self {
            EpistemicProperty::EpsilonNonWidening {
                inputs,
                output,
                max_output,
            } => {
                // Declare epsilon variables
                for (var, _) in inputs {
                    ctx.declare_var(format!("eps_{}", var), SmtSort::Real);
                }
                ctx.declare_var(format!("eps_{}", output), SmtSort::Real);

                // Input bounds
                let mut constraints = Vec::new();
                for (var, bound) in inputs {
                    constraints.push(SmtFormula::And(vec![
                        SmtFormula::Ge(
                            Box::new(SmtTerm::var(format!("eps_{}", var))),
                            Box::new(SmtTerm::real(0.0)),
                        ),
                        SmtFormula::Le(
                            Box::new(SmtTerm::var(format!("eps_{}", var))),
                            Box::new(SmtTerm::real(*bound)),
                        ),
                    ]));
                }

                // Output bound
                constraints.push(SmtFormula::Le(
                    Box::new(SmtTerm::var(format!("eps_{}", output))),
                    Box::new(SmtTerm::real(*max_output)),
                ));

                SmtFormula::And(constraints)
            }

            EpistemicProperty::BoundedUncertainty { var, bound } => {
                ctx.declare_var(format!("eps_{}", var), SmtSort::Real);

                SmtFormula::And(vec![
                    SmtFormula::Ge(
                        Box::new(SmtTerm::var(format!("eps_{}", var))),
                        Box::new(SmtTerm::real(0.0)),
                    ),
                    SmtFormula::Le(
                        Box::new(SmtTerm::var(format!("eps_{}", var))),
                        Box::new(SmtTerm::real(*bound)),
                    ),
                ])
            }

            EpistemicProperty::ProvenanceValid {
                var,
                required_kinds,
            } => {
                // Provenance as uninterpreted sort
                ctx.declare_sort("ProvenanceKind".to_string());
                ctx.declare_var(format!("prov_{}", var), SmtSort::Provenance);

                // Create disjunction: prov_var = k1 ∨ prov_var = k2 ∨ ...
                let kind_constraints: Vec<_> = required_kinds
                    .iter()
                    .map(|kind| {
                        SmtFormula::Eq(
                            Box::new(SmtTerm::var(format!("prov_{}", var))),
                            Box::new(SmtTerm::Provenance(kind.clone())),
                        )
                    })
                    .collect();

                if kind_constraints.is_empty() {
                    SmtFormula::True
                } else {
                    SmtFormula::Or(kind_constraints)
                }
            }

            EpistemicProperty::ValidityMaintained { var: _, predicate } => predicate.clone(),

            EpistemicProperty::CausalConsistency {
                intervention,
                outcome,
                confounders,
            } => {
                ctx.declare_var(intervention.clone(), SmtSort::Real);
                ctx.declare_var(outcome.clone(), SmtSort::Real);
                for c in confounders {
                    ctx.declare_var(c.clone(), SmtSort::Real);
                }

                // Basic consistency: intervention affects outcome only through allowed paths
                // This is a placeholder - full causal reasoning requires a causal graph
                SmtFormula::True
            }

            EpistemicProperty::CounterfactualValid {
                factual,
                intervention,
                relationship,
            } => SmtFormula::Implies(
                Box::new(SmtFormula::And(vec![factual.clone(), intervention.clone()])),
                Box::new(relationship.clone()),
            ),

            EpistemicProperty::ProbabilityNormalized { vars } => {
                for var in vars {
                    ctx.declare_var(var.clone(), SmtSort::Real);
                }

                // All probabilities in [0, 1]
                let mut constraints: Vec<_> = vars
                    .iter()
                    .map(|var| {
                        SmtFormula::And(vec![
                            SmtFormula::Ge(
                                Box::new(SmtTerm::var(var)),
                                Box::new(SmtTerm::real(0.0)),
                            ),
                            SmtFormula::Le(
                                Box::new(SmtTerm::var(var)),
                                Box::new(SmtTerm::real(1.0)),
                            ),
                        ])
                    })
                    .collect();

                // Sum = 1
                if !vars.is_empty() {
                    let sum = vars
                        .iter()
                        .skip(1)
                        .fold(SmtTerm::var(&vars[0]), |acc, var| {
                            SmtTerm::Add(Box::new(acc), Box::new(SmtTerm::var(var)))
                        });
                    constraints.push(SmtFormula::Eq(Box::new(sum), Box::new(SmtTerm::real(1.0))));
                }

                SmtFormula::And(constraints)
            }

            EpistemicProperty::Monotonic {
                input,
                output,
                increasing,
            } => {
                ctx.declare_var(input.clone(), SmtSort::Real);
                ctx.declare_var(format!("{}_prime", input), SmtSort::Real);
                ctx.declare_var(output.clone(), SmtSort::Real);
                ctx.declare_var(format!("{}_prime", output), SmtSort::Real);

                let input_relation = SmtFormula::Lt(
                    Box::new(SmtTerm::var(input)),
                    Box::new(SmtTerm::var(format!("{}_prime", input))),
                );

                let output_relation = if *increasing {
                    SmtFormula::Le(
                        Box::new(SmtTerm::var(output)),
                        Box::new(SmtTerm::var(format!("{}_prime", output))),
                    )
                } else {
                    SmtFormula::Ge(
                        Box::new(SmtTerm::var(output)),
                        Box::new(SmtTerm::var(format!("{}_prime", output))),
                    )
                };

                SmtFormula::Implies(Box::new(input_relation), Box::new(output_relation))
            }
        }
    }
}

impl fmt::Display for EpistemicProperty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EpistemicProperty::EpsilonNonWidening {
                inputs,
                output,
                max_output,
            } => {
                write!(
                    f,
                    "EpsilonNonWidening({:?} → {} ≤ {})",
                    inputs, output, max_output
                )
            }
            EpistemicProperty::BoundedUncertainty { var, bound } => {
                write!(f, "BoundedUncertainty({} ≤ {})", var, bound)
            }
            EpistemicProperty::ProvenanceValid {
                var,
                required_kinds,
            } => {
                write!(f, "ProvenanceValid({} ∈ {:?})", var, required_kinds)
            }
            EpistemicProperty::ValidityMaintained { var, predicate } => {
                write!(f, "ValidityMaintained({}: {})", var, predicate)
            }
            EpistemicProperty::CausalConsistency {
                intervention,
                outcome,
                confounders,
            } => {
                write!(
                    f,
                    "CausalConsistency({} → {} | {:?})",
                    intervention, outcome, confounders
                )
            }
            EpistemicProperty::CounterfactualValid { .. } => {
                write!(f, "CounterfactualValid(...)")
            }
            EpistemicProperty::ProbabilityNormalized { vars } => {
                write!(f, "ProbabilityNormalized({:?})", vars)
            }
            EpistemicProperty::Monotonic {
                input,
                output,
                increasing,
            } => {
                let arrow = if *increasing { "↑" } else { "↓" };
                write!(f, "Monotonic({} {} {})", input, arrow, output)
            }
        }
    }
}

/// Constraint on epistemic values (for constraint collection)
#[derive(Debug, Clone)]
pub struct EpistemicConstraint {
    /// Source location (for error reporting)
    pub location: String,
    /// The property to verify
    pub property: EpistemicProperty,
    /// Optional context (assumptions)
    pub assumptions: Vec<SmtFormula>,
}

impl EpistemicConstraint {
    /// Create a new constraint
    pub fn new(location: impl Into<String>, property: EpistemicProperty) -> Self {
        Self {
            location: location.into(),
            property,
            assumptions: Vec::new(),
        }
    }

    /// Add an assumption
    pub fn with_assumption(mut self, assumption: SmtFormula) -> Self {
        self.assumptions.push(assumption);
        self
    }
}

/// Verifier for epistemic properties
pub struct EpistemicVerifier {
    /// SMT solver to use
    solver: Box<dyn SmtSolver>,
    /// Collected constraints
    constraints: Vec<EpistemicConstraint>,
    /// Known epsilon bounds
    epsilon_bounds: HashMap<String, f64>,
    /// Verification results cache
    results: HashMap<String, VerificationResult>,
}

impl EpistemicVerifier {
    /// Create a new verifier
    pub fn new() -> Self {
        Self {
            solver: Box::new(MockSolver::new()),
            constraints: Vec::new(),
            epsilon_bounds: HashMap::new(),
            results: HashMap::new(),
        }
    }

    /// Create a verifier with a specific solver
    pub fn with_solver(solver: Box<dyn SmtSolver>) -> Self {
        Self {
            solver,
            constraints: Vec::new(),
            epsilon_bounds: HashMap::new(),
            results: HashMap::new(),
        }
    }

    /// Register an epsilon bound for a variable
    pub fn register_epsilon_bound(&mut self, var: &str, bound: f64) {
        self.epsilon_bounds.insert(var.to_string(), bound);
    }

    /// Add a constraint to verify
    pub fn add_constraint(&mut self, constraint: EpistemicConstraint) {
        self.constraints.push(constraint);
    }

    /// Add an epsilon non-widening constraint
    pub fn add_epsilon_nonwidening(
        &mut self,
        location: &str,
        inputs: Vec<(&str, f64)>,
        output: &str,
        max_output: f64,
    ) {
        let inputs: Vec<_> = inputs
            .into_iter()
            .map(|(s, f)| (s.to_string(), f))
            .collect();

        self.add_constraint(EpistemicConstraint::new(
            location,
            EpistemicProperty::EpsilonNonWidening {
                inputs,
                output: output.to_string(),
                max_output,
            },
        ));
    }

    /// Add a bounded uncertainty constraint
    pub fn add_bounded_uncertainty(&mut self, location: &str, var: &str, bound: f64) {
        self.add_constraint(EpistemicConstraint::new(
            location,
            EpistemicProperty::BoundedUncertainty {
                var: var.to_string(),
                bound,
            },
        ));
    }

    /// Verify a single property
    pub fn verify_property(
        &mut self,
        property: &EpistemicProperty,
    ) -> Result<VerificationResult, SolverError> {
        let mut ctx = SmtContext::new();
        let formula = property.to_formula(&mut ctx);
        self.solver.check_valid(&formula)
    }

    /// Verify all collected constraints
    pub fn verify_all(&mut self) -> Vec<(EpistemicConstraint, VerificationResult)> {
        let constraints = std::mem::take(&mut self.constraints);
        let mut results = Vec::new();

        for constraint in constraints {
            let mut ctx = SmtContext::new();

            // Add assumptions
            for assumption in &constraint.assumptions {
                ctx.assert(assumption.clone());
            }

            let formula = constraint.property.to_formula(&mut ctx);

            let result = match self.solver.check_valid(&formula) {
                Ok(r) => r,
                Err(e) => VerificationResult::Error(e.to_string()),
            };

            results.push((constraint, result));
        }

        results
    }

    /// Check epsilon propagation through an arithmetic operation
    pub fn check_epsilon_arithmetic(
        &self,
        op: &str,
        input_epsilons: &[f64],
        input_values: &[Interval],
        output_bound: f64,
    ) -> bool {
        let result_epsilon = match op {
            "add" | "sub" => input_epsilons.iter().sum::<f64>(),
            "mul" if input_epsilons.len() == 2 && input_values.len() == 2 => {
                EpsilonPropagation::mul(
                    input_values[0],
                    input_epsilons[0],
                    input_values[1],
                    input_epsilons[1],
                )
            }
            "div" if input_epsilons.len() == 2 && input_values.len() == 2 => {
                EpsilonPropagation::div(
                    input_values[0],
                    input_epsilons[0],
                    input_values[1],
                    input_epsilons[1],
                )
            }
            _ => f64::INFINITY, // Unknown operation
        };

        result_epsilon <= output_bound
    }

    /// Get statistics about verified constraints
    pub fn statistics(&self) -> VerificationStats {
        let mut stats = VerificationStats::default();

        for result in self.results.values() {
            match result {
                VerificationResult::Sat => stats.valid += 1,
                VerificationResult::Unsat => stats.invalid += 1,
                VerificationResult::Unknown => stats.unknown += 1,
                VerificationResult::Timeout => stats.timeout += 1,
                VerificationResult::Error(_) => stats.error += 1,
            }
        }

        stats.total = self.results.len();
        stats
    }
}

impl Default for EpistemicVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about verification results
#[derive(Debug, Clone, Default)]
pub struct VerificationStats {
    /// Total constraints checked
    pub total: usize,
    /// Constraints proven valid
    pub valid: usize,
    /// Constraints proven invalid
    pub invalid: usize,
    /// Constraints with unknown result
    pub unknown: usize,
    /// Constraints that timed out
    pub timeout: usize,
    /// Constraints that caused errors
    pub error: usize,
}

impl fmt::Display for VerificationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Verification: {} total, {} valid, {} invalid, {} unknown, {} timeout, {} error",
            self.total, self.valid, self.invalid, self.unknown, self.timeout, self.error
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_uncertainty_property() {
        let property = EpistemicProperty::BoundedUncertainty {
            var: "x".to_string(),
            bound: 0.01,
        };

        let mut ctx = SmtContext::new();
        let formula = property.to_formula(&mut ctx);

        // Should be a conjunction of eps_x >= 0 and eps_x <= 0.01
        assert!(matches!(formula, SmtFormula::And(_)));
    }

    #[test]
    fn test_epsilon_nonwidening_property() {
        let property = EpistemicProperty::EpsilonNonWidening {
            inputs: vec![("a".to_string(), 0.01), ("b".to_string(), 0.02)],
            output: "c".to_string(),
            max_output: 0.03,
        };

        let mut ctx = SmtContext::new();
        let formula = property.to_formula(&mut ctx);

        assert!(matches!(formula, SmtFormula::And(_)));
    }

    #[test]
    fn test_probability_normalized_property() {
        let property = EpistemicProperty::ProbabilityNormalized {
            vars: vec!["p1".to_string(), "p2".to_string(), "p3".to_string()],
        };

        let mut ctx = SmtContext::new();
        let formula = property.to_formula(&mut ctx);

        assert!(matches!(formula, SmtFormula::And(_)));
    }

    #[test]
    fn test_verifier_epsilon_check() {
        let verifier = EpistemicVerifier::new();

        // Addition: ε1 + ε2 should be <= 0.03 when ε1 = 0.01, ε2 = 0.02
        let result = verifier.check_epsilon_arithmetic(
            "add",
            &[0.01, 0.02],
            &[Interval::point(1.0), Interval::point(2.0)],
            0.03,
        );
        assert!(result);

        // Should fail if bound is too tight
        let result = verifier.check_epsilon_arithmetic(
            "add",
            &[0.01, 0.02],
            &[Interval::point(1.0), Interval::point(2.0)],
            0.02, // Too tight
        );
        assert!(!result);
    }

    #[test]
    fn test_monotonicity_property() {
        let property = EpistemicProperty::Monotonic {
            input: "dose".to_string(),
            output: "response".to_string(),
            increasing: true,
        };

        let display = format!("{}", property);
        assert!(display.contains("dose"));
        assert!(display.contains("response"));
        assert!(display.contains("↑"));
    }

    #[test]
    fn test_verification_stats() {
        let stats = VerificationStats {
            total: 10,
            valid: 7,
            invalid: 1,
            unknown: 2,
            timeout: 0,
            error: 0,
        };

        let display = format!("{}", stats);
        assert!(display.contains("10 total"));
        assert!(display.contains("7 valid"));
    }
}
