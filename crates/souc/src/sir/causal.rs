//! Causal inference operations for SIR
//!
//! Provides lowering helpers for converting causal HIR expressions
//! into SIR causal operations, and analysis utilities for causal
//! operation sequences.
//!
//! # Three-Step Counterfactual Algorithm
//!
//! Pearl's counterfactual algorithm proceeds in three steps:
//!
//! 1. **Abduction**: Infer exogenous variables U from observed data.
//!    For linear structural equations: U = X - f(Pa(X)).
//!    For nonlinear models: Newton-Raphson or similar iteration.
//!
//! 2. **Action**: Apply the do(X=v) intervention by modifying the
//!    structural equations (removing incoming edges to X, setting X=v).
//!
//! 3. **Prediction**: Forward-propagate through the modified structural
//!    model using the inferred exogenous variables to compute the
//!    counterfactual outcome.
//!
//! Reference: Pearl, J. (2009). Causality. Cambridge University Press,
//!            Chapter 7 "The Logic of Counterfactuals".

use super::ops::{CausalFormulaKind, CausalOp};
use super::values::ValueId;
use crate::types::causal_id::IdentificationFormula;

// ---------------------------------------------------------------------------
// Formula conversion
// ---------------------------------------------------------------------------

/// Convert an `IdentificationFormula` to a `CausalFormulaKind`.
///
/// Maps the type-level identification formula (from the ID algorithm at
/// compile time) to the runtime formula kind used in SIR operations.
/// `Product` formulas are mapped to `BackdoorAdjustment` as a conservative
/// default since they represent composite identification strategies.
pub fn formula_to_sir_kind(formula: &IdentificationFormula) -> CausalFormulaKind {
    match formula {
        IdentificationFormula::Marginal { .. } => CausalFormulaKind::Marginal,
        IdentificationFormula::Adjustment { .. } => CausalFormulaKind::BackdoorAdjustment,
        IdentificationFormula::Frontdoor { .. } => CausalFormulaKind::FrontdoorAdjustment,
        IdentificationFormula::InstrumentalVariable { .. } => {
            CausalFormulaKind::InstrumentalVariable
        }
        IdentificationFormula::Conditional { .. } => CausalFormulaKind::Conditional,
        IdentificationFormula::Product { .. } => {
            // Product formulas are composite; use backdoor as conservative default
            CausalFormulaKind::BackdoorAdjustment
        }
    }
}

// ---------------------------------------------------------------------------
// Lowering helpers
// ---------------------------------------------------------------------------

/// Build the SIR operations for a counterfactual query.
///
/// Returns the three-step sequence: [Abduction, DoIntervention, Prediction].
/// This is the canonical lowering of a `counterfactual { ... }` HIR expression
/// into the SIR causal operation sequence.
///
/// # Arguments
///
/// * `factual_values` - ValueIds for the factual observations
/// * `intervention_var` - Name of the variable being intervened on
/// * `intervention_value` - ValueId for the intervention value
/// * `outcome_var` - Name of the outcome variable to query
/// * `result` - ValueId for the final result
pub fn lower_counterfactual(
    factual_values: Vec<ValueId>,
    intervention_var: String,
    intervention_value: ValueId,
    outcome_var: String,
    result: ValueId,
) -> Vec<CausalOp> {
    // Allocate intermediate ValueIds for the three-step protocol.
    // The caller is responsible for allocating real ValueIds; here we use
    // offsets from the result id to generate distinct temporaries.
    let abduction_result = ValueId::new(result.0 + 1000);
    let post_state = ValueId::new(result.0 + 1001);

    vec![
        // Step 1: Abduction — infer exogenous variables from observations
        CausalOp::Abduction {
            observed: factual_values,
            coefficients: vec![], // Linear coefficients determined at runtime
            result: abduction_result,
        },
        // Step 2: Action — apply intervention
        CausalOp::DoIntervention {
            target_var: intervention_var,
            value: intervention_value,
            pre_state: abduction_result,
            post_state,
        },
        // Step 3: Prediction — forward-propagate through modified model
        CausalOp::Prediction {
            exogenous: abduction_result,
            intervention_state: post_state,
            outcome_var,
            result,
        },
    ]
}

/// Build SIR operations for a `do()` expression.
///
/// Uses the identification formula from the ID algorithm to determine
/// which causal estimation strategy to emit. The formula is determined
/// at compile time; this function translates it into executable SIR ops.
///
/// # Arguments
///
/// * `formula` - The identification formula from compile-time analysis
/// * `treatment_value` - ValueId for the treatment/intervention value
/// * `outcome` - ValueId for the outcome variable
/// * `adjustment_values` - ValueIds for adjustment set variables
/// * `result` - ValueId for the result
pub fn lower_do_expression(
    formula: &IdentificationFormula,
    treatment_value: ValueId,
    outcome: ValueId,
    adjustment_values: Vec<ValueId>,
    result: ValueId,
) -> Vec<CausalOp> {
    let kind = formula_to_sir_kind(formula);

    vec![CausalOp::EstimateEffect {
        formula_kind: kind,
        treatment: treatment_value,
        outcome,
        adjustment_set: adjustment_values,
        result,
    }]
}

// ---------------------------------------------------------------------------
// Analysis utilities
// ---------------------------------------------------------------------------

/// Estimate the computational cost of a causal operation.
///
/// Returns a cost in abstract units, useful for optimization decisions.
/// Higher costs indicate more expensive operations.
///
/// Cost model:
/// - Abduction: 100 base + 10 per observed variable (may require iteration)
/// - DoIntervention: 10 (graph mutation is cheap)
/// - Prediction: 50 (forward propagation)
/// - CounterfactualQuery: 200 (combines all three steps)
/// - EstimateEffect: varies by formula kind
/// - DSeparationTest: 30 + 5 per conditioning variable (graph traversal)
pub fn causal_op_cost(op: &CausalOp) -> u64 {
    match op {
        CausalOp::Abduction { observed, .. } => 100 + 10 * observed.len() as u64,
        CausalOp::DoIntervention { .. } => 10,
        CausalOp::Prediction { .. } => 50,
        CausalOp::CounterfactualQuery {
            factual_observations,
            ..
        } => 200 + 10 * factual_observations.len() as u64,
        CausalOp::EstimateEffect {
            formula_kind,
            adjustment_set,
            ..
        } => {
            let base = match formula_kind {
                CausalFormulaKind::Marginal => 20,
                CausalFormulaKind::Conditional => 30,
                CausalFormulaKind::BackdoorAdjustment => 80,
                CausalFormulaKind::FrontdoorAdjustment => 120,
                CausalFormulaKind::InstrumentalVariable => 60,
            };
            base + 10 * adjustment_set.len() as u64
        }
        CausalOp::DSeparationTest { conditioning, .. } => 30 + 5 * conditioning.len() as u64,
    }
}

/// Check if a causal operation requires iteration (Newton-Raphson).
///
/// Currently only `Abduction` may require iteration when the structural
/// equations are nonlinear. All other operations are single-pass.
pub fn requires_iteration(op: &CausalOp) -> bool {
    matches!(op, CausalOp::Abduction { .. })
}

/// Validate that a sequence of causal ops follows the three-step protocol.
///
/// A valid three-step counterfactual sequence is:
///   1. Exactly one `Abduction` as the first operation
///   2. Exactly one `DoIntervention` as the second operation
///   3. Exactly one `Prediction` as the third operation
///
/// Returns `Ok(())` if the sequence is valid, or `Err(message)` describing
/// the validation failure.
pub fn validate_three_step_sequence(ops: &[CausalOp]) -> Result<(), String> {
    if ops.len() != 3 {
        return Err(format!(
            "three-step protocol requires exactly 3 operations, got {}",
            ops.len()
        ));
    }

    if !matches!(ops[0], CausalOp::Abduction { .. }) {
        return Err(format!(
            "step 1 must be Abduction, got {:?}",
            std::mem::discriminant(&ops[0])
        ));
    }

    if !matches!(ops[1], CausalOp::DoIntervention { .. }) {
        return Err(format!(
            "step 2 must be DoIntervention, got {:?}",
            std::mem::discriminant(&ops[1])
        ));
    }

    if !matches!(ops[2], CausalOp::Prediction { .. }) {
        return Err(format!(
            "step 3 must be Prediction, got {:?}",
            std::mem::discriminant(&ops[2])
        ));
    }

    Ok(())
}

/// Counts of causal operations by type.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CausalOpCounts {
    pub abductions: usize,
    pub interventions: usize,
    pub predictions: usize,
    pub counterfactual_queries: usize,
    pub effect_estimations: usize,
    pub d_separation_tests: usize,
}

/// Count the operations by type in a causal program.
///
/// Useful for profiling and optimization pass metrics.
pub fn count_causal_ops(ops: &[CausalOp]) -> CausalOpCounts {
    let mut counts = CausalOpCounts::default();

    for op in ops {
        match op {
            CausalOp::Abduction { .. } => counts.abductions += 1,
            CausalOp::DoIntervention { .. } => counts.interventions += 1,
            CausalOp::Prediction { .. } => counts.predictions += 1,
            CausalOp::CounterfactualQuery { .. } => counts.counterfactual_queries += 1,
            CausalOp::EstimateEffect { .. } => counts.effect_estimations += 1,
            CausalOp::DSeparationTest { .. } => counts.d_separation_tests += 1,
        }
    }

    counts
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::causal_id::IdentificationFormula;

    // -----------------------------------------------------------------------
    // Helper: create ValueIds for testing
    // -----------------------------------------------------------------------

    fn vid(n: u32) -> ValueId {
        ValueId::new(n)
    }

    // -----------------------------------------------------------------------
    // formula_to_sir_kind tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_formula_to_sir_kind_marginal() {
        let formula = IdentificationFormula::Marginal {
            outcome: vec!["Y".into()],
        };
        assert_eq!(formula_to_sir_kind(&formula), CausalFormulaKind::Marginal);
    }

    #[test]
    fn test_formula_to_sir_kind_adjustment() {
        let formula = IdentificationFormula::Adjustment {
            outcome: vec!["Y".into()],
            treatment: vec!["X".into()],
            adjustment_set: vec!["Z".into()],
        };
        assert_eq!(
            formula_to_sir_kind(&formula),
            CausalFormulaKind::BackdoorAdjustment
        );
    }

    #[test]
    fn test_formula_to_sir_kind_frontdoor() {
        let formula = IdentificationFormula::Frontdoor {
            treatment: vec!["X".into()],
            outcome: vec!["Y".into()],
            mediator: vec!["M".into()],
        };
        assert_eq!(
            formula_to_sir_kind(&formula),
            CausalFormulaKind::FrontdoorAdjustment
        );
    }

    #[test]
    fn test_formula_to_sir_kind_iv() {
        let formula = IdentificationFormula::InstrumentalVariable {
            instrument: vec!["Z".into()],
            treatment: vec!["X".into()],
            outcome: vec!["Y".into()],
        };
        assert_eq!(
            formula_to_sir_kind(&formula),
            CausalFormulaKind::InstrumentalVariable
        );
    }

    #[test]
    fn test_formula_to_sir_kind_conditional() {
        let formula = IdentificationFormula::Conditional {
            outcome: vec!["Y".into()],
            treatment: vec!["X".into()],
        };
        assert_eq!(
            formula_to_sir_kind(&formula),
            CausalFormulaKind::Conditional
        );
    }

    #[test]
    fn test_formula_to_sir_kind_product() {
        let formula = IdentificationFormula::Product {
            factors: vec![
                IdentificationFormula::Marginal {
                    outcome: vec!["Y1".into()],
                },
                IdentificationFormula::Conditional {
                    outcome: vec!["Y2".into()],
                    treatment: vec!["X2".into()],
                },
            ],
        };
        // Product falls back to BackdoorAdjustment
        assert_eq!(
            formula_to_sir_kind(&formula),
            CausalFormulaKind::BackdoorAdjustment
        );
    }

    // -----------------------------------------------------------------------
    // lower_counterfactual tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lower_counterfactual() {
        let ops = lower_counterfactual(
            vec![vid(1), vid(2)],
            "X".to_string(),
            vid(3),
            "Y".to_string(),
            vid(10),
        );

        assert_eq!(ops.len(), 3);
        assert!(matches!(ops[0], CausalOp::Abduction { .. }));
        assert!(matches!(ops[1], CausalOp::DoIntervention { .. }));
        assert!(matches!(ops[2], CausalOp::Prediction { .. }));
    }

    #[test]
    fn test_lower_counterfactual_preserves_vars() {
        let ops = lower_counterfactual(
            vec![vid(1)],
            "Treatment".to_string(),
            vid(5),
            "Outcome".to_string(),
            vid(20),
        );

        // Check that intervention variable name is preserved
        if let CausalOp::DoIntervention { target_var, .. } = &ops[1] {
            assert_eq!(target_var, "Treatment");
        } else {
            panic!("expected DoIntervention at index 1");
        }

        // Check that outcome variable name is preserved
        if let CausalOp::Prediction { outcome_var, .. } = &ops[2] {
            assert_eq!(outcome_var, "Outcome");
        } else {
            panic!("expected Prediction at index 2");
        }

        // Check that the final result id is preserved
        if let CausalOp::Prediction { result, .. } = &ops[2] {
            assert_eq!(*result, vid(20));
        }
    }

    #[test]
    fn test_lower_counterfactual_validates() {
        let ops = lower_counterfactual(
            vec![vid(1), vid(2)],
            "X".to_string(),
            vid(3),
            "Y".to_string(),
            vid(10),
        );

        // The output of lower_counterfactual should always be a valid three-step sequence
        assert!(validate_three_step_sequence(&ops).is_ok());
    }

    // -----------------------------------------------------------------------
    // lower_do_expression tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lower_do_backdoor() {
        let formula = IdentificationFormula::Adjustment {
            outcome: vec!["Y".into()],
            treatment: vec!["X".into()],
            adjustment_set: vec!["Z".into()],
        };

        let ops = lower_do_expression(&formula, vid(1), vid(2), vec![vid(3)], vid(10));

        assert_eq!(ops.len(), 1);
        if let CausalOp::EstimateEffect {
            formula_kind,
            adjustment_set,
            ..
        } = &ops[0]
        {
            assert_eq!(*formula_kind, CausalFormulaKind::BackdoorAdjustment);
            assert_eq!(adjustment_set.len(), 1);
        } else {
            panic!("expected EstimateEffect");
        }
    }

    #[test]
    fn test_lower_do_frontdoor() {
        let formula = IdentificationFormula::Frontdoor {
            treatment: vec!["X".into()],
            outcome: vec!["Y".into()],
            mediator: vec!["M".into()],
        };

        let ops = lower_do_expression(&formula, vid(1), vid(2), vec![vid(3)], vid(10));

        assert_eq!(ops.len(), 1);
        if let CausalOp::EstimateEffect { formula_kind, .. } = &ops[0] {
            assert_eq!(*formula_kind, CausalFormulaKind::FrontdoorAdjustment);
        } else {
            panic!("expected EstimateEffect");
        }
    }

    #[test]
    fn test_lower_do_conditional() {
        let formula = IdentificationFormula::Conditional {
            outcome: vec!["Y".into()],
            treatment: vec!["X".into()],
        };

        let ops = lower_do_expression(&formula, vid(1), vid(2), vec![], vid(10));

        assert_eq!(ops.len(), 1);
        if let CausalOp::EstimateEffect {
            formula_kind,
            adjustment_set,
            ..
        } = &ops[0]
        {
            assert_eq!(*formula_kind, CausalFormulaKind::Conditional);
            assert!(adjustment_set.is_empty());
        } else {
            panic!("expected EstimateEffect");
        }
    }

    // -----------------------------------------------------------------------
    // causal_op_cost tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_causal_op_cost_abduction() {
        let op = CausalOp::Abduction {
            observed: vec![vid(1), vid(2), vid(3)],
            coefficients: vec![1.0, 2.0, 3.0],
            result: vid(10),
        };
        // 100 + 10 * 3 = 130
        assert_eq!(causal_op_cost(&op), 130);
    }

    #[test]
    fn test_causal_op_cost_do_intervention() {
        let op = CausalOp::DoIntervention {
            target_var: "X".to_string(),
            value: vid(1),
            pre_state: vid(2),
            post_state: vid(3),
        };
        assert_eq!(causal_op_cost(&op), 10);
    }

    #[test]
    fn test_causal_op_cost_prediction() {
        let op = CausalOp::Prediction {
            exogenous: vid(1),
            intervention_state: vid(2),
            outcome_var: "Y".to_string(),
            result: vid(10),
        };
        assert_eq!(causal_op_cost(&op), 50);
    }

    #[test]
    fn test_causal_op_cost_counterfactual() {
        let op = CausalOp::CounterfactualQuery {
            factual_observations: vec![vid(1), vid(2)],
            intervention_var: "X".to_string(),
            intervention_value: vid(3),
            outcome_var: "Y".to_string(),
            result: vid(10),
        };
        // 200 + 10 * 2 = 220
        assert_eq!(causal_op_cost(&op), 220);
    }

    #[test]
    fn test_causal_op_cost_estimate_backdoor() {
        let op = CausalOp::EstimateEffect {
            formula_kind: CausalFormulaKind::BackdoorAdjustment,
            treatment: vid(1),
            outcome: vid(2),
            adjustment_set: vec![vid(3), vid(4)],
            result: vid(10),
        };
        // 80 + 10 * 2 = 100
        assert_eq!(causal_op_cost(&op), 100);
    }

    #[test]
    fn test_causal_op_cost_estimate_marginal() {
        let op = CausalOp::EstimateEffect {
            formula_kind: CausalFormulaKind::Marginal,
            treatment: vid(1),
            outcome: vid(2),
            adjustment_set: vec![],
            result: vid(10),
        };
        assert_eq!(causal_op_cost(&op), 20);
    }

    #[test]
    fn test_causal_op_cost_d_separation() {
        let op = CausalOp::DSeparationTest {
            source: "A".to_string(),
            target: "B".to_string(),
            conditioning: vec!["C".to_string(), "D".to_string()],
            result: vid(10),
        };
        // 30 + 5 * 2 = 40
        assert_eq!(causal_op_cost(&op), 40);
    }

    // -----------------------------------------------------------------------
    // requires_iteration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_requires_iteration_abduction() {
        let op = CausalOp::Abduction {
            observed: vec![vid(1)],
            coefficients: vec![1.0],
            result: vid(10),
        };
        assert!(requires_iteration(&op));
    }

    #[test]
    fn test_requires_iteration_do_intervention() {
        let op = CausalOp::DoIntervention {
            target_var: "X".to_string(),
            value: vid(1),
            pre_state: vid(2),
            post_state: vid(3),
        };
        assert!(!requires_iteration(&op));
    }

    #[test]
    fn test_requires_iteration_prediction() {
        let op = CausalOp::Prediction {
            exogenous: vid(1),
            intervention_state: vid(2),
            outcome_var: "Y".to_string(),
            result: vid(10),
        };
        assert!(!requires_iteration(&op));
    }

    #[test]
    fn test_requires_iteration_estimate() {
        let op = CausalOp::EstimateEffect {
            formula_kind: CausalFormulaKind::BackdoorAdjustment,
            treatment: vid(1),
            outcome: vid(2),
            adjustment_set: vec![vid(3)],
            result: vid(10),
        };
        assert!(!requires_iteration(&op));
    }

    // -----------------------------------------------------------------------
    // validate_three_step_sequence tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_three_step_valid() {
        let ops = vec![
            CausalOp::Abduction {
                observed: vec![vid(1)],
                coefficients: vec![],
                result: vid(10),
            },
            CausalOp::DoIntervention {
                target_var: "X".to_string(),
                value: vid(2),
                pre_state: vid(10),
                post_state: vid(11),
            },
            CausalOp::Prediction {
                exogenous: vid(10),
                intervention_state: vid(11),
                outcome_var: "Y".to_string(),
                result: vid(20),
            },
        ];
        assert!(validate_three_step_sequence(&ops).is_ok());
    }

    #[test]
    fn test_validate_three_step_invalid_length_too_short() {
        let ops = vec![CausalOp::Abduction {
            observed: vec![vid(1)],
            coefficients: vec![],
            result: vid(10),
        }];
        let result = validate_three_step_sequence(&ops);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exactly 3"));
    }

    #[test]
    fn test_validate_three_step_invalid_length_too_long() {
        let ops = vec![
            CausalOp::Abduction {
                observed: vec![vid(1)],
                coefficients: vec![],
                result: vid(10),
            },
            CausalOp::DoIntervention {
                target_var: "X".to_string(),
                value: vid(2),
                pre_state: vid(10),
                post_state: vid(11),
            },
            CausalOp::Prediction {
                exogenous: vid(10),
                intervention_state: vid(11),
                outcome_var: "Y".to_string(),
                result: vid(20),
            },
            CausalOp::Prediction {
                exogenous: vid(10),
                intervention_state: vid(11),
                outcome_var: "Z".to_string(),
                result: vid(21),
            },
        ];
        let result = validate_three_step_sequence(&ops);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exactly 3"));
    }

    #[test]
    fn test_validate_three_step_invalid_wrong_first() {
        let ops = vec![
            CausalOp::DoIntervention {
                target_var: "X".to_string(),
                value: vid(2),
                pre_state: vid(10),
                post_state: vid(11),
            },
            CausalOp::DoIntervention {
                target_var: "X".to_string(),
                value: vid(2),
                pre_state: vid(10),
                post_state: vid(11),
            },
            CausalOp::Prediction {
                exogenous: vid(10),
                intervention_state: vid(11),
                outcome_var: "Y".to_string(),
                result: vid(20),
            },
        ];
        let result = validate_three_step_sequence(&ops);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("step 1 must be Abduction"));
    }

    #[test]
    fn test_validate_three_step_invalid_wrong_second() {
        let ops = vec![
            CausalOp::Abduction {
                observed: vec![vid(1)],
                coefficients: vec![],
                result: vid(10),
            },
            CausalOp::Prediction {
                exogenous: vid(10),
                intervention_state: vid(11),
                outcome_var: "Y".to_string(),
                result: vid(20),
            },
            CausalOp::Prediction {
                exogenous: vid(10),
                intervention_state: vid(11),
                outcome_var: "Y".to_string(),
                result: vid(20),
            },
        ];
        let result = validate_three_step_sequence(&ops);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("step 2 must be DoIntervention")
        );
    }

    #[test]
    fn test_validate_three_step_invalid_wrong_third() {
        let ops = vec![
            CausalOp::Abduction {
                observed: vec![vid(1)],
                coefficients: vec![],
                result: vid(10),
            },
            CausalOp::DoIntervention {
                target_var: "X".to_string(),
                value: vid(2),
                pre_state: vid(10),
                post_state: vid(11),
            },
            CausalOp::Abduction {
                observed: vec![vid(1)],
                coefficients: vec![],
                result: vid(10),
            },
        ];
        let result = validate_three_step_sequence(&ops);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("step 3 must be Prediction"));
    }

    // -----------------------------------------------------------------------
    // count_causal_ops tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_count_causal_ops() {
        let ops = vec![
            CausalOp::Abduction {
                observed: vec![vid(1)],
                coefficients: vec![],
                result: vid(10),
            },
            CausalOp::DoIntervention {
                target_var: "X".to_string(),
                value: vid(2),
                pre_state: vid(10),
                post_state: vid(11),
            },
            CausalOp::Prediction {
                exogenous: vid(10),
                intervention_state: vid(11),
                outcome_var: "Y".to_string(),
                result: vid(20),
            },
            CausalOp::EstimateEffect {
                formula_kind: CausalFormulaKind::BackdoorAdjustment,
                treatment: vid(1),
                outcome: vid(2),
                adjustment_set: vec![vid(3)],
                result: vid(30),
            },
            CausalOp::DSeparationTest {
                source: "A".to_string(),
                target: "B".to_string(),
                conditioning: vec![],
                result: vid(40),
            },
            CausalOp::CounterfactualQuery {
                factual_observations: vec![vid(1)],
                intervention_var: "X".to_string(),
                intervention_value: vid(2),
                outcome_var: "Y".to_string(),
                result: vid(50),
            },
        ];

        let counts = count_causal_ops(&ops);
        assert_eq!(counts.abductions, 1);
        assert_eq!(counts.interventions, 1);
        assert_eq!(counts.predictions, 1);
        assert_eq!(counts.effect_estimations, 1);
        assert_eq!(counts.d_separation_tests, 1);
        assert_eq!(counts.counterfactual_queries, 1);
    }

    #[test]
    fn test_count_causal_ops_empty() {
        let counts = count_causal_ops(&[]);
        assert_eq!(counts, CausalOpCounts::default());
    }

    // -----------------------------------------------------------------------
    // CausalOp derive tests (in ops.rs pattern)
    // -----------------------------------------------------------------------

    #[test]
    fn test_causal_op_clone() {
        let op = CausalOp::Abduction {
            observed: vec![vid(1), vid(2)],
            coefficients: vec![0.5, 1.5],
            result: vid(10),
        };
        let cloned = op.clone();
        assert_eq!(op, cloned);
    }

    #[test]
    fn test_causal_formula_kind_eq() {
        assert_eq!(
            CausalFormulaKind::BackdoorAdjustment,
            CausalFormulaKind::BackdoorAdjustment
        );
        assert_ne!(
            CausalFormulaKind::BackdoorAdjustment,
            CausalFormulaKind::FrontdoorAdjustment
        );
        assert_ne!(CausalFormulaKind::Conditional, CausalFormulaKind::Marginal);
        assert_eq!(
            CausalFormulaKind::InstrumentalVariable,
            CausalFormulaKind::InstrumentalVariable
        );
    }

    // -----------------------------------------------------------------------
    // SirInst::Causal integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sir_inst_causal_has_side_effects() {
        use super::super::ops::SirInst;

        let do_op = SirInst::Causal(CausalOp::DoIntervention {
            target_var: "X".to_string(),
            value: vid(1),
            pre_state: vid(2),
            post_state: vid(3),
        });
        assert!(do_op.has_side_effects());

        let abduction = SirInst::Causal(CausalOp::Abduction {
            observed: vec![vid(1)],
            coefficients: vec![],
            result: vid(10),
        });
        assert!(!abduction.has_side_effects());
    }

    #[test]
    fn test_sir_inst_causal_operands() {
        use super::super::ops::SirInst;

        let op = SirInst::Causal(CausalOp::EstimateEffect {
            formula_kind: CausalFormulaKind::BackdoorAdjustment,
            treatment: vid(1),
            outcome: vid(2),
            adjustment_set: vec![vid(3), vid(4)],
            result: vid(10),
        });

        let operands = op.operands();
        assert!(operands.contains(&vid(1)));
        assert!(operands.contains(&vid(2)));
        assert!(operands.contains(&vid(3)));
        assert!(operands.contains(&vid(4)));
        assert!(operands.contains(&vid(10)));
    }

    #[test]
    fn test_sir_inst_causal_display() {
        use super::super::ops::SirInst;

        let op = SirInst::Causal(CausalOp::DSeparationTest {
            source: "A".to_string(),
            target: "B".to_string(),
            conditioning: vec!["C".to_string()],
            result: vid(10),
        });

        let display = format!("{}", op);
        assert!(display.starts_with("causal."));
    }
}
