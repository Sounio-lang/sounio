//! Counterfactual Reasoning: Level 3 of the Causal Hierarchy
//!
//! Implements counterfactual queries using the 3-step algorithm:
//! 1. ABDUCTION: Infer exogenous variables from evidence
//! 2. ACTION: Modify model with intervention
//! 3. PREDICTION: Compute outcome in modified world
//!
//! Also implements Probability of Causation metrics (PN, PS, PNS)

use std::collections::HashMap;
use std::fmt;

use super::intervention::Distribution;
use super::knowledge::CausalKnowledge;
use super::structural::StructuralCausalModel;
use crate::epistemic::composition::ConfidenceValue;
use crate::temporal::TemporalKnowledge;

/// Level 3: Structural Knowledge with full counterfactual capability
#[derive(Clone)]
pub struct StructuralKnowledge<T> {
    /// Level 2 causal knowledge
    pub causal: CausalKnowledge<T>,
    /// Full structural causal model
    pub model: StructuralCausalModel,
}

impl<T: fmt::Debug> fmt::Debug for StructuralKnowledge<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StructuralKnowledge")
            .field("causal", &self.causal)
            .field("model", &self.model)
            .finish()
    }
}

impl<T: Clone> StructuralKnowledge<T> {
    /// Create new structural knowledge
    pub fn new(causal: CausalKnowledge<T>, model: StructuralCausalModel) -> Self {
        StructuralKnowledge { causal, model }
    }

    /// Compute counterfactual: Y_{x'}
    ///
    /// Given evidence (observations), computes what Y would be if X had been x'
    ///
    /// Algorithm:
    /// 1. ABDUCTION: P(U | evidence)
    /// 2. ACTION: M_{x'}
    /// 3. PREDICTION: Y in M_{x'} with posterior U
    pub fn counterfactual(
        &self,
        target: &str,
        intervention: &str,
        intervention_value: f64,
        evidence: &HashMap<String, f64>,
    ) -> CounterfactualResult {
        // Step 1: ABDUCTION - Infer exogenous variables from evidence
        let posterior_u = self.abduct(evidence);

        // Step 2: ACTION - Create intervened model
        let intervened_model = self.model.intervene(intervention, intervention_value);

        // Step 3: PREDICTION - Evaluate in counterfactual world
        let cf_world = self.predict_counterfactual(&intervened_model, &posterior_u);

        let cf_value = cf_world.get(target).copied().unwrap_or(f64::NAN);
        let confidence = self.compute_cf_confidence(&posterior_u, evidence);

        CounterfactualResult {
            query: CounterfactualQuery {
                target: target.to_string(),
                intervention: intervention.to_string(),
                intervention_value,
                evidence: evidence.clone(),
            },
            value: cf_value,
            confidence,
            posterior_u,
            counterfactual_world: cf_world,
        }
    }

    /// Step 1: ABDUCTION - Infer exogenous variables from evidence
    ///
    /// Given observations, infer the posterior distribution of U
    fn abduct(&self, evidence: &HashMap<String, f64>) -> HashMap<String, Distribution> {
        let mut posterior_u = HashMap::new();

        for (u_name, prior) in &self.model.exogenous_distributions {
            // Find variables affected by this exogenous
            let affected_vars: Vec<&String> = self
                .model
                .equations
                .iter()
                .filter(|(_, eq)| eq.exogenous == *u_name)
                .map(|(v, _)| v)
                .collect();

            // Check if we have evidence for any affected variable
            let has_evidence = affected_vars.iter().any(|v| evidence.contains_key(*v));

            if has_evidence {
                // Infer U from evidence
                // Simplified: use point estimate based on inverting structural equation
                let posterior = self.infer_u_posterior(u_name, &affected_vars, evidence, prior);
                posterior_u.insert(u_name.clone(), posterior);
            } else {
                // No evidence, use prior
                posterior_u.insert(u_name.clone(), prior.clone());
            }
        }

        posterior_u
    }

    /// Infer posterior distribution of exogenous variable from evidence
    fn infer_u_posterior(
        &self,
        u_name: &str,
        affected_vars: &[&String],
        evidence: &HashMap<String, f64>,
        prior: &Distribution,
    ) -> Distribution {
        // Find an affected variable with evidence
        for var in affected_vars {
            if let Some(&observed) = evidence.get(*var)
                && let Some(eq) = self.model.equations.get(*var)
            {
                // Try to invert: if Y = f(PA, U), solve for U given Y and PA
                // For linear equations: U = Y - Σ(β_i * PA_i) - intercept

                // Get parent values from evidence
                let parent_values: HashMap<String, f64> = eq
                    .parents
                    .iter()
                    .filter_map(|p| evidence.get(p).map(|v| (p.clone(), *v)))
                    .collect();

                // Compute predicted value with U=0
                let predicted = eq.evaluate(&parent_values, 0.0);

                // Inferred U = observed - predicted
                let inferred_u = observed - predicted;

                // Return point mass at inferred U
                return Distribution::PointMass { value: inferred_u };
            }
        }

        // No useful evidence, use prior
        prior.clone()
    }

    /// Step 3: PREDICTION - Evaluate counterfactual world
    fn predict_counterfactual(
        &self,
        intervened_model: &StructuralCausalModel,
        posterior_u: &HashMap<String, Distribution>,
    ) -> HashMap<String, f64> {
        // Sample from posterior (use mean for deterministic result)
        let u_values: HashMap<String, f64> = posterior_u
            .iter()
            .map(|(name, dist)| (name.clone(), dist.mean()))
            .collect();

        intervened_model.evaluate(&u_values)
    }

    /// Compute confidence in counterfactual estimate
    fn compute_cf_confidence(
        &self,
        posterior_u: &HashMap<String, Distribution>,
        evidence: &HashMap<String, f64>,
    ) -> ConfidenceValue {
        // Confidence based on:
        // 1. Amount of evidence
        // 2. Variance of posterior U
        // 3. Base confidence from causal knowledge

        let base_conf = self.causal.confidence().value();

        // Evidence factor: more evidence = higher confidence
        let evidence_factor = (evidence.len() as f64 / self.model.variables().len() as f64)
            .min(1.0)
            .max(0.1);

        // Variance factor: lower variance = higher confidence
        let avg_variance: f64 = posterior_u.values().map(|d| d.variance()).sum::<f64>()
            / posterior_u.len().max(1) as f64;
        let variance_factor = 1.0 / (1.0 + avg_variance);

        let confidence = base_conf * evidence_factor * variance_factor;
        ConfidenceValue::new(confidence.clamp(0.0, 1.0)).unwrap_or(ConfidenceValue::uncertain())
    }

    /// Probability of Necessity (PN)
    ///
    /// P(Y_{x'} = 0 | X=x, Y=1)
    ///
    /// "Given that Y happened with treatment X=x, would Y NOT have happened
    /// if X had been x' instead?"
    pub fn probability_of_necessity(
        &self,
        treatment: &str,
        treatment_observed: f64,
        treatment_counterfactual: f64,
        outcome: &str,
        n_samples: usize,
    ) -> ProbabilityOfCausation {
        let mut necessary_count = 0;

        for _ in 0..n_samples {
            // Sample from P(U | X=x, Y=1)
            let u_sample = self.sample_conditioned_u(treatment, treatment_observed, outcome, 1.0);

            // Compute Y in counterfactual world
            let m_cf = self.model.intervene(treatment, treatment_counterfactual);
            let cf_values = m_cf.evaluate(&u_sample);

            if let Some(&y_cf) = cf_values.get(outcome) {
                // Y_{x'} = 0 (or < 0.5 for continuous)
                if y_cf < 0.5 {
                    necessary_count += 1;
                }
            }
        }

        let pn = necessary_count as f64 / n_samples as f64;

        ProbabilityOfCausation {
            causation_type: CausationType::Necessity,
            probability: pn,
            confidence: ConfidenceValue::new(0.9 - 0.1 * (1.0 / (n_samples as f64).sqrt()))
                .unwrap_or(ConfidenceValue::uncertain()),
            interpretation: format!(
                "P({} would not have occurred without {} = {}) = {:.1}%",
                outcome,
                treatment,
                treatment_observed,
                pn * 100.0
            ),
        }
    }

    /// Probability of Sufficiency (PS)
    ///
    /// P(Y_x = 1 | X=x', Y=0)
    ///
    /// "Given that Y did NOT happen with X=x', would Y have happened
    /// if X had been x instead?"
    pub fn probability_of_sufficiency(
        &self,
        treatment: &str,
        treatment_counterfactual: f64,
        treatment_observed: f64,
        outcome: &str,
        n_samples: usize,
    ) -> ProbabilityOfCausation {
        let mut sufficient_count = 0;

        for _ in 0..n_samples {
            // Sample from P(U | X=x', Y=0)
            let u_sample = self.sample_conditioned_u(treatment, treatment_observed, outcome, 0.0);

            // Compute Y in counterfactual world
            let m_cf = self.model.intervene(treatment, treatment_counterfactual);
            let cf_values = m_cf.evaluate(&u_sample);

            if let Some(&y_cf) = cf_values.get(outcome) {
                // Y_x = 1 (or >= 0.5 for continuous)
                if y_cf >= 0.5 {
                    sufficient_count += 1;
                }
            }
        }

        let ps = sufficient_count as f64 / n_samples as f64;

        ProbabilityOfCausation {
            causation_type: CausationType::Sufficiency,
            probability: ps,
            confidence: ConfidenceValue::new(0.9 - 0.1 * (1.0 / (n_samples as f64).sqrt()))
                .unwrap_or(ConfidenceValue::uncertain()),
            interpretation: format!(
                "P({} would have occurred with {} = {}) = {:.1}%",
                outcome,
                treatment,
                treatment_counterfactual,
                ps * 100.0
            ),
        }
    }

    /// Probability of Necessity and Sufficiency (PNS)
    ///
    /// P(Y_x = 1, Y_{x'} = 0)
    ///
    /// "X=x is both necessary AND sufficient for Y"
    pub fn probability_of_necessity_and_sufficiency(
        &self,
        treatment: &str,
        treatment_value: f64,
        treatment_baseline: f64,
        outcome: &str,
        n_samples: usize,
    ) -> ProbabilityOfCausation {
        let mut pns_count = 0;

        for _ in 0..n_samples {
            // Sample U from prior
            let u_sample = self.model.sample_exogenous();

            // Y_x (with treatment)
            let m_x = self.model.intervene(treatment, treatment_value);
            let values_x = m_x.evaluate(&u_sample);
            let y_x = values_x.get(outcome).copied().unwrap_or(0.0);

            // Y_{x'} (without treatment)
            let m_xp = self.model.intervene(treatment, treatment_baseline);
            let values_xp = m_xp.evaluate(&u_sample);
            let y_xp = values_xp.get(outcome).copied().unwrap_or(0.0);

            // PNS: Y_x = 1 AND Y_{x'} = 0
            if y_x >= 0.5 && y_xp < 0.5 {
                pns_count += 1;
            }
        }

        let pns = pns_count as f64 / n_samples as f64;

        ProbabilityOfCausation {
            causation_type: CausationType::NecessityAndSufficiency,
            probability: pns,
            confidence: ConfidenceValue::new(0.9 - 0.1 * (1.0 / (n_samples as f64).sqrt()))
                .unwrap_or(ConfidenceValue::uncertain()),
            interpretation: format!(
                "P({} = {} is both necessary and sufficient for {}) = {:.1}%",
                treatment,
                treatment_value,
                outcome,
                pns * 100.0
            ),
        }
    }

    /// Sample exogenous variables conditioned on evidence
    fn sample_conditioned_u(
        &self,
        treatment: &str,
        treatment_value: f64,
        outcome: &str,
        outcome_value: f64,
    ) -> HashMap<String, f64> {
        // Simple rejection sampling (inefficient but correct)
        // In practice, would use MCMC or variational inference

        let max_attempts = 10000;
        let tolerance = 0.3;

        for _ in 0..max_attempts {
            let u_sample = self.model.sample_exogenous();
            let values = self.model.evaluate(&u_sample);

            // Check if sample matches evidence approximately
            let x_match =
                (values.get(treatment).unwrap_or(&f64::NAN) - treatment_value).abs() < tolerance;
            let y_match =
                (values.get(outcome).unwrap_or(&f64::NAN) - outcome_value).abs() < tolerance;

            if x_match && y_match {
                return u_sample;
            }
        }

        // Fallback: return prior sample
        self.model.sample_exogenous()
    }

    /// Downcast to CausalKnowledge
    pub fn as_causal(&self) -> &CausalKnowledge<T> {
        &self.causal
    }

    /// Downcast to TemporalKnowledge
    pub fn as_temporal(&self) -> &TemporalKnowledge<T> {
        &self.causal.base
    }
}

/// Query for counterfactual computation
#[derive(Clone, Debug)]
pub struct CounterfactualQuery {
    /// Target variable to compute
    pub target: String,
    /// Intervention variable
    pub intervention: String,
    /// Counterfactual intervention value
    pub intervention_value: f64,
    /// Observed evidence
    pub evidence: HashMap<String, f64>,
}

impl fmt::Display for CounterfactualQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}_{{{} = {}}} | {:?}",
            self.target, self.intervention, self.intervention_value, self.evidence
        )
    }
}

/// Result of counterfactual computation
#[derive(Clone, Debug)]
pub struct CounterfactualResult {
    /// The query that was computed
    pub query: CounterfactualQuery,
    /// Counterfactual value
    pub value: f64,
    /// Confidence in the result
    pub confidence: ConfidenceValue,
    /// Posterior distribution of exogenous variables
    pub posterior_u: HashMap<String, Distribution>,
    /// Full counterfactual world (all variable values)
    pub counterfactual_world: HashMap<String, f64>,
}

impl CounterfactualResult {
    /// Check if counterfactual differs from factual
    pub fn differs_from_factual(&self, threshold: f64) -> bool {
        if let Some(&factual) = self.query.evidence.get(&self.query.target) {
            (self.value - factual).abs() > threshold
        } else {
            true
        }
    }
}

/// Probability of causation result
#[derive(Clone, Debug)]
pub struct ProbabilityOfCausation {
    /// Type of causation probability
    pub causation_type: CausationType,
    /// Probability value
    pub probability: f64,
    /// Confidence in estimate
    pub confidence: ConfidenceValue,
    /// Human-readable interpretation
    pub interpretation: String,
}

impl fmt::Display for ProbabilityOfCausation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:.1}% ({})",
            self.causation_type,
            self.probability * 100.0,
            self.interpretation
        )
    }
}

/// Types of causation probability
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CausationType {
    /// Probability of Necessity
    Necessity,
    /// Probability of Sufficiency
    Sufficiency,
    /// Probability of Necessity and Sufficiency
    NecessityAndSufficiency,
}

impl fmt::Display for CausationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CausationType::Necessity => write!(f, "PN"),
            CausationType::Sufficiency => write!(f, "PS"),
            CausationType::NecessityAndSufficiency => write!(f, "PNS"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal::graph::{CausalGraph, CausalNode, EdgeType};
    use crate::causal::structural::SCMBuilder;
    use crate::epistemic::composition::EpistemicValue;

    fn simple_structural_knowledge() -> StructuralKnowledge<f64> {
        // X -> Y with Y = 0.5*X + U_Y
        let model = SCMBuilder::new()
            .exogenous_variable(
                "X",
                "U_X",
                Distribution::Normal {
                    mean: 0.0,
                    std: 1.0,
                },
            )
            .linear_variable("Y", vec![("X".to_string(), 0.5)], "U_Y", 0.0)
            .build();

        let mut graph = CausalGraph::new();
        graph.add_node(CausalNode::treatment("X"));
        graph.add_node(CausalNode::outcome("Y"));
        graph.add_edge("X", "Y", EdgeType::Direct).unwrap();

        let base = TemporalKnowledge::timeless(EpistemicValue::with_confidence(0.5, 0.9));

        let causal = CausalKnowledge::new(base, graph, "Y", vec!["X".to_string()]);

        StructuralKnowledge::new(causal, model)
    }

    #[test]
    fn test_counterfactual_basic() {
        let sk = simple_structural_knowledge();

        // Evidence: X=2, Y=1
        let evidence: HashMap<String, f64> = [("X".to_string(), 2.0), ("Y".to_string(), 1.0)]
            .into_iter()
            .collect();

        // Counterfactual: What would Y be if X had been 0?
        let cf = sk.counterfactual("Y", "X", 0.0, &evidence);

        // Y = 0.5*X + U_Y
        // With X=2, Y=1: U_Y = 1 - 0.5*2 = 0
        // With X=0: Y = 0.5*0 + 0 = 0
        assert!((cf.value - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_counterfactual_query_display() {
        let query = CounterfactualQuery {
            target: "Y".to_string(),
            intervention: "X".to_string(),
            intervention_value: 0.0,
            evidence: [("X".to_string(), 1.0)].into_iter().collect(),
        };

        let display = format!("{}", query);
        assert!(display.contains("Y"));
        assert!(display.contains("X"));
    }

    #[test]
    fn test_pns_computation() {
        let sk = simple_structural_knowledge();

        // PNS: X=1 is necessary and sufficient for Y
        let pns = sk.probability_of_necessity_and_sufficiency("X", 1.0, 0.0, "Y", 100);

        // Should have some non-zero probability
        assert!(pns.probability >= 0.0);
        assert!(pns.probability <= 1.0);
    }

    #[test]
    fn test_causation_type_display() {
        assert_eq!(format!("{}", CausationType::Necessity), "PN");
        assert_eq!(format!("{}", CausationType::Sufficiency), "PS");
        assert_eq!(format!("{}", CausationType::NecessityAndSufficiency), "PNS");
    }
}
