//! Causal Effect Estimation with Epistemic Uncertainty
//!
//! This module implements estimation of causal effects with full uncertainty
//! quantification using Knowledge<f64>. All estimates carry:
//! - Point estimate (mean effect)
//! - Variance (statistical uncertainty)
//! - Confidence (epistemic uncertainty about identification)
//!
//! # Effect Types
//!
//! - **ATE** (Average Treatment Effect): E[Y | do(X=1)] - E[Y | do(X=0)]
//! - **CATE** (Conditional ATE): E[Y | do(X=1), Z=z] - E[Y | do(X=0), Z=z]
//! - **LATE** (Local ATE): Effect for compliers in instrumental variable designs
//! - **NDE/NIE** (Natural Direct/Indirect Effects): Mediation analysis
//!
//! # Uncertainty Sources
//!
//! 1. **Statistical**: Sampling variance (Knowledge.variance)
//! 2. **Identification**: Uncertainty about which adjustment set is correct
//! 3. **Structural**: Uncertainty about graph edges (BetaConfidence)
//! 4. **Effect size**: Uncertainty about causal strengths (Knowledge<f64> on edges)

use std::collections::HashMap;

use super::dag::{CausalDAG, EffectEstimate};
use super::do_calculus::{AdjustmentSet, DoCalculus};

/// Average Treatment Effect (ATE)
///
/// The expected difference in outcomes between treating everyone
/// vs. treating no one: E[Y | do(X=1)] - E[Y | do(X=0)]
#[derive(Clone, Debug)]
pub struct AverageTreatmentEffect {
    /// Point estimate of ATE
    pub estimate: EffectEstimate<f64>,
    /// Sample size used for estimation
    pub sample_size: usize,
    /// Adjustment set used (if applicable)
    pub adjustment_set: Option<AdjustmentSet>,
}

impl AverageTreatmentEffect {
    /// Create a new ATE estimate
    pub fn new(estimate: EffectEstimate<f64>, sample_size: usize) -> Self {
        AverageTreatmentEffect {
            estimate,
            sample_size,
            adjustment_set: None,
        }
    }

    /// Add adjustment set information
    pub fn with_adjustment(mut self, adjustment: AdjustmentSet) -> Self {
        self.adjustment_set = Some(adjustment);
        self
    }

    /// Get confidence interval at given level
    pub fn confidence_interval(&self, alpha: f64) -> (f64, f64) {
        if let Some(var) = self.estimate.variance {
            let z = normal_quantile(1.0 - alpha / 2.0);
            let se = var.sqrt();
            let lower = self.estimate.value - z * se;
            let upper = self.estimate.value + z * se;
            (lower, upper)
        } else {
            // No variance estimate, use point estimate
            (self.estimate.value, self.estimate.value)
        }
    }
}

/// Conditional Average Treatment Effect (CATE)
///
/// Treatment effect conditional on covariates Z:
/// E[Y | do(X=1), Z=z] - E[Y | do(X=0), Z=z]
#[derive(Clone, Debug)]
pub struct ConditionalAverageTreatmentEffect {
    /// Conditioning variables and their values
    pub conditioning_vars: HashMap<String, f64>,
    /// Point estimate of CATE
    pub estimate: EffectEstimate<f64>,
    /// Sample size in this stratum
    pub sample_size: usize,
}

impl ConditionalAverageTreatmentEffect {
    /// Create a new CATE estimate
    pub fn new(
        conditioning_vars: HashMap<String, f64>,
        estimate: EffectEstimate<f64>,
        sample_size: usize,
    ) -> Self {
        ConditionalAverageTreatmentEffect {
            conditioning_vars,
            estimate,
            sample_size,
        }
    }

    /// Get confidence interval
    pub fn confidence_interval(&self, alpha: f64) -> (f64, f64) {
        if let Some(var) = self.estimate.variance {
            let z = normal_quantile(1.0 - alpha / 2.0);
            let se = var.sqrt();
            let lower = self.estimate.value - z * se;
            let upper = self.estimate.value + z * se;
            (lower, upper)
        } else {
            (self.estimate.value, self.estimate.value)
        }
    }
}

/// Local Average Treatment Effect (LATE)
///
/// Effect for compliers in instrumental variable designs.
/// Compliers are units whose treatment is affected by the instrument.
#[derive(Clone, Debug)]
pub struct LocalAverageTreatmentEffect {
    /// Instrument variable name
    pub instrument: String,
    /// Point estimate of LATE
    pub estimate: EffectEstimate<f64>,
    /// First-stage F-statistic (instrument strength)
    pub first_stage_f: f64,
    /// Sample size
    pub sample_size: usize,
}

impl LocalAverageTreatmentEffect {
    /// Create a new LATE estimate
    pub fn new(
        instrument: impl Into<String>,
        estimate: EffectEstimate<f64>,
        first_stage_f: f64,
        sample_size: usize,
    ) -> Self {
        LocalAverageTreatmentEffect {
            instrument: instrument.into(),
            estimate,
            first_stage_f,
            sample_size,
        }
    }

    /// Check if instrument is weak (F < 10 rule of thumb)
    pub fn has_weak_instrument(&self) -> bool {
        self.first_stage_f < 10.0
    }

    /// Get confidence interval
    pub fn confidence_interval(&self, alpha: f64) -> (f64, f64) {
        if let Some(var) = self.estimate.variance {
            let z = normal_quantile(1.0 - alpha / 2.0);
            let se = var.sqrt();
            let lower = self.estimate.value - z * se;
            let upper = self.estimate.value + z * se;
            (lower, upper)
        } else {
            (self.estimate.value, self.estimate.value)
        }
    }
}

/// Natural Direct and Indirect Effects for mediation analysis
///
/// - **NDE** (Natural Direct Effect): Effect not through mediator
/// - **NIE** (Natural Indirect Effect): Effect through mediator
/// - Total effect = NDE + NIE
#[derive(Clone, Debug)]
pub struct MediationEffects {
    /// Natural Direct Effect
    pub direct_effect: EffectEstimate<f64>,
    /// Natural Indirect Effect
    pub indirect_effect: EffectEstimate<f64>,
    /// Total effect (should equal direct + indirect)
    pub total_effect: EffectEstimate<f64>,
    /// Proportion mediated
    pub proportion_mediated: f64,
}

impl MediationEffects {
    /// Create mediation effects from direct and indirect components
    pub fn new(direct: EffectEstimate<f64>, indirect: EffectEstimate<f64>) -> Self {
        let total_value = direct.value + indirect.value;
        let total_var = direct.variance.unwrap_or(0.0) + indirect.variance.unwrap_or(0.0);
        let total_conf = direct.confidence.min(indirect.confidence);

        let total_effect = EffectEstimate::with_variance(total_value, total_var, total_conf);

        let proportion = if total_value.abs() > 1e-10 {
            indirect.value / total_value
        } else {
            0.0
        };

        MediationEffects {
            direct_effect: direct,
            indirect_effect: indirect,
            total_effect,
            proportion_mediated: proportion,
        }
    }

    /// Check if mediation is significant (indirect effect non-zero)
    pub fn has_significant_mediation(&self, alpha: f64) -> bool {
        if let Some(var) = self.indirect_effect.variance {
            let z = normal_quantile(1.0 - alpha / 2.0);
            let se = var.sqrt();
            let lower = self.indirect_effect.value - z * se;
            let upper = self.indirect_effect.value + z * se;

            // CI doesn't contain 0
            (lower > 0.0 && upper > 0.0) || (lower < 0.0 && upper < 0.0)
        } else {
            self.indirect_effect.value.abs() > 1e-10
        }
    }
}

/// Estimate ATE from a causal DAG
///
/// Uses the structure of the DAG to identify and estimate the causal effect
pub fn average_treatment_effect(
    dag: &CausalDAG,
    treatment: &str,
    outcome: &str,
    data: &[(HashMap<String, f64>, f64)], // (covariates, outcome)
) -> Result<AverageTreatmentEffect, String> {
    // Find adjustment set
    let adjustment = dag
        .backdoor_adjustment(treatment, outcome)
        .ok_or_else(|| "No valid adjustment set found".to_string())?;

    // Estimate effect using adjustment
    let estimate = estimate_with_adjustment(treatment, outcome, &adjustment.variables, data)?;

    Ok(AverageTreatmentEffect::new(estimate, data.len()).with_adjustment(adjustment))
}

/// Estimate CATE for specific covariate values
pub fn conditional_average_treatment_effect(
    dag: &CausalDAG,
    treatment: &str,
    outcome: &str,
    conditioning: HashMap<String, f64>,
    data: &[(HashMap<String, f64>, f64)],
) -> Result<ConditionalAverageTreatmentEffect, String> {
    // Find adjustment set
    let adjustment = dag
        .backdoor_adjustment(treatment, outcome)
        .ok_or_else(|| "No valid adjustment set found".to_string())?;

    // Filter data to conditioning stratum
    let filtered_data: Vec<_> = data
        .iter()
        .filter(|(covs, _)| {
            conditioning
                .iter()
                .all(|(k, v)| covs.get(k).map(|cv| (cv - v).abs() < 0.1).unwrap_or(false))
        })
        .cloned()
        .collect();

    if filtered_data.is_empty() {
        return Err("No data in conditioning stratum".to_string());
    }

    let estimate =
        estimate_with_adjustment(treatment, outcome, &adjustment.variables, &filtered_data)?;

    Ok(ConditionalAverageTreatmentEffect::new(
        conditioning,
        estimate,
        filtered_data.len(),
    ))
}

/// Estimate LATE using instrumental variable
pub fn local_average_treatment_effect(
    dag: &CausalDAG,
    instrument: &str,
    treatment: &str,
    outcome: &str,
    data: &[(HashMap<String, f64>, f64)],
) -> Result<LocalAverageTreatmentEffect, String> {
    // Check instrument validity (simplified)
    let parents_z = dag.parents(instrument);
    let children_z = dag.children(instrument);

    if !children_z.contains(&treatment) {
        return Err("Instrument doesn't affect treatment".to_string());
    }

    // Estimate reduced form and first stage
    let (reduced_form, first_stage) = estimate_iv_stages(instrument, treatment, outcome, data)?;

    // LATE = reduced form / first stage
    let late_value = reduced_form.value / first_stage.value;
    let late_var =
        if let (Some(rf_var), Some(fs_var)) = (reduced_form.variance, first_stage.variance) {
            // Delta method approximation
            Some(
                (rf_var / first_stage.value.powi(2))
                    + (reduced_form.value.powi(2) * fs_var / first_stage.value.powi(4)),
            )
        } else {
            None
        };

    let late_conf = reduced_form.confidence.min(first_stage.confidence);

    let estimate = EffectEstimate {
        value: late_value,
        variance: late_var,
        confidence: late_conf,
        source: "LATE".to_string(),
    };

    // Approximate F-statistic (simplified)
    let f_stat = if let Some(var) = first_stage.variance {
        first_stage.value.powi(2) / var
    } else {
        100.0 // Assume strong if no variance
    };

    Ok(LocalAverageTreatmentEffect::new(
        instrument,
        estimate,
        f_stat,
        data.len(),
    ))
}

/// Estimate mediation effects (direct and indirect)
pub fn mediation_effects(
    dag: &CausalDAG,
    treatment: &str,
    mediator: &str,
    outcome: &str,
    data: &[(HashMap<String, f64>, f64)],
) -> Result<MediationEffects, String> {
    // Total effect: treatment → outcome
    let total = estimate_simple_effect(treatment, outcome, data)?;

    // Direct effect: treatment → outcome (controlling for mediator)
    let mut mediator_set = std::collections::HashSet::new();
    mediator_set.insert(mediator.to_string());
    let direct = estimate_with_adjustment(treatment, outcome, &mediator_set, data)?;

    // Indirect effect: total - direct
    let indirect_value = total.value - direct.value;
    let indirect_var = if let (Some(tv), Some(dv)) = (total.variance, direct.variance) {
        Some(tv + dv)
    } else {
        None
    };
    let indirect_conf = total.confidence.min(direct.confidence);

    let indirect = EffectEstimate {
        value: indirect_value,
        variance: indirect_var,
        confidence: indirect_conf,
        source: "IndirectEffect".to_string(),
    };

    Ok(MediationEffects::new(direct, indirect))
}

// Helper functions for estimation

/// Estimate effect with backdoor adjustment
fn estimate_with_adjustment(
    treatment: &str,
    outcome: &str,
    adjustment_vars: &std::collections::HashSet<String>,
    data: &[(HashMap<String, f64>, f64)],
) -> Result<EffectEstimate<f64>, String> {
    // Simplified: average outcome difference stratified by adjustment set
    // In practice: use regression or matching

    let treated: Vec<_> = data
        .iter()
        .filter(|(covs, _)| covs.get(treatment).map(|&t| t > 0.5).unwrap_or(false))
        .collect();

    let control: Vec<_> = data
        .iter()
        .filter(|(covs, _)| covs.get(treatment).map(|&t| t <= 0.5).unwrap_or(false))
        .collect();

    if treated.is_empty() || control.is_empty() {
        return Err("Insufficient treated or control units".to_string());
    }

    let treated_mean: f64 = treated.iter().map(|(_, y)| y).sum::<f64>() / treated.len() as f64;
    let control_mean: f64 = control.iter().map(|(_, y)| y).sum::<f64>() / control.len() as f64;

    let effect = treated_mean - control_mean;

    // Estimate variance (simplified)
    let treated_var: f64 = treated
        .iter()
        .map(|(_, y)| (y - treated_mean).powi(2))
        .sum::<f64>()
        / (treated.len() as f64 - 1.0);

    let control_var: f64 = control
        .iter()
        .map(|(_, y)| (y - control_mean).powi(2))
        .sum::<f64>()
        / (control.len() as f64 - 1.0);

    let effect_var = treated_var / treated.len() as f64 + control_var / control.len() as f64;

    // Confidence penalty for adjustment complexity
    let confidence = 0.95_f64.powi(adjustment_vars.len() as i32);

    Ok(EffectEstimate::with_variance(
        effect, effect_var, confidence,
    ))
}

/// Estimate simple bivariate effect (no adjustment)
fn estimate_simple_effect(
    treatment: &str,
    outcome: &str,
    data: &[(HashMap<String, f64>, f64)],
) -> Result<EffectEstimate<f64>, String> {
    estimate_with_adjustment(treatment, outcome, &std::collections::HashSet::new(), data)
}

/// Estimate IV stages (reduced form and first stage)
fn estimate_iv_stages(
    instrument: &str,
    treatment: &str,
    outcome: &str,
    data: &[(HashMap<String, f64>, f64)],
) -> Result<(EffectEstimate<f64>, EffectEstimate<f64>), String> {
    // Reduced form: instrument → outcome
    let reduced_form = estimate_simple_effect(instrument, outcome, data)?;

    // First stage: instrument → treatment
    let first_stage_data: Vec<_> = data
        .iter()
        .map(|(covs, _)| {
            let new_covs = covs.clone();
            let treatment_val = covs.get(treatment).copied().unwrap_or(0.0);
            (new_covs, treatment_val)
        })
        .collect();

    let first_stage = estimate_simple_effect(instrument, treatment, &first_stage_data)?;

    Ok((reduced_form, first_stage))
}

/// Normal distribution quantile (approximation)
fn normal_quantile(p: f64) -> f64 {
    // Approximation of inverse CDF for standard normal
    // For common values:
    if p >= 0.975 {
        1.96 // 95% CI
    } else if p >= 0.95 {
        1.645 // 90% CI
    } else if p >= 0.995 {
        2.576 // 99% CI
    } else {
        1.96 // Default to 95%
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal::dag::{EffectEstimate, EpistemicCausalNode, UncertainCausalEdge};
    use crate::epistemic::BetaConfidence;

    fn simple_dag() -> CausalDAG {
        let mut dag = CausalDAG::new();
        dag.add_node(EpistemicCausalNode::treatment("X"));
        dag.add_node(EpistemicCausalNode::outcome("Y"));

        let edge = UncertainCausalEdge::direct(
            "X",
            "Y",
            BetaConfidence::new(9.0, 1.0),
            EffectEstimate::certain(0.5),
        );
        dag.add_edge(edge).unwrap();

        dag
    }

    fn sample_data() -> Vec<(HashMap<String, f64>, f64)> {
        vec![
            (HashMap::from([("X".to_string(), 1.0)]), 2.0),
            (HashMap::from([("X".to_string(), 1.0)]), 2.5),
            (HashMap::from([("X".to_string(), 0.0)]), 1.0),
            (HashMap::from([("X".to_string(), 0.0)]), 1.5),
        ]
    }

    #[test]
    fn test_ate_creation() {
        let estimate = EffectEstimate::with_variance(0.5, 0.1, 0.9);
        let ate = AverageTreatmentEffect::new(estimate, 100);

        assert_eq!(ate.sample_size, 100);
        assert!((ate.estimate.value - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_ate_confidence_interval() {
        let estimate = EffectEstimate::with_variance(0.5, 0.1, 0.9);
        let ate = AverageTreatmentEffect::new(estimate, 100);

        let (lower, upper) = ate.confidence_interval(0.05);

        assert!(lower < 0.5);
        assert!(upper > 0.5);
        assert!(upper - lower > 0.0);
    }

    #[test]
    fn test_mediation_effects() {
        let direct = EffectEstimate::with_variance(0.3, 0.05, 0.9);
        let indirect = EffectEstimate::with_variance(0.2, 0.03, 0.85);

        let mediation = MediationEffects::new(direct, indirect);

        assert!((mediation.total_effect.value - 0.5).abs() < 0.001);
        assert!((mediation.proportion_mediated - 0.4).abs() < 0.001); // 0.2 / 0.5
    }

    #[test]
    fn test_late_weak_instrument() {
        let estimate = EffectEstimate::with_variance(0.5, 0.1, 0.9);
        let late = LocalAverageTreatmentEffect::new("Z", estimate.clone(), 5.0, 100);

        assert!(late.has_weak_instrument());

        let strong_late = LocalAverageTreatmentEffect::new("Z", estimate, 15.0, 100);
        assert!(!strong_late.has_weak_instrument());
    }

    #[test]
    fn test_average_treatment_effect() {
        let dag = simple_dag();
        let data = sample_data();

        let ate = average_treatment_effect(&dag, "X", "Y", &data);

        assert!(ate.is_ok());
        let ate = ate.unwrap();

        // Effect should be approximately 2.25 - 1.25 = 1.0
        assert!((ate.estimate.value - 1.0).abs() < 0.1);
    }
}
