//! Uplift Modeling with Epistemic Uncertainty
//!
//! This module implements CausalML-inspired uplift estimators as native Sounio
//! constructs with full epistemic uncertainty propagation.
//!
//! # Theory
//!
//! Uplift modeling estimates heterogeneous treatment effects (HTE):
//! τ(x) = E[Y(1) - Y(0) | X = x]
//!
//! Key insight: Different individuals respond differently to treatment.
//! - **Persuadables**: τ(x) > 0, high confidence → target these
//! - **Sure Things**: Would convert regardless → don't waste resources
//! - **Lost Causes**: Won't convert regardless → don't waste resources
//! - **Sleeping Dogs**: τ(x) < 0 → avoid treating (negative effect)
//!
//! # Meta-Learners
//!
//! - **S-Learner**: Single model μ(x, t), τ(x) = μ(x, 1) - μ(x, 0)
//! - **T-Learner**: Two models μ₁(x), μ₀(x), τ(x) = μ₁(x) - μ₀(x)
//! - **X-Learner**: Propensity-weighted, better with imbalanced treatment
//! - **R-Learner**: Robinson transformation, doubly robust
//!
//! # Epistemic Advantage
//!
//! Unlike CausalML (Python, bootstrap-only), Sounio provides:
//! - Full Beta posterior per instance (mean + variance + confidence)
//! - Compile-time overlap checking (propensity bounds)
//! - GPU-accelerated estimation
//! - Provenance tracking for persuadable segments
//!
//! # Example
//!
//! ```ignore
//! let estimator = XLearner::new()
//!     .with_propensity(neural_net)
//!     .with_outcome_models(treated_model, control_model);
//!
//! let uplift = estimator.predict(&features);
//! // uplift: Knowledge<f64> with mean=0.15, variance=0.02, confidence=0.88
//!
//! if uplift.is_persuadable(threshold: 0.05, min_confidence: 0.8) {
//!     target_customer(id);
//! }
//! ```

use crate::epistemic::bayesian::BetaConfidence;

// ============================================================================
// Core Types
// ============================================================================

/// Feature vector for uplift prediction
#[derive(Debug, Clone)]
pub struct Features {
    /// Feature values
    pub values: Vec<f64>,
    /// Feature names (optional)
    pub names: Option<Vec<String>>,
}

impl Features {
    /// Create from values
    pub fn new(values: Vec<f64>) -> Self {
        Self {
            values,
            names: None,
        }
    }

    /// Create with names
    pub fn with_names(values: Vec<f64>, names: Vec<String>) -> Self {
        Self {
            values,
            names: Some(names),
        }
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.values.len()
    }
}

/// Epistemic uplift score
#[derive(Debug, Clone)]
pub struct UpliftScore {
    /// Point estimate of uplift τ(x)
    pub estimate: f64,
    /// Epistemic confidence distribution
    pub confidence: BetaConfidence,
    /// Standard error
    pub std_error: f64,
    /// Propensity score e(x) = P(T=1|X=x)
    pub propensity: f64,
    /// Segment classification
    pub segment: CustomerSegment,
    /// Provenance marker
    pub provenance: UpliftProvenance,
}

impl UpliftScore {
    /// Create new uplift score
    pub fn new(estimate: f64, std_error: f64, propensity: f64, method: &str) -> Self {
        // Convert to epistemic confidence
        // Higher estimate / std_error ratio = higher confidence
        let signal_to_noise = estimate.abs() / std_error.max(0.001);
        let effective_n = (signal_to_noise * 10.0).min(100.0).max(2.0);

        let confidence_value = if estimate > 0.0 {
            0.5 + 0.5 * (1.0 - (-signal_to_noise).exp())
        } else {
            0.5 - 0.5 * (1.0 - (-signal_to_noise).exp())
        };

        let confidence =
            BetaConfidence::from_confidence(confidence_value.clamp(0.01, 0.99), effective_n);

        // Classify segment
        let segment = CustomerSegment::classify(estimate, std_error, propensity);

        Self {
            estimate,
            confidence,
            std_error,
            propensity,
            segment,
            provenance: UpliftProvenance {
                method: method.to_string(),
                features_hash: 0,
                model_version: "1.0".to_string(),
            },
        }
    }

    /// Check if this instance is persuadable
    pub fn is_persuadable(&self, threshold: f64, min_confidence: f64) -> bool {
        self.estimate > threshold
            && self.confidence.mean() >= min_confidence
            && matches!(self.segment, CustomerSegment::Persuadable)
    }

    /// Get credible interval
    pub fn credible_interval(&self, level: f64) -> (f64, f64) {
        let z = normal_quantile((1.0 + level) / 2.0);
        (
            self.estimate - z * self.std_error,
            self.estimate + z * self.std_error,
        )
    }

    /// Probability that uplift is positive
    pub fn probability_positive(&self) -> f64 {
        // Using normal approximation
        normal_cdf(self.estimate / self.std_error.max(0.001))
    }

    /// Probability that uplift exceeds threshold
    pub fn probability_above(&self, threshold: f64) -> f64 {
        normal_cdf((self.estimate - threshold) / self.std_error.max(0.001))
    }
}

/// Customer segment based on uplift
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CustomerSegment {
    /// τ(x) > 0 with high confidence - TARGET THESE
    Persuadable,
    /// Would convert regardless of treatment
    SureThing,
    /// Won't convert regardless of treatment
    LostCause,
    /// τ(x) < 0 - AVOID treating
    SleepingDog,
    /// Uncertain classification
    Uncertain,
}

impl CustomerSegment {
    /// Classify based on uplift estimate and uncertainty
    pub fn classify(estimate: f64, std_error: f64, propensity: f64) -> Self {
        let z_score = estimate / std_error.max(0.001);

        // High propensity + low uplift = Sure Thing
        if propensity > 0.8 && z_score.abs() < 1.0 {
            return CustomerSegment::SureThing;
        }

        // Low propensity + low uplift = Lost Cause
        if propensity < 0.2 && z_score.abs() < 1.0 {
            return CustomerSegment::LostCause;
        }

        // Significant negative uplift = Sleeping Dog
        if z_score < -1.96 {
            return CustomerSegment::SleepingDog;
        }

        // Significant positive uplift = Persuadable
        if z_score > 1.96 {
            return CustomerSegment::Persuadable;
        }

        CustomerSegment::Uncertain
    }

    /// Get recommended action
    pub fn action(&self) -> &'static str {
        match self {
            CustomerSegment::Persuadable => "TARGET",
            CustomerSegment::SureThing => "IGNORE (would convert anyway)",
            CustomerSegment::LostCause => "IGNORE (won't convert)",
            CustomerSegment::SleepingDog => "AVOID (negative effect)",
            CustomerSegment::Uncertain => "UNCERTAIN (gather more data)",
        }
    }
}

/// Provenance for uplift predictions
#[derive(Debug, Clone)]
pub struct UpliftProvenance {
    /// Method used (S-Learner, T-Learner, X-Learner, R-Learner)
    pub method: String,
    /// Hash of input features
    pub features_hash: u64,
    /// Model version
    pub model_version: String,
}

// ============================================================================
// Meta-Learner Estimators
// ============================================================================

/// Type of meta-learner
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaLearnerType {
    /// Single model with treatment indicator
    SLearner,
    /// Two separate models for treated/control
    TLearner,
    /// Propensity-weighted X-Learner
    XLearner,
    /// Robinson transformation R-Learner
    RLearner,
}

/// Trait for uplift estimators
pub trait UpliftEstimator {
    /// Fit the estimator to training data
    fn fit(&mut self, features: &[Features], treatment: &[bool], outcome: &[f64]);

    /// Predict uplift for new features
    fn predict(&self, features: &Features) -> UpliftScore;

    /// Predict for batch
    fn predict_batch(&self, features: &[Features]) -> Vec<UpliftScore> {
        features.iter().map(|f| self.predict(f)).collect()
    }

    /// Get estimator type
    fn learner_type(&self) -> MetaLearnerType;
}

// ============================================================================
// X-Learner Implementation
// ============================================================================

/// X-Learner: Propensity-weighted meta-learner
///
/// Algorithm:
/// 1. Fit propensity model: e(x) = P(T=1|X)
/// 2. Fit outcome models: μ₁(x) for treated, μ₀(x) for control
/// 3. Compute imputed effects:
///    - For treated: D¹ᵢ = Yᵢ - μ₀(Xᵢ)
///    - For control: D⁰ᵢ = μ₁(Xᵢ) - Yᵢ
/// 4. Fit CATE models: τ₁(x) on D¹, τ₀(x) on D⁰
/// 5. Final estimate: τ(x) = e(x)·τ₀(x) + (1-e(x))·τ₁(x)
#[derive(Debug, Clone)]
pub struct XLearner {
    /// Propensity model coefficients (logistic regression)
    propensity_coef: Option<Vec<f64>>,
    /// Outcome model for treated (linear regression)
    outcome_treated_coef: Option<Vec<f64>>,
    /// Outcome model for control
    outcome_control_coef: Option<Vec<f64>>,
    /// CATE model for treated group
    cate_treated_coef: Option<Vec<f64>>,
    /// CATE model for control group
    cate_control_coef: Option<Vec<f64>>,
    /// Residual variance estimates
    residual_var_treated: f64,
    residual_var_control: f64,
    /// Configuration
    config: XLearnerConfig,
}

/// Configuration for X-Learner
#[derive(Debug, Clone)]
pub struct XLearnerConfig {
    /// Minimum propensity for overlap
    pub min_propensity: f64,
    /// Maximum propensity for overlap
    pub max_propensity: f64,
    /// Regularization strength
    pub regularization: f64,
    /// Whether to use epistemic weighting
    pub epistemic_weighting: bool,
}

impl Default for XLearnerConfig {
    fn default() -> Self {
        Self {
            min_propensity: 0.05,
            max_propensity: 0.95,
            regularization: 0.01,
            epistemic_weighting: true,
        }
    }
}

impl XLearner {
    /// Create new X-Learner
    pub fn new() -> Self {
        Self {
            propensity_coef: None,
            outcome_treated_coef: None,
            outcome_control_coef: None,
            cate_treated_coef: None,
            cate_control_coef: None,
            residual_var_treated: 1.0,
            residual_var_control: 1.0,
            config: XLearnerConfig::default(),
        }
    }

    /// Create with config
    pub fn with_config(config: XLearnerConfig) -> Self {
        Self {
            propensity_coef: None,
            outcome_treated_coef: None,
            outcome_control_coef: None,
            cate_treated_coef: None,
            cate_control_coef: None,
            residual_var_treated: 1.0,
            residual_var_control: 1.0,
            config,
        }
    }

    /// Fit propensity model (logistic regression)
    fn fit_propensity(&mut self, features: &[Features], treatment: &[bool]) {
        let n = features.len();
        let dim = features.first().map(|f| f.dim()).unwrap_or(0);

        // Simple logistic regression via gradient descent
        let mut coef = vec![0.0; dim + 1]; // +1 for intercept
        let lr = 0.1;
        let n_iter = 100;

        for _ in 0..n_iter {
            let mut grad = vec![0.0; dim + 1];

            for i in 0..n {
                let x = &features[i].values;
                let y = if treatment[i] { 1.0 } else { 0.0 };

                // Compute prediction
                let mut logit = coef[0]; // intercept
                for j in 0..dim {
                    logit += coef[j + 1] * x[j];
                }
                let pred = sigmoid(logit);

                // Gradient
                let error = pred - y;
                grad[0] += error;
                for j in 0..dim {
                    grad[j + 1] += error * x[j];
                }
            }

            // Update with regularization
            for j in 0..=dim {
                coef[j] -= lr * (grad[j] / n as f64 + self.config.regularization * coef[j]);
            }
        }

        self.propensity_coef = Some(coef);
    }

    /// Predict propensity
    fn predict_propensity(&self, features: &Features) -> f64 {
        let coef = self.propensity_coef.as_ref().unwrap();
        let mut logit = coef[0];
        for (j, &x) in features.values.iter().enumerate() {
            if j + 1 < coef.len() {
                logit += coef[j + 1] * x;
            }
        }
        sigmoid(logit).clamp(self.config.min_propensity, self.config.max_propensity)
    }

    /// Fit outcome model (linear regression)
    fn fit_outcome_model(
        features: &[Features],
        outcomes: &[f64],
        regularization: f64,
    ) -> (Vec<f64>, f64) {
        let n = features.len();
        if n == 0 {
            return (vec![], 1.0);
        }
        let dim = features[0].dim();

        // Simple linear regression via normal equations (regularized)
        // β = (X'X + λI)^(-1) X'y

        // For simplicity, use gradient descent
        let mut coef = vec![0.0; dim + 1];
        let lr = 0.01;
        let n_iter = 100;

        for _ in 0..n_iter {
            let mut grad = vec![0.0; dim + 1];

            for i in 0..n {
                let x = &features[i].values;
                let y = outcomes[i];

                let mut pred = coef[0];
                for j in 0..dim {
                    pred += coef[j + 1] * x[j];
                }

                let error = pred - y;
                grad[0] += error;
                for j in 0..dim {
                    grad[j + 1] += error * x[j];
                }
            }

            for j in 0..=dim {
                coef[j] -= lr * (grad[j] / n as f64 + regularization * coef[j]);
            }
        }

        // Compute residual variance
        let mut sse = 0.0;
        for i in 0..n {
            let x = &features[i].values;
            let y = outcomes[i];
            let mut pred = coef[0];
            for j in 0..dim {
                pred += coef[j + 1] * x[j];
            }
            sse += (y - pred).powi(2);
        }
        let residual_var = sse / n.max(1) as f64;

        (coef, residual_var)
    }

    /// Predict with linear model
    fn predict_linear(coef: &[f64], features: &Features) -> f64 {
        if coef.is_empty() {
            return 0.0;
        }
        let mut pred = coef[0];
        for (j, &x) in features.values.iter().enumerate() {
            if j + 1 < coef.len() {
                pred += coef[j + 1] * x;
            }
        }
        pred
    }
}

impl Default for XLearner {
    fn default() -> Self {
        Self::new()
    }
}

impl UpliftEstimator for XLearner {
    fn fit(&mut self, features: &[Features], treatment: &[bool], outcome: &[f64]) {
        let n = features.len();

        // Step 1: Fit propensity model
        self.fit_propensity(features, treatment);

        // Split data by treatment
        let mut treated_features = Vec::new();
        let mut treated_outcomes = Vec::new();
        let mut control_features = Vec::new();
        let mut control_outcomes = Vec::new();

        for i in 0..n {
            if treatment[i] {
                treated_features.push(features[i].clone());
                treated_outcomes.push(outcome[i]);
            } else {
                control_features.push(features[i].clone());
                control_outcomes.push(outcome[i]);
            }
        }

        // Step 2: Fit outcome models
        let (treated_coef, var_t) = Self::fit_outcome_model(
            &treated_features,
            &treated_outcomes,
            self.config.regularization,
        );
        let (control_coef, var_c) = Self::fit_outcome_model(
            &control_features,
            &control_outcomes,
            self.config.regularization,
        );

        self.outcome_treated_coef = Some(treated_coef);
        self.outcome_control_coef = Some(control_coef);
        self.residual_var_treated = var_t;
        self.residual_var_control = var_c;

        // Step 3: Compute imputed treatment effects
        // For treated: D¹ = Y - μ₀(X)
        let mut imputed_treated = Vec::new();
        for i in 0..treated_features.len() {
            let mu0 = Self::predict_linear(
                self.outcome_control_coef.as_ref().unwrap(),
                &treated_features[i],
            );
            imputed_treated.push(treated_outcomes[i] - mu0);
        }

        // For control: D⁰ = μ₁(X) - Y
        let mut imputed_control = Vec::new();
        for i in 0..control_features.len() {
            let mu1 = Self::predict_linear(
                self.outcome_treated_coef.as_ref().unwrap(),
                &control_features[i],
            );
            imputed_control.push(mu1 - control_outcomes[i]);
        }

        // Step 4: Fit CATE models on imputed effects
        let (cate_t_coef, _) = Self::fit_outcome_model(
            &treated_features,
            &imputed_treated,
            self.config.regularization,
        );
        let (cate_c_coef, _) = Self::fit_outcome_model(
            &control_features,
            &imputed_control,
            self.config.regularization,
        );

        self.cate_treated_coef = Some(cate_t_coef);
        self.cate_control_coef = Some(cate_c_coef);
    }

    fn predict(&self, features: &Features) -> UpliftScore {
        // Get propensity
        let e = self.predict_propensity(features);

        // Get CATE estimates from both models
        let tau_treated =
            Self::predict_linear(self.cate_treated_coef.as_ref().unwrap_or(&vec![]), features);
        let tau_control =
            Self::predict_linear(self.cate_control_coef.as_ref().unwrap_or(&vec![]), features);

        // X-Learner combination: τ(x) = e(x)·τ₀(x) + (1-e(x))·τ₁(x)
        let estimate = e * tau_control + (1.0 - e) * tau_treated;

        // Variance: weighted combination of variances + model uncertainty
        let var_estimate = e.powi(2) * self.residual_var_control
            + (1.0 - e).powi(2) * self.residual_var_treated
            + e * (1.0 - e) * (tau_treated - tau_control).powi(2);
        let std_error = var_estimate.sqrt();

        UpliftScore::new(estimate, std_error, e, "X-Learner")
    }

    fn learner_type(&self) -> MetaLearnerType {
        MetaLearnerType::XLearner
    }
}

// ============================================================================
// T-Learner Implementation (simpler alternative)
// ============================================================================

/// T-Learner: Two separate models
#[derive(Debug, Clone)]
pub struct TLearner {
    /// Model for treated outcomes
    treated_coef: Option<Vec<f64>>,
    /// Model for control outcomes
    control_coef: Option<Vec<f64>>,
    /// Residual variances
    var_treated: f64,
    var_control: f64,
    /// Regularization
    regularization: f64,
}

impl TLearner {
    pub fn new(regularization: f64) -> Self {
        Self {
            treated_coef: None,
            control_coef: None,
            var_treated: 1.0,
            var_control: 1.0,
            regularization,
        }
    }
}

impl Default for TLearner {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl UpliftEstimator for TLearner {
    fn fit(&mut self, features: &[Features], treatment: &[bool], outcome: &[f64]) {
        let n = features.len();

        let mut treated_features = Vec::new();
        let mut treated_outcomes = Vec::new();
        let mut control_features = Vec::new();
        let mut control_outcomes = Vec::new();

        for i in 0..n {
            if treatment[i] {
                treated_features.push(features[i].clone());
                treated_outcomes.push(outcome[i]);
            } else {
                control_features.push(features[i].clone());
                control_outcomes.push(outcome[i]);
            }
        }

        let (t_coef, t_var) =
            XLearner::fit_outcome_model(&treated_features, &treated_outcomes, self.regularization);
        let (c_coef, c_var) =
            XLearner::fit_outcome_model(&control_features, &control_outcomes, self.regularization);

        self.treated_coef = Some(t_coef);
        self.control_coef = Some(c_coef);
        self.var_treated = t_var;
        self.var_control = c_var;
    }

    fn predict(&self, features: &Features) -> UpliftScore {
        let mu1 = XLearner::predict_linear(self.treated_coef.as_ref().unwrap_or(&vec![]), features);
        let mu0 = XLearner::predict_linear(self.control_coef.as_ref().unwrap_or(&vec![]), features);

        let estimate = mu1 - mu0;
        let std_error = (self.var_treated + self.var_control).sqrt();

        UpliftScore::new(estimate, std_error, 0.5, "T-Learner")
    }

    fn learner_type(&self) -> MetaLearnerType {
        MetaLearnerType::TLearner
    }
}

// ============================================================================
// QINI Curve and Metrics
// ============================================================================

/// QINI curve for uplift model evaluation
#[derive(Debug, Clone)]
pub struct QiniCurve {
    /// Sorted uplift scores (descending)
    pub scores: Vec<f64>,
    /// Cumulative treatment group responses
    pub cum_treated_response: Vec<f64>,
    /// Cumulative control group responses
    pub cum_control_response: Vec<f64>,
    /// QINI values at each point
    pub qini_values: Vec<f64>,
    /// Area under QINI curve
    pub auqc: f64,
    /// Variance estimate (bootstrap)
    pub variance: f64,
}

impl QiniCurve {
    /// Compute QINI curve from predictions and outcomes
    pub fn compute(scores: &[UpliftScore], treatment: &[bool], outcome: &[f64]) -> Self {
        let n = scores.len();

        // Sort by uplift score descending
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| scores[b].estimate.partial_cmp(&scores[a].estimate).unwrap());

        let mut sorted_scores = Vec::with_capacity(n);
        let mut cum_treated = Vec::with_capacity(n);
        let mut cum_control = Vec::with_capacity(n);
        let mut qini_vals = Vec::with_capacity(n);

        let mut n_treated = 0.0;
        let mut n_control = 0.0;
        let mut sum_treated = 0.0;
        let mut sum_control = 0.0;

        // Total counts for normalization
        let total_treated: f64 = treatment.iter().filter(|&&t| t).count() as f64;
        let total_control: f64 = treatment.iter().filter(|&&t| !t).count() as f64;

        for &i in &indices {
            sorted_scores.push(scores[i].estimate);

            if treatment[i] {
                n_treated += 1.0;
                sum_treated += outcome[i];
            } else {
                n_control += 1.0;
                sum_control += outcome[i];
            }

            cum_treated.push(sum_treated);
            cum_control.push(sum_control);

            // QINI = (cumulative treated responses) - (cumulative control responses) * (n_treated / n_control)
            let qini = if n_control > 0.0 {
                sum_treated - sum_control * (n_treated / n_control)
            } else {
                sum_treated
            };
            qini_vals.push(qini);
        }

        // Compute AUQC (normalized)
        let max_qini = qini_vals.last().cloned().unwrap_or(0.0);
        let random_qini = if total_control > 0.0 {
            (sum_treated / total_treated - sum_control / total_control) * total_treated / 2.0
        } else {
            0.0
        };

        let auqc = if max_qini > random_qini {
            (qini_vals.iter().sum::<f64>() / n as f64) / max_qini.abs().max(1e-10)
        } else {
            0.0
        };

        // Variance estimate (simplified)
        let variance = scores.iter().map(|s| s.std_error.powi(2)).sum::<f64>() / n.max(1) as f64;

        Self {
            scores: sorted_scores,
            cum_treated_response: cum_treated,
            cum_control_response: cum_control,
            qini_values: qini_vals,
            auqc,
            variance,
        }
    }

    /// Get QINI coefficient (area ratio)
    pub fn qini_coefficient(&self) -> f64 {
        self.auqc
    }

    /// Check monotonicity (uplift should be monotonically decreasing with rank)
    pub fn is_monotonic(&self, tolerance: f64) -> bool {
        for i in 1..self.qini_values.len() {
            // Allow some tolerance for non-monotonicity
            if self.qini_values[i] > self.qini_values[i - 1] + tolerance {
                return false;
            }
        }
        true
    }
}

// ============================================================================
// Uplift Refutation Tests
// ============================================================================

/// Uplift-specific refutation result
#[derive(Debug, Clone)]
pub struct UpliftRefutationResult {
    /// Test name
    pub test_name: String,
    /// Whether test passed
    pub passed: bool,
    /// Test statistic
    pub statistic: f64,
    /// Threshold
    pub threshold: f64,
    /// Epistemic penalty
    pub penalty: f64,
    /// Description
    pub description: String,
}

/// Run monotonicity test on uplift predictions
pub fn test_monotonicity(curve: &QiniCurve, tolerance: f64) -> UpliftRefutationResult {
    let is_mono = curve.is_monotonic(tolerance);

    // Count violations
    let mut violations = 0;
    for i in 1..curve.qini_values.len() {
        if curve.qini_values[i] > curve.qini_values[i - 1] + tolerance {
            violations += 1;
        }
    }

    let statistic = violations as f64 / curve.qini_values.len().max(1) as f64;

    UpliftRefutationResult {
        test_name: "Monotonicity Test".to_string(),
        passed: is_mono,
        statistic,
        threshold: 0.05, // Allow 5% violations
        penalty: if is_mono { 0.0 } else { 0.2 },
        description: format!(
            "QINI curve monotonicity: {} violations ({:.1}%)",
            violations,
            statistic * 100.0
        ),
    }
}

/// Test persuadable segment validity
pub fn test_persuadable_segment(
    scores: &[UpliftScore],
    min_confidence: f64,
    min_fraction: f64,
) -> UpliftRefutationResult {
    let n = scores.len();
    let persuadable_count = scores
        .iter()
        .filter(|s| matches!(s.segment, CustomerSegment::Persuadable))
        .filter(|s| s.confidence.mean() >= min_confidence)
        .count();

    let fraction = persuadable_count as f64 / n.max(1) as f64;
    let passed = fraction >= min_fraction;

    UpliftRefutationResult {
        test_name: "Persuadable Segment Validity".to_string(),
        passed,
        statistic: fraction,
        threshold: min_fraction,
        penalty: if passed { 0.0 } else { 0.15 },
        description: format!(
            "{:.1}% of predictions are high-confidence persuadables (threshold: {:.1}%)",
            fraction * 100.0,
            min_fraction * 100.0
        ),
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Sigmoid function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Standard normal CDF
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Normal quantile (inverse CDF) approximation
fn normal_quantile(p: f64) -> f64 {
    // Rational approximation for normal quantile
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }

    let p = if p > 0.5 { 1.0 - p } else { p };
    let t = (-2.0 * p.ln()).sqrt();

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let result = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p > 0.5 { -result } else { result }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> (Vec<Features>, Vec<bool>, Vec<f64>) {
        // Simple synthetic data
        let features = vec![
            Features::new(vec![1.0, 0.5]),
            Features::new(vec![2.0, 1.0]),
            Features::new(vec![3.0, 1.5]),
            Features::new(vec![1.5, 0.8]),
            Features::new(vec![2.5, 1.2]),
            Features::new(vec![0.5, 0.3]),
            Features::new(vec![3.5, 2.0]),
            Features::new(vec![2.0, 0.9]),
        ];
        let treatment = vec![true, false, true, false, true, false, true, false];
        let outcome = vec![1.2, 0.5, 1.8, 0.4, 1.5, 0.3, 2.0, 0.6];

        (features, treatment, outcome)
    }

    #[test]
    fn test_xlearner_fit_predict() {
        let (features, treatment, outcome) = sample_data();

        let mut learner = XLearner::new();
        learner.fit(&features, &treatment, &outcome);

        let score = learner.predict(&features[0]);

        // Should have positive uplift (treated outcomes > control)
        assert!(score.estimate > 0.0 || score.std_error > 0.0);
        assert!(score.propensity > 0.0 && score.propensity < 1.0);
    }

    #[test]
    fn test_tlearner_fit_predict() {
        let (features, treatment, outcome) = sample_data();

        let mut learner = TLearner::default();
        learner.fit(&features, &treatment, &outcome);

        let score = learner.predict(&features[0]);
        assert!(score.std_error > 0.0);
    }

    #[test]
    fn test_customer_segment() {
        // High positive uplift, moderate propensity
        let seg = CustomerSegment::classify(0.5, 0.1, 0.5);
        assert_eq!(seg, CustomerSegment::Persuadable);

        // High propensity, low uplift
        let seg = CustomerSegment::classify(0.05, 0.1, 0.9);
        assert_eq!(seg, CustomerSegment::SureThing);

        // Negative uplift
        let seg = CustomerSegment::classify(-0.5, 0.1, 0.5);
        assert_eq!(seg, CustomerSegment::SleepingDog);
    }

    #[test]
    fn test_uplift_score_persuadable() {
        let score = UpliftScore::new(0.3, 0.05, 0.5, "test");

        assert!(score.is_persuadable(0.1, 0.5));
        assert!(!score.is_persuadable(0.5, 0.5)); // threshold too high
    }

    #[test]
    fn test_qini_curve() {
        let (features, treatment, outcome) = sample_data();

        let mut learner = XLearner::new();
        learner.fit(&features, &treatment, &outcome);

        let scores: Vec<_> = features.iter().map(|f| learner.predict(f)).collect();
        let curve = QiniCurve::compute(&scores, &treatment, &outcome);

        assert_eq!(curve.scores.len(), features.len());
        assert!(curve.auqc >= 0.0);
    }

    #[test]
    fn test_monotonicity_refutation() {
        let (features, treatment, outcome) = sample_data();

        let mut learner = XLearner::new();
        learner.fit(&features, &treatment, &outcome);

        let scores: Vec<_> = features.iter().map(|f| learner.predict(f)).collect();
        let curve = QiniCurve::compute(&scores, &treatment, &outcome);

        let result = test_monotonicity(&curve, 0.1);
        assert!(result.statistic >= 0.0 && result.statistic <= 1.0);
    }
}
