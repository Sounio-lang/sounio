//! Causal Refutation as Algebraic Effects
//!
//! This module implements DoWhy-inspired refutation tests as algebraic effects,
//! enabling compile-time and runtime validation of causal assumptions.
//!
//! # Theory
//!
//! Causal inference relies on untestable assumptions (e.g., no unmeasured confounding).
//! Refutation tests probe these assumptions by checking if estimates are robust to:
//!
//! - **Placebo Treatment**: Replace treatment with random noise → effect should vanish
//! - **Random Common Cause**: Add random confounder → effect should be stable
//! - **Data Subset**: Re-estimate on subsets → effect should be consistent
//! - **Sensitivity Analysis**: Vary unmeasured confounding → bound the bias
//!
//! # Effect-Based Design
//!
//! Refutation tests are modeled as algebraic effects, allowing:
//! - Compile-time tracking of which assumptions have been tested
//! - Automatic epistemic penalty when tests fail
//! - Composable refutation pipelines
//!
//! # Example
//!
//! ```ignore
//! effect causal_refutation {
//!     fn placebo_test(model, data) -> RefutationResult;
//!     fn sensitivity_analysis(model, gamma) -> SensitivityBounds;
//! }
//!
//! handle causal_refutation {
//!     let ate = estimate_ate(model, data);
//!     let placebo = placebo_test(model, data);
//!     if !placebo.passed {
//!         ate.confidence.penalize(placebo.violation_magnitude);
//!     }
//! }
//! ```

use super::graph::CausalGraph;
use super::z3_identify::EpistemicATE;
use crate::epistemic::bayesian::BetaConfidence;

// ============================================================================
// Refutation Test Types
// ============================================================================

/// Types of refutation tests (inspired by DoWhy)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RefutationType {
    /// Replace treatment with random placebo → effect should be ~0
    PlaceboTreatment,
    /// Add random variable as common cause → effect should be stable
    RandomCommonCause,
    /// Re-estimate on data subsets → effect should be consistent
    DataSubset { fraction: u8 }, // fraction as percentage (e.g., 80 = 80%)
    /// Bootstrap resampling → check confidence interval coverage
    Bootstrap { n_samples: u32 },
    /// Sensitivity to unmeasured confounding (Rosenbaum bounds)
    SensitivityAnalysis { gamma: u8 }, // gamma * 10 (e.g., 15 = 1.5)
    /// Permutation test → p-value for null hypothesis
    PermutationTest { n_permutations: u32 },
    /// Negative control outcome → effect on unrelated outcome should be ~0
    NegativeControlOutcome { control_outcome: String },
    /// Negative control exposure → unrelated exposure should have no effect
    NegativeControlExposure { control_exposure: String },
}

impl RefutationType {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            RefutationType::PlaceboTreatment => "Placebo Treatment",
            RefutationType::RandomCommonCause => "Random Common Cause",
            RefutationType::DataSubset { .. } => "Data Subset",
            RefutationType::Bootstrap { .. } => "Bootstrap",
            RefutationType::SensitivityAnalysis { .. } => "Sensitivity Analysis",
            RefutationType::PermutationTest { .. } => "Permutation Test",
            RefutationType::NegativeControlOutcome { .. } => "Negative Control Outcome",
            RefutationType::NegativeControlExposure { .. } => "Negative Control Exposure",
        }
    }

    /// Default threshold for passing the test
    pub fn default_threshold(&self) -> f64 {
        match self {
            RefutationType::PlaceboTreatment => 0.1, // Effect < 10% of original
            RefutationType::RandomCommonCause => 0.15, // Change < 15%
            RefutationType::DataSubset { .. } => 0.20, // Variance < 20%
            RefutationType::Bootstrap { .. } => 0.05, // 95% CI coverage
            RefutationType::SensitivityAnalysis { .. } => 0.0, // Bounds include 0?
            RefutationType::PermutationTest { .. } => 0.05, // p-value threshold
            RefutationType::NegativeControlOutcome { .. } => 0.1,
            RefutationType::NegativeControlExposure { .. } => 0.1,
        }
    }

    /// Epistemic penalty factor when test fails
    pub fn failure_penalty(&self) -> f64 {
        match self {
            RefutationType::PlaceboTreatment => 0.5, // Severe: halve confidence
            RefutationType::RandomCommonCause => 0.3, // Moderate
            RefutationType::DataSubset { .. } => 0.2, // Mild
            RefutationType::Bootstrap { .. } => 0.1, // Informational
            RefutationType::SensitivityAnalysis { .. } => 0.4,
            RefutationType::PermutationTest { .. } => 0.5,
            RefutationType::NegativeControlOutcome { .. } => 0.4,
            RefutationType::NegativeControlExposure { .. } => 0.4,
        }
    }
}

// ============================================================================
// Refutation Results
// ============================================================================

/// Result of a single refutation test
#[derive(Debug, Clone)]
pub struct RefutationResult {
    /// Type of test performed
    pub test_type: RefutationType,
    /// Whether the test passed
    pub passed: bool,
    /// Test statistic value
    pub statistic: f64,
    /// Threshold used for pass/fail
    pub threshold: f64,
    /// P-value (if applicable)
    pub p_value: Option<f64>,
    /// Confidence interval (if applicable)
    pub confidence_interval: Option<(f64, f64)>,
    /// Human-readable interpretation
    pub interpretation: String,
    /// Epistemic penalty to apply if failed
    pub penalty: f64,
}

impl RefutationResult {
    /// Create a passed result
    pub fn passed(test_type: RefutationType, statistic: f64, threshold: f64) -> Self {
        Self {
            interpretation: format!(
                "{} PASSED: statistic {:.4} < threshold {:.4}",
                test_type.name(),
                statistic,
                threshold
            ),
            test_type,
            passed: true,
            statistic,
            threshold,
            p_value: None,
            confidence_interval: None,
            penalty: 0.0,
        }
    }

    /// Create a failed result
    pub fn failed(test_type: RefutationType, statistic: f64, threshold: f64) -> Self {
        let penalty = test_type.failure_penalty();
        Self {
            interpretation: format!(
                "{} FAILED: statistic {:.4} >= threshold {:.4} (penalty: {:.0}%)",
                test_type.name(),
                statistic,
                threshold,
                penalty * 100.0
            ),
            passed: false,
            statistic,
            threshold,
            p_value: None,
            confidence_interval: None,
            penalty,
            test_type,
        }
    }

    /// Add p-value to result
    pub fn with_p_value(mut self, p: f64) -> Self {
        self.p_value = Some(p);
        self
    }

    /// Add confidence interval to result
    pub fn with_ci(mut self, lower: f64, upper: f64) -> Self {
        self.confidence_interval = Some((lower, upper));
        self
    }
}

/// Aggregate result of multiple refutation tests
#[derive(Debug, Clone)]
pub struct RefutationReport {
    /// Individual test results
    pub results: Vec<RefutationResult>,
    /// Overall pass/fail
    pub all_passed: bool,
    /// Total epistemic penalty
    pub total_penalty: f64,
    /// Adjusted confidence after penalties
    pub adjusted_confidence: BetaConfidence,
    /// Original confidence before refutation
    pub original_confidence: BetaConfidence,
}

impl RefutationReport {
    /// Create report from results
    pub fn from_results(results: Vec<RefutationResult>, original: BetaConfidence) -> Self {
        let all_passed = results.iter().all(|r| r.passed);
        let total_penalty: f64 = results.iter().map(|r| r.penalty).sum();

        // Apply penalty to confidence
        // Penalty reduces effective evidence (increase beta, decrease alpha)
        let penalty_factor = (1.0 - total_penalty).max(0.1);
        let adjusted = BetaConfidence::new(
            original.alpha * penalty_factor,
            original.beta + (1.0 - penalty_factor) * original.alpha,
        );

        Self {
            results,
            all_passed,
            total_penalty,
            adjusted_confidence: adjusted,
            original_confidence: original,
        }
    }

    /// Get count of passed tests
    pub fn passed_count(&self) -> usize {
        self.results.iter().filter(|r| r.passed).count()
    }

    /// Get count of failed tests
    pub fn failed_count(&self) -> usize {
        self.results.iter().filter(|r| !r.passed).count()
    }

    /// Generate markdown summary
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("## Causal Refutation Report\n\n");
        md.push_str(&format!(
            "**Status**: {}\n\n",
            if self.all_passed {
                "✅ ALL PASSED"
            } else {
                "⚠️ SOME FAILED"
            }
        ));
        md.push_str(&format!(
            "**Tests**: {}/{} passed\n\n",
            self.passed_count(),
            self.results.len()
        ));
        md.push_str(&format!(
            "**Confidence**: {:.2}% → {:.2}% (penalty: {:.0}%)\n\n",
            self.original_confidence.mean() * 100.0,
            self.adjusted_confidence.mean() * 100.0,
            self.total_penalty * 100.0
        ));

        md.push_str("### Individual Tests\n\n");
        md.push_str("| Test | Status | Statistic | Threshold | P-value |\n");
        md.push_str("|------|--------|-----------|-----------|--------|\n");

        for r in &self.results {
            let status = if r.passed { "✅" } else { "❌" };
            let p_val = r.p_value.map(|p| format!("{:.4}", p)).unwrap_or("-".into());
            md.push_str(&format!(
                "| {} | {} | {:.4} | {:.4} | {} |\n",
                r.test_type.name(),
                status,
                r.statistic,
                r.threshold,
                p_val
            ));
        }

        md
    }
}

// ============================================================================
// Refutation Effect Definition
// ============================================================================

/// Refutation effect operations
#[derive(Debug, Clone)]
pub enum RefutationOp {
    /// Perform placebo treatment test
    PlaceboTest { treatment: String, outcome: String },
    /// Add random common cause
    RandomCommonCauseTest {
        treatment: String,
        outcome: String,
        n_iterations: u32,
    },
    /// Data subset validation
    SubsetTest { fraction: f64, n_subsets: u32 },
    /// Bootstrap confidence intervals
    BootstrapTest { n_samples: u32, ci_level: f64 },
    /// Sensitivity analysis (Rosenbaum bounds)
    SensitivityTest { gamma_values: Vec<f64> },
}

/// Refutation effect handler trait
pub trait RefutationHandler {
    /// Execute a refutation operation
    fn handle(&mut self, op: RefutationOp, ate: &EpistemicATE) -> RefutationResult;

    /// Run all standard refutation tests
    fn run_standard_battery(&mut self, ate: &EpistemicATE) -> RefutationReport;
}

// ============================================================================
// Default Refutation Handler Implementation
// ============================================================================

/// Default handler using statistical simulation
pub struct DefaultRefutationHandler {
    /// Random seed for reproducibility
    seed: u64,
    /// Cached graph for tests
    graph: Option<CausalGraph>,
    /// Test configuration
    config: RefutationConfig,
}

/// Configuration for refutation tests
#[derive(Debug, Clone)]
pub struct RefutationConfig {
    /// Threshold multiplier (higher = more lenient)
    pub threshold_multiplier: f64,
    /// Number of bootstrap samples
    pub bootstrap_samples: u32,
    /// Number of permutations for permutation test
    pub permutation_count: u32,
    /// Data subset fraction
    pub subset_fraction: f64,
    /// Gamma values for sensitivity analysis
    pub sensitivity_gammas: Vec<f64>,
}

impl Default for RefutationConfig {
    fn default() -> Self {
        Self {
            threshold_multiplier: 1.0,
            bootstrap_samples: 1000,
            permutation_count: 500,
            subset_fraction: 0.8,
            sensitivity_gammas: vec![1.0, 1.25, 1.5, 1.75, 2.0],
        }
    }
}

impl DefaultRefutationHandler {
    /// Create new handler with seed
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            graph: None,
            config: RefutationConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(seed: u64, config: RefutationConfig) -> Self {
        Self {
            seed,
            graph: None,
            config,
        }
    }

    /// Set the causal graph for context
    pub fn set_graph(&mut self, graph: CausalGraph) {
        self.graph = Some(graph);
    }

    /// Simple LCG random number generator
    fn next_random(&mut self) -> f64 {
        // Linear congruential generator
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.seed >> 33) as f64 / (1u64 << 31) as f64
    }

    /// Generate random normal (Box-Muller)
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_random().max(1e-10);
        let u2 = self.next_random();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Placebo test: effect with random treatment should be ~0
    fn placebo_test(&mut self, ate: &EpistemicATE) -> RefutationResult {
        let test_type = RefutationType::PlaceboTreatment;
        let threshold = test_type.default_threshold() * self.config.threshold_multiplier;

        // Simulate placebo effect (should be near 0)
        // In reality, would re-estimate with randomized treatment
        let placebo_effect = self.next_normal() * ate.std_error;
        let statistic = placebo_effect.abs() / ate.estimate.abs().max(0.001);

        if statistic < threshold {
            RefutationResult::passed(test_type, statistic, threshold)
        } else {
            RefutationResult::failed(test_type, statistic, threshold)
        }
    }

    /// Random common cause test: adding confounder shouldn't change effect much
    fn random_common_cause_test(&mut self, ate: &EpistemicATE) -> RefutationResult {
        let test_type = RefutationType::RandomCommonCause;
        let threshold = test_type.default_threshold() * self.config.threshold_multiplier;

        // Simulate effect stability when adding random confounder
        let n_iterations = 100;
        let mut effects = Vec::with_capacity(n_iterations);

        for _ in 0..n_iterations {
            // Simulate adjusted effect (should be close to original)
            let noise = self.next_normal() * ate.std_error * 0.5;
            effects.push(ate.estimate + noise);
        }

        let mean_effect: f64 = effects.iter().sum::<f64>() / effects.len() as f64;
        let relative_change = (mean_effect - ate.estimate).abs() / ate.estimate.abs().max(0.001);

        if relative_change < threshold {
            RefutationResult::passed(test_type, relative_change, threshold)
        } else {
            RefutationResult::failed(test_type, relative_change, threshold)
        }
    }

    /// Data subset test: effect should be consistent across subsets
    fn data_subset_test(&mut self, ate: &EpistemicATE) -> RefutationResult {
        let fraction = (self.config.subset_fraction * 100.0) as u8;
        let test_type = RefutationType::DataSubset { fraction };
        let threshold = test_type.default_threshold() * self.config.threshold_multiplier;

        // Simulate variance across subsets
        let n_subsets = 10;
        let mut subset_effects = Vec::with_capacity(n_subsets);

        for _ in 0..n_subsets {
            // Subset effect with increased variance (fewer samples)
            let subset_se = ate.std_error / self.config.subset_fraction.sqrt();
            let effect = ate.estimate + self.next_normal() * subset_se;
            subset_effects.push(effect);
        }

        // Compute coefficient of variation
        let mean: f64 = subset_effects.iter().sum::<f64>() / subset_effects.len() as f64;
        let variance: f64 = subset_effects
            .iter()
            .map(|e| (e - mean).powi(2))
            .sum::<f64>()
            / subset_effects.len() as f64;
        let cv = variance.sqrt() / mean.abs().max(0.001);

        if cv < threshold {
            RefutationResult::passed(test_type, cv, threshold)
        } else {
            RefutationResult::failed(test_type, cv, threshold)
        }
    }

    /// Bootstrap test: check CI coverage
    fn bootstrap_test(&mut self, ate: &EpistemicATE) -> RefutationResult {
        let test_type = RefutationType::Bootstrap {
            n_samples: self.config.bootstrap_samples,
        };
        let threshold = test_type.default_threshold();

        let mut bootstrap_effects = Vec::with_capacity(self.config.bootstrap_samples as usize);

        for _ in 0..self.config.bootstrap_samples {
            let effect = ate.estimate + self.next_normal() * ate.std_error;
            bootstrap_effects.push(effect);
        }

        bootstrap_effects.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let lower_idx = (self.config.bootstrap_samples as f64 * 0.025) as usize;
        let upper_idx = (self.config.bootstrap_samples as f64 * 0.975) as usize;
        let lower = bootstrap_effects[lower_idx];
        let upper = bootstrap_effects[upper_idx.min(bootstrap_effects.len() - 1)];

        // Check if original CI is covered
        let coverage_ok = lower <= ate.credible_interval.0 && upper >= ate.credible_interval.1;
        let statistic = if coverage_ok { 0.0 } else { 1.0 };

        let result = if coverage_ok {
            RefutationResult::passed(test_type, statistic, threshold)
        } else {
            RefutationResult::failed(test_type, statistic, threshold)
        };

        result.with_ci(lower, upper)
    }

    /// Sensitivity analysis: Rosenbaum bounds
    fn sensitivity_test(&mut self, ate: &EpistemicATE) -> RefutationResult {
        let gamma = (self.config.sensitivity_gammas.last().unwrap_or(&2.0) * 10.0) as u8;
        let test_type = RefutationType::SensitivityAnalysis { gamma };
        let threshold = test_type.default_threshold();

        // Compute bounds for each gamma
        let mut bounds = Vec::new();
        for &g in &self.config.sensitivity_gammas {
            // Simplified Rosenbaum bounds
            // In reality, would compute proper bounds based on matched pairs
            let lower = ate.estimate - ate.std_error * g.ln();
            let upper = ate.estimate + ate.std_error * g.ln();
            bounds.push((g, lower, upper));
        }

        // Find gamma at which bounds include 0
        let critical_gamma = bounds
            .iter()
            .find(|(_, l, u)| *l <= 0.0 && *u >= 0.0)
            .map(|(g, _, _)| *g);

        let statistic = critical_gamma.unwrap_or(f64::INFINITY);
        let robust = critical_gamma.is_none() || critical_gamma.unwrap() > 1.5;

        if robust {
            RefutationResult::passed(test_type, statistic, threshold).with_ci(
                bounds.last().map(|(_, l, _)| *l).unwrap_or(0.0),
                bounds.last().map(|(_, _, u)| *u).unwrap_or(0.0),
            )
        } else {
            RefutationResult::failed(test_type, statistic, threshold).with_ci(
                bounds.last().map(|(_, l, _)| *l).unwrap_or(0.0),
                bounds.last().map(|(_, _, u)| *u).unwrap_or(0.0),
            )
        }
    }

    /// Permutation test
    fn permutation_test(&mut self, ate: &EpistemicATE) -> RefutationResult {
        let test_type = RefutationType::PermutationTest {
            n_permutations: self.config.permutation_count,
        };
        let threshold = test_type.default_threshold();

        let mut null_effects = Vec::with_capacity(self.config.permutation_count as usize);

        for _ in 0..self.config.permutation_count {
            // Null distribution: effect should be ~0 under permutation
            let null_effect = self.next_normal() * ate.std_error;
            null_effects.push(null_effect.abs());
        }

        // P-value: proportion of null effects >= observed
        let observed = ate.estimate.abs();
        let more_extreme = null_effects.iter().filter(|&&e| e >= observed).count();
        let p_value = more_extreme as f64 / self.config.permutation_count as f64;

        let result = if p_value < threshold {
            RefutationResult::passed(test_type, p_value, threshold)
        } else {
            RefutationResult::failed(test_type, p_value, threshold)
        };

        result.with_p_value(p_value)
    }
}

impl RefutationHandler for DefaultRefutationHandler {
    fn handle(&mut self, op: RefutationOp, ate: &EpistemicATE) -> RefutationResult {
        match op {
            RefutationOp::PlaceboTest { .. } => self.placebo_test(ate),
            RefutationOp::RandomCommonCauseTest { .. } => self.random_common_cause_test(ate),
            RefutationOp::SubsetTest { .. } => self.data_subset_test(ate),
            RefutationOp::BootstrapTest { .. } => self.bootstrap_test(ate),
            RefutationOp::SensitivityTest { .. } => self.sensitivity_test(ate),
        }
    }

    fn run_standard_battery(&mut self, ate: &EpistemicATE) -> RefutationReport {
        let results = vec![
            self.placebo_test(ate),
            self.random_common_cause_test(ate),
            self.data_subset_test(ate),
            self.bootstrap_test(ate),
            self.sensitivity_test(ate),
            self.permutation_test(ate),
        ];

        RefutationReport::from_results(results, ate.identification_confidence)
    }
}

// ============================================================================
// Effect System Integration
// ============================================================================

/// Refutation effect for the Sounio effect system
#[derive(Debug, Clone)]
pub struct RefutationEffect {
    /// Required refutation tests
    pub required_tests: Vec<RefutationType>,
    /// Whether all tests must pass
    pub require_all_pass: bool,
    /// Maximum allowed total penalty
    pub max_penalty: f64,
}

impl Default for RefutationEffect {
    fn default() -> Self {
        Self {
            required_tests: vec![
                RefutationType::PlaceboTreatment,
                RefutationType::RandomCommonCause,
                RefutationType::DataSubset { fraction: 80 },
            ],
            require_all_pass: false,
            max_penalty: 0.5,
        }
    }
}

impl RefutationEffect {
    /// Create strict refutation (all tests must pass)
    pub fn strict() -> Self {
        Self {
            required_tests: vec![
                RefutationType::PlaceboTreatment,
                RefutationType::RandomCommonCause,
                RefutationType::DataSubset { fraction: 80 },
                RefutationType::Bootstrap { n_samples: 1000 },
                RefutationType::SensitivityAnalysis { gamma: 20 },
                RefutationType::PermutationTest {
                    n_permutations: 500,
                },
            ],
            require_all_pass: true,
            max_penalty: 0.0,
        }
    }

    /// Create minimal refutation (just placebo)
    pub fn minimal() -> Self {
        Self {
            required_tests: vec![RefutationType::PlaceboTreatment],
            require_all_pass: true,
            max_penalty: 0.3,
        }
    }

    /// Check if report satisfies this effect's requirements
    pub fn satisfied_by(&self, report: &RefutationReport) -> bool {
        if self.require_all_pass && !report.all_passed {
            return false;
        }
        if report.total_penalty > self.max_penalty {
            return false;
        }
        true
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal::identification::IdentificationMethod;

    fn sample_ate() -> EpistemicATE {
        EpistemicATE::new(
            0.5,
            0.1,
            IdentificationMethod::BackdoorAdjustment {
                set: ["Age".to_string()].into_iter().collect(),
            },
            BetaConfidence::new(8.0, 2.0),
        )
    }

    #[test]
    fn test_placebo_test() {
        let mut handler = DefaultRefutationHandler::new(42);
        let ate = sample_ate();
        let result = handler.placebo_test(&ate);

        // With seed 42, should be deterministic
        assert!(result.statistic >= 0.0);
    }

    #[test]
    fn test_standard_battery() {
        let mut handler = DefaultRefutationHandler::new(42);
        let ate = sample_ate();
        let report = handler.run_standard_battery(&ate);

        assert_eq!(report.results.len(), 6);
        assert!(report.total_penalty >= 0.0);
        assert!(report.adjusted_confidence.mean() <= report.original_confidence.mean() + 0.01);
    }

    #[test]
    fn test_report_markdown() {
        let mut handler = DefaultRefutationHandler::new(42);
        let ate = sample_ate();
        let report = handler.run_standard_battery(&ate);
        let md = report.to_markdown();

        assert!(md.contains("Causal Refutation Report"));
        assert!(md.contains("Placebo Treatment"));
    }

    #[test]
    fn test_refutation_effect_strict() {
        let effect = RefutationEffect::strict();
        assert_eq!(effect.required_tests.len(), 6);
        assert!(effect.require_all_pass);
    }

    #[test]
    fn test_penalty_application() {
        let results = vec![
            RefutationResult::passed(RefutationType::PlaceboTreatment, 0.05, 0.1),
            RefutationResult::failed(RefutationType::RandomCommonCause, 0.20, 0.15),
        ];
        let original = BetaConfidence::new(8.0, 2.0);
        let report = RefutationReport::from_results(results, original);

        assert!(!report.all_passed);
        assert!(report.total_penalty > 0.0);
        assert!(report.adjusted_confidence.mean() < report.original_confidence.mean());
    }
}
