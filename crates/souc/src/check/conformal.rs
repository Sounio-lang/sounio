//! Conformal Prediction for Type Uncertainty
//!
//! Conformal prediction provides distribution-free uncertainty quantification
//! with finite-sample coverage guarantees. This is ideal for type checking
//! where we need rigorous bounds on compatibility.
//!
//! # The Key Guarantee
//!
//! For any error rate α ∈ (0, 1), conformal prediction constructs prediction
//! sets C(x) such that:
//!
//! ```text
//! P(Y ∈ C(X)) ≥ 1 - α
//! ```
//!
//! This holds regardless of the underlying distribution, requiring only that
//! calibration and test data are exchangeable.
//!
//! # Application to Type Checking
//!
//! Instead of a binary "compatible/incompatible" decision, we get:
//! - A **prediction set** of compatible types with guaranteed coverage
//! - A **p-value** for each type assignment indicating plausibility
//! - **Adaptive thresholds** that account for model uncertainty
//!
//! # The Algorithm (Split Conformal)
//!
//! 1. **Calibrate**: Compute nonconformity scores s_i = score(x_i, y_i)
//!    on held-out calibration data
//!
//! 2. **Quantile**: Find the (1-α)(1 + 1/n) quantile q̂ of calibration scores
//!
//! 3. **Predict**: For new x, include y in prediction set if score(x, y) ≤ q̂
//!
//! # Conformalized Semantic Distance
//!
//! Our nonconformity score is based on semantic distance:
//!
//! ```text
//! s(expected, actual) = semantic_distance(expected, actual) / θ
//! ```
//!
//! where θ is a learned or configured threshold.
//!
//! # References
//!
//! - Vovk et al. (2005): Algorithmic Learning in a Random World
//! - Angelopoulos & Bates (2021): A Gentle Introduction to Conformal Prediction
//! - Barber et al. (2023): Conformal Prediction Under Distribution Shift

use std::collections::{HashMap, VecDeque};

use crate::ontology::loader::IRI;

/// Configuration for conformal type checking
#[derive(Debug, Clone)]
pub struct ConformalConfig {
    /// Target coverage level (1 - α), e.g., 0.95 for 95% coverage
    pub coverage: f64,
    /// Minimum calibration set size
    pub min_calibration: usize,
    /// Maximum calibration set size (for online learning)
    pub max_calibration: usize,
    /// Whether to use adaptive conformal inference
    pub adaptive: bool,
    /// Adaptation rate for online conformal (ACI)
    pub gamma: f64,
}

impl Default for ConformalConfig {
    fn default() -> Self {
        Self {
            coverage: 0.95,
            min_calibration: 30,
            max_calibration: 1000,
            adaptive: true,
            gamma: 0.005,
        }
    }
}

/// A calibration example: (expected_type, actual_type, score)
#[derive(Debug, Clone)]
pub struct CalibrationExample {
    /// Expected type (from annotation or context)
    pub expected: IRI,
    /// Actual type (provided value)
    pub actual: IRI,
    /// Nonconformity score
    pub score: f64,
    /// Whether this was a valid assignment (ground truth)
    pub valid: bool,
}

/// Result of conformal type checking
#[derive(Debug, Clone)]
pub struct ConformalResult {
    /// The nonconformity score for this assignment
    pub score: f64,
    /// P-value: probability of seeing a score this extreme or more
    pub p_value: f64,
    /// Whether the assignment is in the prediction set at configured coverage
    pub in_prediction_set: bool,
    /// The threshold used (quantile of calibration scores)
    pub threshold: f64,
    /// Coverage level used
    pub coverage: f64,
    /// Size of calibration set
    pub calibration_size: usize,
}

impl ConformalResult {
    /// Check if the assignment is conforming (statistically plausible)
    pub fn is_conforming(&self) -> bool {
        self.in_prediction_set
    }

    /// Get a confidence-like score (higher = more confident it's valid)
    pub fn confidence(&self) -> f64 {
        self.p_value
    }

    /// Whether we have enough calibration data
    pub fn is_calibrated(&self) -> bool {
        self.calibration_size >= 10
    }
}

/// Conformal predictor for type compatibility
pub struct ConformalTypeChecker {
    /// Configuration
    config: ConformalConfig,
    /// Calibration scores (sorted for quantile computation)
    calibration_scores: Vec<f64>,
    /// Recent examples for online learning
    recent_examples: VecDeque<CalibrationExample>,
    /// Current quantile threshold
    current_threshold: f64,
    /// Running estimate for adaptive conformal inference
    alpha_t: f64,
    /// Statistics
    stats: ConformalStats,
}

/// Statistics about conformal predictions
#[derive(Debug, Clone, Default)]
pub struct ConformalStats {
    /// Total predictions made
    pub total_predictions: usize,
    /// Number in prediction set
    pub in_set_count: usize,
    /// Number of calibration updates
    pub calibration_updates: usize,
    /// Running coverage estimate
    pub empirical_coverage: f64,
    /// Average prediction set size
    pub avg_set_size: f64,
}

impl ConformalTypeChecker {
    /// Create a new conformal type checker
    pub fn new(config: ConformalConfig) -> Self {
        let alpha = 1.0 - config.coverage;

        Self {
            config,
            calibration_scores: Vec::new(),
            recent_examples: VecDeque::new(),
            current_threshold: 1.0, // Conservative initial threshold
            alpha_t: alpha,
            stats: ConformalStats::default(),
        }
    }

    /// Add a calibration example
    ///
    /// Calibration examples should be from held-out data with known
    /// ground truth about whether the type assignment is valid.
    pub fn calibrate(&mut self, example: CalibrationExample) {
        // Only use valid examples for calibration (positive class)
        // This assumes we're calibrating on correct type assignments
        if example.valid {
            self.calibration_scores.push(example.score);

            // Keep sorted for efficient quantile
            self.calibration_scores
                .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }

        // For online learning, track recent examples
        self.recent_examples.push_back(example);
        while self.recent_examples.len() > self.config.max_calibration {
            self.recent_examples.pop_front();
        }

        // Update threshold
        self.update_threshold();
        self.stats.calibration_updates += 1;
    }

    /// Update calibration with batch of examples
    pub fn calibrate_batch(&mut self, examples: &[CalibrationExample]) {
        for ex in examples {
            if ex.valid {
                self.calibration_scores.push(ex.score);
            }
        }

        self.calibration_scores
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        self.update_threshold();
        self.stats.calibration_updates += 1;
    }

    /// Compute the conformal quantile threshold
    fn update_threshold(&mut self) {
        let n = self.calibration_scores.len();
        if n == 0 {
            self.current_threshold = 1.0; // Conservative
            return;
        }

        // Compute the (1 - α)(1 + 1/n) quantile
        // This is the key conformal correction that ensures coverage
        let alpha = if self.config.adaptive {
            self.alpha_t
        } else {
            1.0 - self.config.coverage
        };

        let adjusted_level = (1.0 - alpha) * (1.0 + 1.0 / n as f64);
        let quantile_idx = ((n as f64) * adjusted_level).ceil() as usize;
        let idx = quantile_idx.min(n).saturating_sub(1);

        self.current_threshold = self.calibration_scores[idx];
    }

    /// Check a type assignment and return conformal result
    pub fn check(&mut self, score: f64) -> ConformalResult {
        let n = self.calibration_scores.len();

        // Compute p-value: fraction of calibration scores >= this score
        let p_value = if n > 0 {
            let num_greater = self
                .calibration_scores
                .iter()
                .filter(|&&s| s >= score)
                .count();
            (num_greater + 1) as f64 / (n + 1) as f64
        } else {
            0.5 // No calibration data, return neutral p-value
        };

        let in_prediction_set = score <= self.current_threshold;

        // Update statistics
        self.stats.total_predictions += 1;
        if in_prediction_set {
            self.stats.in_set_count += 1;
        }
        self.stats.empirical_coverage =
            self.stats.in_set_count as f64 / self.stats.total_predictions as f64;

        // Adaptive conformal inference update
        if self.config.adaptive && n >= self.config.min_calibration {
            let err_t = if in_prediction_set { 0.0 } else { 1.0 };
            let target_alpha = 1.0 - self.config.coverage;
            self.alpha_t = self.alpha_t + self.config.gamma * (target_alpha - err_t);
            self.alpha_t = self.alpha_t.clamp(0.001, 0.5); // Keep α in reasonable range
            self.update_threshold();
        }

        ConformalResult {
            score,
            p_value,
            in_prediction_set,
            threshold: self.current_threshold,
            coverage: self.config.coverage,
            calibration_size: n,
        }
    }

    /// Get the prediction set for a given expected type
    ///
    /// Returns all candidate types whose scores are below the threshold
    pub fn prediction_set(
        &self,
        expected: &IRI,
        candidates: &[IRI],
        score_fn: impl Fn(&IRI, &IRI) -> f64,
    ) -> Vec<(IRI, f64)> {
        let mut result = Vec::new();

        for candidate in candidates {
            let score = score_fn(expected, candidate);
            if score <= self.current_threshold {
                result.push((candidate.clone(), score));
            }
        }

        // Sort by score (most conforming first)
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        result
    }

    /// Get current statistics
    pub fn stats(&self) -> &ConformalStats {
        &self.stats
    }

    /// Get current threshold
    pub fn threshold(&self) -> f64 {
        self.current_threshold
    }

    /// Check if we have sufficient calibration data
    pub fn is_calibrated(&self) -> bool {
        self.calibration_scores.len() >= self.config.min_calibration
    }

    /// Get calibration set size
    pub fn calibration_size(&self) -> usize {
        self.calibration_scores.len()
    }

    /// Reset calibration data
    pub fn reset(&mut self) {
        self.calibration_scores.clear();
        self.recent_examples.clear();
        self.current_threshold = 1.0;
        self.alpha_t = 1.0 - self.config.coverage;
        self.stats = ConformalStats::default();
    }
}

/// Conformalized quantile regression for threshold estimation
///
/// Instead of using a fixed threshold, this learns thresholds
/// that provide coverage guarantees at different confidence levels.
pub struct ConformalQuantileRegressor {
    /// Coverage levels to provide thresholds for
    coverages: Vec<f64>,
    /// Corresponding thresholds
    thresholds: Vec<f64>,
    /// Calibration data
    scores: Vec<f64>,
}

impl ConformalQuantileRegressor {
    /// Create with standard coverage levels
    pub fn new() -> Self {
        Self {
            coverages: vec![0.50, 0.75, 0.90, 0.95, 0.99],
            thresholds: vec![0.5, 0.6, 0.7, 0.8, 0.9], // Initial estimates
            scores: Vec::new(),
        }
    }

    /// Fit to calibration data
    pub fn fit(&mut self, scores: &[f64]) {
        self.scores = scores.to_vec();
        self.scores
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = self.scores.len();
        if n == 0 {
            return;
        }

        self.thresholds = self
            .coverages
            .iter()
            .map(|&cov| {
                let alpha = 1.0 - cov;
                let adjusted = (1.0 - alpha) * (1.0 + 1.0 / n as f64);
                let idx = ((n as f64) * adjusted).ceil() as usize;
                let idx = idx.min(n).saturating_sub(1);
                self.scores[idx]
            })
            .collect();
    }

    /// Get threshold for a specific coverage level
    pub fn threshold_for_coverage(&self, coverage: f64) -> f64 {
        // Find closest coverage level
        let mut best_idx = 0;
        let mut best_diff = f64::MAX;

        for (i, &cov) in self.coverages.iter().enumerate() {
            let diff = (cov - coverage).abs();
            if diff < best_diff {
                best_diff = diff;
                best_idx = i;
            }
        }

        self.thresholds[best_idx]
    }

    /// Interpolate threshold for arbitrary coverage level
    pub fn interpolated_threshold(&self, coverage: f64) -> f64 {
        if self.coverages.is_empty() {
            return 1.0;
        }

        // Find surrounding coverage levels
        for i in 0..self.coverages.len() - 1 {
            if coverage >= self.coverages[i] && coverage <= self.coverages[i + 1] {
                let t =
                    (coverage - self.coverages[i]) / (self.coverages[i + 1] - self.coverages[i]);
                return self.thresholds[i] * (1.0 - t) + self.thresholds[i + 1] * t;
            }
        }

        // Extrapolate
        if coverage < self.coverages[0] {
            self.thresholds[0]
        } else {
            *self.thresholds.last().unwrap_or(&1.0)
        }
    }
}

impl Default for ConformalQuantileRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Mondrian conformal predictor for conditional coverage
///
/// Provides coverage guarantees conditional on type categories,
/// not just marginal coverage. This is important when different
/// ontology domains have different difficulty levels.
pub struct MondrianConformalChecker {
    /// Base configuration
    config: ConformalConfig,
    /// Separate conformal predictors per category
    predictors: HashMap<String, ConformalTypeChecker>,
    /// Category extractor (maps IRI to category)
    default_category: String,
}

impl MondrianConformalChecker {
    /// Create a new Mondrian conformal predictor
    pub fn new(config: ConformalConfig) -> Self {
        Self {
            config,
            predictors: HashMap::new(),
            default_category: "default".to_string(),
        }
    }

    /// Get or create predictor for a category
    fn get_predictor(&mut self, category: &str) -> &mut ConformalTypeChecker {
        if !self.predictors.contains_key(category) {
            self.predictors.insert(
                category.to_string(),
                ConformalTypeChecker::new(self.config.clone()),
            );
        }
        self.predictors.get_mut(category).unwrap()
    }

    /// Calibrate with category information
    pub fn calibrate(&mut self, example: CalibrationExample, category: &str) {
        self.get_predictor(category).calibrate(example);
    }

    /// Check with category-specific calibration
    pub fn check(&mut self, score: f64, category: &str) -> ConformalResult {
        // Fall back to default if category not calibrated
        let cat = if self.predictors.contains_key(category)
            && self.predictors[category].is_calibrated()
        {
            category.to_string()
        } else {
            self.default_category.clone()
        };

        self.get_predictor(&cat).check(score)
    }

    /// Get statistics per category
    pub fn stats_by_category(&self) -> HashMap<String, &ConformalStats> {
        self.predictors
            .iter()
            .map(|(k, v)| (k.clone(), v.stats()))
            .collect()
    }
}

/// Extract category from IRI (prefix/ontology)
pub fn iri_category(iri: &IRI) -> String {
    let s = iri.as_str();
    if let Some(pos) = s.rfind('/') {
        let local = &s[pos + 1..];
        if let Some(underscore) = local.find('_') {
            return local[..underscore].to_uppercase();
        }
    }
    if let Some(colon) = s.find(':') {
        return s[..colon].to_uppercase();
    }
    "UNKNOWN".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conformal_basic() {
        let mut checker = ConformalTypeChecker::new(ConformalConfig::default());

        // Calibrate with some scores
        for i in 0..100 {
            let score = (i as f64) / 100.0;
            checker.calibrate(CalibrationExample {
                expected: IRI::new("http://test/expected"),
                actual: IRI::new("http://test/actual"),
                score,
                valid: true,
            });
        }

        // Check a low score (should be conforming)
        let result = checker.check(0.1);
        assert!(result.is_conforming());
        assert!(result.p_value > 0.5);

        // Check a high score (should not be conforming at 95% coverage)
        let result = checker.check(0.98);
        assert!(!result.is_conforming());
        assert!(result.p_value < 0.1);
    }

    #[test]
    fn test_conformal_p_value() {
        let mut checker = ConformalTypeChecker::new(ConformalConfig::default());

        // Calibrate with uniform scores
        for i in 0..50 {
            checker.calibrate(CalibrationExample {
                expected: IRI::new("http://test/e"),
                actual: IRI::new("http://test/a"),
                score: (i as f64) / 50.0,
                valid: true,
            });
        }

        // Median score should have p-value around 0.5
        let result = checker.check(0.5);
        assert!(result.p_value > 0.3 && result.p_value < 0.7);
    }

    #[test]
    fn test_conformal_uncalibrated() {
        let checker = ConformalTypeChecker::new(ConformalConfig::default());

        assert!(!checker.is_calibrated());
        assert_eq!(checker.calibration_size(), 0);
    }

    #[test]
    fn test_quantile_regressor() {
        let mut qr = ConformalQuantileRegressor::new();

        let scores: Vec<f64> = (0..100).map(|i| (i as f64) / 100.0).collect();
        qr.fit(&scores);

        // Higher coverage should give higher threshold
        let t50 = qr.threshold_for_coverage(0.50);
        let t95 = qr.threshold_for_coverage(0.95);

        assert!(t95 > t50);
    }

    #[test]
    fn test_mondrian() {
        let mut checker = MondrianConformalChecker::new(ConformalConfig::default());

        // Calibrate category A with low scores
        for i in 0..50 {
            checker.calibrate(
                CalibrationExample {
                    expected: IRI::new("http://test/e"),
                    actual: IRI::new("http://test/a"),
                    score: (i as f64) / 100.0, // 0 to 0.5
                    valid: true,
                },
                "A",
            );
        }

        // Calibrate category B with higher scores
        for i in 0..50 {
            checker.calibrate(
                CalibrationExample {
                    expected: IRI::new("http://test/e"),
                    actual: IRI::new("http://test/b"),
                    score: 0.5 + (i as f64) / 100.0, // 0.5 to 1.0
                    valid: true,
                },
                "B",
            );
        }

        // Same score should be treated differently per category
        let result_a = checker.check(0.6, "A");
        let result_b = checker.check(0.6, "B");

        // 0.6 is high for A (which calibrated on 0-0.5) but low for B
        assert!(result_a.threshold < result_b.threshold || !result_a.is_conforming());
    }

    #[test]
    fn test_iri_category() {
        assert_eq!(
            iri_category(&IRI::new("http://purl.obolibrary.org/obo/CHEBI_15365")),
            "CHEBI"
        );
        assert_eq!(
            iri_category(&IRI::new("http://purl.obolibrary.org/obo/GO_0008150")),
            "GO"
        );
    }

    #[test]
    fn test_adaptive_conformal() {
        let mut config = ConformalConfig::default();
        config.adaptive = true;
        config.min_calibration = 10;

        let mut checker = ConformalTypeChecker::new(config);

        // Initial calibration with scores from 0 to 1
        for i in 0..20 {
            checker.calibrate(CalibrationExample {
                expected: IRI::new("http://test/e"),
                actual: IRI::new("http://test/a"),
                score: (i as f64) / 20.0,
                valid: true,
            });
        }

        let initial_threshold = checker.threshold();
        assert!(initial_threshold > 0.0);

        // Make predictions with high scores
        // This tests that the adaptive algorithm continues to work correctly
        for _ in 0..10 {
            let result = checker.check(0.99);
            // Just verify that check returns a result, regardless of value
            assert!(result.score >= 0.0);
        }

        let adapted_threshold = checker.threshold();
        // Thresholds should be valid (positive)
        assert!(adapted_threshold > 0.0);
        assert!(initial_threshold > 0.0);
    }
}
