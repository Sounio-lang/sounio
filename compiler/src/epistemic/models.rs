//! Configurable Uncertainty Models for Epistemic Computing
//!
//! This module provides multiple uncertainty propagation models that can be
//! selected at compile-time or runtime. Each model has different mathematical
//! foundations and trade-offs.
//!
//! # Available Models
//!
//! | Model | Foundation | Use Case |
//! |-------|-----------|----------|
//! | Probabilistic | Probability theory | General uncertainty |
//! | Interval | Interval arithmetic | Bounded uncertainty |
//! | Affine | Affine arithmetic | Correlated errors |
//! | Fuzzy | Possibility theory | Vague concepts |
//! | DempsterShafer | Evidence theory | Conflicting sources |
//! | Bayesian | Bayesian inference | Prior + evidence |
//!
//! # Configuration
//!
//! Models can be configured at multiple levels:
//! - Global (compiler flag or config file)
//! - Module level (`#[epistemic(model = "Interval")]`)
//! - Function level (`#[epistemic(model = "Affine")]`)
//! - Expression level (inline model selection)
//!
//! # Example
//!
//! ```sounio
//! #[epistemic(model = "Interval", rounding = "outward")]
//! fn critical_calculation(x: Knowledge[f64]) -> Knowledge[f64] {
//!     // Uses interval arithmetic for this function
//!     x * 2.0
//! }
//!
//! #[epistemic(model = "Probabilistic")]
//! fn statistical_analysis(data: &[Knowledge[f64]]) -> Knowledge[f64] {
//!     // Uses probabilistic model
//!     data.iter().sum() / data.len() as f64
//! }
//! ```

use std::fmt;

/// Uncertainty model selection
///
/// Each model represents a different mathematical framework for
/// representing and propagating uncertainty.
#[derive(Debug, Clone, PartialEq)]
pub enum UncertaintyModel {
    /// Probabilistic model using confidence values [0, 1]
    ///
    /// Default model. Represents uncertainty as a single probability
    /// value that propagates through operations.
    Probabilistic(ProbabilisticConfig),

    /// Interval arithmetic model
    ///
    /// Represents uncertainty as bounded intervals [a, b].
    /// Guarantees that the true value lies within the computed interval.
    /// Best for safety-critical computations.
    Interval(IntervalConfig),

    /// Affine arithmetic model
    ///
    /// Extends interval arithmetic to track correlations between errors.
    /// Reduces overestimation in complex expressions.
    /// Best for numerical analysis with correlated uncertainties.
    Affine(AffineConfig),

    /// Fuzzy set / possibility theory model
    ///
    /// Represents vague or imprecise concepts using membership functions.
    /// Best for linguistic uncertainty ("approximately 5", "very high").
    Fuzzy(FuzzyConfig),

    /// Dempster-Shafer evidence theory
    ///
    /// Represents uncertainty as belief/plausibility intervals.
    /// Supports explicit "don't know" states.
    /// Best for combining evidence from multiple sources.
    DempsterShafer(DempsterShaferConfig),

    /// Full Bayesian inference
    ///
    /// Maintains probability distributions instead of point estimates.
    /// Most expressive but computationally expensive.
    /// Best for probabilistic modeling.
    Bayesian(BayesianConfig),
}

impl Default for UncertaintyModel {
    fn default() -> Self {
        UncertaintyModel::Probabilistic(ProbabilisticConfig::default())
    }
}

impl UncertaintyModel {
    /// Get a human-readable name for the model
    pub fn name(&self) -> &'static str {
        match self {
            UncertaintyModel::Probabilistic(_) => "Probabilistic",
            UncertaintyModel::Interval(_) => "Interval",
            UncertaintyModel::Affine(_) => "Affine",
            UncertaintyModel::Fuzzy(_) => "Fuzzy",
            UncertaintyModel::DempsterShafer(_) => "DempsterShafer",
            UncertaintyModel::Bayesian(_) => "Bayesian",
        }
    }

    /// Check if this model supports interval bounds
    pub fn supports_intervals(&self) -> bool {
        matches!(
            self,
            UncertaintyModel::Interval(_)
                | UncertaintyModel::Affine(_)
                | UncertaintyModel::DempsterShafer(_)
        )
    }

    /// Check if this model supports distribution tracking
    pub fn supports_distributions(&self) -> bool {
        matches!(self, UncertaintyModel::Bayesian(_))
    }

    /// Get the default confidence degradation factor for operations
    pub fn default_degradation(&self) -> f64 {
        match self {
            UncertaintyModel::Probabilistic(c) => c.default_factor,
            UncertaintyModel::Interval(_) => 1.0, // No degradation, interval grows
            UncertaintyModel::Affine(_) => 1.0,
            UncertaintyModel::Fuzzy(c) => c.default_degradation,
            UncertaintyModel::DempsterShafer(_) => 1.0,
            UncertaintyModel::Bayesian(_) => 1.0,
        }
    }
}

// =============================================================================
// Probabilistic Model Configuration
// =============================================================================

/// Configuration for probabilistic uncertainty model
#[derive(Debug, Clone, PartialEq)]
pub struct ProbabilisticConfig {
    /// Default confidence factor for operations
    pub default_factor: f64,

    /// Combination strategy for multiple sources
    pub combination: CombinationRule,

    /// Whether to track confidence bounds
    pub track_bounds: bool,

    /// Minimum confidence threshold (below this triggers warning)
    pub min_threshold: f64,
}

impl Default for ProbabilisticConfig {
    fn default() -> Self {
        Self {
            default_factor: 1.0,
            combination: CombinationRule::Multiplicative,
            track_bounds: false,
            min_threshold: 0.0,
        }
    }
}

/// Rule for combining confidence from multiple sources
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombinationRule {
    /// Product: ε₁ × ε₂ (independent)
    Multiplicative,
    /// Minimum: min(ε₁, ε₂) (conservative)
    Minimum,
    /// Maximum: max(ε₁, ε₂) (optimistic)
    Maximum,
    /// Weighted average
    WeightedAverage,
    /// Dempster's rule of combination
    Dempster,
}

// =============================================================================
// Interval Arithmetic Configuration
// =============================================================================

/// Configuration for interval arithmetic model
#[derive(Debug, Clone, PartialEq)]
pub struct IntervalConfig {
    /// Rounding direction for interval bounds
    pub rounding: RoundingMode,

    /// Maximum interval width before warning
    pub max_width: Option<f64>,

    /// Whether to use extended intervals (include infinities)
    pub extended: bool,

    /// Subdivision strategy for non-monotonic functions
    pub subdivision: SubdivisionStrategy,
}

impl Default for IntervalConfig {
    fn default() -> Self {
        Self {
            rounding: RoundingMode::Outward,
            max_width: None,
            extended: true,
            subdivision: SubdivisionStrategy::None,
        }
    }
}

/// Rounding mode for interval bounds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundingMode {
    /// Round lower bound down, upper bound up (safe)
    Outward,
    /// Round both bounds down
    Down,
    /// Round both bounds up
    Up,
    /// Round to nearest (not guaranteed safe)
    Nearest,
}

/// Strategy for subdividing intervals
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubdivisionStrategy {
    /// No subdivision
    None,
    /// Bisection
    Bisection,
    /// Adaptive based on function behavior
    Adaptive,
}

// =============================================================================
// Affine Arithmetic Configuration
// =============================================================================

/// Configuration for affine arithmetic model
///
/// Affine arithmetic represents values as:
///   x = x₀ + Σᵢ xᵢεᵢ
/// where εᵢ are noise symbols in [-1, 1]
#[derive(Debug, Clone, PartialEq)]
pub struct AffineConfig {
    /// Maximum number of noise symbols to track
    pub max_noise_symbols: usize,

    /// Strategy for reducing noise symbols when limit exceeded
    pub condensation: CondensationStrategy,

    /// Whether to track correlations across computations
    pub track_correlations: bool,
}

impl Default for AffineConfig {
    fn default() -> Self {
        Self {
            max_noise_symbols: 64,
            condensation: CondensationStrategy::MergeSmallest,
            track_correlations: true,
        }
    }
}

/// Strategy for reducing noise symbols
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CondensationStrategy {
    /// Merge smallest coefficients into single term
    MergeSmallest,
    /// Convert to interval (loses correlation info)
    ToInterval,
    /// Keep most recent symbols
    KeepRecent,
}

// =============================================================================
// Fuzzy Set Configuration
// =============================================================================

/// Configuration for fuzzy / possibility theory model
#[derive(Debug, Clone, PartialEq)]
pub struct FuzzyConfig {
    /// Type of membership function
    pub membership: MembershipFunction,

    /// Alpha-cut threshold for defuzzification
    pub alpha_cut: f64,

    /// Default degradation for operations
    pub default_degradation: f64,
}

impl Default for FuzzyConfig {
    fn default() -> Self {
        Self {
            membership: MembershipFunction::Triangular,
            alpha_cut: 0.5,
            default_degradation: 0.95,
        }
    }
}

/// Type of fuzzy membership function
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MembershipFunction {
    /// Triangular: (a, b, c) where b is peak
    Triangular,
    /// Trapezoidal: (a, b, c, d) where [b,c] is peak plateau
    Trapezoidal,
    /// Gaussian: (μ, σ)
    Gaussian,
    /// Sigmoidal: for "large" or "small" concepts
    Sigmoidal,
}

// =============================================================================
// Dempster-Shafer Configuration
// =============================================================================

/// Configuration for Dempster-Shafer evidence theory
#[derive(Debug, Clone, PartialEq)]
pub struct DempsterShaferConfig {
    /// How to handle high conflict between sources
    pub conflict_handling: ConflictHandling,

    /// Threshold for considering conflict significant
    pub conflict_threshold: f64,

    /// Whether to track focal elements explicitly
    pub track_focal_elements: bool,
}

impl Default for DempsterShaferConfig {
    fn default() -> Self {
        Self {
            conflict_handling: ConflictHandling::Normalize,
            conflict_threshold: 0.3,
            track_focal_elements: false,
        }
    }
}

/// How to handle conflicting evidence
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictHandling {
    /// Normalize (standard Dempster's rule)
    Normalize,
    /// Yager's rule (assign to unknown)
    Yager,
    /// Dubois-Prade (disjunctive combination)
    DuboisPrade,
    /// Reject if conflict exceeds threshold
    Reject,
}

// =============================================================================
// Bayesian Configuration
// =============================================================================

/// Configuration for full Bayesian inference
#[derive(Debug, Clone, PartialEq)]
pub struct BayesianConfig {
    /// Prior distribution family
    pub prior: PriorFamily,

    /// Number of samples for Monte Carlo methods
    pub num_samples: usize,

    /// Inference method
    pub inference: InferenceMethod,
}

impl Default for BayesianConfig {
    fn default() -> Self {
        Self {
            prior: PriorFamily::Uniform,
            num_samples: 1000,
            inference: InferenceMethod::Analytical,
        }
    }
}

/// Family of prior distributions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriorFamily {
    /// Uniform (uninformative)
    Uniform,
    /// Jeffreys (reference)
    Jeffreys,
    /// Conjugate (depends on likelihood)
    Conjugate,
    /// Empirical (from data)
    Empirical,
}

/// Inference method for posterior computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceMethod {
    /// Closed-form analytical (when available)
    Analytical,
    /// Markov Chain Monte Carlo
    MCMC,
    /// Variational inference
    Variational,
    /// Importance sampling
    ImportanceSampling,
}

// =============================================================================
// Uncertainty Value Types
// =============================================================================

/// A value with associated uncertainty in the selected model
#[derive(Debug, Clone)]
pub enum UncertainValue {
    /// Point estimate with confidence
    Point { value: f64, confidence: f64 },

    /// Interval bounds
    Interval { lower: f64, upper: f64 },

    /// Affine form
    Affine {
        center: f64,
        noise_terms: Vec<(u32, f64)>, // (symbol_id, coefficient)
    },

    /// Fuzzy number (represented by support bounds and peak)
    Fuzzy {
        /// Lower bound of support
        support_lower: f64,
        /// Upper bound of support
        support_upper: f64,
        /// Peak value (where membership = 1.0)
        peak: f64,
    },

    /// Belief interval [Bel, Pl]
    BeliefInterval { belief: f64, plausibility: f64 },

    /// Full distribution (samples or parameters)
    Distribution {
        samples: Vec<f64>,
        mean: f64,
        variance: f64,
    },
}

impl UncertainValue {
    /// Get a point estimate (central value)
    pub fn point_estimate(&self) -> f64 {
        match self {
            UncertainValue::Point { value, .. } => *value,
            UncertainValue::Interval { lower, upper } => (lower + upper) / 2.0,
            UncertainValue::Affine { center, .. } => *center,
            UncertainValue::Fuzzy { peak, .. } => *peak,
            UncertainValue::BeliefInterval {
                belief,
                plausibility,
            } => (belief + plausibility) / 2.0,
            UncertainValue::Distribution { mean, .. } => *mean,
        }
    }

    /// Get confidence/certainty measure [0, 1]
    pub fn certainty(&self) -> f64 {
        match self {
            UncertainValue::Point { confidence, .. } => *confidence,
            UncertainValue::Interval { lower, upper } => {
                // Narrower intervals = higher certainty
                let width = upper - lower;
                if width <= 0.0 {
                    1.0
                } else {
                    1.0 / (1.0 + width)
                }
            }
            UncertainValue::Affine { noise_terms, .. } => {
                let total_error: f64 = noise_terms.iter().map(|(_, c)| c.abs()).sum();
                1.0 / (1.0 + total_error)
            }
            UncertainValue::Fuzzy {
                support_lower,
                support_upper,
                ..
            } => {
                // Narrower support = higher certainty
                let width = support_upper - support_lower;
                if width <= 0.0 {
                    1.0
                } else {
                    1.0 / (1.0 + width)
                }
            }
            UncertainValue::BeliefInterval {
                belief,
                plausibility,
            } => {
                // Use pignistic probability
                (belief + plausibility) / 2.0
            }
            UncertainValue::Distribution { variance, .. } => {
                // Lower variance = higher certainty
                1.0 / (1.0 + variance.sqrt())
            }
        }
    }
}

// =============================================================================
// Model Selection and Configuration
// =============================================================================

/// Global epistemic configuration
#[derive(Debug, Clone)]
pub struct EpistemicConfig {
    /// Default uncertainty model
    pub default_model: UncertaintyModel,

    /// Per-operation model overrides
    pub operation_models: std::collections::HashMap<String, UncertaintyModel>,

    /// Confidence threshold for warnings
    pub warning_threshold: f64,

    /// Confidence threshold for errors
    pub error_threshold: f64,

    /// Whether to track provenance
    pub track_provenance: bool,

    /// Whether to enable epistemic firewalls
    pub enable_firewalls: bool,
}

impl Default for EpistemicConfig {
    fn default() -> Self {
        Self {
            default_model: UncertaintyModel::default(),
            operation_models: std::collections::HashMap::new(),
            warning_threshold: 0.3,
            error_threshold: 0.1,
            track_provenance: true,
            enable_firewalls: true,
        }
    }
}

impl EpistemicConfig {
    /// Create config with specific default model
    pub fn with_model(model: UncertaintyModel) -> Self {
        Self {
            default_model: model,
            ..Default::default()
        }
    }

    /// Set model for specific operation
    pub fn set_operation_model(&mut self, operation: &str, model: UncertaintyModel) {
        self.operation_models.insert(operation.to_string(), model);
    }

    /// Get model for operation (uses default if not overridden)
    pub fn get_model(&self, operation: &str) -> &UncertaintyModel {
        self.operation_models
            .get(operation)
            .unwrap_or(&self.default_model)
    }

    /// Create interval arithmetic config
    pub fn interval() -> Self {
        Self::with_model(UncertaintyModel::Interval(IntervalConfig::default()))
    }

    /// Create affine arithmetic config
    pub fn affine() -> Self {
        Self::with_model(UncertaintyModel::Affine(AffineConfig::default()))
    }

    /// Create Bayesian inference config
    pub fn bayesian() -> Self {
        Self::with_model(UncertaintyModel::Bayesian(BayesianConfig::default()))
    }
}

// =============================================================================
// Propagation Functions
// =============================================================================

/// Propagate uncertainty through a binary operation
pub fn propagate_binary(
    left: &UncertainValue,
    right: &UncertainValue,
    op: BinaryOp,
    model: &UncertaintyModel,
) -> UncertainValue {
    match model {
        UncertaintyModel::Probabilistic(config) => propagate_probabilistic(left, right, op, config),
        UncertaintyModel::Interval(config) => propagate_interval(left, right, op, config),
        UncertaintyModel::Affine(config) => propagate_affine(left, right, op, config),
        _ => {
            // Fallback to probabilistic for unimplemented models
            propagate_probabilistic(left, right, op, &ProbabilisticConfig::default())
        }
    }
}

/// Binary operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

fn propagate_probabilistic(
    left: &UncertainValue,
    right: &UncertainValue,
    op: BinaryOp,
    config: &ProbabilisticConfig,
) -> UncertainValue {
    let l_val = left.point_estimate();
    let r_val = right.point_estimate();
    let l_conf = left.certainty();
    let r_conf = right.certainty();

    let value = match op {
        BinaryOp::Add => l_val + r_val,
        BinaryOp::Sub => l_val - r_val,
        BinaryOp::Mul => l_val * r_val,
        BinaryOp::Div => l_val / r_val,
    };

    let confidence = match config.combination {
        CombinationRule::Multiplicative => l_conf * r_conf * config.default_factor,
        CombinationRule::Minimum => l_conf.min(r_conf) * config.default_factor,
        CombinationRule::Maximum => l_conf.max(r_conf),
        CombinationRule::WeightedAverage => (l_conf + r_conf) / 2.0 * config.default_factor,
        CombinationRule::Dempster => {
            // 1 - (1-ε₁)(1-ε₂)
            1.0 - (1.0 - l_conf) * (1.0 - r_conf)
        }
    };

    UncertainValue::Point {
        value,
        confidence: confidence.clamp(0.0, 1.0),
    }
}

fn propagate_interval(
    left: &UncertainValue,
    right: &UncertainValue,
    op: BinaryOp,
    _config: &IntervalConfig,
) -> UncertainValue {
    // Convert to intervals if needed
    let (l_lo, l_hi) = match left {
        UncertainValue::Interval { lower, upper } => (*lower, *upper),
        _ => {
            let v = left.point_estimate();
            (v, v)
        }
    };

    let (r_lo, r_hi) = match right {
        UncertainValue::Interval { lower, upper } => (*lower, *upper),
        _ => {
            let v = right.point_estimate();
            (v, v)
        }
    };

    let (lower, upper) = match op {
        BinaryOp::Add => (l_lo + r_lo, l_hi + r_hi),
        BinaryOp::Sub => (l_lo - r_hi, l_hi - r_lo),
        BinaryOp::Mul => {
            let products = [l_lo * r_lo, l_lo * r_hi, l_hi * r_lo, l_hi * r_hi];
            (
                products.iter().cloned().fold(f64::INFINITY, f64::min),
                products.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            )
        }
        BinaryOp::Div => {
            if r_lo <= 0.0 && r_hi >= 0.0 {
                // Division by interval containing zero
                (f64::NEG_INFINITY, f64::INFINITY)
            } else {
                let quotients = [l_lo / r_lo, l_lo / r_hi, l_hi / r_lo, l_hi / r_hi];
                (
                    quotients.iter().cloned().fold(f64::INFINITY, f64::min),
                    quotients.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                )
            }
        }
    };

    UncertainValue::Interval { lower, upper }
}

fn propagate_affine(
    left: &UncertainValue,
    right: &UncertainValue,
    op: BinaryOp,
    config: &AffineConfig,
) -> UncertainValue {
    // Convert to affine forms
    let (l_center, l_noise) = match left {
        UncertainValue::Affine {
            center,
            noise_terms,
        } => (*center, noise_terms.clone()),
        _ => (left.point_estimate(), vec![]),
    };

    let (r_center, r_noise) = match right {
        UncertainValue::Affine {
            center,
            noise_terms,
        } => (*center, noise_terms.clone()),
        _ => (right.point_estimate(), vec![]),
    };

    match op {
        BinaryOp::Add | BinaryOp::Sub => {
            let center = if op == BinaryOp::Add {
                l_center + r_center
            } else {
                l_center - r_center
            };

            // Merge noise terms
            let mut noise_map: std::collections::HashMap<u32, f64> =
                std::collections::HashMap::new();

            for (id, coef) in &l_noise {
                *noise_map.entry(*id).or_insert(0.0) += coef;
            }

            for (id, coef) in &r_noise {
                let factor = if op == BinaryOp::Add { 1.0 } else { -1.0 };
                *noise_map.entry(*id).or_insert(0.0) += factor * coef;
            }

            let mut noise_terms: Vec<_> = noise_map.into_iter().collect();
            condense_noise_terms(&mut noise_terms, config.max_noise_symbols);

            UncertainValue::Affine {
                center,
                noise_terms,
            }
        }
        BinaryOp::Mul => {
            // Affine * Affine introduces nonlinearity
            // Approximate: (a + Σaᵢεᵢ)(b + Σbⱼεⱼ) ≈ ab + bΣaᵢεᵢ + aΣbⱼεⱼ + new_noise
            let center = l_center * r_center;

            let mut noise_terms: Vec<(u32, f64)> = Vec::new();

            // Scale left noise by right center
            for (id, coef) in &l_noise {
                noise_terms.push((*id, coef * r_center));
            }

            // Scale right noise by left center
            for (id, coef) in &r_noise {
                noise_terms.push((*id, coef * l_center));
            }

            // Add new noise term for nonlinear part
            let l_error: f64 = l_noise.iter().map(|(_, c)| c.abs()).sum();
            let r_error: f64 = r_noise.iter().map(|(_, c)| c.abs()).sum();
            let new_noise_coef = l_error * r_error;

            if new_noise_coef > 1e-10 {
                // Generate new unique symbol ID (simplified - in practice use global counter)
                let new_id = noise_terms.iter().map(|(id, _)| *id).max().unwrap_or(0) + 1;
                noise_terms.push((new_id, new_noise_coef));
            }

            condense_noise_terms(&mut noise_terms, config.max_noise_symbols);

            UncertainValue::Affine {
                center,
                noise_terms,
            }
        }
        BinaryOp::Div => {
            // Division is complex in affine arithmetic
            // Approximate using interval then convert back
            let interval = propagate_interval(left, right, op, &IntervalConfig::default());

            if let UncertainValue::Interval { lower, upper } = interval {
                let center = (lower + upper) / 2.0;
                let radius = (upper - lower) / 2.0;
                let new_id = l_noise
                    .iter()
                    .chain(r_noise.iter())
                    .map(|(id, _)| *id)
                    .max()
                    .unwrap_or(0)
                    + 1;

                UncertainValue::Affine {
                    center,
                    noise_terms: vec![(new_id, radius)],
                }
            } else {
                interval
            }
        }
    }
}

/// Condense noise terms when exceeding limit
fn condense_noise_terms(terms: &mut Vec<(u32, f64)>, max: usize) {
    if terms.len() <= max {
        return;
    }

    // Sort by absolute coefficient (ascending)
    terms.sort_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap());

    // Merge smallest terms into one
    let to_merge = terms.len() - max + 1;
    let merged_coef: f64 = terms[..to_merge].iter().map(|(_, c)| c.abs()).sum();

    // Remove merged terms
    terms.drain(..to_merge);

    // Add consolidated term
    let new_id = terms.iter().map(|(id, _)| *id).max().unwrap_or(0) + 1;
    terms.push((new_id, merged_coef));
}

// =============================================================================
// Display implementations
// =============================================================================

impl fmt::Display for UncertaintyModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl fmt::Display for UncertainValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UncertainValue::Point { value, confidence } => {
                write!(f, "{:.4} (ε={:.2}%)", value, confidence * 100.0)
            }
            UncertainValue::Interval { lower, upper } => {
                write!(f, "[{:.4}, {:.4}]", lower, upper)
            }
            UncertainValue::Affine {
                center,
                noise_terms,
            } => {
                write!(f, "{:.4}", center)?;
                for (id, coef) in noise_terms {
                    write!(f, " + {:.4}ε{}", coef, id)?;
                }
                Ok(())
            }
            UncertainValue::BeliefInterval {
                belief,
                plausibility,
            } => {
                write!(f, "Bel=[{:.2}, {:.2}]", belief, plausibility)
            }
            UncertainValue::Distribution { mean, variance, .. } => {
                write!(f, "μ={:.4}, σ²={:.4}", mean, variance)
            }
            UncertainValue::Fuzzy {
                support_lower,
                support_upper,
                peak,
            } => {
                write!(
                    f,
                    "Fuzzy[{:.4}, {:.4}] peak={:.4}",
                    support_lower, support_upper, peak
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probabilistic_propagation() {
        let left = UncertainValue::Point {
            value: 10.0,
            confidence: 0.9,
        };
        let right = UncertainValue::Point {
            value: 5.0,
            confidence: 0.8,
        };

        let model = UncertaintyModel::Probabilistic(ProbabilisticConfig::default());
        let result = propagate_binary(&left, &right, BinaryOp::Add, &model);

        if let UncertainValue::Point { value, confidence } = result {
            assert!((value - 15.0).abs() < 1e-10);
            assert!((confidence - 0.72).abs() < 1e-10); // 0.9 * 0.8
        } else {
            panic!("Expected Point value");
        }
    }

    #[test]
    fn test_interval_propagation() {
        let left = UncertainValue::Interval {
            lower: 9.0,
            upper: 11.0,
        };
        let right = UncertainValue::Interval {
            lower: 4.0,
            upper: 6.0,
        };

        let model = UncertaintyModel::Interval(IntervalConfig::default());
        let result = propagate_binary(&left, &right, BinaryOp::Add, &model);

        if let UncertainValue::Interval { lower, upper } = result {
            assert!((lower - 13.0).abs() < 1e-10);
            assert!((upper - 17.0).abs() < 1e-10);
        } else {
            panic!("Expected Interval value");
        }
    }

    #[test]
    fn test_interval_multiplication() {
        let left = UncertainValue::Interval {
            lower: 2.0,
            upper: 3.0,
        };
        let right = UncertainValue::Interval {
            lower: 4.0,
            upper: 5.0,
        };

        let model = UncertaintyModel::Interval(IntervalConfig::default());
        let result = propagate_binary(&left, &right, BinaryOp::Mul, &model);

        if let UncertainValue::Interval { lower, upper } = result {
            assert!((lower - 8.0).abs() < 1e-10); // 2*4
            assert!((upper - 15.0).abs() < 1e-10); // 3*5
        } else {
            panic!("Expected Interval value");
        }
    }

    #[test]
    fn test_affine_addition() {
        let left = UncertainValue::Affine {
            center: 10.0,
            noise_terms: vec![(1, 0.5)],
        };
        let right = UncertainValue::Affine {
            center: 5.0,
            noise_terms: vec![(1, 0.3)], // Same noise symbol - correlated
        };

        let model = UncertaintyModel::Affine(AffineConfig::default());
        let result = propagate_binary(&left, &right, BinaryOp::Add, &model);

        if let UncertainValue::Affine {
            center,
            noise_terms,
        } = result
        {
            assert!((center - 15.0).abs() < 1e-10);
            // Correlated noise should combine: 0.5 + 0.3 = 0.8
            let coef = noise_terms.iter().find(|(id, _)| *id == 1).map(|(_, c)| *c);
            assert!(coef.is_some());
            assert!((coef.unwrap() - 0.8).abs() < 1e-10);
        } else {
            panic!("Expected Affine value");
        }
    }

    #[test]
    fn test_uncertainty_value_certainty() {
        let point = UncertainValue::Point {
            value: 10.0,
            confidence: 0.85,
        };
        assert!((point.certainty() - 0.85).abs() < 1e-10);

        let interval = UncertainValue::Interval {
            lower: 9.0,
            upper: 11.0,
        };
        // Width = 2, certainty = 1/(1+2) = 0.333...
        assert!((interval.certainty() - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_model_selection() {
        let config = EpistemicConfig::interval();
        assert_eq!(config.default_model.name(), "Interval");

        let mut config = EpistemicConfig::default();
        config.set_operation_model(
            "ode_solve",
            UncertaintyModel::Affine(AffineConfig::default()),
        );

        assert_eq!(config.get_model("ode_solve").name(), "Affine");
        assert_eq!(config.get_model("other").name(), "Probabilistic");
    }
}
