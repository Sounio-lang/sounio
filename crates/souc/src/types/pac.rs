//! PAC Learning Types for Sounio
//!
//! Provides compile-time verification of PAC learning guarantees:
//! - VC dimension tracking
//! - Sample complexity verification
//! - Generalization bounds in types
//!
//! References:
//! - Valiant (1984) "A Theory of the Learnable"
//! - Vapnik & Chervonenkis (1971) VC dimension
//! - Blumer et al. (1989) Sample complexity bounds
//! - McAllester (1999) PAC-Bayesian bounds

use std::collections::HashMap;
use std::fmt;

/// Represents the VC (Vapnik-Chervonenkis) dimension of a hypothesis class.
/// The VC dimension d measures the capacity of the class; finite d implies PAC-learnability.
#[derive(Debug, Clone, PartialEq)]
pub enum VCDimension {
    /// Known constant VC dimension d.
    Const(u64),
    /// Symbolic VC dimension from a type parameter or variable.
    Symbolic(String),
    /// Expressive VC dimension as a computed expression.
    Expr(Box<VCDimExpr>),
    /// Infinite VC dimension, implying the class is not PAC-learnable.
    Infinite,
    /// Unknown VC dimension, to be inferred or bounded later.
    Unknown,
}

impl fmt::Display for VCDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VCDimension::Const(d) => write!(f, "{}", d),
            VCDimension::Symbolic(s) => write!(f, "{}", s),
            VCDimension::Expr(expr) => write!(f, "{}", expr),
            VCDimension::Infinite => write!(f, "∞"),
            VCDimension::Unknown => write!(f, "?"),
        }
    }
}

impl Default for VCDimension {
    fn default() -> Self {
        VCDimension::Unknown
    }
}

/// Expression tree for symbolic VC dimension computations.
/// Supports basic arithmetic and logarithmic operations common in VC bounds,
/// e.g., d = O(W² log W) for neural networks with W weights.
#[derive(Debug, Clone, PartialEq)]
pub enum VCDimExpr {
    /// Constant value.
    Const(u64),
    /// Variable (e.g., from type parameter).
    Var(String),
    /// Addition: e₁ + e₂.
    Add(Box<VCDimExpr>, Box<VCDimExpr>),
    /// Multiplication: e₁ × e₂.
    Mul(Box<VCDimExpr>, Box<VCDimExpr>),
    /// Logarithm (base-2): log₂(e), floored to integer.
    Log(Box<VCDimExpr>),
    /// Exponentiation: e^b where b is a small integer exponent.
    Pow(Box<VCDimExpr>, u32),
}

impl VCDimExpr {
    /// Attempts to evaluate the expression to a constant u64 using provided variable bindings.
    /// Returns None if evaluation fails (e.g., unbound variable, non-integer log, overflow).
    pub fn try_eval(&self, bindings: &HashMap<String, u64>) -> Option<u64> {
        match self {
            VCDimExpr::Const(c) => Some(*c),
            VCDimExpr::Var(v) => bindings.get(v).copied(),
            VCDimExpr::Add(left, right) => {
                let l = left.try_eval(bindings)?;
                let r = right.try_eval(bindings)?;
                l.checked_add(r)
            }
            VCDimExpr::Mul(left, right) => {
                let l = left.try_eval(bindings)?;
                let r = right.try_eval(bindings)?;
                l.checked_mul(r)
            }
            VCDimExpr::Log(expr) => {
                let val = expr.try_eval(bindings)? as f64;
                if val <= 0.0 {
                    return None;
                }
                Some(val.log2().floor() as u64)
            }
            VCDimExpr::Pow(base, exp) => {
                let b = base.try_eval(bindings)?;
                let mut res = 1u64;
                for _ in 0..*exp {
                    res = res.checked_mul(b)?;
                }
                Some(res)
            }
        }
    }

    /// Create a simple addition expression: a + b
    pub fn add(a: VCDimExpr, b: VCDimExpr) -> Self {
        VCDimExpr::Add(Box::new(a), Box::new(b))
    }

    /// Create a simple multiplication expression: a * b
    pub fn mul(a: VCDimExpr, b: VCDimExpr) -> Self {
        VCDimExpr::Mul(Box::new(a), Box::new(b))
    }

    /// Create d + 1 (common for linear classifiers)
    pub fn d_plus_one(var: &str) -> Self {
        VCDimExpr::add(VCDimExpr::Var(var.to_string()), VCDimExpr::Const(1))
    }
}

impl fmt::Display for VCDimExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VCDimExpr::Const(c) => write!(f, "{}", c),
            VCDimExpr::Var(v) => write!(f, "{}", v),
            VCDimExpr::Add(l, r) => write!(f, "({} + {})", l, r),
            VCDimExpr::Mul(l, r) => write!(f, "({} * {})", l, r),
            VCDimExpr::Log(e) => write!(f, "log₂({})", e),
            VCDimExpr::Pow(b, e) => write!(f, "{}^{}", b, e),
        }
    }
}

/// Bound on the approximation error ε in PAC learning.
#[derive(Debug, Clone, PartialEq)]
pub enum EpsilonBound {
    /// Constant error bound.
    Const(f64),
    /// Symbolic bound (e.g., from parameters).
    Symbolic(String),
    /// Unknown bound.
    Unknown,
}

impl Default for EpsilonBound {
    fn default() -> Self {
        EpsilonBound::Unknown
    }
}

/// Bound on the confidence parameter δ (failure probability).
#[derive(Debug, Clone, PartialEq)]
pub enum DeltaBound {
    /// Constant confidence bound (0 < δ ≤ 1).
    Const(f64),
    /// Symbolic bound.
    Symbolic(String),
    /// Unknown bound.
    Unknown,
}

impl Default for DeltaBound {
    fn default() -> Self {
        DeltaBound::Unknown
    }
}

/// Bound on the sample complexity m.
#[derive(Debug, Clone, PartialEq)]
pub enum SampleBound {
    /// Constant sample size.
    Const(u64),
    /// Symbolic bound.
    Symbolic(String),
    /// Computed from expression.
    Expr(Box<SampleBoundExpr>),
    /// Unknown bound.
    Unknown,
}

impl Default for SampleBound {
    fn default() -> Self {
        SampleBound::Unknown
    }
}

/// Expression for computing sample bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct SampleBoundExpr {
    pub vc_dim: VCDimension,
    pub epsilon: EpsilonBound,
    pub delta: DeltaBound,
}

/// Bound on the error rate (e.g., training or generalization error).
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorBound {
    /// Constant error rate (0 ≤ err ≤ 1).
    Const(f64),
    /// Symbolic bound.
    Symbolic(String),
    /// VC-based bound: O(√(d/m))
    VCBound {
        vc_dim: VCDimension,
        samples: SampleBound,
    },
    /// Unknown bound.
    Unknown,
}

impl Default for ErrorBound {
    fn default() -> Self {
        ErrorBound::Unknown
    }
}

/// Represents the sample complexity requirements for PAC learning.
///
/// The fundamental theorem of PAC learning states that for a hypothesis class
/// with VC dimension d, the sample complexity for (ε, δ)-PAC learning is:
///
/// m ≥ (1/ε²) * (d·ln(1/δ) + ln(1/ε))
#[derive(Debug, Clone, PartialEq)]
pub struct SampleComplexity {
    /// VC dimension of the hypothesis class.
    pub vc_dim: VCDimension,
    /// Desired approximation error ε.
    pub epsilon: EpsilonBound,
    /// Confidence parameter δ.
    pub delta: DeltaBound,
    /// Minimum required samples m.
    pub min_samples: SampleBound,
}

impl SampleComplexity {
    /// Create a new sample complexity bound.
    pub fn new(vc_dim: VCDimension, epsilon: EpsilonBound, delta: DeltaBound) -> Self {
        let min_samples = Self::compute_bound(&vc_dim, &epsilon, &delta);
        Self {
            vc_dim,
            epsilon,
            delta,
            min_samples,
        }
    }

    fn compute_bound(
        vc_dim: &VCDimension,
        epsilon: &EpsilonBound,
        delta: &DeltaBound,
    ) -> SampleBound {
        match (vc_dim, epsilon, delta) {
            (VCDimension::Const(d), EpsilonBound::Const(eps), DeltaBound::Const(del)) => {
                SampleBound::Const(compute_sample_complexity(*d, *eps, *del))
            }
            _ => SampleBound::Expr(Box::new(SampleBoundExpr {
                vc_dim: vc_dim.clone(),
                epsilon: epsilon.clone(),
                delta: delta.clone(),
            })),
        }
    }
}

/// Generalization bound, relating training error to true error.
///
/// With probability ≥ 1-δ:
/// test_error ≤ train_error + √((d·ln(2m/d) + ln(4/δ)) / m)
#[derive(Debug, Clone, PartialEq)]
pub struct GeneralizationBound {
    /// Training error.
    pub train_error: ErrorBound,
    /// Gap or generalization gap bound.
    pub gap_bound: ErrorBound,
    /// Confidence (1 - δ) for the bound.
    pub confidence: f64,
}

impl GeneralizationBound {
    /// Create a new generalization bound.
    pub fn new(train_error: ErrorBound, gap_bound: ErrorBound, confidence: f64) -> Self {
        Self {
            train_error,
            gap_bound,
            confidence,
        }
    }

    /// Create from PAC parameters.
    pub fn from_pac(vc_dim: u64, samples: u64, delta: f64, train_error: f64) -> Self {
        let gap = compute_generalization_gap(vc_dim, samples, delta);
        Self {
            train_error: ErrorBound::Const(train_error),
            gap_bound: ErrorBound::Const(gap),
            confidence: 1.0 - delta,
        }
    }

    /// Get the upper bound on test error (train_error + gap).
    pub fn test_error_bound(&self) -> Option<f64> {
        match (&self.train_error, &self.gap_bound) {
            (ErrorBound::Const(train), ErrorBound::Const(gap)) => Some(train + gap),
            _ => None,
        }
    }
}

impl Default for GeneralizationBound {
    fn default() -> Self {
        Self {
            train_error: ErrorBound::Unknown,
            gap_bound: ErrorBound::Unknown,
            confidence: 0.0,
        }
    }
}

/// PAC-Bayesian bound, providing tighter posterior guarantees.
///
/// McAllester bound:
/// test_error ≤ train_error + √((KL(posterior || prior) + ln(m/δ)) / (2m))
#[derive(Debug, Clone, PartialEq)]
pub struct PACBayesBound {
    /// KL divergence between posterior and prior.
    pub kl_divergence: f64,
    /// Number of samples m.
    pub samples: u64,
    /// Confidence δ.
    pub delta: f64,
}

impl PACBayesBound {
    /// Compute the generalization gap from PAC-Bayes bound.
    pub fn gap(&self) -> f64 {
        if self.samples == 0 || self.delta <= 0.0 || self.delta >= 1.0 {
            return f64::INFINITY;
        }
        let m = self.samples as f64;
        ((self.kl_divergence + (m / self.delta).ln()) / (2.0 * m)).sqrt()
    }
}

/// Hypothesis class descriptor, with VC dimension and related properties.
#[derive(Debug, Clone)]
pub struct HypothesisClass {
    /// Name of the class (e.g., "LinearClassifier").
    pub name: String,
    /// VC dimension d.
    pub vc_dim: VCDimension,
    /// Model type name.
    pub model_type: String,
    /// Whether the class is finite (affects learnability).
    pub is_finite: bool,
    /// Rademacher complexity bound.
    pub rademacher: Option<RademacherBound>,
}

impl HypothesisClass {
    /// Create a new hypothesis class.
    pub fn new(
        name: impl Into<String>,
        vc_dim: VCDimension,
        model_type: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            vc_dim,
            model_type: model_type.into(),
            is_finite: false,
            rademacher: None,
        }
    }

    /// Create a linear classifier hypothesis class.
    /// VC dimension = d + 1 for d-dimensional input.
    pub fn linear_classifier(dimensions: u64) -> Self {
        Self {
            name: format!("LinearClassifier<{}>", dimensions),
            vc_dim: VCDimension::Const(dimensions + 1),
            model_type: "LinearClassifier".to_string(),
            is_finite: false,
            rademacher: Some(RademacherBound::Sqrt {
                numerator: VCDimension::Const(dimensions + 1),
            }),
        }
    }

    /// Create a decision stump (single threshold) hypothesis class.
    /// VC dimension = 2.
    pub fn decision_stump() -> Self {
        Self {
            name: "DecisionStump".to_string(),
            vc_dim: VCDimension::Const(2),
            model_type: "DecisionStump".to_string(),
            is_finite: false,
            rademacher: None,
        }
    }

    /// Create an axis-aligned rectangle hypothesis class.
    /// VC dimension = 2d for d-dimensional input.
    pub fn axis_aligned_rectangle(dimensions: u64) -> Self {
        Self {
            name: format!("AxisAlignedRectangle<{}>", dimensions),
            vc_dim: VCDimension::Const(2 * dimensions),
            model_type: "AxisAlignedRectangle".to_string(),
            is_finite: false,
            rademacher: None,
        }
    }

    /// Create a neural network hypothesis class.
    /// VC dimension = O(W·L·log(W)) where W = weights, L = layers.
    pub fn neural_network(layers: u64, width: u64) -> Self {
        let weights = layers * width * width;
        Self {
            name: format!("NeuralNetwork<{},{}>", layers, width),
            vc_dim: VCDimension::Expr(Box::new(VCDimExpr::mul(
                VCDimExpr::mul(VCDimExpr::Const(weights), VCDimExpr::Const(layers)),
                VCDimExpr::Log(Box::new(VCDimExpr::Const(weights))),
            ))),
            model_type: "NeuralNetwork".to_string(),
            is_finite: false,
            rademacher: None,
        }
    }

    /// Create a decision tree hypothesis class.
    /// VC dimension = O(2^depth · log(features)).
    pub fn decision_tree(depth: u64, features: u64) -> Self {
        Self {
            name: format!("DecisionTree<{},{}>", depth, features),
            vc_dim: VCDimension::Expr(Box::new(VCDimExpr::mul(
                VCDimExpr::Pow(Box::new(VCDimExpr::Const(2)), depth as u32),
                VCDimExpr::Log(Box::new(VCDimExpr::Const(features))),
            ))),
            model_type: "DecisionTree".to_string(),
            is_finite: false,
            rademacher: None,
        }
    }
}

/// Rademacher complexity bound for the hypothesis class.
/// Relates to generalization via: test_error ≤ train_error + 2·R_n + O(√(ln(1/δ)/n))
#[derive(Debug, Clone, PartialEq)]
pub enum RademacherBound {
    /// Constant finite bound.
    Const(f64),
    /// O(√(d/n)) style bound.
    Sqrt { numerator: VCDimension },
    /// Infinite (unbounded class).
    Infinite,
    /// Unknown.
    Unknown,
}

// ============================================================================
// Computation Functions
// ============================================================================

/// Computes the minimum sample complexity m using the standard PAC bound.
///
/// Formula: m ≥ (1/ε²) * (d·ln(1/δ) + ln(1/ε))
///
/// This is the fundamental theorem of PAC learning.
pub fn compute_sample_complexity(vc_dim: u64, epsilon: f64, delta: f64) -> u64 {
    if epsilon <= 0.0 || epsilon >= 1.0 || delta <= 0.0 || delta >= 1.0 {
        return u64::MAX;
    }

    let eps_sq = epsilon * epsilon;
    let ln_delta_inv = (1.0 / delta).ln();
    let ln_eps_inv = (1.0 / epsilon).ln();

    let bound = (1.0 / eps_sq) * (vc_dim as f64 * ln_delta_inv + ln_eps_inv);
    bound.ceil() as u64
}

/// Computes the tighter Hanneke (2016) sample complexity bound.
///
/// Formula: m ≥ C * (d + ln(1/δ)) / ε²
/// where C is a universal constant (typically ~2).
pub fn compute_sample_complexity_tight(vc_dim: u64, epsilon: f64, delta: f64) -> u64 {
    if epsilon <= 0.0 || epsilon >= 1.0 || delta <= 0.0 || delta >= 1.0 {
        return u64::MAX;
    }

    let c = 2.0; // Universal constant
    let eps_sq = epsilon * epsilon;
    let d = vc_dim as f64;

    let bound = c * (d + (1.0 / delta).ln()) / eps_sq;
    bound.ceil() as u64
}

/// Computes the generalization gap from VC dimension and sample size.
///
/// Gap ≤ √((d·ln(2m/d) + ln(4/δ)) / m)
pub fn compute_generalization_gap(vc_dim: u64, samples: u64, delta: f64) -> f64 {
    if samples == 0 || delta <= 0.0 || delta >= 1.0 {
        return f64::INFINITY;
    }

    let d = vc_dim as f64;
    let m = samples as f64;

    let term1 = d * (2.0 * m / d).ln();
    let term2 = (4.0 / delta).ln();

    ((term1 + term2) / m).sqrt()
}

/// Computes the PAC-Bayes generalization gap.
///
/// Gap ≤ √((KL + ln(m/δ)) / (2m))
pub fn compute_pac_bayes_gap(kl_divergence: f64, samples: u64, delta: f64) -> f64 {
    if samples == 0 || delta <= 0.0 || delta >= 1.0 {
        return f64::INFINITY;
    }

    let m = samples as f64;
    ((kl_divergence + (m / delta).ln()) / (2.0 * m)).sqrt()
}

/// Computes sample complexity for finite hypothesis classes.
///
/// For |H| finite: m ≥ (1/ε) * ln(|H|/δ)
pub fn compute_finite_sample_complexity(hypothesis_size: u64, epsilon: f64, delta: f64) -> u64 {
    if epsilon <= 0.0 || epsilon >= 1.0 || delta <= 0.0 || delta >= 1.0 {
        return u64::MAX;
    }

    let bound = (1.0 / epsilon) * (hypothesis_size as f64 / delta).ln();
    bound.ceil() as u64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_complexity_basic() {
        // Linear classifier in 10D: VC dim = 11
        // epsilon = 0.01, delta = 0.05
        let required = compute_sample_complexity(11, 0.01, 0.05);

        // Should be approximately 3000-6000
        assert!(required >= 3000 && required <= 7000, "Got {}", required);
    }

    #[test]
    fn test_sample_complexity_monotonic_in_vc_dim() {
        let samples_d5 = compute_sample_complexity(5, 0.01, 0.05);
        let samples_d10 = compute_sample_complexity(10, 0.01, 0.05);
        let samples_d20 = compute_sample_complexity(20, 0.01, 0.05);

        assert!(samples_d5 < samples_d10);
        assert!(samples_d10 < samples_d20);
    }

    #[test]
    fn test_sample_complexity_monotonic_in_epsilon() {
        let samples_eps_001 = compute_sample_complexity(10, 0.01, 0.05);
        let samples_eps_01 = compute_sample_complexity(10, 0.1, 0.05);

        assert!(samples_eps_01 < samples_eps_001);
    }

    #[test]
    fn test_sample_complexity_monotonic_in_delta() {
        let samples_delta_001 = compute_sample_complexity(10, 0.01, 0.01);
        let samples_delta_01 = compute_sample_complexity(10, 0.01, 0.1);

        assert!(samples_delta_01 < samples_delta_001);
    }

    #[test]
    fn test_vc_dim_expr_eval() {
        // d + 1 with d = 10
        let expr = VCDimExpr::d_plus_one("d");
        let mut bindings = HashMap::new();
        bindings.insert("d".to_string(), 10);

        assert_eq!(expr.try_eval(&bindings), Some(11));
    }

    #[test]
    fn test_vc_dim_expr_neural_network() {
        // W * L * log(W) with W = 100, L = 3
        let expr = VCDimExpr::mul(
            VCDimExpr::mul(
                VCDimExpr::Var("W".to_string()),
                VCDimExpr::Var("L".to_string()),
            ),
            VCDimExpr::Log(Box::new(VCDimExpr::Var("W".to_string()))),
        );

        let mut bindings = HashMap::new();
        bindings.insert("W".to_string(), 100);
        bindings.insert("L".to_string(), 3);

        // 100 * 3 * log2(100) = 300 * 6 = 1800
        let result = expr.try_eval(&bindings).unwrap();
        assert!(result >= 1800 && result <= 2100);
    }

    #[test]
    fn test_generalization_bound() {
        let bound = GeneralizationBound::from_pac(10, 5000, 0.05, 0.02);

        assert!(bound.confidence > 0.94);
        let test_bound = bound.test_error_bound().unwrap();
        assert!(
            test_bound < 0.1,
            "Test error bound should be reasonable: {}",
            test_bound
        );
    }

    #[test]
    fn test_pac_bayes_bound() {
        let bound = PACBayesBound {
            kl_divergence: 10.0,
            samples: 1000,
            delta: 0.05,
        };

        let gap = bound.gap();
        assert!(gap > 0.0 && gap < 0.5, "PAC-Bayes gap: {}", gap);
    }

    #[test]
    fn test_hypothesis_class_linear() {
        let h = HypothesisClass::linear_classifier(10);
        assert_eq!(h.vc_dim, VCDimension::Const(11));
    }

    #[test]
    fn test_hypothesis_class_decision_stump() {
        let h = HypothesisClass::decision_stump();
        assert_eq!(h.vc_dim, VCDimension::Const(2));
    }

    #[test]
    fn test_finite_sample_complexity() {
        // Finite hypothesis class with 1000 hypotheses
        let required = compute_finite_sample_complexity(1000, 0.1, 0.05);

        // Should be much smaller than infinite VC case
        assert!(required < 1000);
    }

    #[test]
    fn test_vc_dimension_display() {
        assert_eq!(format!("{}", VCDimension::Const(10)), "10");
        assert_eq!(format!("{}", VCDimension::Symbolic("d".to_string())), "d");
        assert_eq!(format!("{}", VCDimension::Infinite), "∞");
        assert_eq!(format!("{}", VCDimension::Unknown), "?");
    }

    #[test]
    fn test_invalid_inputs() {
        // Invalid epsilon
        assert_eq!(compute_sample_complexity(10, 0.0, 0.05), u64::MAX);
        assert_eq!(compute_sample_complexity(10, 1.0, 0.05), u64::MAX);

        // Invalid delta
        assert_eq!(compute_sample_complexity(10, 0.01, 0.0), u64::MAX);
        assert_eq!(compute_sample_complexity(10, 0.01, 1.0), u64::MAX);
    }
}
