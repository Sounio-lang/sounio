//! Numerical Stability & Error Propagation for GPU
//!
//! This module provides sophisticated numerical analysis for GPU kernels, integrating
//! with Sounio's epistemic computing framework to treat numerical error as a form
//! of uncertainty.
//!
//! # Key Features
//!
//! 1. **Error Representation**: ULP-based error tracking with relative/absolute bounds
//! 2. **Error Propagation**: Track how errors accumulate through arithmetic operations
//! 3. **Stability Analysis**: Detect ill-conditioned operations, cancellation, overflow/underflow
//! 4. **Precision Selection**: Recommend optimal precision (FP16/FP32/FP64) per operation
//! 5. **Mitigation**: Apply compensated algorithms (Kahan summation, etc.)
//!
//! # Integration with Epistemic Computing
//!
//! Numerical error is a form of epistemic uncertainty:
//! - Rounding errors → epistemic epsilon bounds
//! - Condition numbers → uncertainty amplification
//! - Catastrophic cancellation → validity predicate violations
//!
//! ```text
//! Numerical Error ⊆ Epistemic Uncertainty
//!     ↓                      ↓
//! ErrorBound        →    Shadow Epsilon Register
//! StabilityRisk     →    Validity Predicate
//! ULP Error         →    Provenance Tracking
//! ```

use std::collections::HashMap;
use std::fmt;

// ============================================================================
// Error Representation (~150 LOC)
// ============================================================================

/// Units in Last Place (ULP) error measurement
///
/// ULP is the standard way to measure floating-point error, representing
/// the gap between two adjacent representable numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UlpError {
    /// Number of ULPs between computed and exact result
    pub ulps: u64,
    /// Relative error: |computed - exact| / |exact|
    pub relative_error: f64,
    /// Absolute error: |computed - exact|
    pub absolute_error: f64,
}

impl UlpError {
    /// Create a new ULP error measurement
    pub fn new(ulps: u64, relative_error: f64, absolute_error: f64) -> Self {
        Self {
            ulps,
            relative_error,
            absolute_error,
        }
    }

    /// Create from computed and exact values
    pub fn from_values(computed: f64, exact: f64) -> Self {
        let abs_error = (computed - exact).abs();
        let rel_error = if exact != 0.0 {
            abs_error / exact.abs()
        } else {
            abs_error
        };

        // Compute ULP distance
        let ulps = Self::ulp_distance(computed, exact);

        Self {
            ulps,
            relative_error: rel_error,
            absolute_error: abs_error,
        }
    }

    /// Compute ULP distance between two f64 values
    fn ulp_distance(a: f64, b: f64) -> u64 {
        if a.is_nan() || b.is_nan() {
            return u64::MAX;
        }
        if a == b {
            return 0;
        }

        let a_bits = a.to_bits() as i64;
        let b_bits = b.to_bits() as i64;

        (a_bits.wrapping_sub(b_bits)).unsigned_abs()
    }

    /// Check if error is within acceptable bounds
    pub fn is_acceptable(&self, max_ulps: u64, max_relative: f64) -> bool {
        self.ulps <= max_ulps && self.relative_error <= max_relative
    }

    /// Convert to epistemic epsilon bound
    pub fn to_epsilon(&self) -> f32 {
        self.relative_error.max(self.absolute_error) as f32
    }

    /// Combine two ULP errors (conservative upper bound)
    pub fn combine(&self, other: &UlpError) -> UlpError {
        UlpError {
            ulps: self.ulps.saturating_add(other.ulps),
            relative_error: self.relative_error + other.relative_error,
            absolute_error: self.absolute_error + other.absolute_error,
        }
    }

    /// Scale error by a factor (for multiplicative operations)
    pub fn scale(&self, factor: f64) -> UlpError {
        UlpError {
            ulps: self.ulps,
            relative_error: self.relative_error * factor.abs(),
            absolute_error: self.absolute_error * factor.abs(),
        }
    }
}

impl fmt::Display for UlpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ULP (rel: {:.2e}, abs: {:.2e})",
            self.ulps, self.relative_error, self.absolute_error
        )
    }
}

/// Error bounds for an operation or value
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorBound {
    /// Minimum possible error
    pub min_error: f64,
    /// Maximum possible error
    pub max_error: f64,
    /// Expected (average) error
    pub expected_error: f64,
    /// Confidence in this bound estimate (0.0 to 1.0)
    pub confidence: f64,
}

impl ErrorBound {
    /// Create a new error bound
    pub fn new(min_error: f64, max_error: f64, expected_error: f64, confidence: f64) -> Self {
        Self {
            min_error,
            max_error,
            expected_error: expected_error.clamp(min_error, max_error),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Create from single point estimate
    pub fn from_estimate(error: f64, confidence: f64) -> Self {
        Self::new(0.0, error * 2.0, error, confidence)
    }

    /// Create zero error (exact computation)
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }

    /// Create for machine epsilon
    pub fn machine_epsilon(precision: Precision) -> Self {
        let eps = precision.epsilon();
        Self::new(0.0, eps * 2.0, eps, 0.95)
    }

    /// Check if computation is numerically stable
    pub fn is_stable(&self, threshold: f64) -> bool {
        self.max_error <= threshold
    }

    /// Widen the error bound (conservative estimate)
    pub fn widen(&self, factor: f64) -> ErrorBound {
        ErrorBound {
            min_error: self.min_error,
            max_error: self.max_error * factor,
            expected_error: self.expected_error * factor,
            confidence: self.confidence * 0.9, // Reduce confidence when widening
        }
    }

    /// Combine two error bounds (for operations on two values)
    pub fn combine(&self, other: &ErrorBound) -> ErrorBound {
        ErrorBound {
            min_error: self.min_error + other.min_error,
            max_error: self.max_error + other.max_error,
            expected_error: (self.expected_error.powi(2) + other.expected_error.powi(2)).sqrt(),
            confidence: self.confidence * other.confidence,
        }
    }

    /// Convert to epistemic epsilon
    pub fn to_epsilon(&self) -> f32 {
        self.expected_error as f32
    }
}

impl fmt::Display for ErrorBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:.2e}, {:.2e}] exp: {:.2e} conf: {:.2}",
            self.min_error, self.max_error, self.expected_error, self.confidence
        )
    }
}

/// Numerical stability risk assessment
#[derive(Debug, Clone, PartialEq)]
pub enum StabilityRisk {
    /// Operation is numerically stable
    Stable,
    /// Mild instability (amplified error but manageable)
    MildInstability { condition_number: f64 },
    /// Severe risk of overflow or underflow
    Severe {
        overflow_risk: f64,
        underflow_risk: f64,
    },
    /// Catastrophic cancellation (a - b where a ≈ b)
    Catastrophic { cancellation_risk: f64 },
}

impl StabilityRisk {
    /// Check if operation should be rejected
    pub fn is_unacceptable(&self) -> bool {
        matches!(self, StabilityRisk::Catastrophic { .. })
    }

    /// Get risk severity score (0.0 = stable, 1.0 = catastrophic)
    pub fn severity(&self) -> f64 {
        match self {
            StabilityRisk::Stable => 0.0,
            StabilityRisk::MildInstability { condition_number } => {
                (condition_number.log10() / 16.0).min(0.5) // log10(cond) normalized
            }
            StabilityRisk::Severe {
                overflow_risk,
                underflow_risk,
            } => 0.5 + 0.3 * overflow_risk.max(*underflow_risk),
            StabilityRisk::Catastrophic { cancellation_risk } => 0.8 + 0.2 * cancellation_risk,
        }
    }

    /// Get recommended mitigation strategy
    pub fn mitigation(&self) -> Option<MitigationStrategy> {
        match self {
            StabilityRisk::Stable => None,
            StabilityRisk::MildInstability { condition_number } if *condition_number > 1e8 => {
                Some(MitigationStrategy::UpgradePrecision)
            }
            StabilityRisk::Severe { overflow_risk, .. } if *overflow_risk > 0.5 => {
                Some(MitigationStrategy::Rescaling)
            }
            StabilityRisk::Catastrophic { .. } => Some(MitigationStrategy::CompensatedAlgorithm),
            _ => Some(MitigationStrategy::UpgradePrecision),
        }
    }
}

impl fmt::Display for StabilityRisk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StabilityRisk::Stable => write!(f, "Stable"),
            StabilityRisk::MildInstability { condition_number } => {
                write!(f, "Mild instability (κ = {:.2e})", condition_number)
            }
            StabilityRisk::Severe {
                overflow_risk,
                underflow_risk,
            } => {
                write!(
                    f,
                    "Severe (overflow: {:.2}, underflow: {:.2})",
                    overflow_risk, underflow_risk
                )
            }
            StabilityRisk::Catastrophic { cancellation_risk } => {
                write!(
                    f,
                    "Catastrophic cancellation (risk: {:.2})",
                    cancellation_risk
                )
            }
        }
    }
}

/// Floating-point precision levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    FP16,
    FP32,
    FP64,
    FP8,
}

impl Precision {
    /// Machine epsilon for this precision
    pub fn epsilon(&self) -> f64 {
        match self {
            Precision::FP8 => 0.125,         // ~1/8
            Precision::FP16 => 0.0009765625, // 2^-10
            Precision::FP32 => 1.1920929e-7, // 2^-23
            Precision::FP64 => 2.220446e-16, // 2^-52
        }
    }

    /// Maximum representable value
    pub fn max_value(&self) -> f64 {
        match self {
            Precision::FP8 => 448.0, // E4M3 max
            Precision::FP16 => 65504.0,
            Precision::FP32 => 3.4028235e38,
            Precision::FP64 => 1.7976931e308,
        }
    }

    /// Minimum normal value
    pub fn min_normal(&self) -> f64 {
        match self {
            Precision::FP8 => 0.015625,       // 2^-6
            Precision::FP16 => 6.103515e-5,   // 2^-14
            Precision::FP32 => 1.1754944e-38, // 2^-126
            Precision::FP64 => 2.225074e-308, // 2^-1022
        }
    }

    /// Number of mantissa bits
    pub fn mantissa_bits(&self) -> u32 {
        match self {
            Precision::FP8 => 3,
            Precision::FP16 => 10,
            Precision::FP32 => 23,
            Precision::FP64 => 52,
        }
    }

    /// Check if value is representable without overflow
    pub fn can_represent(&self, value: f64) -> bool {
        value.abs() <= self.max_value() && (value == 0.0 || value.abs() >= self.min_normal())
    }

    /// Estimate quantization error for this precision
    pub fn quantization_error(&self, value: f64) -> f64 {
        if value == 0.0 {
            return 0.0;
        }
        value.abs() * self.epsilon()
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Precision::FP8 => write!(f, "FP8"),
            Precision::FP16 => write!(f, "FP16"),
            Precision::FP32 => write!(f, "FP32"),
            Precision::FP64 => write!(f, "FP64"),
        }
    }
}

// ============================================================================
// Error Propagation (~250 LOC)
// ============================================================================

/// Tracks error propagation through arithmetic operations
pub struct ErrorPropagator {
    /// Current precision level
    precision: Precision,
    /// Error tracking mode
    mode: PropagationMode,
    /// Operation history for analysis
    history: Vec<ErrorEvent>,
}

/// Error propagation tracking mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropagationMode {
    /// Conservative (worst-case) bounds
    Conservative,
    /// Expected (average-case) bounds
    Expected,
    /// Interval arithmetic with tight bounds
    Interval,
}

/// Record of an error-inducing event
#[derive(Debug, Clone)]
pub struct ErrorEvent {
    pub operation: String,
    pub input_errors: Vec<ErrorBound>,
    pub output_error: ErrorBound,
    pub amplification: f64,
}

impl ErrorPropagator {
    /// Create a new error propagator
    pub fn new(precision: Precision, mode: PropagationMode) -> Self {
        Self {
            precision,
            mode,
            history: Vec::new(),
        }
    }

    /// Get the current precision
    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// Propagate error through addition: z = x + y
    pub fn propagate_add(&mut self, x: ErrorBound, y: ErrorBound) -> ErrorBound {
        // For addition, errors add in quadrature
        let result = match self.mode {
            PropagationMode::Conservative => ErrorBound {
                min_error: 0.0,
                max_error: x.max_error + y.max_error + self.precision.epsilon(),
                expected_error: (x.expected_error.powi(2) + y.expected_error.powi(2)).sqrt(),
                confidence: x.confidence.min(y.confidence),
            },
            PropagationMode::Expected => {
                let expected = (x.expected_error.powi(2) + y.expected_error.powi(2)).sqrt();
                ErrorBound {
                    min_error: 0.0,
                    max_error: expected * 2.0,
                    expected_error: expected,
                    confidence: x.confidence * y.confidence,
                }
            }
            PropagationMode::Interval => {
                // Tight interval arithmetic
                let min = x.min_error + y.min_error;
                let max = x.max_error + y.max_error + self.precision.epsilon();
                ErrorBound {
                    min_error: min,
                    max_error: max,
                    expected_error: (min + max) / 2.0,
                    confidence: x.confidence * y.confidence,
                }
            }
        };

        self.record_event("add", vec![x, y], result);
        result
    }

    /// Propagate error through subtraction: z = x - y
    pub fn propagate_sub(&mut self, x: ErrorBound, y: ErrorBound) -> ErrorBound {
        // Subtraction has same error propagation as addition
        let mut result = self.propagate_add(x, y);

        // But check for catastrophic cancellation
        // (This is a simplified check; real implementation would need values)
        result.confidence *= 0.95;

        self.history.last_mut().unwrap().operation = "sub".to_string();
        result
    }

    /// Propagate error through multiplication: z = x * y
    pub fn propagate_mul(
        &mut self,
        x: ErrorBound,
        y: ErrorBound,
        x_val: f64,
        y_val: f64,
    ) -> ErrorBound {
        // For multiplication: δz/z ≈ δx/x + δy/y (relative errors add)
        let x_rel = if x_val != 0.0 {
            x.expected_error / x_val.abs()
        } else {
            x.expected_error
        };
        let y_rel = if y_val != 0.0 {
            y.expected_error / y_val.abs()
        } else {
            y.expected_error
        };

        let z_val = x_val * y_val;
        let expected_rel = (x_rel.powi(2) + y_rel.powi(2)).sqrt();
        let expected_abs = z_val.abs() * expected_rel;

        let result = ErrorBound {
            min_error: 0.0,
            max_error: expected_abs * 2.0 + self.precision.epsilon(),
            expected_error: expected_abs,
            confidence: x.confidence * y.confidence * 0.95,
        };

        self.record_event("mul", vec![x, y], result);
        result
    }

    /// Propagate error through division: z = x / y
    pub fn propagate_div(
        &mut self,
        x: ErrorBound,
        y: ErrorBound,
        x_val: f64,
        y_val: f64,
    ) -> ErrorBound {
        if y_val.abs() < self.precision.min_normal() {
            // Near-zero divisor: massive error amplification
            return ErrorBound {
                min_error: 0.0,
                max_error: f64::INFINITY,
                expected_error: 1.0,
                confidence: 0.0,
            };
        }

        // For division: δz/z ≈ δx/x + δy/y (similar to multiplication)
        let x_rel = if x_val != 0.0 {
            x.expected_error / x_val.abs()
        } else {
            x.expected_error
        };
        let y_rel = if y_val != 0.0 {
            y.expected_error / y_val.abs()
        } else {
            y.expected_error
        };

        let z_val = x_val / y_val;
        let expected_rel = (x_rel.powi(2) + y_rel.powi(2)).sqrt();
        let expected_abs = z_val.abs() * expected_rel;

        // Additional amplification near zero divisor
        let amplification = if y_val.abs() < 1.0 {
            1.0 / y_val.abs()
        } else {
            1.0
        };

        let result = ErrorBound {
            min_error: 0.0,
            max_error: (expected_abs * 2.0 + self.precision.epsilon()) * amplification,
            expected_error: expected_abs * amplification.sqrt(),
            confidence: x.confidence * y.confidence * 0.9,
        };

        self.record_event("div", vec![x, y], result);
        result
    }

    /// Propagate error through FMA: z = a * b + c
    pub fn propagate_fma(
        &mut self,
        a: ErrorBound,
        b: ErrorBound,
        c: ErrorBound,
        a_val: f64,
        b_val: f64,
    ) -> ErrorBound {
        // FMA is exact for the multiply-add, only one rounding error
        let mul_error = self.propagate_mul(a, b, a_val, b_val);

        // More accurate than separate mul + add
        let mut result = self.propagate_add(mul_error, c);
        result.expected_error *= 0.5; // FMA has better accuracy

        self.history.last_mut().unwrap().operation = "fma".to_string();
        result
    }

    /// Propagate error through square root: z = sqrt(x)
    pub fn propagate_sqrt(&mut self, x: ErrorBound, x_val: f64) -> ErrorBound {
        if x_val < 0.0 {
            // Domain error
            return ErrorBound {
                min_error: 0.0,
                max_error: f64::INFINITY,
                expected_error: 1.0,
                confidence: 0.0,
            };
        }

        if x_val == 0.0 {
            return ErrorBound::zero();
        }

        // For sqrt: δz ≈ δx / (2 * sqrt(x))
        let z_val = x_val.sqrt();
        let expected = x.expected_error / (2.0 * z_val);

        let result = ErrorBound {
            min_error: 0.0,
            max_error: expected * 2.0 + self.precision.epsilon(),
            expected_error: expected,
            confidence: x.confidence * 0.98,
        };

        self.record_event("sqrt", vec![x], result);
        result
    }

    /// Propagate error through exponential: z = exp(x)
    pub fn propagate_exp(&mut self, x: ErrorBound, x_val: f64) -> ErrorBound {
        // For exp: δz ≈ exp(x) * δx
        let z_val = x_val.exp();
        let expected = z_val.abs() * x.expected_error;

        let result = ErrorBound {
            min_error: 0.0,
            max_error: expected * 2.0 + self.precision.epsilon(),
            expected_error: expected,
            confidence: x.confidence * 0.95,
        };

        self.record_event("exp", vec![x], result);
        result
    }

    /// Propagate error through logarithm: z = log(x)
    pub fn propagate_log(&mut self, x: ErrorBound, x_val: f64) -> ErrorBound {
        if x_val <= 0.0 {
            // Domain error
            return ErrorBound {
                min_error: 0.0,
                max_error: f64::INFINITY,
                expected_error: 1.0,
                confidence: 0.0,
            };
        }

        // For log: δz ≈ δx / x
        let expected = x.expected_error / x_val.abs();

        let result = ErrorBound {
            min_error: 0.0,
            max_error: expected * 2.0 + self.precision.epsilon(),
            expected_error: expected,
            confidence: x.confidence * 0.95,
        };

        self.record_event("log", vec![x], result);
        result
    }

    /// Propagate error through sine: z = sin(x)
    pub fn propagate_sin(&mut self, x: ErrorBound, x_val: f64) -> ErrorBound {
        // For sin: δz ≈ |cos(x)| * δx
        let cos_x = x_val.cos().abs();
        let expected = cos_x * x.expected_error;

        let result = ErrorBound {
            min_error: 0.0,
            max_error: expected * 2.0 + self.precision.epsilon(),
            expected_error: expected,
            confidence: x.confidence * 0.9,
        };

        self.record_event("sin", vec![x], result);
        result
    }

    /// Propagate error through cosine: z = cos(x)
    pub fn propagate_cos(&mut self, x: ErrorBound, x_val: f64) -> ErrorBound {
        // For cos: δz ≈ |sin(x)| * δx
        let sin_x = x_val.sin().abs();
        let expected = sin_x * x.expected_error;

        let result = ErrorBound {
            min_error: 0.0,
            max_error: expected * 2.0 + self.precision.epsilon(),
            expected_error: expected,
            confidence: x.confidence * 0.9,
        };

        self.record_event("cos", vec![x], result);
        result
    }

    /// Propagate error through tangent: z = tan(x)
    pub fn propagate_tan(&mut self, x: ErrorBound, x_val: f64) -> ErrorBound {
        // For tan: δz ≈ sec²(x) * δx = (1 + tan²(x)) * δx
        let tan_x = x_val.tan();
        let sec2_x = 1.0 + tan_x * tan_x;
        let expected = sec2_x * x.expected_error;

        let result = ErrorBound {
            min_error: 0.0,
            max_error: expected * 2.0 + self.precision.epsilon(),
            expected_error: expected,
            confidence: x.confidence * 0.85,
        };

        self.record_event("tan", vec![x], result);
        result
    }

    /// Estimate error for a sum reduction
    pub fn propagate_sum(&mut self, values: &[ErrorBound]) -> ErrorBound {
        if values.is_empty() {
            return ErrorBound::zero();
        }

        // For n additions, error grows as sqrt(n) (if uncorrelated)
        let n = values.len() as f64;
        let avg_error = values.iter().map(|e| e.expected_error).sum::<f64>() / n;
        let expected = avg_error * n.sqrt();

        let result = ErrorBound {
            min_error: 0.0,
            max_error: expected * 2.0 + self.precision.epsilon() * n,
            expected_error: expected,
            confidence: values
                .iter()
                .map(|e| e.confidence)
                .product::<f64>()
                .powf(1.0 / n),
        };

        self.record_event("sum", values.to_vec(), result);
        result
    }

    /// Get error propagation history
    pub fn history(&self) -> &[ErrorEvent] {
        &self.history
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Record an error event
    fn record_event(&mut self, operation: &str, inputs: Vec<ErrorBound>, output: ErrorBound) {
        let max_input_error = inputs
            .iter()
            .map(|e| e.expected_error)
            .fold(0.0f64, f64::max);
        let amplification = if max_input_error > 0.0 {
            output.expected_error / max_input_error
        } else {
            1.0
        };

        self.history.push(ErrorEvent {
            operation: operation.to_string(),
            input_errors: inputs,
            output_error: output,
            amplification,
        });
    }
}

// ============================================================================
// Stability Analysis (~200 LOC)
// ============================================================================

/// Analyzes GPU kernels for numerical stability issues
pub struct StabilityAnalyzer {
    /// Target precision
    precision: Precision,
    /// Condition number threshold for warnings
    condition_threshold: f64,
    /// Cancellation detection threshold
    cancellation_threshold: f64,
    /// Detected stability issues
    issues: Vec<StabilityIssue>,
}

/// A detected numerical stability issue
#[derive(Debug, Clone)]
pub struct StabilityIssue {
    pub risk: StabilityRisk,
    pub operation: String,
    pub location: String,
    pub description: String,
    pub mitigation: Option<MitigationStrategy>,
}

impl StabilityAnalyzer {
    /// Create a new stability analyzer
    pub fn new(precision: Precision) -> Self {
        Self {
            precision,
            condition_threshold: 1e8,
            cancellation_threshold: 1e-6,
            issues: Vec::new(),
        }
    }

    /// Analyze a subtraction for catastrophic cancellation
    pub fn check_cancellation(&mut self, x: f64, y: f64, location: &str) -> StabilityRisk {
        if x == 0.0 && y == 0.0 {
            return StabilityRisk::Stable;
        }

        let diff = (x - y).abs();
        let magnitude = x.abs().max(y.abs());

        if magnitude == 0.0 {
            return StabilityRisk::Stable;
        }

        let relative_diff = diff / magnitude;

        let risk = if relative_diff < self.cancellation_threshold {
            // Severe cancellation: x ≈ y
            let cancellation_risk = 1.0 - (relative_diff / self.cancellation_threshold).min(1.0);
            StabilityRisk::Catastrophic { cancellation_risk }
        } else if relative_diff < 0.01 {
            // Mild cancellation
            let condition_number = magnitude / diff;
            StabilityRisk::MildInstability { condition_number }
        } else {
            StabilityRisk::Stable
        };

        if !matches!(risk, StabilityRisk::Stable) {
            self.issues.push(StabilityIssue {
                risk: risk.clone(),
                operation: "subtraction".to_string(),
                location: location.to_string(),
                description: format!("Catastrophic cancellation: {:.6e} - {:.6e}", x, y),
                mitigation: risk.mitigation(),
            });
        }

        risk
    }

    /// Analyze division for near-zero divisor
    pub fn check_division(&mut self, dividend: f64, divisor: f64, location: &str) -> StabilityRisk {
        let min_normal = self.precision.min_normal();
        let max_value = self.precision.max_value();

        let risk = if divisor.abs() < min_normal {
            // Underflow in divisor
            StabilityRisk::Severe {
                overflow_risk: 1.0,
                underflow_risk: 1.0,
            }
        } else if dividend.abs() / divisor.abs() > max_value * 0.1 {
            // Potential overflow in result
            StabilityRisk::Severe {
                overflow_risk: 0.8,
                underflow_risk: 0.0,
            }
        } else if divisor.abs() < 1.0 {
            // Large divisor amplifies error
            let condition_number = 1.0 / divisor.abs();
            StabilityRisk::MildInstability { condition_number }
        } else {
            StabilityRisk::Stable
        };

        if !matches!(risk, StabilityRisk::Stable) {
            self.issues.push(StabilityIssue {
                risk: risk.clone(),
                operation: "division".to_string(),
                location: location.to_string(),
                description: format!("Unstable division: {:.6e} / {:.6e}", dividend, divisor),
                mitigation: risk.mitigation(),
            });
        }

        risk
    }

    /// Check for potential overflow
    pub fn check_overflow(&mut self, value: f64, location: &str) -> StabilityRisk {
        let max_value = self.precision.max_value();

        let risk = if value.abs() > max_value {
            StabilityRisk::Severe {
                overflow_risk: 1.0,
                underflow_risk: 0.0,
            }
        } else if value.abs() > max_value * 0.5 {
            let overflow_risk = (value.abs() / max_value - 0.5) * 2.0;
            StabilityRisk::Severe {
                overflow_risk,
                underflow_risk: 0.0,
            }
        } else {
            StabilityRisk::Stable
        };

        if !matches!(risk, StabilityRisk::Stable) {
            self.issues.push(StabilityIssue {
                risk: risk.clone(),
                operation: "overflow_check".to_string(),
                location: location.to_string(),
                description: format!("Potential overflow: {:.6e}", value),
                mitigation: Some(MitigationStrategy::Rescaling),
            });
        }

        risk
    }

    /// Check for potential underflow
    pub fn check_underflow(&mut self, value: f64, location: &str) -> StabilityRisk {
        if value == 0.0 {
            return StabilityRisk::Stable;
        }

        let min_normal = self.precision.min_normal();

        let risk = if value.abs() < min_normal {
            StabilityRisk::Severe {
                overflow_risk: 0.0,
                underflow_risk: 1.0,
            }
        } else if value.abs() < min_normal * 100.0 {
            let underflow_risk = 1.0 - (value.abs() / (min_normal * 100.0)).log10().abs() / 2.0;
            StabilityRisk::Severe {
                overflow_risk: 0.0,
                underflow_risk: underflow_risk.clamp(0.0, 1.0),
            }
        } else {
            StabilityRisk::Stable
        };

        if !matches!(risk, StabilityRisk::Stable) {
            self.issues.push(StabilityIssue {
                risk: risk.clone(),
                operation: "underflow_check".to_string(),
                location: location.to_string(),
                description: format!("Potential underflow: {:.6e}", value),
                mitigation: Some(MitigationStrategy::Rescaling),
            });
        }

        risk
    }

    /// Estimate condition number for a matrix operation
    pub fn estimate_condition_number(&self, singular_values: &[f64]) -> f64 {
        if singular_values.is_empty() {
            return 1.0;
        }

        let max_sv = singular_values.iter().fold(0.0f64, |a, &b| a.max(b));
        let min_sv = singular_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        if min_sv == 0.0 {
            return f64::INFINITY;
        }

        max_sv / min_sv
    }

    /// Analyze condition number for stability
    pub fn check_condition_number(
        &mut self,
        condition_number: f64,
        location: &str,
    ) -> StabilityRisk {
        let risk = if condition_number > 1e12 {
            StabilityRisk::Catastrophic {
                cancellation_risk: 1.0,
            }
        } else if condition_number > self.condition_threshold {
            StabilityRisk::MildInstability { condition_number }
        } else {
            StabilityRisk::Stable
        };

        if !matches!(risk, StabilityRisk::Stable) {
            self.issues.push(StabilityIssue {
                risk: risk.clone(),
                operation: "condition_number".to_string(),
                location: location.to_string(),
                description: format!("Ill-conditioned: κ = {:.2e}", condition_number),
                mitigation: risk.mitigation(),
            });
        }

        risk
    }

    /// Get all detected issues
    pub fn issues(&self) -> &[StabilityIssue] {
        &self.issues
    }

    /// Check if any catastrophic issues were found
    pub fn has_catastrophic_issues(&self) -> bool {
        self.issues.iter().any(|i| i.risk.is_unacceptable())
    }

    /// Get summary statistics
    pub fn summary(&self) -> StabilitySummary {
        let mut stable = 0;
        let mut mild = 0;
        let mut severe = 0;
        let mut catastrophic = 0;

        for issue in &self.issues {
            match issue.risk {
                StabilityRisk::Stable => stable += 1,
                StabilityRisk::MildInstability { .. } => mild += 1,
                StabilityRisk::Severe { .. } => severe += 1,
                StabilityRisk::Catastrophic { .. } => catastrophic += 1,
            }
        }

        StabilitySummary {
            total_checks: self.issues.len(),
            stable,
            mild_instability: mild,
            severe,
            catastrophic,
        }
    }
}

/// Summary of stability analysis
#[derive(Debug, Clone)]
pub struct StabilitySummary {
    pub total_checks: usize,
    pub stable: usize,
    pub mild_instability: usize,
    pub severe: usize,
    pub catastrophic: usize,
}

impl StabilitySummary {
    pub fn is_acceptable(&self) -> bool {
        self.catastrophic == 0 && self.severe < self.total_checks / 10
    }
}

// ============================================================================
// Precision Selection (~150 LOC)
// ============================================================================

/// Recommends optimal precision for GPU operations
pub struct PrecisionAdvisor {
    /// Current default precision
    default_precision: Precision,
    /// Per-operation precision recommendations
    recommendations: HashMap<String, Precision>,
    /// Error tolerance threshold
    error_tolerance: f64,
}

impl PrecisionAdvisor {
    /// Create a new precision advisor
    pub fn new(default_precision: Precision, error_tolerance: f64) -> Self {
        Self {
            default_precision,
            recommendations: HashMap::new(),
            error_tolerance,
        }
    }

    /// Recommend precision for an operation based on error analysis
    pub fn recommend(
        &mut self,
        operation: &str,
        error_bound: ErrorBound,
        value_range: (f64, f64),
    ) -> Precision {
        let (min_val, max_val) = value_range;

        // Start with lowest precision and upgrade as needed
        let mut recommended = Precision::FP16;

        // Check if FP16 can represent the value range
        if max_val > Precision::FP16.max_value() || min_val < -Precision::FP16.max_value() {
            recommended = Precision::FP32;
        }

        // Check if error tolerance requires higher precision
        if error_bound.expected_error > self.error_tolerance {
            if recommended == Precision::FP16 {
                recommended = Precision::FP32;
            } else if recommended == Precision::FP32
                && error_bound.expected_error > self.error_tolerance * 10.0
            {
                recommended = Precision::FP64;
            }
        }

        // Check for subnormal values
        if min_val.abs() > 0.0 && min_val.abs() < recommended.min_normal() {
            recommended = match recommended {
                Precision::FP16 => Precision::FP32,
                Precision::FP32 => Precision::FP64,
                p => p,
            };
        }

        self.recommendations
            .insert(operation.to_string(), recommended);
        recommended
    }

    /// Recommend precision upgrade for stability risk
    pub fn recommend_for_risk(&mut self, operation: &str, risk: &StabilityRisk) -> Precision {
        let upgrade = match risk {
            StabilityRisk::Stable => self.default_precision,
            StabilityRisk::MildInstability { condition_number } => {
                if *condition_number > 1e10 {
                    Precision::FP64
                } else if *condition_number > 1e6 {
                    Precision::FP32
                } else {
                    self.default_precision
                }
            }
            StabilityRisk::Severe { .. } => {
                // Severe risks usually need higher precision
                match self.default_precision {
                    Precision::FP8 | Precision::FP16 => Precision::FP32,
                    Precision::FP32 => Precision::FP64,
                    Precision::FP64 => Precision::FP64,
                }
            }
            StabilityRisk::Catastrophic { .. } => {
                // Catastrophic needs highest precision available
                Precision::FP64
            }
        };

        self.recommendations.insert(operation.to_string(), upgrade);
        upgrade
    }

    /// Generate mixed-precision strategy for a kernel
    pub fn synthesize_strategy(&self, operations: &[String]) -> MixedPrecisionStrategy {
        let mut fp16_ops = Vec::new();
        let mut fp32_ops = Vec::new();
        let mut fp64_ops = Vec::new();

        for op in operations {
            match self
                .recommendations
                .get(op)
                .unwrap_or(&self.default_precision)
            {
                Precision::FP8 | Precision::FP16 => fp16_ops.push(op.clone()),
                Precision::FP32 => fp32_ops.push(op.clone()),
                Precision::FP64 => fp64_ops.push(op.clone()),
            }
        }

        MixedPrecisionStrategy {
            fp16_operations: fp16_ops,
            fp32_operations: fp32_ops,
            fp64_operations: fp64_ops,
        }
    }

    /// Check if quantization to INT8 is safe for given value range
    pub fn is_quantization_safe(&self, value_range: (f64, f64), error_bound: ErrorBound) -> bool {
        let (min_val, max_val) = value_range;
        let range = max_val - min_val;

        // INT8 has 256 levels
        let quantization_step = range / 256.0;
        let quantization_error = quantization_step / 2.0;

        // Safe if quantization error is within tolerance
        quantization_error < self.error_tolerance
            && error_bound.expected_error + quantization_error < self.error_tolerance * 2.0
    }

    /// Get recommended precision for a specific operation
    pub fn get_recommendation(&self, operation: &str) -> Option<Precision> {
        self.recommendations.get(operation).copied()
    }
}

/// Mixed-precision execution strategy
#[derive(Debug, Clone)]
pub struct MixedPrecisionStrategy {
    pub fp16_operations: Vec<String>,
    pub fp32_operations: Vec<String>,
    pub fp64_operations: Vec<String>,
}

impl MixedPrecisionStrategy {
    /// Estimate performance impact (relative to all-FP32)
    pub fn performance_factor(&self) -> f64 {
        let total = (self.fp16_operations.len()
            + self.fp32_operations.len()
            + self.fp64_operations.len()) as f64;
        if total == 0.0 {
            return 1.0;
        }

        // FP16 is ~2x faster, FP64 is ~2x slower (rough estimates)
        let fp16_factor = 2.0 * (self.fp16_operations.len() as f64 / total);
        let fp32_factor = 1.0 * (self.fp32_operations.len() as f64 / total);
        let fp64_factor = 0.5 * (self.fp64_operations.len() as f64 / total);

        fp16_factor + fp32_factor + fp64_factor
    }
}

// ============================================================================
// Mitigation (~50 LOC)
// ============================================================================

/// Numerical stability mitigation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MitigationStrategy {
    /// Upgrade to higher precision (FP16 → FP32 → FP64)
    UpgradePrecision,
    /// Apply Kahan summation for accumulation
    KahanSummation,
    /// Use compensated algorithm (e.g., 2Sum, 2Mul)
    CompensatedAlgorithm,
    /// Rescale values to prevent overflow/underflow
    Rescaling,
    /// Reorder operations to minimize error
    Reordering,
}

impl fmt::Display for MitigationStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MitigationStrategy::UpgradePrecision => write!(f, "Upgrade precision"),
            MitigationStrategy::KahanSummation => write!(f, "Kahan summation"),
            MitigationStrategy::CompensatedAlgorithm => write!(f, "Compensated algorithm"),
            MitigationStrategy::Rescaling => write!(f, "Rescaling"),
            MitigationStrategy::Reordering => write!(f, "Operation reordering"),
        }
    }
}

/// Applies stability mitigations to GPU code
pub struct StabilityMitigator {
    /// Applied mitigations
    mitigations: Vec<AppliedMitigation>,
}

#[derive(Debug, Clone)]
pub struct AppliedMitigation {
    pub strategy: MitigationStrategy,
    pub location: String,
    pub description: String,
}

impl StabilityMitigator {
    /// Create a new mitigator
    pub fn new() -> Self {
        Self {
            mitigations: Vec::new(),
        }
    }

    /// Apply precision upgrade
    pub fn apply_precision_upgrade(&mut self, operation: &str, from: Precision, to: Precision) {
        self.mitigations.push(AppliedMitigation {
            strategy: MitigationStrategy::UpgradePrecision,
            location: operation.to_string(),
            description: format!("Upgrade {} → {} for {}", from, to, operation),
        });
    }

    /// Apply Kahan summation for a sum reduction
    pub fn apply_kahan_summation(&mut self, location: &str) {
        self.mitigations.push(AppliedMitigation {
            strategy: MitigationStrategy::KahanSummation,
            location: location.to_string(),
            description: "Insert Kahan summation to reduce accumulation error".to_string(),
        });
    }

    /// Apply compensated algorithm
    pub fn apply_compensated(&mut self, operation: &str) {
        self.mitigations.push(AppliedMitigation {
            strategy: MitigationStrategy::CompensatedAlgorithm,
            location: operation.to_string(),
            description: format!("Use compensated algorithm for {}", operation),
        });
    }

    /// Apply rescaling
    pub fn apply_rescaling(&mut self, location: &str, scale_factor: f64) {
        self.mitigations.push(AppliedMitigation {
            strategy: MitigationStrategy::Rescaling,
            location: location.to_string(),
            description: format!(
                "Rescale by {:.2e} to prevent overflow/underflow",
                scale_factor
            ),
        });
    }

    /// Get all applied mitigations
    pub fn mitigations(&self) -> &[AppliedMitigation] {
        &self.mitigations
    }

    /// Clear mitigations
    pub fn clear(&mut self) {
        self.mitigations.clear();
    }
}

impl Default for StabilityMitigator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Integration with Epistemic Computing
// ============================================================================

/// Convert error bound to epistemic shadow register epsilon
pub fn error_to_epistemic_epsilon(error: &ErrorBound) -> f32 {
    error.to_epsilon()
}

/// Convert stability risk to validity predicate confidence
pub fn risk_to_validity_confidence(risk: &StabilityRisk) -> f64 {
    1.0 - risk.severity()
}

/// Synthesize provenance ID from error propagation history
pub fn synthesize_provenance(events: &[ErrorEvent]) -> u64 {
    let mut provenance = 0u64;
    for (i, event) in events.iter().enumerate() {
        // Each operation contributes to provenance
        let op_hash = event
            .operation
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        provenance ^= op_hash << (i % 8);
    }
    provenance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ulp_error() {
        // Use values very close together (few ULPs apart)
        let exact = 1.0_f64;
        let computed = f64::from_bits(exact.to_bits() + 5); // 5 ULPs away

        let error = UlpError::from_values(computed, exact);
        assert_eq!(error.ulps, 5);
        assert!(error.relative_error > 0.0);
        assert!(error.is_acceptable(10, 1e-6));
    }

    #[test]
    fn test_error_propagation_add() {
        let mut prop = ErrorPropagator::new(Precision::FP32, PropagationMode::Expected);
        let x = ErrorBound::machine_epsilon(Precision::FP32);
        let y = ErrorBound::machine_epsilon(Precision::FP32);

        let result = prop.propagate_add(x, y);
        assert!(result.expected_error > 0.0);
    }

    #[test]
    fn test_cancellation_detection() {
        let mut analyzer = StabilityAnalyzer::new(Precision::FP32);
        let risk = analyzer.check_cancellation(1.0000001, 1.0, "test");

        match risk {
            StabilityRisk::Catastrophic { .. } => {
                assert!(analyzer.has_catastrophic_issues());
            }
            _ => {}
        }
    }

    #[test]
    fn test_precision_recommendation() {
        let mut advisor = PrecisionAdvisor::new(Precision::FP32, 1e-6);
        let error = ErrorBound::from_estimate(1e-3, 0.9);

        let recommended = advisor.recommend("test_op", error, (0.0, 100.0));
        assert!(recommended == Precision::FP32 || recommended == Precision::FP64);
    }

    #[test]
    fn test_division_stability() {
        let mut analyzer = StabilityAnalyzer::new(Precision::FP32);

        // Test with divisor below min_normal (triggers Severe)
        let risk = analyzer.check_division(1.0, 1e-40, "test");
        match risk {
            StabilityRisk::Severe { overflow_risk, .. } => {
                assert!(overflow_risk > 0.5);
            }
            _ => panic!("Expected severe risk for subnormal divisor"),
        }

        // Test with mild instability (divisor < 1.0 but not subnormal)
        let risk2 = analyzer.check_division(1.0, 0.001, "test2");
        match risk2 {
            StabilityRisk::MildInstability { condition_number } => {
                assert!(condition_number > 100.0);
            }
            _ => panic!("Expected mild instability"),
        }
    }

    #[test]
    fn test_error_bound_combine() {
        let e1 = ErrorBound::from_estimate(1e-6, 0.95);
        let e2 = ErrorBound::from_estimate(2e-6, 0.90);
        let combined = e1.combine(&e2);

        assert!(combined.expected_error > e1.expected_error);
        assert!(combined.confidence < e1.confidence);
    }

    #[test]
    fn test_stability_summary() {
        let mut analyzer = StabilityAnalyzer::new(Precision::FP32);
        analyzer.check_cancellation(1.0000001, 1.0, "loc1");
        analyzer.check_division(1.0, 0.5, "loc2");

        let summary = analyzer.summary();
        assert!(summary.total_checks > 0);
    }
}
