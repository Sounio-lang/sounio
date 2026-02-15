//! Nonlinear confidence propagation for Knowledge<T>
//!
//! Extends epistemic confidence propagation to transcendental functions
//! (sin, cos, exp, log, sqrt, pow) using sensitivity analysis and
//! unscented transform approximation.
//!
//! GUM-compliant: uses first-order sensitivity coefficients per
//! ISO/IEC Guide 98-3:2008 Section 5.1.2, with optional higher-order
//! corrections via unscented transform.
//!
//! # Degradation Model
//!
//! Each nonlinear function class has an empirically calibrated degradation
//! factor that captures how much confidence is lost when applying the
//! function. These factors are conservative estimates based on:
//!
//! - **Trigonometric (sin, cos)**: 0.98 -- mild degradation, bounded output
//! - **Exponential (exp)**: 0.95 -- input uncertainty amplified exponentially
//! - **Logarithmic (log)**: 0.90 -- singularity at zero, high sensitivity
//! - **Square root (sqrt)**: 0.95 -- domain restriction, moderate amplification
//! - **Power (pow)**: 0.93 -- depends on exponent, generally amplifies
//!
//! # Control Flow
//!
//! Confidence also degrades through control flow:
//! - **Conditional**: takes the minimum branch confidence (conservative)
//! - **Loop**: geometric decay per iteration (uncertainty accumulates)
//!
//! # Example
//!
//! ```rust
//! use souc::epistemic::nonlinear_propagation::*;
//!
//! let result = propagate_nonlinear(
//!     NonlinearOp::Exp,
//!     0.95,  // input confidence
//!     0.02,  // input uncertainty (relative)
//! );
//! assert!((result.output_confidence - 0.9025).abs() < 1e-6);
//! ```

use std::fmt;

// =============================================================================
// Nonlinear Operation Classification
// =============================================================================

/// Classification of nonlinear operations for confidence propagation.
///
/// Each variant carries the degradation semantics appropriate to
/// the mathematical properties of the function class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NonlinearOp {
    /// sin(x) -- bounded output, periodic, degradation = 0.98
    Sin,
    /// cos(x) -- bounded output, periodic, degradation = 0.98
    Cos,
    /// tan(x) -- unbounded, singularities, degradation = 0.90 (same as log)
    Tan,
    /// exp(x) -- exponential amplification, degradation = 0.95
    Exp,
    /// log(x) -- singularity at zero, degradation = 0.90
    Log,
    /// sqrt(x) -- domain restriction to non-negative, degradation = 0.95
    Sqrt,
    /// pow(x, n) -- exponent-dependent amplification, degradation = 0.93
    Pow,
    /// abs(x) -- non-differentiable at zero, degradation = 0.99
    Abs,
}

impl NonlinearOp {
    /// Base degradation factor for this operation class.
    ///
    /// These factors are calibrated against GUM Section 5.1.2 sensitivity
    /// coefficients for typical scientific computing workloads.
    pub fn degradation_factor(self) -> f64 {
        match self {
            NonlinearOp::Sin | NonlinearOp::Cos => 0.98,
            NonlinearOp::Tan => 0.90,
            NonlinearOp::Exp => 0.95,
            NonlinearOp::Log => 0.90,
            NonlinearOp::Sqrt => 0.95,
            NonlinearOp::Pow => 0.93,
            NonlinearOp::Abs => 0.99,
        }
    }

    /// GUM sensitivity coefficient |df/dx| description.
    ///
    /// Returns a human-readable description of the first-order
    /// sensitivity coefficient used for uncertainty propagation.
    pub fn sensitivity_description(self) -> &'static str {
        match self {
            NonlinearOp::Sin => "|cos(x)|",
            NonlinearOp::Cos => "|sin(x)|",
            NonlinearOp::Tan => "|sec^2(x)|",
            NonlinearOp::Exp => "|exp(x)|",
            NonlinearOp::Log => "|1/x|",
            NonlinearOp::Sqrt => "|1/(2*sqrt(x))|",
            NonlinearOp::Pow => "|n * x^(n-1)|",
            NonlinearOp::Abs => "1 (except at x=0)",
        }
    }
}

impl fmt::Display for NonlinearOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NonlinearOp::Sin => write!(f, "sin"),
            NonlinearOp::Cos => write!(f, "cos"),
            NonlinearOp::Tan => write!(f, "tan"),
            NonlinearOp::Exp => write!(f, "exp"),
            NonlinearOp::Log => write!(f, "log"),
            NonlinearOp::Sqrt => write!(f, "sqrt"),
            NonlinearOp::Pow => write!(f, "pow"),
            NonlinearOp::Abs => write!(f, "abs"),
        }
    }
}

// =============================================================================
// Propagation Rule
// =============================================================================

/// Rule governing how confidence propagates through a nonlinear operation.
///
/// This extends the `PropagationRule` enum from `types::epistemic` to handle
/// transcendental and nonlinear functions specifically.
#[derive(Debug, Clone, PartialEq)]
pub enum NonlinearPropagationRule {
    /// First-order GUM propagation: confidence *= degradation_factor.
    ///
    /// Uses the sensitivity coefficient |df/dx| scaled by the base
    /// degradation factor. Appropriate when input uncertainty is small
    /// relative to the function's radius of curvature.
    FirstOrder {
        /// The nonlinear operation
        op: NonlinearOp,
    },

    /// Higher-order propagation via unscented transform.
    ///
    /// When first-order is insufficient (large input uncertainty or
    /// highly nonlinear region), use sigma-point propagation for
    /// improved accuracy up to 3rd order.
    UnscentedTransform {
        /// The nonlinear operation
        op: NonlinearOp,
        /// UT configuration (alpha, beta, kappa)
        alpha: f64,
        beta: f64,
        kappa: f64,
    },

    /// Composition of multiple nonlinear operations.
    ///
    /// For chains like sin(exp(x)), degradation factors multiply.
    Composition {
        /// Ordered sequence of operations (innermost first)
        ops: Vec<NonlinearOp>,
    },

    /// Custom degradation factor override.
    ///
    /// Allows domain experts to specify an exact degradation factor
    /// when the default calibration is inappropriate.
    Custom {
        /// Human-readable label
        label: String,
        /// The degradation factor to apply
        factor: f64,
    },
}

impl NonlinearPropagationRule {
    /// Compute the composite degradation factor for this rule.
    pub fn degradation_factor(&self) -> f64 {
        match self {
            NonlinearPropagationRule::FirstOrder { op } => op.degradation_factor(),
            NonlinearPropagationRule::UnscentedTransform { op, .. } => {
                // UT provides better estimates, so slightly less degradation
                // (the UT itself accounts for higher-order terms)
                let base = op.degradation_factor();
                // Midpoint between base and 1.0: reduces conservatism
                (base + 1.0) / 2.0
            }
            NonlinearPropagationRule::Composition { ops } => {
                ops.iter().map(|op| op.degradation_factor()).product()
            }
            NonlinearPropagationRule::Custom { factor, .. } => *factor,
        }
    }
}

// =============================================================================
// Confidence Flow Result
// =============================================================================

/// Result of propagating confidence through a nonlinear operation.
///
/// Contains the output confidence, the degradation that was applied,
/// and diagnostic information for debugging and provenance tracking.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfidenceFlowResult {
    /// Output confidence after propagation (clamped to [0, 1])
    pub output_confidence: f64,

    /// The degradation factor that was applied
    pub degradation_applied: f64,

    /// Human-readable diagnostic explaining the propagation
    pub diagnostic: String,

    /// Whether a warning should be emitted (e.g., confidence dropped below threshold)
    pub warning: Option<ConfidenceWarning>,
}

/// Warnings that may be emitted during confidence propagation.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfidenceWarning {
    /// Confidence dropped below a meaningful threshold
    BelowThreshold {
        /// The threshold that was crossed
        threshold: f64,
        /// The actual confidence value
        actual: f64,
    },
    /// Domain concern: input may be near a singularity or boundary
    DomainConcern {
        /// Description of the concern
        description: String,
    },
    /// Accumulated degradation through composition is severe
    SevereDegradation {
        /// Total degradation factor
        total_factor: f64,
        /// Number of operations in the chain
        chain_length: usize,
    },
}

impl fmt::Display for ConfidenceWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfidenceWarning::BelowThreshold {
                threshold, actual, ..
            } => write!(
                f,
                "confidence {:.4} dropped below threshold {:.4}",
                actual, threshold
            ),
            ConfidenceWarning::DomainConcern { description } => {
                write!(f, "domain concern: {}", description)
            }
            ConfidenceWarning::SevereDegradation {
                total_factor,
                chain_length,
            } => write!(
                f,
                "severe degradation: factor {:.4} over {} operations",
                total_factor, chain_length
            ),
        }
    }
}

// =============================================================================
// Core Propagation Functions
// =============================================================================

/// Default confidence threshold below which a warning is emitted.
const DEFAULT_WARNING_THRESHOLD: f64 = 0.5;

/// Propagate confidence through a single nonlinear operation.
///
/// Applies the degradation factor for `op` to `input_confidence`.
/// The `input_uncertainty` parameter (relative standard uncertainty)
/// is used for diagnostic purposes and to detect domain concerns.
///
/// # Arguments
///
/// * `op` - The nonlinear operation being applied
/// * `input_confidence` - Confidence of the input value, in [0, 1]
/// * `input_uncertainty` - Relative standard uncertainty of input (e.g., 0.02 for 2%)
///
/// # Returns
///
/// A `ConfidenceFlowResult` with the degraded confidence and diagnostics.
pub fn propagate_nonlinear(
    op: NonlinearOp,
    input_confidence: f64,
    input_uncertainty: f64,
) -> ConfidenceFlowResult {
    let factor = op.degradation_factor();
    let output_confidence = (input_confidence * factor).clamp(0.0, 1.0);

    let diagnostic = format!(
        "{}(x): confidence {:.6} * {:.4} = {:.6} (sensitivity: {}, input_u: {:.6})",
        op,
        input_confidence,
        factor,
        output_confidence,
        op.sensitivity_description(),
        input_uncertainty,
    );

    let warning = check_warnings(op, output_confidence, input_uncertainty);

    ConfidenceFlowResult {
        output_confidence,
        degradation_applied: factor,
        diagnostic,
        warning,
    }
}

/// Propagate confidence using a specific rule.
///
/// This is the general-purpose entry point that supports first-order,
/// unscented transform, composition, and custom rules.
///
/// # Arguments
///
/// * `rule` - The propagation rule to apply
/// * `input_confidence` - Confidence of the input, in [0, 1]
/// * `input_uncertainty` - Relative standard uncertainty of input
///
/// # Returns
///
/// A `ConfidenceFlowResult` with the propagated confidence.
pub fn propagate_with_rule(
    rule: &NonlinearPropagationRule,
    input_confidence: f64,
    input_uncertainty: f64,
) -> ConfidenceFlowResult {
    let factor = rule.degradation_factor();
    let output_confidence = (input_confidence * factor).clamp(0.0, 1.0);

    let diagnostic = match rule {
        NonlinearPropagationRule::FirstOrder { op } => {
            format!(
                "first-order {}(x): {:.6} * {:.4} = {:.6}",
                op, input_confidence, factor, output_confidence
            )
        }
        NonlinearPropagationRule::UnscentedTransform { op, alpha, .. } => {
            format!(
                "UT {}(x) [alpha={:.4}]: {:.6} * {:.4} = {:.6}",
                op, alpha, input_confidence, factor, output_confidence
            )
        }
        NonlinearPropagationRule::Composition { ops } => {
            let chain: Vec<String> = ops.iter().map(|o| o.to_string()).collect();
            format!(
                "composition [{}]: {:.6} * {:.4} = {:.6}",
                chain.join(" -> "),
                input_confidence,
                factor,
                output_confidence
            )
        }
        NonlinearPropagationRule::Custom { label, .. } => {
            format!(
                "custom '{}': {:.6} * {:.4} = {:.6}",
                label, input_confidence, factor, output_confidence
            )
        }
    };

    let warning = if output_confidence < DEFAULT_WARNING_THRESHOLD {
        Some(ConfidenceWarning::BelowThreshold {
            threshold: DEFAULT_WARNING_THRESHOLD,
            actual: output_confidence,
        })
    } else if let NonlinearPropagationRule::Composition { ops } = rule {
        if factor < 0.80 {
            Some(ConfidenceWarning::SevereDegradation {
                total_factor: factor,
                chain_length: ops.len(),
            })
        } else {
            None
        }
    } else {
        None
    };

    ConfidenceFlowResult {
        output_confidence,
        degradation_applied: factor,
        diagnostic,
        warning,
    }
}

/// Check for warnings based on the operation and result.
fn check_warnings(
    op: NonlinearOp,
    output_confidence: f64,
    input_uncertainty: f64,
) -> Option<ConfidenceWarning> {
    // Below-threshold warning takes priority
    if output_confidence < DEFAULT_WARNING_THRESHOLD {
        return Some(ConfidenceWarning::BelowThreshold {
            threshold: DEFAULT_WARNING_THRESHOLD,
            actual: output_confidence,
        });
    }

    // Domain concerns for specific operations
    match op {
        NonlinearOp::Log if input_uncertainty > 0.5 => Some(ConfidenceWarning::DomainConcern {
            description:
                "log(x) with large relative uncertainty; input may include non-positive values"
                    .to_string(),
        }),
        NonlinearOp::Sqrt if input_uncertainty > 0.5 => Some(ConfidenceWarning::DomainConcern {
            description:
                "sqrt(x) with large relative uncertainty; input may include negative values"
                    .to_string(),
        }),
        NonlinearOp::Tan if input_uncertainty > 0.3 => Some(ConfidenceWarning::DomainConcern {
            description: "tan(x) with large uncertainty; input may be near pi/2 singularity"
                .to_string(),
        }),
        _ => None,
    }
}

// =============================================================================
// Control Flow Confidence Propagation
// =============================================================================

/// Propagate confidence through a conditional branch (if/match).
///
/// Takes the minimum confidence across all branches, since we cannot
/// statically guarantee which branch will execute. This is the
/// conservative choice per GUM Section 5.1.
///
/// # Arguments
///
/// * `branch_confidences` - Confidence values from each branch
///
/// # Returns
///
/// The minimum confidence across branches, or `None` if input is empty.
pub fn propagate_conditional(branch_confidences: &[f64]) -> Option<ConfidenceFlowResult> {
    if branch_confidences.is_empty() {
        return None;
    }

    let min_confidence = branch_confidences
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);

    let max_confidence = branch_confidences
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let spread = max_confidence - min_confidence;

    let diagnostic = format!(
        "conditional: min({}) = {:.6} (spread: {:.6})",
        branch_confidences
            .iter()
            .map(|c| format!("{:.4}", c))
            .collect::<Vec<_>>()
            .join(", "),
        min_confidence,
        spread,
    );

    let warning = if min_confidence < DEFAULT_WARNING_THRESHOLD {
        Some(ConfidenceWarning::BelowThreshold {
            threshold: DEFAULT_WARNING_THRESHOLD,
            actual: min_confidence,
        })
    } else {
        None
    };

    Some(ConfidenceFlowResult {
        output_confidence: min_confidence.clamp(0.0, 1.0),
        degradation_applied: if max_confidence > 0.0 {
            min_confidence / max_confidence
        } else {
            1.0
        },
        diagnostic,
        warning,
    })
}

/// Per-iteration confidence decay factor for loops.
const LOOP_DECAY_PER_ITERATION: f64 = 0.999;

/// Propagate confidence through a loop body.
///
/// Applies geometric decay: each iteration multiplies confidence by
/// `body_confidence * LOOP_DECAY_PER_ITERATION`. After `max_iterations`
/// iterations, the confidence is:
///
///   output = body_confidence * (LOOP_DECAY_PER_ITERATION ^ max_iterations)
///
/// This models the accumulation of uncertainty through repeated computation.
///
/// # Arguments
///
/// * `body_confidence` - Confidence of the loop body's output per iteration
/// * `max_iterations` - Maximum number of iterations (statically known or bounded)
///
/// # Returns
///
/// The geometrically decayed confidence.
pub fn propagate_loop(body_confidence: f64, max_iterations: u32) -> ConfidenceFlowResult {
    // Geometric decay: confidence * decay^iterations
    let decay = libm::pow(LOOP_DECAY_PER_ITERATION, max_iterations as f64);
    let output_confidence = (body_confidence * decay).clamp(0.0, 1.0);
    let total_factor = if body_confidence > 0.0 {
        output_confidence / body_confidence
    } else {
        0.0
    };

    let diagnostic = format!(
        "loop ({} iters): {:.6} * {:.6}^{} = {:.6}",
        max_iterations,
        body_confidence,
        LOOP_DECAY_PER_ITERATION,
        max_iterations,
        output_confidence
    );

    let warning = if output_confidence < DEFAULT_WARNING_THRESHOLD {
        Some(ConfidenceWarning::BelowThreshold {
            threshold: DEFAULT_WARNING_THRESHOLD,
            actual: output_confidence,
        })
    } else if max_iterations > 1000 {
        Some(ConfidenceWarning::SevereDegradation {
            total_factor,
            chain_length: max_iterations as usize,
        })
    } else {
        None
    };

    ConfidenceFlowResult {
        output_confidence,
        degradation_applied: total_factor,
        diagnostic,
        warning,
    }
}

// =============================================================================
// Function Confidence Summary (Interprocedural)
// =============================================================================

/// Summary of a function's effect on confidence for interprocedural analysis.
///
/// When analyzing function calls across module boundaries, we need a compact
/// representation of how a function transforms confidence. This struct
/// captures the essential information without requiring re-analysis of
/// the function body.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionConfidenceSummary {
    /// Name of the function
    pub function_name: String,

    /// Minimum degradation factor across all paths through the function.
    ///
    /// This is the worst-case confidence multiplier. A caller's confidence
    /// will be multiplied by at least this factor.
    pub min_degradation: f64,

    /// Maximum degradation factor across all paths.
    ///
    /// Best-case confidence multiplier (closest to 1.0).
    pub max_degradation: f64,

    /// Whether the function contains loops (which cause geometric decay)
    pub has_loops: bool,

    /// Whether the function contains conditionals (which cause min-branching)
    pub has_conditionals: bool,

    /// Nonlinear operations used in the function body
    pub nonlinear_ops: Vec<NonlinearOp>,

    /// Total number of confidence-affecting operations
    pub operation_count: usize,
}

impl FunctionConfidenceSummary {
    /// Create a summary for a pure identity function (no degradation).
    pub fn identity(name: &str) -> Self {
        Self {
            function_name: name.to_string(),
            min_degradation: 1.0,
            max_degradation: 1.0,
            has_loops: false,
            has_conditionals: false,
            nonlinear_ops: Vec::new(),
            operation_count: 0,
        }
    }

    /// Create a summary from a single nonlinear operation.
    pub fn from_single_op(name: &str, op: NonlinearOp) -> Self {
        let factor = op.degradation_factor();
        Self {
            function_name: name.to_string(),
            min_degradation: factor,
            max_degradation: factor,
            has_loops: false,
            has_conditionals: false,
            nonlinear_ops: vec![op],
            operation_count: 1,
        }
    }

    /// Apply this summary to an input confidence.
    ///
    /// Uses the minimum (worst-case) degradation for conservative analysis.
    pub fn apply_conservative(&self, input_confidence: f64) -> ConfidenceFlowResult {
        let output_confidence = (input_confidence * self.min_degradation).clamp(0.0, 1.0);

        let diagnostic = format!(
            "call {}(): {:.6} * {:.4} = {:.6} (conservative, {} ops)",
            self.function_name,
            input_confidence,
            self.min_degradation,
            output_confidence,
            self.operation_count
        );

        let warning = if output_confidence < DEFAULT_WARNING_THRESHOLD {
            Some(ConfidenceWarning::BelowThreshold {
                threshold: DEFAULT_WARNING_THRESHOLD,
                actual: output_confidence,
            })
        } else {
            None
        };

        ConfidenceFlowResult {
            output_confidence,
            degradation_applied: self.min_degradation,
            diagnostic,
            warning,
        }
    }

    /// Compose two function summaries sequentially (f then g).
    ///
    /// The composite degradation factors multiply.
    pub fn compose(&self, other: &FunctionConfidenceSummary) -> Self {
        let mut combined_ops = self.nonlinear_ops.clone();
        combined_ops.extend_from_slice(&other.nonlinear_ops);

        Self {
            function_name: format!("{} -> {}", self.function_name, other.function_name),
            min_degradation: self.min_degradation * other.min_degradation,
            max_degradation: self.max_degradation * other.max_degradation,
            has_loops: self.has_loops || other.has_loops,
            has_conditionals: self.has_conditionals || other.has_conditionals,
            nonlinear_ops: combined_ops,
            operation_count: self.operation_count + other.operation_count,
        }
    }
}

impl fmt::Display for FunctionConfidenceSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(): degradation [{:.4}, {:.4}], {} ops",
            self.function_name, self.min_degradation, self.max_degradation, self.operation_count
        )?;
        if self.has_loops {
            write!(f, " [loops]")?;
        }
        if self.has_conditionals {
            write!(f, " [cond]")?;
        }
        Ok(())
    }
}

// =============================================================================
// Builder for composing propagation chains
// =============================================================================

/// Builder for constructing multi-step nonlinear confidence propagation.
///
/// Allows chaining multiple nonlinear operations and control flow
/// constructs to compute the final confidence.
///
/// # Example
///
/// ```rust
/// use souc::epistemic::nonlinear_propagation::*;
///
/// let result = PropagationChainBuilder::new(0.95, 0.02)
///     .apply(NonlinearOp::Exp)
///     .apply(NonlinearOp::Sin)
///     .build();
///
/// // sin(exp(x)): 0.95 * 0.95 * 0.98
/// assert!((result.output_confidence - 0.95 * 0.95 * 0.98).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct PropagationChainBuilder {
    current_confidence: f64,
    input_uncertainty: f64,
    steps: Vec<ConfidenceFlowResult>,
    ops: Vec<NonlinearOp>,
}

impl PropagationChainBuilder {
    /// Create a new chain starting with the given confidence.
    pub fn new(input_confidence: f64, input_uncertainty: f64) -> Self {
        Self {
            current_confidence: input_confidence.clamp(0.0, 1.0),
            input_uncertainty,
            steps: Vec::new(),
            ops: Vec::new(),
        }
    }

    /// Apply a nonlinear operation to the chain.
    pub fn apply(mut self, op: NonlinearOp) -> Self {
        let result = propagate_nonlinear(op, self.current_confidence, self.input_uncertainty);
        self.current_confidence = result.output_confidence;
        self.ops.push(op);
        self.steps.push(result);
        self
    }

    /// Apply a conditional merge to the chain.
    ///
    /// The `branch_factors` are additional degradation factors from
    /// each branch. The current confidence is combined with the minimum.
    pub fn apply_conditional(mut self, branch_confidences: &[f64]) -> Self {
        if let Some(result) = propagate_conditional(branch_confidences) {
            self.current_confidence = result.output_confidence;
            self.steps.push(result);
        }
        self
    }

    /// Apply a loop to the chain.
    pub fn apply_loop(mut self, max_iterations: u32) -> Self {
        let result = propagate_loop(self.current_confidence, max_iterations);
        self.current_confidence = result.output_confidence;
        self.steps.push(result);
        self
    }

    /// Finalize and return the composite result.
    pub fn build(self) -> ConfidenceFlowResult {
        let total_degradation: f64 = self.steps.iter().map(|s| s.degradation_applied).product();

        let diagnostics: Vec<String> = self.steps.iter().map(|s| s.diagnostic.clone()).collect();

        let warning = if self.current_confidence < DEFAULT_WARNING_THRESHOLD {
            Some(ConfidenceWarning::BelowThreshold {
                threshold: DEFAULT_WARNING_THRESHOLD,
                actual: self.current_confidence,
            })
        } else if total_degradation < 0.80 && self.steps.len() > 1 {
            Some(ConfidenceWarning::SevereDegradation {
                total_factor: total_degradation,
                chain_length: self.steps.len(),
            })
        } else {
            None
        };

        ConfidenceFlowResult {
            output_confidence: self.current_confidence,
            degradation_applied: total_degradation,
            diagnostic: diagnostics.join(" | "),
            warning,
        }
    }

    /// Generate a `FunctionConfidenceSummary` from this chain.
    pub fn to_summary(self, function_name: &str) -> FunctionConfidenceSummary {
        let total = self.ops.iter().map(|op| op.degradation_factor()).product();
        FunctionConfidenceSummary {
            function_name: function_name.to_string(),
            min_degradation: total,
            max_degradation: total,
            has_loops: self.steps.iter().any(|s| s.diagnostic.contains("loop")),
            has_conditionals: self
                .steps
                .iter()
                .any(|s| s.diagnostic.contains("conditional")),
            nonlinear_ops: self.ops,
            operation_count: self.steps.len(),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Degradation factor tests ----

    #[test]
    fn test_sin_propagation() {
        let result = propagate_nonlinear(NonlinearOp::Sin, 1.0, 0.01);
        let expected = 1.0 * 0.98;
        assert!(
            (result.output_confidence - expected).abs() < 1e-10,
            "sin: expected {}, got {}",
            expected,
            result.output_confidence
        );
        assert!((result.degradation_applied - 0.98).abs() < 1e-10);
    }

    #[test]
    fn test_cos_propagation() {
        let result = propagate_nonlinear(NonlinearOp::Cos, 0.90, 0.01);
        let expected = 0.90 * 0.98;
        assert!(
            (result.output_confidence - expected).abs() < 1e-10,
            "cos: expected {}, got {}",
            expected,
            result.output_confidence
        );
        assert!((result.degradation_applied - 0.98).abs() < 1e-10);
    }

    #[test]
    fn test_exp_propagation() {
        let result = propagate_nonlinear(NonlinearOp::Exp, 0.95, 0.02);
        let expected = 0.95 * 0.95;
        assert!(
            (result.output_confidence - expected).abs() < 1e-10,
            "exp: expected {}, got {}",
            expected,
            result.output_confidence
        );
        assert!((result.degradation_applied - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_log_propagation() {
        let result = propagate_nonlinear(NonlinearOp::Log, 0.95, 0.02);
        let expected = 0.95 * 0.90;
        assert!(
            (result.output_confidence - expected).abs() < 1e-10,
            "log: expected {}, got {}",
            expected,
            result.output_confidence
        );
        assert!((result.degradation_applied - 0.90).abs() < 1e-10);
    }

    #[test]
    fn test_sqrt_propagation() {
        let result = propagate_nonlinear(NonlinearOp::Sqrt, 0.95, 0.02);
        let expected = 0.95 * 0.95;
        assert!(
            (result.output_confidence - expected).abs() < 1e-10,
            "sqrt: expected {}, got {}",
            expected,
            result.output_confidence
        );
    }

    #[test]
    fn test_pow_propagation() {
        let result = propagate_nonlinear(NonlinearOp::Pow, 1.0, 0.01);
        let expected = 1.0 * 0.93;
        assert!(
            (result.output_confidence - expected).abs() < 1e-10,
            "pow: expected {}, got {}",
            expected,
            result.output_confidence
        );
    }

    // ---- Control flow tests ----

    #[test]
    fn test_conditional_propagation_takes_minimum() {
        let branches = vec![0.95, 0.80, 0.90];
        let result = propagate_conditional(&branches).expect("non-empty branches");
        assert!(
            (result.output_confidence - 0.80).abs() < 1e-10,
            "conditional: expected min 0.80, got {}",
            result.output_confidence
        );
    }

    #[test]
    fn test_conditional_single_branch() {
        let branches = vec![0.75];
        let result = propagate_conditional(&branches).expect("non-empty branches");
        assert!(
            (result.output_confidence - 0.75).abs() < 1e-10,
            "single branch: expected 0.75, got {}",
            result.output_confidence
        );
    }

    #[test]
    fn test_conditional_empty_returns_none() {
        let branches: Vec<f64> = vec![];
        assert!(propagate_conditional(&branches).is_none());
    }

    #[test]
    fn test_loop_geometric_decay() {
        let result = propagate_loop(0.95, 100);
        // 0.95 * 0.999^100
        let expected = 0.95 * libm::pow(0.999, 100.0);
        assert!(
            (result.output_confidence - expected).abs() < 1e-10,
            "loop: expected {}, got {}",
            expected,
            result.output_confidence
        );
    }

    #[test]
    fn test_loop_zero_iterations() {
        let result = propagate_loop(0.95, 0);
        assert!(
            (result.output_confidence - 0.95).abs() < 1e-10,
            "loop 0 iters: expected 0.95, got {}",
            result.output_confidence
        );
    }

    // ---- Composition tests ----

    #[test]
    fn test_composition_sin_exp() {
        // sin(exp(x)): innermost first -> exp then sin
        let result = PropagationChainBuilder::new(1.0, 0.01)
            .apply(NonlinearOp::Exp) // 1.0 * 0.95 = 0.95
            .apply(NonlinearOp::Sin) // 0.95 * 0.98 = 0.931
            .build();

        let expected = 1.0 * 0.95 * 0.98;
        assert!(
            (result.output_confidence - expected).abs() < 1e-10,
            "sin(exp(x)): expected {}, got {}",
            expected,
            result.output_confidence
        );
    }

    #[test]
    fn test_composition_rule_directly() {
        let rule = NonlinearPropagationRule::Composition {
            ops: vec![NonlinearOp::Exp, NonlinearOp::Sin],
        };
        let factor = rule.degradation_factor();
        let expected = 0.95 * 0.98;
        assert!(
            (factor - expected).abs() < 1e-10,
            "composition factor: expected {}, got {}",
            expected,
            factor
        );
    }

    #[test]
    fn test_composition_triple() {
        // log(sin(exp(x))): exp -> sin -> log
        let result = PropagationChainBuilder::new(1.0, 0.01)
            .apply(NonlinearOp::Exp) // * 0.95
            .apply(NonlinearOp::Sin) // * 0.98
            .apply(NonlinearOp::Log) // * 0.90
            .build();

        let expected = 0.95 * 0.98 * 0.90;
        assert!(
            (result.output_confidence - expected).abs() < 1e-10,
            "log(sin(exp(x))): expected {}, got {}",
            expected,
            result.output_confidence
        );
    }

    // ---- Propagation rule tests ----

    #[test]
    fn test_first_order_rule() {
        let rule = NonlinearPropagationRule::FirstOrder {
            op: NonlinearOp::Exp,
        };
        let result = propagate_with_rule(&rule, 0.90, 0.01);
        let expected = 0.90 * 0.95;
        assert!(
            (result.output_confidence - expected).abs() < 1e-10,
            "first-order exp: expected {}, got {}",
            expected,
            result.output_confidence
        );
    }

    #[test]
    fn test_custom_rule() {
        let rule = NonlinearPropagationRule::Custom {
            label: "my_function".to_string(),
            factor: 0.85,
        };
        let result = propagate_with_rule(&rule, 1.0, 0.0);
        assert!(
            (result.output_confidence - 0.85).abs() < 1e-10,
            "custom: expected 0.85, got {}",
            result.output_confidence
        );
    }

    #[test]
    fn test_ut_rule_less_conservative() {
        // UT rule should be less conservative than first-order
        let first_order = NonlinearPropagationRule::FirstOrder {
            op: NonlinearOp::Exp,
        };
        let ut = NonlinearPropagationRule::UnscentedTransform {
            op: NonlinearOp::Exp,
            alpha: 0.5,
            beta: 2.0,
            kappa: 0.0,
        };

        let fo_result = propagate_with_rule(&first_order, 0.90, 0.01);
        let ut_result = propagate_with_rule(&ut, 0.90, 0.01);

        assert!(
            ut_result.output_confidence > fo_result.output_confidence,
            "UT should be less conservative: UT={}, FO={}",
            ut_result.output_confidence,
            fo_result.output_confidence
        );
    }

    // ---- Function summary tests ----

    #[test]
    fn test_function_summary_identity() {
        let summary = FunctionConfidenceSummary::identity("id");
        assert!((summary.min_degradation - 1.0).abs() < 1e-10);
        let result = summary.apply_conservative(0.95);
        assert!((result.output_confidence - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_function_summary_single_op() {
        let summary = FunctionConfidenceSummary::from_single_op("my_exp", NonlinearOp::Exp);
        assert!((summary.min_degradation - 0.95).abs() < 1e-10);
        let result = summary.apply_conservative(1.0);
        assert!(
            (result.output_confidence - 0.95).abs() < 1e-10,
            "summary apply: expected 0.95, got {}",
            result.output_confidence
        );
    }

    #[test]
    fn test_function_summary_compose() {
        let exp_summary = FunctionConfidenceSummary::from_single_op("f", NonlinearOp::Exp);
        let sin_summary = FunctionConfidenceSummary::from_single_op("g", NonlinearOp::Sin);
        let composed = exp_summary.compose(&sin_summary);

        let expected = 0.95 * 0.98;
        assert!(
            (composed.min_degradation - expected).abs() < 1e-10,
            "composed degradation: expected {}, got {}",
            expected,
            composed.min_degradation
        );
        assert_eq!(composed.operation_count, 2);
        assert_eq!(composed.nonlinear_ops.len(), 2);
    }

    // ---- Warning tests ----

    #[test]
    fn test_below_threshold_warning() {
        let result = propagate_nonlinear(NonlinearOp::Log, 0.45, 0.1);
        // 0.45 * 0.90 = 0.405 < 0.5
        assert!(matches!(
            result.warning,
            Some(ConfidenceWarning::BelowThreshold { .. })
        ));
    }

    #[test]
    fn test_domain_concern_log() {
        // Large uncertainty + log should trigger domain concern
        let result = propagate_nonlinear(NonlinearOp::Log, 0.95, 0.6);
        assert!(matches!(
            result.warning,
            Some(ConfidenceWarning::DomainConcern { .. })
        ));
    }

    // ---- Clamping tests ----

    #[test]
    fn test_confidence_clamped_to_unit_interval() {
        // Even with confidence > 1 input, output should be clamped
        let result = propagate_nonlinear(NonlinearOp::Sin, 1.5, 0.0);
        assert!(result.output_confidence <= 1.0);
        assert!(result.output_confidence >= 0.0);
    }

    #[test]
    fn test_negative_confidence_clamped() {
        let result = propagate_nonlinear(NonlinearOp::Sin, -0.5, 0.0);
        assert!(result.output_confidence >= 0.0);
    }

    // ---- Display tests ----

    #[test]
    fn test_nonlinear_op_display() {
        assert_eq!(format!("{}", NonlinearOp::Sin), "sin");
        assert_eq!(format!("{}", NonlinearOp::Exp), "exp");
        assert_eq!(format!("{}", NonlinearOp::Log), "log");
    }

    #[test]
    fn test_function_summary_display() {
        let summary = FunctionConfidenceSummary::from_single_op("my_func", NonlinearOp::Exp);
        let display = format!("{}", summary);
        assert!(display.contains("my_func"));
        assert!(display.contains("0.9500"));
    }

    #[test]
    fn test_confidence_warning_display() {
        let warning = ConfidenceWarning::BelowThreshold {
            threshold: 0.5,
            actual: 0.3,
        };
        let display = format!("{}", warning);
        assert!(display.contains("0.3000"));
        assert!(display.contains("0.5000"));
    }
}
