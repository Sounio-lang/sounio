//! Validation Framework for Phase 2 GPU Optimizations
//!
//! Comprehensive testing to ensure:
//! - Numerical correctness
//! - Feature isolation and orthogonality
//! - Integration compatibility
//! - Accuracy preservation

// Compiler internals often require deep nesting for pattern matching and control flow.
#![allow(clippy::excessive_nesting)]

use std::collections::HashMap;
use std::fmt;

// ============================================================================
// Validation Errors and Issues
// ============================================================================

/// Validation error types
#[derive(Clone, Debug)]
pub enum ValidationError {
    /// Output size mismatch
    SizeMismatch { expected: usize, actual: usize },
    /// Value mismatch at specific index
    ValueMismatch {
        index: usize,
        expected: f32,
        actual: f32,
    },
    /// Precision loss exceeded threshold
    PrecisionLoss { max_error: f32, threshold: f32 },
    /// Invalid output detected
    InvalidOutput { description: String },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::SizeMismatch { expected, actual } => {
                write!(f, "Size mismatch: expected {}, got {}", expected, actual)
            }
            ValidationError::ValueMismatch {
                index,
                expected,
                actual,
            } => write!(
                f,
                "Value mismatch at index {}: expected {}, got {}",
                index, expected, actual
            ),
            ValidationError::PrecisionLoss {
                max_error,
                threshold,
            } => write!(
                f,
                "Precision loss: max error {} exceeds threshold {}",
                max_error, threshold
            ),
            ValidationError::InvalidOutput { description } => {
                write!(f, "Invalid output: {}", description)
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Validation issues (warnings, not errors)
#[derive(Clone, Debug)]
pub enum ValidationIssue {
    /// Value mismatch detected
    ValueMismatch {
        index: usize,
        expected: f32,
        actual: f32,
        relative_error: f32,
    },
    /// NaN detected in output
    NaNDetected { index: usize, context: String },
    /// Infinity detected in output
    InfDetected { index: usize, context: String },
    /// Accumulated precision loss
    PrecisionLoss { max_error: f32, mean_error: f32 },
}

// ============================================================================
// Validation Configuration
// ============================================================================

/// Configuration for validation
#[derive(Clone, Debug)]
pub struct ValidationConfig {
    /// Relative tolerance for floating-point comparisons
    pub relative_tolerance: f32,
    /// Absolute tolerance for floating-point comparisons
    pub absolute_tolerance: f32,
    /// Check for NaN values
    pub check_nan: bool,
    /// Check for infinity values
    pub check_inf: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            relative_tolerance: 1e-5,
            absolute_tolerance: 1e-8,
            check_nan: true,
            check_inf: true,
        }
    }
}

/// Tolerance configuration for precision comparisons
#[derive(Clone, Debug)]
pub struct ToleranceConfig {
    /// Relative tolerance
    pub rtol: f32,
    /// Absolute tolerance
    pub atol: f32,
}

impl Default for ToleranceConfig {
    fn default() -> Self {
        Self {
            rtol: 1e-5,
            atol: 1e-8,
        }
    }
}

/// Precision statistics
#[derive(Clone, Debug, Default)]
pub struct PrecisionStats {
    /// Maximum absolute error
    pub max_error: f32,
    /// Mean absolute error
    pub mean_error: f32,
    /// Root mean square error
    pub rms_error: f32,
    /// Number of mismatches
    pub mismatch_count: usize,
}

/// Buffer comparison result
#[derive(Clone, Debug)]
pub struct BufferComparison {
    /// Whether buffers match within tolerance
    pub matches: bool,
    /// Precision statistics
    pub stats: PrecisionStats,
    /// Detected issues
    pub issues: Vec<ValidationIssue>,
}

/// Correctness validator
pub struct CorrectnessValidator {
    config: ValidationConfig,
}

impl CorrectnessValidator {
    /// Create a new correctness validator
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self {
            config: ValidationConfig::default(),
        }
    }

    /// Validate buffer correctness
    pub fn validate_buffers(
        &self,
        expected: &[f32],
        actual: &[f32],
    ) -> Result<BufferComparison, ValidationError> {
        if expected.len() != actual.len() {
            return Err(ValidationError::SizeMismatch {
                expected: expected.len(),
                actual: actual.len(),
            });
        }

        let mut stats = PrecisionStats::default();
        let mut issues = Vec::new();
        let mut sum_error = 0.0f32;
        let mut sum_sq_error = 0.0f32;

        for (i, (&exp, &act)) in expected.iter().zip(actual.iter()).enumerate() {
            // Check for NaN
            if self.config.check_nan && act.is_nan() {
                issues.push(ValidationIssue::NaNDetected {
                    index: i,
                    context: format!("value at index {}", i),
                });
                continue;
            }

            // Check for infinity
            if self.config.check_inf && act.is_infinite() {
                issues.push(ValidationIssue::InfDetected {
                    index: i,
                    context: format!("value at index {}", i),
                });
                continue;
            }

            let abs_error = (exp - act).abs();
            let rel_error = if exp.abs() > 1e-8 {
                abs_error / exp.abs()
            } else {
                abs_error
            };

            sum_error += abs_error;
            sum_sq_error += abs_error * abs_error;

            if abs_error > stats.max_error {
                stats.max_error = abs_error;
            }

            // Check tolerance
            if rel_error > self.config.relative_tolerance
                && abs_error > self.config.absolute_tolerance
            {
                stats.mismatch_count += 1;
                issues.push(ValidationIssue::ValueMismatch {
                    index: i,
                    expected: exp,
                    actual: act,
                    relative_error: rel_error,
                });
            }
        }

        let n = expected.len() as f32;
        stats.mean_error = sum_error / n;
        stats.rms_error = (sum_sq_error / n).sqrt();

        let matches = issues.is_empty();

        Ok(BufferComparison {
            matches,
            stats,
            issues,
        })
    }
}

// ============================================================================
// Existing validators follow below
// ============================================================================

/// Validation result for a single test
#[derive(Clone, Debug)]
pub struct ValidationResult {
    pub test_name: String,
    pub passed: bool,
    pub metrics: HashMap<String, f32>,
    pub diagnostics: Vec<String>,
}

impl ValidationResult {
    pub fn new(name: &str, passed: bool) -> Self {
        Self {
            test_name: name.to_string(),
            passed,
            metrics: HashMap::new(),
            diagnostics: vec![],
        }
    }

    pub fn add_metric(&mut self, key: &str, value: f32) {
        self.metrics.insert(key.to_string(), value);
    }

    pub fn add_diagnostic(&mut self, msg: &str) {
        self.diagnostics.push(msg.to_string());
    }

    pub fn display(&self) -> String {
        let status = if self.passed { "PASS" } else { "FAIL" };
        let mut output = format!("[{}] {}\n", status, self.test_name);

        for (key, val) in &self.metrics {
            output.push_str(&format!("  {}: {:.6}\n", key, val));
        }

        for diag in &self.diagnostics {
            output.push_str(&format!("  ! {}\n", diag));
        }

        output
    }
}

/// Numerical correctness validator
pub struct NumericalCorrectnessValidator;

impl NumericalCorrectnessValidator {
    pub fn validate(input: &[f32], output_fp32: &[f32], output_mixed: &[f32]) -> ValidationResult {
        let mut result = ValidationResult::new("numerical_correctness", true);

        if output_fp32.len() != output_mixed.len() {
            result.passed = false;
            result.add_diagnostic("Output size mismatch");
            return result;
        }

        let mut max_diff = 0.0f32;
        let mut rms_error = 0.0f32;

        for (fp32, mixed) in output_fp32.iter().zip(output_mixed.iter()) {
            let diff = (fp32 - mixed).abs();
            max_diff = max_diff.max(diff);
            rms_error += diff * diff;

            if !mixed.is_finite() {
                result.passed = false;
                result.add_diagnostic("Non-finite value in output");
            }
        }

        rms_error = (rms_error / output_fp32.len() as f32).sqrt();

        result.add_metric("max_diff", max_diff);
        result.add_metric("rms_error", rms_error);

        // Threshold: mixed-precision within 0.5% of FP32
        if rms_error
            > 0.005 * output_fp32.iter().map(|x| x.abs()).sum::<f32>() / output_fp32.len() as f32
        {
            result.passed = false;
            result.add_diagnostic(&format!("RMS error {:.6} exceeds threshold", rms_error));
        }

        result
    }
}

/// Feature isolation validator
pub struct FeatureIsolationValidator;

impl FeatureIsolationValidator {
    pub fn validate(baseline_loss: f32, individual_losses: &[(String, f32)]) -> ValidationResult {
        let mut result = ValidationResult::new("feature_isolation", true);

        for (feature_name, feature_loss) in individual_losses {
            let rel_impact = ((feature_loss - baseline_loss).abs() / baseline_loss)
                .abs()
                .max(0.0);
            result.add_metric(&format!("impact_{}", feature_name), rel_impact);

            // Each feature should have <1% individual impact
            if rel_impact > 0.01 {
                result.passed = false;
                result.add_diagnostic(&format!(
                    "{} has {}% impact, expected <1%",
                    feature_name,
                    (rel_impact * 100.0) as i32
                ));
            }
        }

        result
    }
}

/// Integration validator
pub struct IntegrationValidator;

impl IntegrationValidator {
    pub fn validate(
        baseline_loss: f32,
        combined_losses: &[(Vec<String>, f32)],
    ) -> ValidationResult {
        let mut result = ValidationResult::new("integration", true);

        for (features, loss) in combined_losses {
            let rel_diff = ((loss - baseline_loss).abs() / baseline_loss)
                .abs()
                .max(0.0);
            result.add_metric(&format!("combined_{}", features.join("+")), rel_diff);

            // Combined should stay within 2% deviation
            if rel_diff > 0.02 {
                result.passed = false;
                result.add_diagnostic(&format!(
                    "{} causes {:.1}% deviation",
                    features.join("+"),
                    rel_diff * 100.0
                ));
            }
        }

        result
    }
}

/// Accuracy preservation validator
pub struct AccuracyPreservationValidator;

impl AccuracyPreservationValidator {
    pub fn validate(
        baseline_accuracy: f32,
        feature_accuracies: &[(String, f32)],
    ) -> ValidationResult {
        let mut result = ValidationResult::new("accuracy_preservation", true);

        // Thresholds per feature
        let thresholds = [
            ("mixed_precision", 0.005), // <0.5% loss
            ("fusion", 0.001),          // <0.1% loss
            ("sparsity", 0.01),         // <1% loss
            ("qat", 0.01),              // <1% loss
        ];

        for (feature_name, accuracy) in feature_accuracies {
            let acc_loss = ((baseline_accuracy - accuracy) / baseline_accuracy)
                .abs()
                .max(0.0);
            result.add_metric(&format!("acc_loss_{}", feature_name), acc_loss);

            let threshold = thresholds
                .iter()
                .find(|(name, _)| feature_name.contains(name))
                .map(|(_, t)| t)
                .unwrap_or(&0.02);

            if acc_loss > *threshold {
                result.passed = false;
                result.add_diagnostic(&format!(
                    "{} causes {:.2}% accuracy loss (threshold {:.2}%)",
                    feature_name,
                    acc_loss * 100.0,
                    threshold * 100.0
                ));
            }
        }

        result
    }
}

/// Sparsity pattern validator
pub struct SparsityValidator;

impl SparsityValidator {
    pub fn validate(weights: &[f32], pattern: &str) -> ValidationResult {
        let mut result = ValidationResult::new(&format!("sparsity_{}", pattern), true);

        match pattern {
            "2:4" => {
                // Verify 2:4 pattern: max 2 non-zeros per 4 elements
                for chunk in weights.chunks(4) {
                    let non_zeros = chunk.iter().filter(|&&w| w.abs() > 1e-6).count();
                    result.add_metric("chunk_non_zeros", non_zeros as f32);

                    if non_zeros > 2 {
                        result.passed = false;
                        result.add_diagnostic(&format!(
                            "2:4 pattern violated: {} non-zeros in chunk",
                            non_zeros
                        ));
                    }
                }

                // Overall sparsity should be ~50%
                let non_zeros = weights.iter().filter(|&&w| w.abs() > 1e-6).count();
                let sparsity = 1.0 - (non_zeros as f32 / weights.len() as f32);
                result.add_metric("overall_sparsity", sparsity);

                if (sparsity - 0.5).abs() > 0.15 {
                    result.passed = false;
                    result.add_diagnostic(&format!(
                        "Overall sparsity {:.1}% deviates from expected 50%",
                        sparsity * 100.0
                    ));
                }
            }
            "csr" => {
                // Verify CSR indices are sorted and in bounds
                let mut prev_idx = 0;
                for (i, &w) in weights.iter().enumerate() {
                    if w.abs() > 1e-6 {
                        if i < prev_idx {
                            result.passed = false;
                            result.add_diagnostic("CSR indices not sorted");
                        }
                        prev_idx = i;
                    }
                }
                result.add_metric(
                    "nnz",
                    weights.iter().filter(|&&w| w.abs() > 1e-6).count() as f32,
                );
            }
            _ => {
                result.add_diagnostic(&format!("Unknown sparsity pattern: {}", pattern));
            }
        }

        result
    }
}

/// Loss scaling validator
pub struct LossScalingValidator;

impl LossScalingValidator {
    pub fn validate(initial_scale: f32, scale_history: &[f32]) -> ValidationResult {
        let mut result = ValidationResult::new("loss_scaling", true);

        if scale_history.is_empty() {
            result.passed = false;
            result.add_diagnostic("No scale history provided");
            return result;
        }

        let min_scale = *scale_history
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&initial_scale);

        let max_scale = *scale_history
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&initial_scale);

        result.add_metric("min_scale", min_scale);
        result.add_metric("max_scale", max_scale);

        // Scale should stay within bounds: [1.0, 16M]
        let max_allowed = (1u32 << 24) as f32; // 16M
        if min_scale < 1.0 || max_scale > max_allowed {
            result.passed = false;
            result.add_diagnostic(&format!(
                "Scale out of bounds: [{:.0}, {:.0}], expected [1, {:.0}]",
                min_scale, max_scale, max_allowed
            ));
        }

        result
    }
}

/// Quantization validator
pub struct QuantizationValidator;

impl QuantizationValidator {
    pub fn validate(
        original: &[f32],
        quantized: &[f32],
        scale: f32,
        bit_width: usize,
    ) -> ValidationResult {
        let mut result = ValidationResult::new(&format!("quantization_int{}", bit_width), true);

        if original.len() != quantized.len() {
            result.passed = false;
            result.add_diagnostic("Length mismatch between original and quantized");
            return result;
        }

        let mut max_error = 0.0f32;
        let mut rms_error = 0.0f32;

        for (orig, quant) in original.iter().zip(quantized.iter()) {
            let error = (orig - quant).abs();
            max_error = max_error.max(error);
            rms_error += error * error;

            if !quant.is_finite() {
                result.passed = false;
                result.add_diagnostic("Non-finite quantized value");
            }
        }

        rms_error = (rms_error / original.len() as f32).sqrt();

        result.add_metric("max_error", max_error);
        result.add_metric("rms_error", rms_error);

        // Error should be bounded by scale/2
        let max_allowed_error = scale * 0.5;
        if max_error > max_allowed_error {
            result.passed = false;
            result.add_diagnostic(&format!(
                "Max error {:.6} exceeds bound {:.6}",
                max_error, max_allowed_error
            ));
        }

        result
    }
}

/// Fusion validator
pub struct FusionValidator;

impl FusionValidator {
    pub fn validate(
        unfused_output: &[f32],
        fused_output: &[f32],
        epsilon: f32,
    ) -> ValidationResult {
        let mut result = ValidationResult::new("fusion", true);

        if unfused_output.len() != fused_output.len() {
            result.passed = false;
            result.add_diagnostic("Output size mismatch");
            return result;
        }

        let mut max_diff = 0.0f32;

        for (unfused, fused) in unfused_output.iter().zip(fused_output.iter()) {
            let diff = (unfused - fused).abs();
            max_diff = max_diff.max(diff);

            if !fused.is_finite() {
                result.passed = false;
                result.add_diagnostic("Non-finite fused output");
            }
        }

        result.add_metric("max_diff", max_diff);

        if max_diff > epsilon {
            result.passed = false;
            result.add_diagnostic(&format!(
                "Max difference {:.6} exceeds epsilon {:.6}",
                max_diff, epsilon
            ));
        }

        result
    }
}

/// Test suite runner
pub struct ValidationSuite {
    results: Vec<ValidationResult>,
}

impl ValidationSuite {
    pub fn new() -> Self {
        Self { results: vec![] }
    }

    pub fn add_result(&mut self, result: ValidationResult) {
        self.results.push(result);
    }

    pub fn summary(&self) -> String {
        let passed = self.results.iter().filter(|r| r.passed).count();
        let total = self.results.len();

        let mut output = format!(
            "\n=== Validation Summary ===\n{}/{} tests passed\n\n",
            passed, total
        );

        for result in &self.results {
            output.push_str(&result.display());
        }

        output
    }

    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed)
    }
}

impl fmt::Display for ValidationSuite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}
