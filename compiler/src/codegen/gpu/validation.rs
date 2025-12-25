//! GPU Correctness Validation
//!
//! Validates that optimized GPU kernels produce correct results by comparing
//! their outputs against baseline (unoptimized) versions.
//!
//! # Features
//!
//! - Configurable numerical tolerance (absolute, relative, ULP)
//! - Type-specific comparison (f32, f64, integers)
//! - Detection of NaN/Infinity in outputs
//! - Precision loss tracking for reductions
//! - Detailed mismatch reporting

use std::fmt;

use super::ir::GpuType;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during validation
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// Output buffer sizes don't match
    SizeMismatch { expected: usize, actual: usize },
    /// Values don't match within tolerance
    ValueMismatch {
        index: usize,
        expected: f64,
        actual: f64,
    },
    /// Numerical precision degraded beyond threshold
    PrecisionLoss { max_error: f64, threshold: f64 },
    /// Output contains invalid values (NaN, Inf)
    InvalidOutput { description: String },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::SizeMismatch { expected, actual } => {
                write!(
                    f,
                    "Size mismatch: expected {} bytes, got {}",
                    expected, actual
                )
            }
            ValidationError::ValueMismatch {
                index,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Value mismatch at index {}: expected {}, got {}",
                    index, expected, actual
                )
            }
            ValidationError::PrecisionLoss {
                max_error,
                threshold,
            } => {
                write!(
                    f,
                    "Precision loss: max error {} exceeds threshold {}",
                    max_error, threshold
                )
            }
            ValidationError::InvalidOutput { description } => {
                write!(f, "Invalid output: {}", description)
            }
        }
    }
}

impl std::error::Error for ValidationError {}

// ============================================================================
// Validation Issues
// ============================================================================

/// A specific issue detected during validation
#[derive(Debug, Clone)]
pub enum ValidationIssue {
    /// Values don't match within tolerance
    ValueMismatch {
        index: usize,
        expected: f64,
        actual: f64,
        absolute_error: f64,
        relative_error: f64,
    },
    /// NaN detected in output
    NaNDetected { index: usize, buffer: String },
    /// Infinity detected in output
    InfDetected {
        index: usize,
        buffer: String,
        positive: bool,
    },
    /// Accumulated precision loss
    PrecisionLoss { max_error: f64, mean_error: f64 },
}

// ============================================================================
// Tolerance Configuration
// ============================================================================

/// Configuration for numerical comparison tolerances
#[derive(Debug, Clone)]
pub struct ToleranceConfig {
    /// Absolute tolerance for comparisons
    pub absolute: f64,
    /// Relative tolerance for comparisons
    pub relative: f64,
    /// ULP (units in last place) tolerance
    pub ulp: u32,
    /// Allow NaN == NaN
    pub nan_equal: bool,
    /// Allow Inf == Inf (with same sign)
    pub inf_equal: bool,
}

impl Default for ToleranceConfig {
    fn default() -> Self {
        Self {
            absolute: 1e-6,
            relative: 1e-5,
            ulp: 4,
            nan_equal: false,
            inf_equal: true,
        }
    }
}

impl ToleranceConfig {
    /// Strict tolerance for regression testing
    pub fn strict() -> Self {
        Self {
            absolute: 1e-10,
            relative: 1e-9,
            ulp: 1,
            nan_equal: false,
            inf_equal: true,
        }
    }

    /// Relaxed tolerance for fast-math kernels
    pub fn relaxed() -> Self {
        Self {
            absolute: 1e-4,
            relative: 1e-3,
            ulp: 16,
            nan_equal: false,
            inf_equal: true,
        }
    }

    /// Check if two f32 values are equal within tolerance
    pub fn f32_equal(&self, expected: f32, actual: f32) -> bool {
        // Handle special cases
        if expected.is_nan() && actual.is_nan() {
            return self.nan_equal;
        }
        if expected.is_infinite() && actual.is_infinite() {
            return self.inf_equal && (expected.signum() == actual.signum());
        }
        if expected.is_nan() || actual.is_nan() {
            return false;
        }
        if expected.is_infinite() || actual.is_infinite() {
            return false;
        }

        // Check absolute tolerance
        let abs_diff = (expected - actual).abs();
        if abs_diff <= self.absolute as f32 {
            return true;
        }

        // Check relative tolerance
        let max_abs = expected.abs().max(actual.abs());
        if max_abs > 0.0 && abs_diff / max_abs <= self.relative as f32 {
            return true;
        }

        // Check ULP tolerance
        let ulp_diff = ulp_diff_f32(expected, actual);
        ulp_diff <= self.ulp as i64
    }

    /// Check if two f64 values are equal within tolerance
    pub fn f64_equal(&self, expected: f64, actual: f64) -> bool {
        // Handle special cases
        if expected.is_nan() && actual.is_nan() {
            return self.nan_equal;
        }
        if expected.is_infinite() && actual.is_infinite() {
            return self.inf_equal && (expected.signum() == actual.signum());
        }
        if expected.is_nan() || actual.is_nan() {
            return false;
        }
        if expected.is_infinite() || actual.is_infinite() {
            return false;
        }

        // Check absolute tolerance
        let abs_diff = (expected - actual).abs();
        if abs_diff <= self.absolute {
            return true;
        }

        // Check relative tolerance
        let max_abs = expected.abs().max(actual.abs());
        if max_abs > 0.0 && abs_diff / max_abs <= self.relative {
            return true;
        }

        // Check ULP tolerance
        let ulp_diff = ulp_diff_f64(expected, actual);
        ulp_diff <= self.ulp as i64
    }
}

/// Calculate ULP difference between two f32 values
fn ulp_diff_f32(a: f32, b: f32) -> i64 {
    let a_bits = a.to_bits() as i32;
    let b_bits = b.to_bits() as i32;

    // Handle negative values by flipping to lexicographic ordering
    // 0x80000000 as i32 wraps to i32::MIN
    let a_bits = if a_bits < 0 {
        i32::MIN.wrapping_sub(a_bits)
    } else {
        a_bits
    };
    let b_bits = if b_bits < 0 {
        i32::MIN.wrapping_sub(b_bits)
    } else {
        b_bits
    };

    (a_bits as i64 - b_bits as i64).abs()
}

/// Calculate ULP difference between two f64 values
fn ulp_diff_f64(a: f64, b: f64) -> i64 {
    let a_bits = a.to_bits() as i64;
    let b_bits = b.to_bits() as i64;

    // Handle negative values by flipping to lexicographic ordering
    // 0x8000000000000000 as i64 wraps to i64::MIN
    let a_bits = if a_bits < 0 {
        i64::MIN.wrapping_sub(a_bits)
    } else {
        a_bits
    };
    let b_bits = if b_bits < 0 {
        i64::MIN.wrapping_sub(b_bits)
    } else {
        b_bits
    };

    (a_bits - b_bits).abs()
}

// ============================================================================
// Validation Configuration
// ============================================================================

/// Configuration for correctness validation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Enable validation
    pub enabled: bool,
    /// Tolerance configuration
    pub tolerance: ToleranceConfig,
    /// Maximum elements to compare (0 = unlimited)
    pub max_elements: usize,
    /// Stop on first mismatch
    pub stop_on_first: bool,
    /// Track precision statistics
    pub track_precision: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            tolerance: ToleranceConfig::default(),
            max_elements: 0, // Unlimited
            stop_on_first: false,
            track_precision: true,
        }
    }
}

// ============================================================================
// Buffer Comparison
// ============================================================================

/// Result of comparing a single buffer
#[derive(Debug, Clone)]
pub struct BufferComparison {
    /// Buffer name
    pub name: String,
    /// Element type
    pub element_type: GpuType,
    /// Total elements compared
    pub total_elements: usize,
    /// Number of matching elements
    pub matching_elements: usize,
    /// Maximum absolute error
    pub max_absolute_error: f64,
    /// Maximum relative error
    pub max_relative_error: f64,
    /// Mean absolute error
    pub mean_absolute_error: f64,
    /// Issues found
    pub issues: Vec<ValidationIssue>,
}

impl BufferComparison {
    /// Check if comparison passed (all elements match)
    pub fn passed(&self) -> bool {
        self.matching_elements == self.total_elements && self.issues.is_empty()
    }

    /// Get match percentage
    pub fn match_percentage(&self) -> f64 {
        if self.total_elements == 0 {
            100.0
        } else {
            (self.matching_elements as f64 / self.total_elements as f64) * 100.0
        }
    }
}

// ============================================================================
// Validation Result
// ============================================================================

/// Result of correctness validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Did validation pass?
    pub passed: bool,
    /// Buffer comparisons
    pub comparisons: Vec<BufferComparison>,
    /// All issues found
    pub issues: Vec<ValidationIssue>,
    /// Precision statistics (if tracked)
    pub precision_stats: Option<PrecisionStats>,
}

impl ValidationResult {
    /// Create a passed result
    pub fn passed() -> Self {
        Self {
            passed: true,
            comparisons: Vec::new(),
            issues: Vec::new(),
            precision_stats: None,
        }
    }

    /// Create a failed result
    pub fn failed(issues: Vec<ValidationIssue>) -> Self {
        Self {
            passed: false,
            comparisons: Vec::new(),
            issues,
            precision_stats: None,
        }
    }

    /// Get total number of issues
    pub fn issue_count(&self) -> usize {
        self.issues.len()
    }
}

/// Precision statistics
#[derive(Debug, Clone, Default)]
pub struct PrecisionStats {
    /// Maximum absolute error
    pub max_absolute_error: f64,
    /// Maximum relative error
    pub max_relative_error: f64,
    /// Mean absolute error
    pub mean_absolute_error: f64,
    /// Mean relative error
    pub mean_relative_error: f64,
    /// Number of comparisons
    pub comparison_count: usize,
}

// ============================================================================
// Correctness Validator
// ============================================================================

/// Validates GPU kernel output correctness
pub struct CorrectnessValidator {
    config: ValidationConfig,
}

impl CorrectnessValidator {
    /// Create a new validator with default configuration
    pub fn new() -> Self {
        Self {
            config: ValidationConfig::default(),
        }
    }

    /// Create a validator with specific configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Validate f32 buffers
    pub fn validate_f32(&self, name: &str, expected: &[f32], actual: &[f32]) -> BufferComparison {
        let mut issues = Vec::new();
        let mut matching = 0usize;
        let mut sum_abs_error = 0.0f64;
        let mut max_abs_error = 0.0f64;
        let mut max_rel_error = 0.0f64;

        let count = expected.len().min(actual.len());
        let max_compare = if self.config.max_elements > 0 {
            count.min(self.config.max_elements)
        } else {
            count
        };

        for i in 0..max_compare {
            let exp = expected[i];
            let act = actual[i];

            // Check for NaN
            if act.is_nan() && !exp.is_nan() {
                issues.push(ValidationIssue::NaNDetected {
                    index: i,
                    buffer: name.to_string(),
                });
                if self.config.stop_on_first {
                    break;
                }
                continue;
            }

            // Check for Inf
            if act.is_infinite() && !exp.is_infinite() {
                issues.push(ValidationIssue::InfDetected {
                    index: i,
                    buffer: name.to_string(),
                    positive: act > 0.0,
                });
                if self.config.stop_on_first {
                    break;
                }
                continue;
            }

            // Compare with tolerance
            if self.config.tolerance.f32_equal(exp, act) {
                matching += 1;
            } else {
                let abs_error = (exp as f64 - act as f64).abs();
                let rel_error = if exp != 0.0 {
                    abs_error / exp.abs() as f64
                } else {
                    abs_error
                };

                issues.push(ValidationIssue::ValueMismatch {
                    index: i,
                    expected: exp as f64,
                    actual: act as f64,
                    absolute_error: abs_error,
                    relative_error: rel_error,
                });

                if self.config.stop_on_first {
                    break;
                }
            }

            // Track precision stats
            if self.config.track_precision && !exp.is_nan() && !act.is_nan() {
                let abs_error = (exp as f64 - act as f64).abs();
                sum_abs_error += abs_error;
                max_abs_error = max_abs_error.max(abs_error);
                if exp != 0.0 {
                    let rel_error = abs_error / exp.abs() as f64;
                    max_rel_error = max_rel_error.max(rel_error);
                }
            }
        }

        BufferComparison {
            name: name.to_string(),
            element_type: GpuType::F32,
            total_elements: max_compare,
            matching_elements: matching,
            max_absolute_error: max_abs_error,
            max_relative_error: max_rel_error,
            mean_absolute_error: if max_compare > 0 {
                sum_abs_error / max_compare as f64
            } else {
                0.0
            },
            issues,
        }
    }

    /// Validate f64 buffers
    pub fn validate_f64(&self, name: &str, expected: &[f64], actual: &[f64]) -> BufferComparison {
        let mut issues = Vec::new();
        let mut matching = 0usize;
        let mut sum_abs_error = 0.0f64;
        let mut max_abs_error = 0.0f64;
        let mut max_rel_error = 0.0f64;

        let count = expected.len().min(actual.len());
        let max_compare = if self.config.max_elements > 0 {
            count.min(self.config.max_elements)
        } else {
            count
        };

        for i in 0..max_compare {
            let exp = expected[i];
            let act = actual[i];

            // Check for NaN
            if act.is_nan() && !exp.is_nan() {
                issues.push(ValidationIssue::NaNDetected {
                    index: i,
                    buffer: name.to_string(),
                });
                if self.config.stop_on_first {
                    break;
                }
                continue;
            }

            // Check for Inf
            if act.is_infinite() && !exp.is_infinite() {
                issues.push(ValidationIssue::InfDetected {
                    index: i,
                    buffer: name.to_string(),
                    positive: act > 0.0,
                });
                if self.config.stop_on_first {
                    break;
                }
                continue;
            }

            // Compare with tolerance
            if self.config.tolerance.f64_equal(exp, act) {
                matching += 1;
            } else {
                let abs_error = (exp - act).abs();
                let rel_error = if exp != 0.0 {
                    abs_error / exp.abs()
                } else {
                    abs_error
                };

                issues.push(ValidationIssue::ValueMismatch {
                    index: i,
                    expected: exp,
                    actual: act,
                    absolute_error: abs_error,
                    relative_error: rel_error,
                });

                if self.config.stop_on_first {
                    break;
                }
            }

            // Track precision stats
            if self.config.track_precision && !exp.is_nan() && !act.is_nan() {
                let abs_error = (exp - act).abs();
                sum_abs_error += abs_error;
                max_abs_error = max_abs_error.max(abs_error);
                if exp != 0.0 {
                    let rel_error = abs_error / exp.abs();
                    max_rel_error = max_rel_error.max(rel_error);
                }
            }
        }

        BufferComparison {
            name: name.to_string(),
            element_type: GpuType::F64,
            total_elements: max_compare,
            matching_elements: matching,
            max_absolute_error: max_abs_error,
            max_relative_error: max_rel_error,
            mean_absolute_error: if max_compare > 0 {
                sum_abs_error / max_compare as f64
            } else {
                0.0
            },
            issues,
        }
    }

    /// Validate i32 buffers (exact comparison)
    pub fn validate_i32(&self, name: &str, expected: &[i32], actual: &[i32]) -> BufferComparison {
        let mut issues = Vec::new();
        let mut matching = 0usize;

        let count = expected.len().min(actual.len());
        let max_compare = if self.config.max_elements > 0 {
            count.min(self.config.max_elements)
        } else {
            count
        };

        for i in 0..max_compare {
            if expected[i] == actual[i] {
                matching += 1;
            } else {
                issues.push(ValidationIssue::ValueMismatch {
                    index: i,
                    expected: expected[i] as f64,
                    actual: actual[i] as f64,
                    absolute_error: (expected[i] - actual[i]).abs() as f64,
                    relative_error: 0.0,
                });

                if self.config.stop_on_first {
                    break;
                }
            }
        }

        BufferComparison {
            name: name.to_string(),
            element_type: GpuType::I32,
            total_elements: max_compare,
            matching_elements: matching,
            max_absolute_error: 0.0,
            max_relative_error: 0.0,
            mean_absolute_error: 0.0,
            issues,
        }
    }

    /// Validate raw byte buffers as f32
    pub fn validate_bytes_as_f32(
        &self,
        name: &str,
        expected: &[u8],
        actual: &[u8],
    ) -> Result<BufferComparison, ValidationError> {
        if expected.len() != actual.len() {
            return Err(ValidationError::SizeMismatch {
                expected: expected.len(),
                actual: actual.len(),
            });
        }

        // Interpret as f32 slices
        let exp_f32: &[f32] = bytemuck_cast_slice(expected)?;
        let act_f32: &[f32] = bytemuck_cast_slice(actual)?;

        Ok(self.validate_f32(name, exp_f32, act_f32))
    }

    /// Validate and produce a full result
    pub fn validate_result(&self, comparisons: Vec<BufferComparison>) -> ValidationResult {
        let mut all_issues = Vec::new();
        let mut passed = true;

        for comp in &comparisons {
            if !comp.passed() {
                passed = false;
            }
            all_issues.extend(comp.issues.clone());
        }

        let precision_stats = if self.config.track_precision && !comparisons.is_empty() {
            let mut stats = PrecisionStats::default();
            for comp in &comparisons {
                stats.max_absolute_error = stats.max_absolute_error.max(comp.max_absolute_error);
                stats.max_relative_error = stats.max_relative_error.max(comp.max_relative_error);
                stats.mean_absolute_error += comp.mean_absolute_error;
                stats.comparison_count += comp.total_elements;
            }
            if !comparisons.is_empty() {
                stats.mean_absolute_error /= comparisons.len() as f64;
            }
            Some(stats)
        } else {
            None
        };

        ValidationResult {
            passed,
            comparisons,
            issues: all_issues,
            precision_stats,
        }
    }
}

impl Default for CorrectnessValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to cast byte slices to typed slices
fn bytemuck_cast_slice<T: Copy>(bytes: &[u8]) -> Result<&[T], ValidationError> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<T>()) {
        return Err(ValidationError::InvalidOutput {
            description: format!(
                "Buffer size {} not aligned to type size {}",
                bytes.len(),
                std::mem::size_of::<T>()
            ),
        });
    }

    // Safety: We've verified alignment and size
    let ptr = bytes.as_ptr() as *const T;
    let len = bytes.len() / std::mem::size_of::<T>();
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tolerance_f32_equal() {
        let tol = ToleranceConfig::default();

        // Exact match
        assert!(tol.f32_equal(1.0, 1.0));

        // Within absolute tolerance
        assert!(tol.f32_equal(1.0, 1.0 + 1e-7));

        // Outside tolerance
        assert!(!tol.f32_equal(1.0, 1.01));

        // NaN handling
        assert!(!tol.nan_equal); // Default doesn't allow NaN == NaN
        assert!(!tol.f32_equal(f32::NAN, f32::NAN));

        // Inf handling
        assert!(tol.f32_equal(f32::INFINITY, f32::INFINITY));
        assert!(!tol.f32_equal(f32::INFINITY, f32::NEG_INFINITY));
    }

    #[test]
    fn test_tolerance_strict() {
        let tol = ToleranceConfig::strict();
        assert!(tol.absolute < ToleranceConfig::default().absolute);
    }

    #[test]
    fn test_tolerance_relaxed() {
        let tol = ToleranceConfig::relaxed();
        assert!(tol.absolute > ToleranceConfig::default().absolute);
    }

    #[test]
    fn test_validate_f32_pass() {
        let validator = CorrectnessValidator::new();
        let expected = vec![1.0f32, 2.0, 3.0, 4.0];
        let actual = vec![1.0f32, 2.0, 3.0, 4.0];

        let result = validator.validate_f32("test", &expected, &actual);
        assert!(result.passed());
        assert_eq!(result.match_percentage(), 100.0);
    }

    #[test]
    fn test_validate_f32_fail() {
        let validator = CorrectnessValidator::new();
        let expected = vec![1.0f32, 2.0, 3.0, 4.0];
        let actual = vec![1.0f32, 2.0, 999.0, 4.0]; // Mismatch at index 2

        let result = validator.validate_f32("test", &expected, &actual);
        assert!(!result.passed());
        assert_eq!(result.matching_elements, 3);
        assert!(!result.issues.is_empty());
    }

    #[test]
    fn test_validate_f32_nan_detection() {
        let validator = CorrectnessValidator::new();
        let expected = vec![1.0f32, 2.0, 3.0];
        let actual = vec![1.0f32, f32::NAN, 3.0];

        let result = validator.validate_f32("test", &expected, &actual);
        assert!(!result.passed());
        assert!(matches!(
            result.issues[0],
            ValidationIssue::NaNDetected { .. }
        ));
    }

    #[test]
    fn test_validate_i32_exact() {
        let validator = CorrectnessValidator::new();
        let expected = vec![1i32, 2, 3, 4];
        let actual = vec![1i32, 2, 3, 4];

        let result = validator.validate_i32("test", &expected, &actual);
        assert!(result.passed());
    }

    #[test]
    fn test_validation_result() {
        let validator = CorrectnessValidator::new();

        let comp1 = validator.validate_f32("buf1", &[1.0, 2.0], &[1.0, 2.0]);
        let comp2 = validator.validate_f32("buf2", &[3.0, 4.0], &[3.0, 4.0]);

        let result = validator.validate_result(vec![comp1, comp2]);
        assert!(result.passed);
        assert!(result.precision_stats.is_some());
    }

    #[test]
    fn test_ulp_diff() {
        let a = 1.0f32;
        let b = 1.0f32 + f32::EPSILON;
        let diff = ulp_diff_f32(a, b);
        assert!(diff > 0);
        assert!(diff <= 2);
    }

    #[test]
    fn test_buffer_comparison_percentage() {
        let comp = BufferComparison {
            name: "test".to_string(),
            element_type: GpuType::F32,
            total_elements: 100,
            matching_elements: 95,
            max_absolute_error: 0.001,
            max_relative_error: 0.0001,
            mean_absolute_error: 0.0005,
            issues: Vec::new(),
        };

        assert_eq!(comp.match_percentage(), 95.0);
    }
}
