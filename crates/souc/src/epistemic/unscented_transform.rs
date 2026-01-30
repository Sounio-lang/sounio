//! Unscented Transform for Nonlinear Uncertainty Propagation
//!
//! Implements the Scaled Unscented Transform (SUT) for propagating uncertainty
//! through nonlinear transformations without linearization.
//!
//! # Overview
//!
//! The Unscented Transform is superior to first-order Taylor series (GUM) for
//! highly nonlinear functions because it:
//! - Captures uncertainty to 3rd order (vs 1st order for linearization)
//! - Uses deterministic sigma points (no Monte Carlo sampling)
//! - Scales efficiently: O(n) points for n dimensions vs O(n²) for 2nd-order methods
//!
//! # Mathematical Foundation
//!
//! Given a random variable x ~ N(μ, Σ) and nonlinear function y = f(x):
//!
//! 1. Generate 2n+1 sigma points: χᵢ
//! 2. Transform: Yᵢ = f(χᵢ)
//! 3. Compute: E[y] = Σ Wᵢᵐ Yᵢ, Cov[y] = Σ Wᵢᶜ (Yᵢ - E[y])(Yᵢ - E[y])ᵀ
//!
//! # Example
//!
//! ```rust
//! use sounio::epistemic::unscented_transform::{UTConfig, UnscentedTransform};
//!
//! let config = UTConfig::default(); // α=1e-3, β=2, κ=0
//! let mean = vec![0.0, 0.0];
//! let cov = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
//!
//! let ut = UnscentedTransform::new(config, mean, cov);
//!
//! // Nonlinear transformation
//! let transformed = ut.propagate(|x| vec![x[0] * x[0] + x[1]]);
//! ```
//!
//! # References
//!
//! - Julier & Uhlmann (2004): "Unscented Filtering and Nonlinear Estimation"
//! - Wan & van der Merwe (2000): "The Unscented Kalman Filter"

/// Configuration for Unscented Transform
#[derive(Debug, Clone, PartialEq)]
pub struct UTConfig {
    /// Spread parameter: controls sigma point spread (typically 1e-4 to 1)
    /// Small α → points close to mean (better for strong nonlinearity)
    /// Large α → points farther from mean (better for weak nonlinearity)
    pub alpha: f64,

    /// Prior knowledge parameter: β = 2 is optimal for Gaussian
    pub beta: f64,

    /// Secondary scaling parameter: κ = 3 - n is common choice
    /// Often set to 0 or 3 - n_dims
    pub kappa: f64,

    /// Sigma point method
    pub method: SigmaPointMethod,
}

impl Default for UTConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5, // Moderate spread (1e-3 is too small for numerical stability)
            beta: 2.0,  // Gaussian prior (optimal for Gaussian)
            kappa: 0.0, // Will be adjusted to 3-n based on dimension
            method: SigmaPointMethod::Symmetric,
        }
    }
}

impl UTConfig {
    /// Create config with automatic κ adjustment
    pub fn new_auto(alpha: f64, beta: f64) -> Self {
        Self {
            alpha,
            beta,
            kappa: 0.0, // Will be set to 3 - n when creating UT
            method: SigmaPointMethod::Symmetric,
        }
    }

    /// Compute λ = α²(n + κ) - n
    pub fn lambda(&self, n_dims: usize) -> f64 {
        self.alpha * self.alpha * (n_dims as f64 + self.kappa) - n_dims as f64
    }

    /// Compute γ = sqrt(n + λ)
    pub fn gamma(&self, n_dims: usize) -> f64 {
        libm::sqrt(n_dims as f64 + self.lambda(n_dims))
    }
}

/// Sigma point generation method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SigmaPointMethod {
    /// Symmetric: 2n+1 points (standard UT)
    Symmetric,

    /// Simplex: n+2 points (minimum set)
    Simplex,

    /// Spherical: 2n points (no central point)
    Spherical,
}

/// Unscented Transform propagator
#[derive(Debug, Clone)]
pub struct UnscentedTransform {
    pub config: UTConfig,
    pub n_dims: usize,
    pub mean: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
    pub sigma_points: Vec<Vec<f64>>,
    pub weights_mean: Vec<f64>,
    pub weights_cov: Vec<f64>,
}

impl UnscentedTransform {
    /// Create new UT propagator
    pub fn new(config: UTConfig, mean: Vec<f64>, covariance: Vec<Vec<f64>>) -> Self {
        let n_dims = mean.len();

        // Auto-adjust kappa if needed
        let mut config = config;
        if config.kappa == 0.0 {
            config.kappa = 3.0 - n_dims as f64;
        }

        let (sigma_points, weights_mean, weights_cov) =
            generate_sigma_points(&config, &mean, &covariance);

        Self {
            config,
            n_dims,
            mean,
            covariance,
            sigma_points,
            weights_mean,
            weights_cov,
        }
    }

    /// Propagate uncertainty through nonlinear transformation
    ///
    /// # Arguments
    /// * `f` - Nonlinear function: R^n → R^m
    ///
    /// # Returns
    /// Propagated mean and covariance
    pub fn propagate<F>(&self, f: F) -> UTPropagationResult
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        // Transform sigma points
        let transformed_points: Vec<Vec<f64>> =
            self.sigma_points.iter().map(|point| f(point)).collect();

        // Output dimension
        let m_dims = transformed_points[0].len();

        // Compute mean: E[y] = Σ Wᵢᵐ Yᵢ
        let mut mean = vec![0.0; m_dims];
        for (i, y_i) in transformed_points.iter().enumerate() {
            for j in 0..m_dims {
                mean[j] += self.weights_mean[i] * y_i[j];
            }
        }

        // Compute covariance: Cov[y] = Σ Wᵢᶜ (Yᵢ - μ)(Yᵢ - μ)ᵀ
        let mut covariance = vec![vec![0.0; m_dims]; m_dims];
        for (i, y_i) in transformed_points.iter().enumerate() {
            for j in 0..m_dims {
                for k in 0..m_dims {
                    let dev_j = y_i[j] - mean[j];
                    let dev_k = y_i[k] - mean[k];
                    covariance[j][k] += self.weights_cov[i] * dev_j * dev_k;
                }
            }
        }

        UTPropagationResult {
            mean,
            covariance,
            sigma_points: self.sigma_points.clone(),
            transformed_points,
        }
    }

    /// Propagate with cross-covariance (for Kalman filtering)
    pub fn propagate_with_cross<F>(&self, f: F) -> UTCrossPropagationResult
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let result = self.propagate(f);

        // Compute cross-covariance: Cov[x,y] = Σ Wᵢᶜ (Xᵢ - μₓ)(Yᵢ - μᵧ)ᵀ
        let m_dims = result.mean.len();
        let mut cross_covariance = vec![vec![0.0; m_dims]; self.n_dims];

        for i in 0..self.sigma_points.len() {
            for j in 0..self.n_dims {
                for k in 0..m_dims {
                    let dev_x = self.sigma_points[i][j] - self.mean[j];
                    let dev_y = result.transformed_points[i][k] - result.mean[k];
                    cross_covariance[j][k] += self.weights_cov[i] * dev_x * dev_y;
                }
            }
        }

        UTCrossPropagationResult {
            mean: result.mean,
            covariance: result.covariance,
            cross_covariance,
            sigma_points: result.sigma_points,
            transformed_points: result.transformed_points,
        }
    }

    /// Compute standard deviations from covariance matrix
    pub fn std_devs(&self) -> Vec<f64> {
        (0..self.n_dims)
            .map(|i| libm::sqrt(self.covariance[i][i]))
            .collect()
    }
}

/// Result of UT propagation
#[derive(Debug, Clone)]
pub struct UTPropagationResult {
    pub mean: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
    pub sigma_points: Vec<Vec<f64>>,
    pub transformed_points: Vec<Vec<f64>>,
}

impl UTPropagationResult {
    /// Get standard deviations
    pub fn std_devs(&self) -> Vec<f64> {
        (0..self.mean.len())
            .map(|i| libm::sqrt(self.covariance[i][i]))
            .collect()
    }

    /// Get correlation matrix
    pub fn correlation_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.mean.len();
        let std_devs = self.std_devs();
        let mut corr = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                if std_devs[i] > 1e-10 && std_devs[j] > 1e-10 {
                    corr[i][j] = self.covariance[i][j] / (std_devs[i] * std_devs[j]);
                } else if i == j {
                    corr[i][j] = 1.0;
                }
            }
        }

        corr
    }
}

/// Result with cross-covariance (for filtering)
#[derive(Debug, Clone)]
pub struct UTCrossPropagationResult {
    pub mean: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
    pub cross_covariance: Vec<Vec<f64>>,
    pub sigma_points: Vec<Vec<f64>>,
    pub transformed_points: Vec<Vec<f64>>,
}

// =============================================================================
// Sigma Point Generation
// =============================================================================

/// Generate sigma points and weights
fn generate_sigma_points(
    config: &UTConfig,
    mean: &[f64],
    covariance: &[Vec<f64>],
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    match config.method {
        SigmaPointMethod::Symmetric => generate_symmetric_points(config, mean, covariance),
        SigmaPointMethod::Simplex => generate_simplex_points(config, mean, covariance),
        SigmaPointMethod::Spherical => generate_spherical_points(config, mean, covariance),
    }
}

/// Generate symmetric sigma points (standard UT with 2n+1 points)
fn generate_symmetric_points(
    config: &UTConfig,
    mean: &[f64],
    covariance: &[Vec<f64>],
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let n = mean.len();
    let lambda = config.lambda(n);
    let gamma = config.gamma(n);

    // Cholesky decomposition: Σ = LLᵀ
    let l = cholesky_decomposition(covariance);

    // Generate 2n+1 sigma points
    let mut sigma_points = Vec::with_capacity(2 * n + 1);
    let mut weights_mean = Vec::with_capacity(2 * n + 1);
    let mut weights_cov = Vec::with_capacity(2 * n + 1);

    // Central point: χ₀ = μ
    sigma_points.push(mean.to_vec());
    weights_mean.push(lambda / (n as f64 + lambda));
    weights_cov
        .push(lambda / (n as f64 + lambda) + (1.0 - config.alpha * config.alpha + config.beta));

    // Positive spread: χᵢ = μ + γ·Lᵢ for i = 1..n
    // Negative spread: χᵢ = μ - γ·Lᵢ for i = n+1..2n
    for i in 0..n {
        let mut point_pos = mean.to_vec();
        let mut point_neg = mean.to_vec();

        for j in 0..n {
            point_pos[j] += gamma * l[j][i];
            point_neg[j] -= gamma * l[j][i];
        }

        sigma_points.push(point_pos);
        sigma_points.push(point_neg);

        let w = 1.0 / (2.0 * (n as f64 + lambda));
        weights_mean.push(w);
        weights_mean.push(w);
        weights_cov.push(w);
        weights_cov.push(w);
    }

    (sigma_points, weights_mean, weights_cov)
}

/// Generate simplex sigma points (n+2 points)
fn generate_simplex_points(
    config: &UTConfig,
    mean: &[f64],
    covariance: &[Vec<f64>],
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    // Simplified: fall back to symmetric for now
    // Full simplex implementation requires more complex geometry
    generate_symmetric_points(config, mean, covariance)
}

/// Generate spherical sigma points (2n points, no central)
fn generate_spherical_points(
    config: &UTConfig,
    mean: &[f64],
    covariance: &[Vec<f64>],
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    // Simplified: fall back to symmetric for now
    generate_symmetric_points(config, mean, covariance)
}

/// Cholesky decomposition: A = LLᵀ
///
/// Returns lower triangular matrix L such that A = LLᵀ
fn cholesky_decomposition(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let mut l = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;

            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }

            if i == j {
                // Diagonal element
                let val = matrix[i][i] - sum;
                if val < 0.0 {
                    // Matrix not positive definite - use small epsilon
                    l[i][j] = 1e-10;
                } else {
                    l[i][j] = libm::sqrt(val);
                }
            } else {
                // Off-diagonal element
                if l[j][j].abs() > 1e-10 {
                    l[i][j] = (matrix[i][j] - sum) / l[j][j];
                }
            }
        }
    }

    l
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compare UT with first-order Taylor (GUM) propagation
pub fn compare_with_taylor<F>(
    ut: &UnscentedTransform,
    f: &F,
    jacobian: &[Vec<f64>],
) -> ComparisonResult
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    // UT propagation
    let ut_result = ut.propagate(f);

    // First-order Taylor: μᵧ ≈ f(μₓ), Σᵧ ≈ J Σₓ Jᵀ
    let taylor_mean = f(&ut.mean);
    let taylor_cov = propagate_covariance_taylor(&ut.covariance, jacobian);

    ComparisonResult {
        ut_mean: ut_result.mean,
        ut_covariance: ut_result.covariance,
        taylor_mean,
        taylor_covariance: taylor_cov,
    }
}

/// Propagate covariance using first-order Taylor: Σᵧ = J Σₓ Jᵀ
fn propagate_covariance_taylor(cov: &[Vec<f64>], jacobian: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = jacobian.len(); // Output dimension
    let n = jacobian[0].len(); // Input dimension

    // Compute J * Σₓ
    let mut j_sigma = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            for k in 0..n {
                j_sigma[i][j] += jacobian[i][k] * cov[k][j];
            }
        }
    }

    // Compute (J * Σₓ) * Jᵀ
    let mut result = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            for k in 0..n {
                result[i][j] += j_sigma[i][k] * jacobian[j][k];
            }
        }
    }

    result
}

/// Comparison between UT and Taylor approximation
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub ut_mean: Vec<f64>,
    pub ut_covariance: Vec<Vec<f64>>,
    pub taylor_mean: Vec<f64>,
    pub taylor_covariance: Vec<Vec<f64>>,
}

impl ComparisonResult {
    /// Compute relative error in mean
    pub fn mean_error(&self) -> Vec<f64> {
        self.ut_mean
            .iter()
            .zip(&self.taylor_mean)
            .map(|(ut, taylor)| {
                if taylor.abs() > 1e-10 {
                    (ut - taylor).abs() / taylor.abs()
                } else {
                    (ut - taylor).abs()
                }
            })
            .collect()
    }

    /// Compute relative error in variance (diagonal elements)
    pub fn variance_error(&self) -> Vec<f64> {
        (0..self.ut_mean.len())
            .map(|i| {
                let ut_var = self.ut_covariance[i][i];
                let taylor_var = self.taylor_covariance[i][i];

                if taylor_var.abs() > 1e-10 {
                    (ut_var - taylor_var).abs() / taylor_var.abs()
                } else {
                    (ut_var - taylor_var).abs()
                }
            })
            .collect()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigma_point_generation() {
        let config = UTConfig::default();
        let mean = vec![0.0, 0.0];
        let cov = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let ut = UnscentedTransform::new(config, mean, cov);

        // Should have 2n+1 = 5 points for 2D
        assert_eq!(ut.sigma_points.len(), 5);

        // Mean weights should sum to 1
        let sum_mean: f64 = ut.weights_mean.iter().sum();
        assert!((sum_mean - 1.0).abs() < 1e-10);

        // Covariance weights don't necessarily sum to 1 (due to β parameter)
        // They incorporate higher-order moment information
        let sum_cov: f64 = ut.weights_cov.iter().sum();
        assert!(sum_cov > 0.0); // Just check they're positive
    }

    #[test]
    fn test_linear_propagation() {
        let config = UTConfig::default();
        let mean = vec![1.0, 2.0];
        let cov = vec![vec![0.1, 0.0], vec![0.0, 0.2]];

        let ut = UnscentedTransform::new(config, mean, cov);

        // Linear transformation: y = 2x₁ + x₂
        let result = ut.propagate(|x| vec![2.0 * x[0] + x[1]]);

        // Mean should be exact for linear
        let expected_mean = 2.0 * 1.0 + 2.0;
        assert!((result.mean[0] - expected_mean).abs() < 1e-6);

        // Variance: Var[2x₁ + x₂] = 4·Var[x₁] + Var[x₂] = 4·0.1 + 0.2 = 0.6
        let expected_var = 4.0 * 0.1 + 0.2;
        assert!((result.covariance[0][0] - expected_var).abs() < 1e-6);
    }

    #[test]
    fn test_nonlinear_propagation() {
        let config = UTConfig::default();
        let mean = vec![1.0];
        let cov = vec![vec![0.01]]; // Small variance

        let ut = UnscentedTransform::new(config, mean, cov);

        // Nonlinear: y = x²
        let result = ut.propagate(|x| vec![x[0] * x[0]]);

        // Mean should be approximately 1.0² = 1.0 (for small variance)
        assert!((result.mean[0] - 1.0).abs() < 0.05);

        // UT captures nonlinearity better than Taylor
    }

    #[test]
    fn test_multivariate_propagation() {
        let config = UTConfig::default();
        let mean = vec![1.0, 2.0];
        let cov = vec![vec![0.1, 0.05], vec![0.05, 0.2]];

        let ut = UnscentedTransform::new(config, mean, cov);

        // Nonlinear: y₁ = x₁·x₂, y₂ = x₁² + x₂²
        let result = ut.propagate(|x| vec![x[0] * x[1], x[0] * x[0] + x[1] * x[1]]);

        // Check dimensions
        assert_eq!(result.mean.len(), 2);
        assert_eq!(result.covariance.len(), 2);
        assert_eq!(result.covariance[0].len(), 2);
    }

    #[test]
    fn test_cholesky_decomposition() {
        let matrix = vec![vec![4.0, 2.0], vec![2.0, 3.0]];

        let l = cholesky_decomposition(&matrix);

        // Verify LLᵀ = A
        let mut reconstructed = vec![vec![0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    reconstructed[i][j] += l[i][k] * l[j][k];
                }
            }
        }

        for i in 0..2 {
            for j in 0..2 {
                assert!((reconstructed[i][j] - matrix[i][j]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_ut_vs_taylor() {
        let config = UTConfig::default();
        let mean = vec![0.0];
        let cov = vec![vec![0.1]];

        let ut = UnscentedTransform::new(config, mean, cov);

        // Highly nonlinear: f(x) = sin(x)
        let f = |x: &[f64]| vec![libm::sin(x[0])];

        // Jacobian at x=0: df/dx = cos(0) = 1
        let jacobian = vec![vec![1.0]];

        let comparison = compare_with_taylor(&ut, &f, &jacobian);

        // For small variance and x near 0, both should be similar
        // But UT should capture higher-order effects
        assert!((comparison.ut_mean[0] - comparison.taylor_mean[0]).abs() < 0.01);
    }

    #[test]
    fn test_cross_covariance() {
        let config = UTConfig::default();
        let mean = vec![1.0, 2.0];
        let cov = vec![vec![0.1, 0.0], vec![0.0, 0.2]];

        let ut = UnscentedTransform::new(config, mean, cov);

        let result = ut.propagate_with_cross(|x| vec![x[0] + x[1]]);

        // Cross-covariance should have correct dimensions (2x1)
        assert_eq!(result.cross_covariance.len(), 2);
        assert_eq!(result.cross_covariance[0].len(), 1);
    }
}
