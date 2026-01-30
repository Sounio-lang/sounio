//! Gaussian Process Regression for Epistemic Uncertainty Quantification
//!
//! Implements Gaussian Processes (GPs) for surrogate modeling and uncertainty
//! quantification in epistemic computing.
//!
//! # Overview
//!
//! Gaussian Processes provide a Bayesian non-parametric framework for modeling
//! unknown functions with quantified uncertainty. They are ideal for:
//! - Surrogate modeling of expensive `Knowledge<T>` computations
//! - Uncertainty quantification in scientific simulations
//! - Multi-fidelity fusion (combining cheap and expensive models)
//! - Adaptive sampling for experimental design
//!
//! # Mathematical Background
//!
//! A Gaussian Process is fully specified by:
//! - Mean function: μ(x) (typically 0)
//! - Covariance/kernel function: k(x, x')
//!
//! Given training data (X, y), predictions at new points x* follow:
//!
//! ```text
//! f(x*) ~ N(μ*, σ²*)
//! μ* = k(x*, X) [K + σ²I]⁻¹ y
//! σ²* = k(x*, x*) - k(x*, X) [K + σ²I]⁻¹ k(X, x*)
//! ```
//!
//! where K is the kernel matrix K_ij = k(x_i, x_j).
//!
//! # Example
//!
//! ```rust
//! use sounio::epistemic::gaussian_process::{GaussianProcess, GPConfig, Kernel};
//!
//! // Training data
//! let x_train = vec![vec![0.0], vec![1.0], vec![2.0]];
//! let y_train = vec![0.0, 1.0, 0.5];
//!
//! // Configure GP with RBF kernel
//! let config = GPConfig::default();
//! let mut gp = GaussianProcess::new(config);
//! gp.fit(&x_train, &y_train);
//!
//! // Predict with uncertainty
//! let x_test = vec![vec![1.5]];
//! let predictions = gp.predict(&x_test);
//! println!("Mean: {}, Std: {}", predictions.mean[0], predictions.std_dev[0]);
//! ```
//!
//! # References
//!
//! - Rasmussen & Williams (2006): "Gaussian Processes for Machine Learning"
//! - Titsias (2009): "Variational Learning of Inducing Variables in Sparse GPs"
//! - Snelson & Ghahramani (2006): "Sparse Gaussian Processes using Pseudo-inputs"

use std::f64::consts::PI;

/// Configuration for Gaussian Process
#[derive(Debug, Clone, PartialEq)]
pub struct GPConfig {
    /// Kernel function
    pub kernel: Kernel,

    /// Observation noise variance
    pub noise_variance: f64,

    /// Whether to optimize hyperparameters
    pub optimize_hyperparameters: bool,

    /// Maximum iterations for optimization
    pub max_iterations: usize,

    /// Learning rate for gradient-based optimization
    pub learning_rate: f64,

    /// Use sparse approximation with inducing points
    pub use_sparse: bool,

    /// Number of inducing points for sparse GP
    pub n_inducing: usize,

    /// Convergence tolerance for optimization
    pub tolerance: f64,
}

impl Default for GPConfig {
    fn default() -> Self {
        Self {
            kernel: Kernel::rbf(1.0, 1.0), // length_scale=1, variance=1
            noise_variance: 1e-4,
            optimize_hyperparameters: true,
            max_iterations: 100,
            learning_rate: 0.01,
            use_sparse: false,
            n_inducing: 50,
            tolerance: 1e-6,
        }
    }
}

/// Kernel function for Gaussian Process
#[derive(Debug, Clone, PartialEq)]
pub enum Kernel {
    /// Radial Basis Function (RBF) / Squared Exponential kernel
    ///
    /// k(x, x') = σ² exp(-||x - x'||² / (2ℓ²))
    ///
    /// - ℓ: length scale (controls smoothness)
    /// - σ²: signal variance (output scale)
    RBF { length_scale: f64, variance: f64 },

    /// Matérn kernel (ν=3/2)
    ///
    /// k(x, x') = σ² (1 + √3 r/ℓ) exp(-√3 r/ℓ)
    /// where r = ||x - x'||
    ///
    /// Less smooth than RBF, allows for non-differentiable functions
    Matern32 { length_scale: f64, variance: f64 },

    /// Matérn kernel (ν=5/2)
    ///
    /// k(x, x') = σ² (1 + √5 r/ℓ + 5r²/(3ℓ²)) exp(-√5 r/ℓ)
    ///
    /// Twice differentiable, smoother than ν=3/2
    Matern52 { length_scale: f64, variance: f64 },

    /// Periodic kernel
    ///
    /// k(x, x') = σ² exp(-2 sin²(π|x-x'|/p) / ℓ²)
    ///
    /// - p: period
    /// - ℓ: length scale
    Periodic {
        length_scale: f64,
        period: f64,
        variance: f64,
    },

    /// Linear kernel
    ///
    /// k(x, x') = σ² (x · x')
    Linear { variance: f64 },

    /// Sum of two kernels
    Sum {
        kernel1: Box<Kernel>,
        kernel2: Box<Kernel>,
    },

    /// Product of two kernels
    Product {
        kernel1: Box<Kernel>,
        kernel2: Box<Kernel>,
    },
}

impl Kernel {
    /// Create RBF kernel
    pub fn rbf(length_scale: f64, variance: f64) -> Self {
        Kernel::RBF {
            length_scale,
            variance,
        }
    }

    /// Create Matérn-3/2 kernel
    pub fn matern32(length_scale: f64, variance: f64) -> Self {
        Kernel::Matern32 {
            length_scale,
            variance,
        }
    }

    /// Create Matérn-5/2 kernel
    pub fn matern52(length_scale: f64, variance: f64) -> Self {
        Kernel::Matern52 {
            length_scale,
            variance,
        }
    }

    /// Create periodic kernel
    pub fn periodic(length_scale: f64, period: f64, variance: f64) -> Self {
        Kernel::Periodic {
            length_scale,
            period,
            variance,
        }
    }

    /// Evaluate kernel function k(x, x')
    pub fn eval(&self, x: &[f64], x_prime: &[f64]) -> f64 {
        match self {
            Kernel::RBF {
                length_scale,
                variance,
            } => {
                let r_squared = squared_distance(x, x_prime);
                variance * libm::exp(-r_squared / (2.0 * length_scale * length_scale))
            }

            Kernel::Matern32 {
                length_scale,
                variance,
            } => {
                let r = libm::sqrt(squared_distance(x, x_prime));
                let sqrt3_r_l = 1.732050808 * r / length_scale; // √3 ≈ 1.732
                variance * (1.0 + sqrt3_r_l) * libm::exp(-sqrt3_r_l)
            }

            Kernel::Matern52 {
                length_scale,
                variance,
            } => {
                let r = libm::sqrt(squared_distance(x, x_prime));
                let sqrt5_r_l = 2.236067977 * r / length_scale; // √5 ≈ 2.236
                let term1 = 1.0 + sqrt5_r_l;
                let term2 = 5.0 * r * r / (3.0 * length_scale * length_scale);
                variance * (term1 + term2) * libm::exp(-sqrt5_r_l)
            }

            Kernel::Periodic {
                length_scale,
                period,
                variance,
            } => {
                let diff = (x[0] - x_prime[0]).abs(); // Assumes 1D for simplicity
                let sin_term = libm::sin(PI * diff / period);
                variance * libm::exp(-2.0 * sin_term * sin_term / (length_scale * length_scale))
            }

            Kernel::Linear { variance } => {
                let dot_product: f64 = x.iter().zip(x_prime.iter()).map(|(a, b)| a * b).sum();
                variance * dot_product
            }

            Kernel::Sum { kernel1, kernel2 } => kernel1.eval(x, x_prime) + kernel2.eval(x, x_prime),

            Kernel::Product { kernel1, kernel2 } => {
                kernel1.eval(x, x_prime) * kernel2.eval(x, x_prime)
            }
        }
    }

    /// Compute kernel matrix K for training data
    pub fn kernel_matrix(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = x.len();
        let mut k = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..=i {
                // Symmetric, only compute lower triangle
                let k_ij = self.eval(&x[i], &x[j]);
                k[i][j] = k_ij;
                k[j][i] = k_ij;
            }
        }

        k
    }

    /// Compute cross-covariance k(x*, X)
    pub fn cross_covariance(&self, x_star: &[Vec<f64>], x_train: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n_star = x_star.len();
        let n_train = x_train.len();
        let mut k_star = vec![vec![0.0; n_train]; n_star];

        for i in 0..n_star {
            for j in 0..n_train {
                k_star[i][j] = self.eval(&x_star[i], &x_train[j]);
            }
        }

        k_star
    }
}

/// Gaussian Process regressor
pub struct GaussianProcess {
    config: GPConfig,
    x_train: Option<Vec<Vec<f64>>>,
    y_train: Option<Vec<f64>>,
    alpha: Option<Vec<f64>>,      // (K + σ²I)⁻¹ y
    k_inv: Option<Vec<Vec<f64>>>, // (K + σ²I)⁻¹
    log_marginal_likelihood: f64,
}

impl GaussianProcess {
    /// Create new Gaussian Process
    pub fn new(config: GPConfig) -> Self {
        Self {
            config,
            x_train: None,
            y_train: None,
            alpha: None,
            k_inv: None,
            log_marginal_likelihood: f64::NEG_INFINITY,
        }
    }

    /// Fit GP to training data
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        assert_eq!(x.len(), y.len(), "X and y must have same length");

        self.x_train = Some(x.to_vec());
        self.y_train = Some(y.to_vec());

        if self.config.optimize_hyperparameters {
            self.optimize_hyperparameters();
        }

        self.compute_alpha();
        self.log_marginal_likelihood = self.compute_log_marginal_likelihood();
    }

    /// Predict at new points
    pub fn predict(&self, x_star: &[Vec<f64>]) -> GPPrediction {
        assert!(self.x_train.is_some(), "Must fit GP before predicting");

        let x_train = self.x_train.as_ref().unwrap();
        let alpha = self.alpha.as_ref().unwrap();

        // Compute k(x*, X)
        let k_star = self.config.kernel.cross_covariance(x_star, x_train);

        // Mean: μ* = k(x*, X) α
        let mut mean = vec![0.0; x_star.len()];
        for i in 0..x_star.len() {
            for j in 0..x_train.len() {
                mean[i] += k_star[i][j] * alpha[j];
            }
        }

        // Variance: σ²* = k(x*, x*) - k(x*, X) K⁻¹ k(X, x*)
        let mut variance = vec![0.0; x_star.len()];
        for i in 0..x_star.len() {
            // Prior variance
            let k_star_star = self.config.kernel.eval(&x_star[i], &x_star[i]);

            // Compute v = K⁻¹ k(X, x*)
            let k_inv = self.k_inv.as_ref().unwrap();
            let mut v = vec![0.0; x_train.len()];
            for j in 0..x_train.len() {
                for k in 0..x_train.len() {
                    v[j] += k_inv[j][k] * k_star[i][k];
                }
            }

            // Posterior variance reduction
            let var_reduction: f64 = k_star[i].iter().zip(&v).map(|(a, b)| a * b).sum();
            variance[i] = k_star_star - var_reduction;
        }

        let std_dev: Vec<f64> = variance.iter().map(|&v| libm::sqrt(v.max(0.0))).collect();

        GPPrediction {
            mean,
            variance,
            std_dev,
        }
    }

    /// Compute α = (K + σ²I)⁻¹ y
    fn compute_alpha(&mut self) {
        let x_train = self.x_train.as_ref().unwrap();
        let y_train = self.y_train.as_ref().unwrap();
        let n = x_train.len();

        // Compute kernel matrix
        let mut k = self.config.kernel.kernel_matrix(x_train);

        // Add noise: K + σ²I
        for i in 0..n {
            k[i][i] += self.config.noise_variance;
        }

        // Cholesky decomposition: K = L Lᵀ
        let l = cholesky_decomposition(&k);

        // Solve L α' = y
        let alpha_prime = forward_substitution(&l, y_train);

        // Solve Lᵀ α = α'
        let alpha = backward_substitution(&transpose(&l), &alpha_prime);

        // Also store K⁻¹ for variance computation
        let k_inv = cholesky_inverse(&l);

        self.alpha = Some(alpha);
        self.k_inv = Some(k_inv);
    }

    /// Compute log marginal likelihood for hyperparameter optimization
    fn compute_log_marginal_likelihood(&self) -> f64 {
        let x_train = self.x_train.as_ref().unwrap();
        let y_train = self.y_train.as_ref().unwrap();
        let n = x_train.len();

        // Kernel matrix with noise
        let mut k = self.config.kernel.kernel_matrix(x_train);
        for i in 0..n {
            k[i][i] += self.config.noise_variance;
        }

        // Cholesky decomposition
        let l = cholesky_decomposition(&k);

        // Log determinant: log |K| = 2 Σ log L_ii
        let log_det: f64 = l
            .iter()
            .map(|row| libm::log(row[row.len() - 1]))
            .sum::<f64>()
            * 2.0;

        // Data fit term: yᵀ K⁻¹ y = αᵀ y
        let alpha = self.alpha.as_ref().unwrap();
        let data_fit: f64 = y_train.iter().zip(alpha.iter()).map(|(y, a)| y * a).sum();

        // Log marginal likelihood: -0.5 (yᵀ K⁻¹ y + log|K| + n log(2π))
        -0.5 * (data_fit + log_det + n as f64 * libm::log(2.0 * PI))
    }

    /// Optimize hyperparameters by maximizing log marginal likelihood
    fn optimize_hyperparameters(&mut self) {
        // Simple gradient ascent on length_scale (demonstration)
        // Full implementation would optimize all kernel hyperparameters

        let mut best_lml = f64::NEG_INFINITY;
        let mut best_length_scale = match self.config.kernel {
            Kernel::RBF { length_scale, .. } => length_scale,
            _ => 1.0,
        };

        // Grid search over length scales (simplified)
        let length_scales = vec![0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0];

        for &ls in &length_scales {
            // Update kernel
            match &mut self.config.kernel {
                Kernel::RBF { length_scale, .. } => *length_scale = ls,
                _ => {}
            }

            // Recompute alpha
            self.compute_alpha();

            // Evaluate log marginal likelihood
            let lml = self.compute_log_marginal_likelihood();

            if lml > best_lml {
                best_lml = lml;
                best_length_scale = ls;
            }
        }

        // Set best hyperparameters
        match &mut self.config.kernel {
            Kernel::RBF { length_scale, .. } => *length_scale = best_length_scale,
            _ => {}
        }
    }
}

/// Prediction result from Gaussian Process
#[derive(Debug, Clone)]
pub struct GPPrediction {
    /// Predictive mean
    pub mean: Vec<f64>,

    /// Predictive variance
    pub variance: Vec<f64>,

    /// Predictive standard deviation
    pub std_dev: Vec<f64>,
}

impl GPPrediction {
    /// Get 95% confidence interval
    pub fn confidence_interval_95(&self) -> Vec<(f64, f64)> {
        self.mean
            .iter()
            .zip(&self.std_dev)
            .map(|(m, s)| (m - 1.96 * s, m + 1.96 * s))
            .collect()
    }
}

// =============================================================================
// Linear Algebra Utilities
// =============================================================================

/// Compute squared Euclidean distance
fn squared_distance(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(a, b)| (a - b) * (a - b)).sum()
}

/// Cholesky decomposition: A = L Lᵀ
fn cholesky_decomposition(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }

            if i == j {
                l[i][j] = libm::sqrt((a[i][i] - sum).max(1e-10));
            } else {
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }

    l
}

/// Forward substitution: solve L x = b
fn forward_substitution(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut x = vec![0.0; n];

    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / l[i][i];
    }

    x
}

/// Backward substitution: solve Uᵀ x = b
fn backward_substitution(u: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = u.len();
    let mut x = vec![0.0; n];

    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += u[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / u[i][i];
    }

    x
}

/// Matrix transpose
fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    let mut at = vec![vec![0.0; n]; m];

    for i in 0..n {
        for j in 0..m {
            at[j][i] = a[i][j];
        }
    }

    at
}

/// Compute matrix inverse from Cholesky factor: A⁻¹ where A = L Lᵀ
fn cholesky_inverse(l: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = l.len();

    // Compute L⁻¹
    let mut l_inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        l_inv[i][i] = 1.0 / l[i][i];
        for j in (i + 1)..n {
            let mut sum = 0.0;
            for k in i..j {
                sum += l[j][k] * l_inv[k][i];
            }
            l_inv[j][i] = -sum / l[j][j];
        }
    }

    // Compute A⁻¹ = L⁻ᵀ L⁻¹
    let mut a_inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0.max(i).max(j)..n {
                a_inv[i][j] += l_inv[k][i] * l_inv[k][j];
            }
        }
    }

    a_inv
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rbf_kernel() {
        let kernel = Kernel::rbf(1.0, 1.0);

        // Same point should give variance
        let k_same = kernel.eval(&[0.0], &[0.0]);
        assert!((k_same - 1.0).abs() < 1e-10);

        // Different points should decay with distance
        let k_diff = kernel.eval(&[0.0], &[1.0]);
        assert!(k_diff < 1.0);
        assert!(k_diff > 0.0);
    }

    #[test]
    fn test_gp_fit_predict() {
        // Simple 1D regression
        let x_train = vec![vec![0.0], vec![1.0], vec![2.0]];
        let y_train = vec![0.0, 1.0, 0.0]; // Quadratic-ish

        let config = GPConfig::default();
        let mut gp = GaussianProcess::new(config);
        gp.fit(&x_train, &y_train);

        // Predict at training points (should have low variance)
        let pred = gp.predict(&x_train);

        for (i, &y_true) in y_train.iter().enumerate() {
            assert!((pred.mean[i] - y_true).abs() < 0.1);
            assert!(pred.std_dev[i] < 0.5); // Low uncertainty at training points
        }
    }

    #[test]
    fn test_gp_interpolation() {
        // GP should interpolate between points
        let x_train = vec![vec![0.0], vec![2.0]];
        let y_train = vec![0.0, 2.0];

        let mut gp = GaussianProcess::new(GPConfig::default());
        gp.fit(&x_train, &y_train);

        // Predict at midpoint
        let x_test = vec![vec![1.0]];
        let pred = gp.predict(&x_test);

        // Should be approximately linear interpolation
        assert!((pred.mean[0] - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_cholesky_decomposition() {
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];

        let l = cholesky_decomposition(&a);

        // Verify L Lᵀ = A
        let l_t = transpose(&l);
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += l[i][k] * l_t[k][j];
                }
                assert!((sum - a[i][j]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_matern_kernels() {
        let k32 = Kernel::matern32(1.0, 1.0);
        let k52 = Kernel::matern52(1.0, 1.0);

        // Both should equal variance at same point
        assert!((k32.eval(&[0.0], &[0.0]) - 1.0).abs() < 1e-10);
        assert!((k52.eval(&[0.0], &[0.0]) - 1.0).abs() < 1e-10);

        // Matern-5/2 should be smoother (decay slower) than 3/2
        let r = 0.5;
        let v32 = k32.eval(&[0.0], &[r]);
        let v52 = k52.eval(&[0.0], &[r]);
        assert!(v52 > v32); // 5/2 is smoother
    }
}
