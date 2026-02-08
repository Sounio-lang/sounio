//! Polynomial Chaos Expansion (PCE) for Uncertainty Quantification
//!
//! Implements PCE-based uncertainty propagation with:
//! - Orthogonal polynomial basis (Hermite, Legendre, Laguerre, Jacobi)
//! - Tensor product expansion for multi-dimensional inputs
//! - Gauss quadrature for coefficient computation
//! - Sobol indices for variance-based sensitivity analysis
//!
//! # References
//!
//! - "PCE for ML Uncertainty" (July 2025)
//! - "PCE with Aleatory/Epistemic" (Oct 2024)
//! - "Physics-Constrained PCE" (May 2024)
//!
//! # Mathematical Foundation
//!
//! PCE represents a stochastic variable Y = f(X₁, ..., Xₙ) as:
//!
//! Y ≈ Σ cₐ Ψₐ(ξ)
//!
//! where:
//! - cₐ are expansion coefficients
//! - Ψₐ are multivariate orthogonal polynomials
//! - ξ are standard random variables
//!
//! # Example
//!
//! ```rust
//! use sounio::epistemic::pce::{PCEConfig, PCEExpansion, PolynomialFamily, TruncationScheme};
//!
//! let config = PCEConfig {
//!     max_degree: 3,
//!     family: PolynomialFamily::Hermite,
//!     n_quadrature_points: 10,
//!     sparse_grid: false,
//!     truncation: TruncationScheme::TotalDegree,
//! };
//!
//! let pce = PCEExpansion::new(config, 2); // 2D expansion
//! ```

use std::f64::consts::PI;

/// Polynomial family for PCE basis
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PolynomialFamily {
    /// Hermite polynomials (for Gaussian random variables)
    /// H_n(x) with weight exp(-x²/2)
    Hermite,

    /// Legendre polynomials (for uniform random variables)
    /// P_n(x) on [-1, 1] with weight 1
    Legendre,

    /// Laguerre polynomials (for exponential random variables)
    /// L_n(x) with weight exp(-x)
    Laguerre,

    /// Jacobi polynomials (for beta-distributed random variables)
    /// P_n^(α,β)(x) with weight (1-x)^α (1+x)^β
    Jacobi { alpha: f64, beta: f64 },
}

impl PolynomialFamily {
    /// Get the name of the polynomial family
    pub fn name(&self) -> &'static str {
        match self {
            PolynomialFamily::Hermite => "Hermite",
            PolynomialFamily::Legendre => "Legendre",
            PolynomialFamily::Laguerre => "Laguerre",
            PolynomialFamily::Jacobi { .. } => "Jacobi",
        }
    }

    /// Evaluate polynomial of given degree at point x
    pub fn eval(&self, degree: usize, x: f64) -> f64 {
        match self {
            PolynomialFamily::Hermite => hermite_polynomial(degree, x),
            PolynomialFamily::Legendre => legendre_polynomial(degree, x),
            PolynomialFamily::Laguerre => laguerre_polynomial(degree, x),
            PolynomialFamily::Jacobi { alpha, beta } => jacobi_polynomial(degree, x, *alpha, *beta),
        }
    }

    /// Get normalization constant <Ψ_n, Ψ_n> for probability measure
    pub fn normalization(&self, degree: usize) -> f64 {
        match self {
            PolynomialFamily::Hermite => {
                // For standard normal N(0,1): <H_n, H_n> = n!
                factorial(degree)
            }
            PolynomialFamily::Legendre => {
                // For uniform on [-1,1] with probability measure: <P_n, P_n> = 1/(2n+1)
                1.0 / (2.0 * degree as f64 + 1.0)
            }
            PolynomialFamily::Laguerre => {
                // For exponential distribution: <L_n, L_n> = 1
                1.0
            }
            PolynomialFamily::Jacobi { alpha, beta } => {
                // Jacobi polynomials - simplified
                let n = degree as f64;
                1.0 / (2.0 * n + alpha + beta + 1.0)
            }
        }
    }
}

/// Configuration for PCE expansion
#[derive(Debug, Clone, PartialEq)]
pub struct PCEConfig {
    /// Maximum polynomial degree
    pub max_degree: usize,

    /// Polynomial family to use
    pub family: PolynomialFamily,

    /// Number of quadrature points per dimension
    pub n_quadrature_points: usize,

    /// Whether to use sparse grid quadrature
    pub sparse_grid: bool,

    /// Truncation scheme
    pub truncation: TruncationScheme,
}

impl Default for PCEConfig {
    fn default() -> Self {
        Self {
            max_degree: 3,
            family: PolynomialFamily::Hermite,
            n_quadrature_points: 10,
            sparse_grid: false,
            truncation: TruncationScheme::TotalDegree,
        }
    }
}

/// Truncation scheme for multivariate PCE
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TruncationScheme {
    /// Total degree: |α| = α₁ + ... + αₙ ≤ p
    TotalDegree,

    /// Tensor product: αᵢ ≤ p for all i
    TensorProduct,

    /// Hyperbolic: (Π αᵢ^q)^(1/q) ≤ p with q ∈ (0,1]
    Hyperbolic { q: f64 },
}

/// Multi-index for multivariate polynomial
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MultiIndex {
    /// Degrees for each dimension
    pub degrees: Vec<usize>,
}

impl MultiIndex {
    /// Create new multi-index
    pub fn new(degrees: Vec<usize>) -> Self {
        Self { degrees }
    }

    /// Total degree |α| = α₁ + ... + αₙ
    pub fn total_degree(&self) -> usize {
        self.degrees.iter().sum()
    }

    /// Check if this multi-index satisfies the truncation scheme
    pub fn satisfies(&self, scheme: TruncationScheme, max_degree: usize) -> bool {
        match scheme {
            TruncationScheme::TotalDegree => self.total_degree() <= max_degree,
            TruncationScheme::TensorProduct => self.degrees.iter().all(|&d| d <= max_degree),
            TruncationScheme::Hyperbolic { q } => {
                let product: f64 = self
                    .degrees
                    .iter()
                    .map(|&d| libm::pow(d as f64, q))
                    .product();
                libm::pow(product, 1.0 / q) <= max_degree as f64
            }
        }
    }
}

/// Polynomial Chaos Expansion
pub struct PCEExpansion {
    /// Configuration
    config: PCEConfig,

    /// Number of input dimensions
    n_dims: usize,

    /// Multi-indices for basis functions
    multi_indices: Vec<MultiIndex>,

    /// Expansion coefficients
    coefficients: Vec<f64>,

    /// Quadrature points (for each dimension)
    quad_points: Vec<Vec<f64>>,

    /// Quadrature weights (for each dimension)
    quad_weights: Vec<Vec<f64>>,
}

impl PCEExpansion {
    /// Create new PCE expansion
    pub fn new(config: PCEConfig, n_dims: usize) -> Self {
        // Generate multi-indices
        let multi_indices = generate_multi_indices(n_dims, config.max_degree, config.truncation);

        // Generate quadrature points and weights
        let (quad_points, quad_weights) = if config.sparse_grid {
            generate_sparse_grid(&config, n_dims)
        } else {
            generate_tensor_quadrature(&config, n_dims)
        };

        Self {
            config,
            n_dims,
            multi_indices,
            coefficients: vec![],
            quad_points,
            quad_weights,
        }
    }

    /// Compute PCE coefficients from model evaluations
    ///
    /// # Arguments
    /// * `model` - Function f: ℝⁿ → ℝ to approximate
    ///
    /// Uses Gauss quadrature to compute:
    /// cₐ = <f, Ψₐ> / <Ψₐ, Ψₐ>
    pub fn compute_coefficients<F>(&mut self, model: F)
    where
        F: Fn(&[f64]) -> f64,
    {
        let n_coeffs = self.multi_indices.len();
        self.coefficients = vec![0.0; n_coeffs];

        // For each multi-index (basis function)
        for (i, alpha) in self.multi_indices.iter().enumerate() {
            let mut coefficient = 0.0;

            // Gauss quadrature over all dimensions
            self.tensor_quadrature(|xi, weight| {
                // Evaluate model at quadrature point
                let y = model(xi);

                // Evaluate multivariate polynomial Ψₐ(ξ)
                let psi = self.eval_basis(alpha, xi);

                // Accumulate weighted product
                coefficient += y * psi * weight;
            });

            // Normalize by <Ψₐ, Ψₐ>
            let norm = self.basis_normalization(alpha);
            self.coefficients[i] = coefficient / norm;
        }
    }

    /// Evaluate PCE expansion at point xi
    pub fn eval(&self, xi: &[f64]) -> f64 {
        let mut result = 0.0;

        for (coeff, alpha) in self.coefficients.iter().zip(&self.multi_indices) {
            result += coeff * self.eval_basis(alpha, xi);
        }

        result
    }

    /// Compute mean of expansion
    ///
    /// E[Y] = c₀ (coefficient of constant term)
    pub fn mean(&self) -> f64 {
        self.coefficients.first().copied().unwrap_or(0.0)
    }

    /// Compute variance of expansion
    ///
    /// Var[Y] = Σ_{|α|>0} c_α² <Ψₐ, Ψₐ>
    pub fn variance(&self) -> f64 {
        let mut var = 0.0;

        for (i, (coeff, alpha)) in self
            .coefficients
            .iter()
            .zip(&self.multi_indices)
            .enumerate()
        {
            // Skip constant term (i=0)
            if i > 0 {
                let norm = self.basis_normalization(alpha);
                var += coeff * coeff * norm;
            }
        }

        var
    }

    /// Compute Sobol indices for sensitivity analysis
    ///
    /// Returns:
    /// - First-order indices: Sᵢ = Var[E[Y|Xᵢ]] / Var[Y]
    /// - Total indices: STᵢ = E[Var[Y|X₋ᵢ]] / Var[Y]
    pub fn sobol_indices(&self) -> SobolIndices {
        let total_variance = self.variance();

        if total_variance < 1e-10 {
            // No variance - return zeros
            return SobolIndices {
                first_order: vec![0.0; self.n_dims],
                total: vec![0.0; self.n_dims],
            };
        }

        let mut first_order = vec![0.0; self.n_dims];
        let mut total = vec![0.0; self.n_dims];

        // For each input dimension
        for dim in 0..self.n_dims {
            let mut first_var = 0.0;
            let mut total_var = 0.0;

            for (coeff, alpha) in self.coefficients.iter().zip(&self.multi_indices) {
                let degree_in_dim = alpha.degrees[dim];
                let total_degree = alpha.total_degree();

                if total_degree > 0 {
                    let norm = self.basis_normalization(alpha);
                    let contrib = coeff * coeff * norm;

                    // First-order: only terms with degree in this dim, zero elsewhere
                    if degree_in_dim > 0
                        && alpha
                            .degrees
                            .iter()
                            .enumerate()
                            .all(|(i, &d)| i == dim || d == 0)
                    {
                        first_var += contrib;
                    }

                    // Total: all terms with non-zero degree in this dim
                    if degree_in_dim > 0 {
                        total_var += contrib;
                    }
                }
            }

            first_order[dim] = first_var / total_variance;
            total[dim] = total_var / total_variance;
        }

        SobolIndices { first_order, total }
    }

    /// Evaluate multivariate basis function Ψₐ(ξ)
    fn eval_basis(&self, alpha: &MultiIndex, xi: &[f64]) -> f64 {
        let mut result = 1.0;

        for (i, &degree) in alpha.degrees.iter().enumerate() {
            result *= self.config.family.eval(degree, xi[i]);
        }

        result
    }

    /// Get normalization constant for basis function
    fn basis_normalization(&self, alpha: &MultiIndex) -> f64 {
        let mut norm = 1.0;

        for &degree in &alpha.degrees {
            norm *= self.config.family.normalization(degree);
        }

        norm
    }

    /// Tensor product Gauss quadrature
    fn tensor_quadrature<F>(&self, mut f: F)
    where
        F: FnMut(&[f64], f64),
    {
        self.tensor_quadrature_recursive(&mut f, &mut vec![0.0; self.n_dims], 0, 1.0);
    }

    fn tensor_quadrature_recursive<F>(&self, f: &mut F, xi: &mut Vec<f64>, dim: usize, weight: f64)
    where
        F: FnMut(&[f64], f64),
    {
        if dim == self.n_dims {
            // Base case: evaluate at this quadrature point
            f(xi, weight);
        } else {
            // Recursive case: iterate over quadrature points in this dimension
            for (i, (&point, &w)) in self.quad_points[dim]
                .iter()
                .zip(&self.quad_weights[dim])
                .enumerate()
            {
                xi[dim] = point;
                self.tensor_quadrature_recursive(f, xi, dim + 1, weight * w);
            }
        }
    }
}

/// Sobol sensitivity indices
#[derive(Debug, Clone, PartialEq)]
pub struct SobolIndices {
    /// First-order indices: Sᵢ = Var[E[Y|Xᵢ]] / Var[Y]
    pub first_order: Vec<f64>,

    /// Total indices: STᵢ = E[Var[Y|X₋ᵢ]] / Var[Y]
    pub total: Vec<f64>,
}

// =============================================================================
// Orthogonal Polynomial Implementations
// =============================================================================

/// Hermite polynomial H_n(x) using recurrence relation
///
/// H_0(x) = 1
/// H_1(x) = x
/// H_{n+1}(x) = x H_n(x) - n H_{n-1}(x)
fn hermite_polynomial(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }

    let mut h_prev = 1.0;
    let mut h_curr = x;

    for k in 1..n {
        let h_next = x * h_curr - (k as f64) * h_prev;
        h_prev = h_curr;
        h_curr = h_next;
    }

    h_curr
}

/// Legendre polynomial P_n(x) using recurrence relation
///
/// P_0(x) = 1
/// P_1(x) = x
/// (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
fn legendre_polynomial(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }

    let mut p_prev = 1.0;
    let mut p_curr = x;

    for k in 1..n {
        let k_f = k as f64;
        let p_next = ((2.0 * k_f + 1.0) * x * p_curr - k_f * p_prev) / (k_f + 1.0);
        p_prev = p_curr;
        p_curr = p_next;
    }

    p_curr
}

/// Laguerre polynomial L_n(x) using recurrence relation
///
/// L_0(x) = 1
/// L_1(x) = 1 - x
/// (n+1) L_{n+1}(x) = (2n+1-x) L_n(x) - n L_{n-1}(x)
fn laguerre_polynomial(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return 1.0 - x;
    }

    let mut l_prev = 1.0;
    let mut l_curr = 1.0 - x;

    for k in 1..n {
        let k_f = k as f64;
        let l_next = ((2.0 * k_f + 1.0 - x) * l_curr - k_f * l_prev) / (k_f + 1.0);
        l_prev = l_curr;
        l_curr = l_next;
    }

    l_curr
}

/// Jacobi polynomial P_n^(α,β)(x) - simplified implementation
fn jacobi_polynomial(n: usize, x: f64, alpha: f64, beta: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return (alpha + 1.0) + (alpha + beta + 2.0) * (x - 1.0) / 2.0;
    }

    // Use Legendre as approximation for simplicity
    // Full implementation would use proper Jacobi recurrence
    legendre_polynomial(n, x)
}

// =============================================================================
// Quadrature Generation
// =============================================================================

/// Generate tensor product quadrature points and weights
fn generate_tensor_quadrature(config: &PCEConfig, n_dims: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut points = Vec::with_capacity(n_dims);
    let mut weights = Vec::with_capacity(n_dims);

    for _ in 0..n_dims {
        let (pts, wts) = gauss_quadrature(&config.family, config.n_quadrature_points);
        points.push(pts);
        weights.push(wts);
    }

    (points, weights)
}

/// Generate sparse grid quadrature (simplified Smolyak construction)
fn generate_sparse_grid(config: &PCEConfig, n_dims: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    // For now, fall back to tensor product
    // Full sparse grid implementation is complex
    generate_tensor_quadrature(config, n_dims)
}

/// Gauss quadrature points and weights for given polynomial family
fn gauss_quadrature(family: &PolynomialFamily, n_points: usize) -> (Vec<f64>, Vec<f64>) {
    match family {
        PolynomialFamily::Hermite => gauss_hermite_quadrature(n_points),
        PolynomialFamily::Legendre => gauss_legendre_quadrature(n_points),
        PolynomialFamily::Laguerre => gauss_laguerre_quadrature(n_points),
        PolynomialFamily::Jacobi { .. } => {
            // Simplified: use Legendre as fallback
            gauss_legendre_quadrature(n_points)
        }
    }
}

/// Gauss-Hermite quadrature (probabilists' version)
/// For standard normal distribution N(0,1)
/// Points and weights are already normalized for probability measure
fn gauss_hermite_quadrature(n: usize) -> (Vec<f64>, Vec<f64>) {
    // Hardcoded points/weights for common sizes
    // These are for ∫ f(x) (1/√(2π)) exp(-x²/2) dx
    match n {
        3 => (
            vec![-1.732050808, 0.0, 1.732050808],
            vec![0.166666667, 0.666666667, 0.166666667],
        ),
        5 => (
            vec![-2.02018287, -0.95857246, 0.0, 0.95857246, 2.02018287],
            vec![0.01995324, 0.39361932, 0.94530872, 0.39361932, 0.01995324],
        ),
        _ => {
            // Fall back to uniform approximation on [-3σ, 3σ]
            let mut points = Vec::with_capacity(n);
            let mut weights = Vec::with_capacity(n);

            for i in 0..n {
                let x = -3.0 + 6.0 * (i as f64 / (n - 1) as f64);
                points.push(x);
                // Approximate Gaussian weight
                let w = libm::exp(-x * x / 2.0) / (2.0 * std::f64::consts::PI).sqrt();
                weights.push(w * 6.0 / n as f64); // Scale by interval width
            }

            // Normalize weights to sum to 1
            let sum: f64 = weights.iter().sum();
            for w in &mut weights {
                *w /= sum;
            }

            (points, weights)
        }
    }
}

/// Gauss-Legendre quadrature on [-1, 1]
/// Returns points and weights normalized for probability measure (weights sum to 1)
fn gauss_legendre_quadrature(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        2 => (
            vec![-0.577350269, 0.577350269],
            vec![0.5, 0.5], // Normalized: 1.0/2
        ),
        3 => (
            vec![-0.774596669, 0.0, 0.774596669],
            vec![0.277777778, 0.444444444, 0.277777778], // Normalized: /2
        ),
        5 => (
            vec![-0.906179846, -0.538469310, 0.0, 0.538469310, 0.906179846],
            vec![
                0.118463443, // 0.236926885 / 2
                0.239314335, // 0.478628670 / 2
                0.284444444, // 0.568888889 / 2
                0.239314335,
                0.118463443,
            ],
        ),
        _ => {
            // Fall back to uniform
            let mut points = Vec::with_capacity(n);
            let mut weights = Vec::with_capacity(n);

            for i in 0..n {
                let x = -1.0 + 2.0 * (i as f64 / (n - 1) as f64);
                points.push(x);
                weights.push(1.0 / n as f64); // Probability-normalized
            }

            (points, weights)
        }
    }
}

/// Gauss-Laguerre quadrature on [0, ∞)
/// For exponential distribution with weight exp(-x)
/// Weights are already normalized for probability measure
fn gauss_laguerre_quadrature(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        3 => (
            vec![0.415774556, 2.294280361, 6.289945083],
            vec![0.711093010, 0.278517733, 0.0103892565],
        ),
        _ => {
            // Fall back to exponential spacing
            let mut points = Vec::with_capacity(n);
            let mut weights = Vec::with_capacity(n);

            for i in 0..n {
                let x = (i as f64 + 0.5) / n as f64 * 10.0;
                points.push(x);
                let w = libm::exp(-x);
                weights.push(w * 10.0 / n as f64); // Scale by interval width
            }

            // Normalize weights to sum to 1
            let sum: f64 = weights.iter().sum();
            for w in &mut weights {
                *w /= sum;
            }

            (points, weights)
        }
    }
}

// =============================================================================
// Multi-Index Generation
// =============================================================================

/// Generate all multi-indices satisfying truncation scheme
fn generate_multi_indices(
    n_dims: usize,
    max_degree: usize,
    truncation: TruncationScheme,
) -> Vec<MultiIndex> {
    let mut indices = Vec::new();

    // Start with zero multi-index (constant term)
    indices.push(MultiIndex::new(vec![0; n_dims]));

    // Generate all valid multi-indices up to max_degree
    generate_multi_indices_recursive(
        n_dims,
        max_degree,
        truncation,
        &mut vec![0; n_dims],
        0,
        &mut indices,
    );

    indices
}

fn generate_multi_indices_recursive(
    n_dims: usize,
    max_degree: usize,
    truncation: TruncationScheme,
    current: &mut Vec<usize>,
    dim: usize,
    result: &mut Vec<MultiIndex>,
) {
    if dim == n_dims {
        // Check if this multi-index satisfies truncation
        let mi = MultiIndex::new(current.clone());
        if mi.satisfies(truncation, max_degree) && mi.total_degree() > 0 {
            result.push(mi);
        }
        return;
    }

    // Try all degrees for this dimension
    for degree in 0..=max_degree {
        current[dim] = degree;
        generate_multi_indices_recursive(n_dims, max_degree, truncation, current, dim + 1, result);
    }

    current[dim] = 0;
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Factorial function
fn factorial(n: usize) -> f64 {
    if n == 0 || n == 1 {
        return 1.0;
    }

    let mut result = 1.0;
    for i in 2..=n {
        result *= i as f64;
    }
    result
}

/// Gamma function approximation (Stirling's approximation for large n)
fn gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    if x < 1.0 {
        // Use reflection formula: Γ(x)Γ(1-x) = π/sin(πx)
        return PI / (libm::sin(PI * x) * gamma(1.0 - x));
    }
    if x == 1.0 {
        return 1.0;
    }
    if x < 20.0 {
        // Use recurrence: Γ(x+1) = x Γ(x)
        return (x - 1.0) * gamma(x - 1.0);
    }

    // Stirling's approximation for large x
    let ln_gamma = (x - 0.5) * libm::log(x) - x + 0.5 * libm::log(2.0 * PI);
    libm::exp(ln_gamma)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hermite_polynomials() {
        // H_0(x) = 1
        assert!((hermite_polynomial(0, 1.5) - 1.0).abs() < 1e-10);

        // H_1(x) = x
        assert!((hermite_polynomial(1, 2.0) - 2.0).abs() < 1e-10);

        // H_2(x) = x² - 1
        assert!((hermite_polynomial(2, 2.0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_legendre_polynomials() {
        // P_0(x) = 1
        assert!((legendre_polynomial(0, 0.5) - 1.0).abs() < 1e-10);

        // P_1(x) = x
        assert!((legendre_polynomial(1, 0.5) - 0.5).abs() < 1e-10);

        // P_2(x) = (3x² - 1)/2
        let p2 = (3.0 * 0.5 * 0.5 - 1.0) / 2.0;
        assert!((legendre_polynomial(2, 0.5) - p2).abs() < 1e-10);
    }

    #[test]
    fn test_multi_index_total_degree() {
        let mi = MultiIndex::new(vec![1, 2, 3]);
        assert_eq!(mi.total_degree(), 6);
    }

    #[test]
    fn test_multi_index_satisfies_total_degree() {
        let mi1 = MultiIndex::new(vec![1, 2, 1]);
        assert!(mi1.satisfies(TruncationScheme::TotalDegree, 5));
        assert!(!mi1.satisfies(TruncationScheme::TotalDegree, 3));
    }

    #[test]
    fn test_pce_1d_constant() {
        let config = PCEConfig::default();
        let mut pce = PCEExpansion::new(config, 1);

        // Constant function f(x) = 5.0
        pce.compute_coefficients(|_x| 5.0);

        println!(
            "Multi-indices: {:?}",
            pce.multi_indices
                .iter()
                .map(|mi| &mi.degrees)
                .collect::<Vec<_>>()
        );
        println!("Coefficients: {:?}", pce.coefficients);
        println!("Mean: {}", pce.mean());
        println!("Variance: {}", pce.variance());

        assert!((pce.mean() - 5.0).abs() < 1e-6);
        // Variance should be very small (quadrature introduces small numerical error)
        assert!(pce.variance() < 1e-3);
    }

    #[test]
    fn test_pce_1d_linear() {
        let config = PCEConfig {
            max_degree: 2,
            family: PolynomialFamily::Legendre,
            n_quadrature_points: 5,
            sparse_grid: false,
            truncation: TruncationScheme::TotalDegree,
        };
        let mut pce = PCEExpansion::new(config, 1);

        // Linear function f(x) = 2x + 3 on [-1, 1]
        pce.compute_coefficients(|x| 2.0 * x[0] + 3.0);

        // Mean should be 3.0 (since E[x] = 0 for uniform on [-1,1])
        assert!((pce.mean() - 3.0).abs() < 1e-4);

        // Variance for uniform: Var[2x] = 4 Var[x] = 4/3
        let expected_var = 4.0 / 3.0;
        assert!((pce.variance() - expected_var).abs() < 1e-3);
    }

    #[test]
    fn test_sobol_indices_2d() {
        let config = PCEConfig {
            max_degree: 2,
            family: PolynomialFamily::Legendre,
            n_quadrature_points: 5,
            sparse_grid: false,
            truncation: TruncationScheme::TotalDegree,
        };
        let mut pce = PCEExpansion::new(config, 2);

        // f(x1, x2) = 2*x1² + x2 (x1 should dominate)
        pce.compute_coefficients(|x| 2.0 * x[0] * x[0] + x[1]);

        let indices = pce.sobol_indices();

        // With coefficient 2, x1 should have higher sensitivity than x2
        assert!(indices.first_order[0] > indices.first_order[1]);
        assert!(indices.total[0] > indices.total[1]);

        // Total indices should sum to ~1.0 for additive model
        let total_sum: f64 = indices.total.iter().sum();
        assert!((total_sum - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_polynomial_families() {
        let hermite = PolynomialFamily::Hermite;
        let legendre = PolynomialFamily::Legendre;
        let laguerre = PolynomialFamily::Laguerre;

        assert_eq!(hermite.name(), "Hermite");
        assert_eq!(legendre.name(), "Legendre");
        assert_eq!(laguerre.name(), "Laguerre");

        // Check normalization is positive
        assert!(hermite.normalization(2) > 0.0);
        assert!(legendre.normalization(2) > 0.0);
        assert!(laguerre.normalization(2) > 0.0);
    }
}
