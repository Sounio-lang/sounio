//! Model Discovery Runtime (SINDy-like)
//!
//! This module provides data-driven model discovery for Sounio,
//! enabling automatic discovery of governing equations from data.
//!
//! # Core Operations
//!
//! - `discover(data, library)` - Discover sparse model from data
//! - `build_library(terms)` - Build function library for discovery
//! - `sparse_regression(X, y, threshold)` - Sparse regression (STLS)
//!
//! # Example
//!
//! ```d
//! // Discover Lorenz attractor dynamics
//! let library = build_library([
//!     poly(1), poly(2), poly(3),  // x, x², x³
//!     |x, y| x * y,               // interactions
//! ]);
//!
//! let model = discover(trajectory, library,
//!     sparsity: 0.1,
//!     threshold: 0.05
//! );
//!
//! print(model.symbolic_form());  // dx/dt = σ(y - x)
//! ```

use std::fmt;

/// Library term (basis function)
pub struct LibraryTerm {
    /// Name/description of the term
    pub name: String,
    /// Function to evaluate the term
    pub eval: Box<dyn Fn(&[f64]) -> f64 + Send + Sync>,
    /// Number of input variables
    pub n_vars: usize,
}

impl LibraryTerm {
    /// Create a new library term
    pub fn new<F>(name: &str, n_vars: usize, f: F) -> Self
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        Self {
            name: name.to_string(),
            eval: Box::new(f),
            n_vars,
        }
    }

    /// Evaluate the term
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        (self.eval)(x)
    }
}

impl fmt::Debug for LibraryTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LibraryTerm({})", self.name)
    }
}

/// Function library for model discovery
#[derive(Default)]
pub struct FunctionLibrary {
    terms: Vec<LibraryTerm>,
}

impl FunctionLibrary {
    /// Create a new empty library
    pub fn new() -> Self {
        Self { terms: Vec::new() }
    }

    /// Add a term to the library
    pub fn add_term(&mut self, term: LibraryTerm) {
        self.terms.push(term);
    }

    /// Add constant term (1)
    pub fn add_constant(&mut self) {
        self.add_term(LibraryTerm::new("1", 0, |_| 1.0));
    }

    /// Add polynomial terms up to degree n for each variable
    pub fn add_polynomials(&mut self, n_vars: usize, max_degree: usize) {
        // Individual variable polynomials
        for var in 0..n_vars {
            let var_name = format!("x{}", var);
            for deg in 1..=max_degree {
                let name = if deg == 1 {
                    var_name.clone()
                } else {
                    format!("{}^{}", var_name, deg)
                };
                self.add_term(LibraryTerm::new(&name, n_vars, move |x| {
                    x.get(var).copied().unwrap_or(0.0).powi(deg as i32)
                }));
            }
        }
    }

    /// Add interaction terms (products of variables)
    pub fn add_interactions(&mut self, n_vars: usize) {
        for i in 0..n_vars {
            for j in (i + 1)..n_vars {
                let name = format!("x{}*x{}", i, j);
                self.add_term(LibraryTerm::new(&name, n_vars, move |x| {
                    x.get(i).copied().unwrap_or(0.0) * x.get(j).copied().unwrap_or(0.0)
                }));
            }
        }
    }

    /// Add trigonometric terms
    pub fn add_trig(&mut self, n_vars: usize) {
        for var in 0..n_vars {
            let name_sin = format!("sin(x{})", var);
            self.add_term(LibraryTerm::new(&name_sin, n_vars, move |x| {
                x.get(var).copied().unwrap_or(0.0).sin()
            }));

            let name_cos = format!("cos(x{})", var);
            self.add_term(LibraryTerm::new(&name_cos, n_vars, move |x| {
                x.get(var).copied().unwrap_or(0.0).cos()
            }));
        }
    }

    /// Build the library matrix from data
    /// Returns matrix where each row is a data point and each column is a library term
    pub fn build_matrix(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n_samples = data.len();
        let n_terms = self.terms.len();

        let mut matrix = vec![vec![0.0; n_terms]; n_samples];

        for (i, x) in data.iter().enumerate() {
            for (j, term) in self.terms.iter().enumerate() {
                matrix[i][j] = term.evaluate(x);
            }
        }

        matrix
    }

    /// Get number of terms
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Get term names
    pub fn term_names(&self) -> Vec<&str> {
        self.terms.iter().map(|t| t.name.as_str()).collect()
    }
}

/// Create a standard polynomial library
pub fn polynomial_library(n_vars: usize, max_degree: usize) -> FunctionLibrary {
    let mut lib = FunctionLibrary::new();
    lib.add_constant();
    lib.add_polynomials(n_vars, max_degree);
    lib.add_interactions(n_vars);
    lib
}

/// Create a library for dynamical systems (includes trig)
pub fn dynamics_library(n_vars: usize, max_degree: usize) -> FunctionLibrary {
    let mut lib = FunctionLibrary::new();
    lib.add_constant();
    lib.add_polynomials(n_vars, max_degree);
    lib.add_interactions(n_vars);
    lib.add_trig(n_vars);
    lib
}

/// Discovered model
#[derive(Debug, Clone)]
pub struct DiscoveredModel {
    /// Coefficient for each term in the library
    pub coefficients: Vec<f64>,
    /// Term names
    pub term_names: Vec<String>,
    /// Active terms (non-zero coefficients)
    pub active_terms: Vec<(String, f64)>,
    /// Number of active terms
    pub sparsity: usize,
    /// Training error (MSE)
    pub mse: f64,
}

impl DiscoveredModel {
    /// Get symbolic representation
    pub fn symbolic_form(&self) -> String {
        let mut terms: Vec<String> = Vec::new();

        for (name, coef) in &self.active_terms {
            if coef.abs() < 1e-10 {
                continue;
            }

            let term = if name == "1" {
                format!("{:.4}", coef)
            } else if (coef - 1.0).abs() < 1e-10 {
                name.clone()
            } else if (coef + 1.0).abs() < 1e-10 {
                format!("-{}", name)
            } else {
                format!("{:.4}*{}", coef, name)
            };

            terms.push(term);
        }

        if terms.is_empty() {
            "0".to_string()
        } else {
            terms.join(" + ").replace("+ -", "- ")
        }
    }

    /// Predict using the discovered model
    pub fn predict(&self, library: &FunctionLibrary, x: &[f64]) -> f64 {
        let mut result = 0.0;
        for (i, term) in library.terms.iter().enumerate() {
            result += self.coefficients[i] * term.evaluate(x);
        }
        result
    }
}

impl fmt::Display for DiscoveredModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.symbolic_form())
    }
}

/// SINDy options
#[derive(Debug, Clone)]
pub struct SINDyOptions {
    /// Sparsity threshold for coefficients
    pub threshold: f64,
    /// Maximum iterations for STLS
    pub max_iterations: usize,
    /// Regularization parameter (alpha)
    pub alpha: f64,
    /// Normalize library columns
    pub normalize: bool,
}

impl Default for SINDyOptions {
    fn default() -> Self {
        Self {
            threshold: 0.05,
            max_iterations: 10,
            alpha: 0.0,
            normalize: true,
        }
    }
}

/// Sparse Identification of Nonlinear Dynamics (SINDy)
///
/// Discovers governing equations of the form:
/// dx/dt = Θ(x) * ξ
///
/// Where Θ(x) is the library matrix and ξ are sparse coefficients.
pub fn sindy(
    data: &[Vec<f64>],
    derivatives: &[f64],
    library: &FunctionLibrary,
    options: &SINDyOptions,
) -> DiscoveredModel {
    // Build library matrix
    let theta = library.build_matrix(data);
    let n_samples = theta.len();
    let n_terms = library.len();

    // Normalize columns if requested
    let (theta_norm, scales) = if options.normalize {
        normalize_columns(&theta)
    } else {
        (theta.clone(), vec![1.0; n_terms])
    };

    // Sequential Thresholded Least Squares (STLS)
    let mut xi = least_squares(&theta_norm, derivatives);

    for _ in 0..options.max_iterations {
        // Threshold small coefficients
        for coef in &mut xi {
            if coef.abs() < options.threshold {
                *coef = 0.0;
            }
        }

        // Re-fit with only active terms
        let active: Vec<usize> = xi
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c.abs() >= options.threshold)
            .map(|(i, _)| i)
            .collect();

        if active.is_empty() {
            break;
        }

        // Build reduced library
        let theta_reduced: Vec<Vec<f64>> = theta_norm
            .iter()
            .map(|row| active.iter().map(|&i| row[i]).collect())
            .collect();

        let xi_reduced = least_squares(&theta_reduced, derivatives);

        // Update full coefficient vector
        for (j, &i) in active.iter().enumerate() {
            xi[i] = xi_reduced[j];
        }
    }

    // Rescale coefficients
    for i in 0..n_terms {
        if scales[i].abs() > 1e-10 {
            xi[i] /= scales[i];
        }
    }

    // Compute MSE
    let mut mse = 0.0;
    for (i, row) in theta.iter().enumerate() {
        let pred: f64 = row.iter().zip(xi.iter()).map(|(a, b)| a * b).sum();
        mse += (derivatives[i] - pred).powi(2);
    }
    mse /= n_samples as f64;

    // Build active terms list
    let term_names: Vec<String> = library.term_names().iter().map(|s| s.to_string()).collect();
    let active_terms: Vec<(String, f64)> = xi
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c.abs() >= options.threshold)
        .map(|(i, &c)| (term_names[i].clone(), c))
        .collect();

    DiscoveredModel {
        coefficients: xi.clone(),
        term_names,
        active_terms: active_terms.clone(),
        sparsity: active_terms.len(),
        mse,
    }
}

/// Normalize matrix columns to unit norm
fn normalize_columns(matrix: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>) {
    if matrix.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let n_cols = matrix[0].len();
    let n_rows = matrix.len();

    // Compute column norms
    let mut norms = vec![0.0; n_cols];
    for row in matrix {
        for (j, &val) in row.iter().enumerate() {
            norms[j] += val * val;
        }
    }
    for norm in &mut norms {
        *norm = norm.sqrt().max(1e-10);
    }

    // Normalize
    let mut result = vec![vec![0.0; n_cols]; n_rows];
    for (i, row) in matrix.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[i][j] = val / norms[j];
        }
    }

    (result, norms)
}

/// Solve least squares: min ||Ax - b||²
/// Using normal equations: x = (A^T A)^{-1} A^T b
fn least_squares(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || a[0].is_empty() {
        return Vec::new();
    }

    let n = a[0].len();
    let m = a.len();

    // Compute A^T A
    let mut ata = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..m {
                ata[i][j] += a[k][i] * a[k][j];
            }
        }
    }

    // Compute A^T b
    let mut atb = vec![0.0; n];
    for i in 0..n {
        for k in 0..m {
            atb[i] += a[k][i] * b[k];
        }
    }

    // Solve using Cholesky or regularized inverse
    solve_symmetric(&ata, &atb)
}

/// Solve symmetric positive definite system Ax = b
fn solve_symmetric(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    if n == 0 {
        return Vec::new();
    }

    // Add small regularization for numerical stability
    let mut a_reg = a.to_vec();
    for i in 0..n {
        a_reg[i][i] += 1e-8;
    }

    // Cholesky decomposition: A = L L^T
    let mut l = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a_reg[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }

            if i == j {
                if sum <= 0.0 {
                    // Not positive definite - fall back to pseudo-inverse
                    return pseudo_inverse_solve(a, b);
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    // Forward substitution: L y = b
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i][j] * y[j];
        }
        y[i] = sum / l[i][i];
    }

    // Backward substitution: L^T x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j][i] * x[j];
        }
        x[i] = sum / l[i][i];
    }

    x
}

/// Pseudo-inverse solution for non-positive-definite systems
fn pseudo_inverse_solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    // Simple gradient descent as fallback
    let n = b.len();
    let mut x = vec![0.0; n];
    let lr = 0.01;
    let max_iter = 1000;

    for _ in 0..max_iter {
        // Compute residual
        let mut r = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                r[i] += a[i][j] * x[j];
            }
            r[i] = b[i] - r[i];
        }

        // Gradient step
        for i in 0..n {
            let mut grad = 0.0;
            for j in 0..n {
                grad += a[j][i] * r[j];
            }
            x[i] += lr * grad;
        }
    }

    x
}

/// Compute numerical derivatives using finite differences
pub fn compute_derivatives(data: &[Vec<f64>], dt: f64) -> Vec<Vec<f64>> {
    if data.len() < 2 {
        return Vec::new();
    }

    let n_samples = data.len();
    let n_vars = data[0].len();
    let mut derivatives = vec![vec![0.0; n_vars]; n_samples - 2];

    // Central differences
    for i in 1..(n_samples - 1) {
        for j in 0..n_vars {
            derivatives[i - 1][j] = (data[i + 1][j] - data[i - 1][j]) / (2.0 * dt);
        }
    }

    derivatives
}

/// Discover ODE system from trajectory data
pub fn discover_ode(
    trajectory: &[Vec<f64>],
    dt: f64,
    library: &FunctionLibrary,
    options: &SINDyOptions,
) -> Vec<DiscoveredModel> {
    if trajectory.len() < 3 {
        return Vec::new();
    }

    let n_vars = trajectory[0].len();

    // Compute derivatives
    let derivatives = compute_derivatives(trajectory, dt);

    // Use interior points (matching derivatives)
    let interior_data: Vec<Vec<f64>> = trajectory[1..(trajectory.len() - 1)].to_vec();

    // Discover each component
    let mut models = Vec::new();
    for var in 0..n_vars {
        let derivs: Vec<f64> = derivatives.iter().map(|d| d[var]).collect();
        let model = sindy(&interior_data, &derivs, library, options);
        models.push(model);
    }

    models
}

/// Discovered ODE system
#[derive(Debug)]
pub struct DiscoveredODE {
    /// Models for each state variable
    pub components: Vec<DiscoveredModel>,
    /// Variable names
    pub var_names: Vec<String>,
}

impl DiscoveredODE {
    /// Get symbolic form of the ODE system
    pub fn symbolic_form(&self) -> String {
        let mut lines = Vec::new();
        for (i, model) in self.components.iter().enumerate() {
            let default_name = format!("x{}", i);
            let var = self
                .var_names
                .get(i)
                .map(|s| s.as_str())
                .unwrap_or(&default_name);
            lines.push(format!("d{}/dt = {}", var, model.symbolic_form()));
        }
        lines.join("\n")
    }
}

impl fmt::Display for DiscoveredODE {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.symbolic_form())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_creation() {
        let lib = polynomial_library(2, 2);

        // Should have: 1, x0, x0^2, x1, x1^2, x0*x1
        assert!(lib.len() >= 6);
    }

    #[test]
    fn test_library_evaluation() {
        let mut lib = FunctionLibrary::new();
        lib.add_constant();
        lib.add_polynomials(2, 2);

        let x = vec![2.0, 3.0];
        let matrix = lib.build_matrix(&[x]);

        // Check constant term
        assert_eq!(matrix[0][0], 1.0);
    }

    #[test]
    fn test_sindy_linear() {
        // Generate data for dx/dt = -x
        let n = 100;
        let dt = 0.01;
        let data: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let t = i as f64 * dt;
                vec![(-t).exp()]
            })
            .collect();

        // Derivatives: dx/dt = -x
        let derivatives: Vec<f64> = data.iter().map(|x| -x[0]).collect();

        let lib = polynomial_library(1, 2);
        let options = SINDyOptions {
            threshold: 0.1,
            ..Default::default()
        };

        let model = sindy(&data, &derivatives, &lib, &options);

        // Should discover coefficient close to -1 for x
        let x_coef = model
            .coefficients
            .iter()
            .enumerate()
            .find(|(i, _)| lib.term_names()[*i] == "x0")
            .map(|(_, &c)| c)
            .unwrap_or(0.0);

        assert!((x_coef + 1.0).abs() < 0.2, "Expected -1, got {}", x_coef);
    }

    #[test]
    fn test_normalize_columns() {
        let matrix = vec![vec![3.0, 0.0], vec![4.0, 0.0], vec![0.0, 5.0]];

        let (normalized, norms) = normalize_columns(&matrix);

        // First column: [3, 4, 0], norm = 5
        assert!((norms[0] - 5.0).abs() < 1e-10);

        // Check normalized values
        assert!((normalized[0][0] - 0.6).abs() < 1e-10);
        assert!((normalized[1][0] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_least_squares_simple() {
        // Solve 2x = 4 => x = 2
        let a = vec![vec![2.0]];
        let b = vec![4.0];

        let x = least_squares(&a, &b);

        assert!((x[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_derivatives() {
        // Linear function: x(t) = t, dx/dt = 1
        let data: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let dt = 1.0;

        let derivs = compute_derivatives(&data, dt);

        for d in &derivs {
            assert!((d[0] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_discovered_model_display() {
        let model = DiscoveredModel {
            coefficients: vec![0.0, 2.5, -1.0],
            term_names: vec!["1".into(), "x".into(), "x^2".into()],
            active_terms: vec![("x".into(), 2.5), ("x^2".into(), -1.0)],
            sparsity: 2,
            mse: 0.01,
        };

        let s = model.symbolic_form();
        assert!(s.contains("2.5"));
        assert!(s.contains("x"));
    }

    #[test]
    fn test_dynamics_library() {
        let lib = dynamics_library(2, 2);

        // Should include trig terms
        let names = lib.term_names();
        assert!(names.iter().any(|n| n.contains("sin")));
        assert!(names.iter().any(|n| n.contains("cos")));
    }
}
