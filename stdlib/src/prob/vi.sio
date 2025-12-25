//! Variational Inference algorithms

use super::distributions::{Distribution, Normal, MultivariateNormal}
use linalg::{Vector, Matrix}
use autodiff::reverse::{Var, gradient}
use numerics::optimize::{OptResult, GradientDescent}

/// Variational family trait
pub trait VariationalFamily {
    /// Sample from the variational distribution
    fn sample(&self, rng: &mut impl rand::Rng) -> Vector<f64> with Prob;
    
    /// Log probability density
    fn log_prob(&self, x: &Vector<f64>) -> f64;
    
    /// Get variational parameters
    fn parameters(&self) -> Vector<f64>;
    
    /// Update variational parameters
    fn update_parameters(&mut self, params: &Vector<f64>);
    
    /// Number of parameters
    fn n_params(&self) -> usize;
}

/// Mean-field Gaussian variational family
pub struct MeanFieldGaussian {
    /// Mean parameters
    pub mu: Vector<f64>,
    
    /// Log standard deviation parameters (for numerical stability)
    pub log_sigma: Vector<f64>,
}

impl MeanFieldGaussian {
    pub fn new(dim: usize) -> Self {
        MeanFieldGaussian {
            mu: Vector::zeros(dim),
            log_sigma: Vector::zeros(dim),
        }
    }
    
    pub fn sigma(&self) -> Vector<f64> {
        self.log_sigma.map(|x| x.exp())
    }
}

impl VariationalFamily for MeanFieldGaussian {
    fn sample(&self, rng: &mut impl rand::Rng) -> Vector<f64> with Prob {
        let dim = self.mu.len();
        let mut sample = Vector::zeros(dim);
        
        for i in 0..dim {
            let normal = Normal::new(self.mu[i], self.log_sigma[i].exp());
            sample[i] = normal.sample(rng);
        }
        
        sample
    }
    
    fn log_prob(&self, x: &Vector<f64>) -> f64 {
        let dim = self.mu.len();
        let mut log_prob = 0.0;
        
        for i in 0..dim {
            let normal = Normal::new(self.mu[i], self.log_sigma[i].exp());
            log_prob += normal.log_pdf(x[i]);
        }
        
        log_prob
    }
    
    fn parameters(&self) -> Vector<f64> {
        let mut params = Vector::zeros(2 * self.mu.len());
        for i in 0..self.mu.len() {
            params[i] = self.mu[i];
            params[i + self.mu.len()] = self.log_sigma[i];
        }
        params
    }
    
    fn update_parameters(&mut self, params: &Vector<f64>) {
        let dim = self.mu.len();
        for i in 0..dim {
            self.mu[i] = params[i];
            self.log_sigma[i] = params[i + dim];
        }
    }
    
    fn n_params(&self) -> usize {
        2 * self.mu.len()
    }
}

/// Full-rank Gaussian variational family
pub struct FullRankGaussian {
    /// Mean parameters
    pub mu: Vector<f64>,
    
    /// Cholesky factor of covariance (lower triangular)
    pub l: Matrix<f64>,
}

impl FullRankGaussian {
    pub fn new(dim: usize) -> Self {
        FullRankGaussian {
            mu: Vector::zeros(dim),
            l: Matrix::eye(dim),
        }
    }
}

impl VariationalFamily for FullRankGaussian {
    fn sample(&self, rng: &mut impl rand::Rng) -> Vector<f64> with Prob {
        let dim = self.mu.len();
        
        // Sample from standard normal
        let mut z = Vector::zeros(dim);
        let standard = Normal::standard();
        for i in 0..dim {
            z[i] = standard.sample(rng);
        }
        
        // Transform: x = mu + L * z
        &self.mu + &(&self.l * &z)
    }
    
    fn log_prob(&self, x: &Vector<f64>) -> f64 {
        let dim = self.mu.len();
        let diff = x - &self.mu;
        
        // Solve L * y = diff
        let y = linalg::solve_triangular(&self.l, &diff, linalg::Lower).unwrap();
        let quad_form = y.dot(&y);
        
        // Log determinant = sum(log(diag(L)))
        let log_det: f64 = (0..dim)
            .map(|i| self.l[(i, i)].ln())
            .sum();
        
        -0.5 * (dim as f64 * (2.0 * std::math::PI).ln() + 2.0 * log_det + quad_form)
    }
    
    fn parameters(&self) -> Vector<f64> {
        let dim = self.mu.len();
        let mut params = Vector::zeros(dim + dim * (dim + 1) / 2);
        
        // Mean parameters
        for i in 0..dim {
            params[i] = self.mu[i];
        }
        
        // Lower triangular elements of L
        let mut idx = dim;
        for i in 0..dim {
            for j in 0..=i {
                params[idx] = self.l[(i, j)];
                idx += 1;
            }
        }
        
        params
    }
    
    fn update_parameters(&mut self, params: &Vector<f64>) {
        let dim = self.mu.len();
        
        // Update mean
        for i in 0..dim {
            self.mu[i] = params[i];
        }
        
        // Update L (lower triangular)
        let mut idx = dim;
        for i in 0..dim {
            for j in 0..=i {
                self.l[(i, j)] = params[idx];
                idx += 1;
            }
            // Zero out upper triangular part
            for j in (i+1)..dim {
                self.l[(i, j)] = 0.0;
            }
        }
    }
    
    fn n_params(&self) -> usize {
        let dim = self.mu.len();
        dim + dim * (dim + 1) / 2
    }
}

/// Variational Inference result
pub struct VIResult {
    /// Final variational parameters
    pub params: Vector<f64>,
    
    /// ELBO values during optimization
    pub elbo_history: Vector<f64>,
    
    /// Final ELBO value
    pub final_elbo: f64,
    
    /// Number of iterations
    pub n_iter: usize,
    
    /// Whether optimization converged
    pub converged: bool,
}

/// Automatic Differentiation Variational Inference (ADVI)
pub struct ADVI {
    /// Learning rate
    pub learning_rate: f64,
    
    /// Number of Monte Carlo samples for ELBO estimation
    pub n_samples: usize,
    
    /// Maximum iterations
    pub max_iter: usize,
    
    /// Convergence tolerance
    pub tol: f64,
}

impl ADVI {
    pub fn new() -> Self {
        ADVI {
            learning_rate: 0.01,
            n_samples: 10,
            max_iter: 1000,
            tol: 1e-6,
        }
    }
    
    /// Fit variational distribution using ADVI
    pub fn fit<F, V>(
        &self,
        log_prob: F,
        mut variational: V,
        rng: &mut impl rand::Rng,
    ) -> VIResult
    where 
        F: Fn(&Vector<Var>) -> Var + Clone,
        V: VariationalFamily,
    {
        let mut elbo_history = Vec::new();
        let mut params = variational.parameters();
        let mut prev_elbo = f64::NEG_INFINITY;
        
        for iter in 0..self.max_iter {
            // Compute ELBO and its gradient
            let elbo_fn = |theta: &Vector<Var>| -> Var {
                // Update variational parameters
                let theta_vals: Vector<f64> = theta.iter().map(|t| t.value()).collect();
                variational.update_parameters(&theta_vals);
                
                let mut elbo = Var::new(0.0);
                
                // Monte Carlo estimate of ELBO
                for _ in 0..self.n_samples {
                    let z = variational.sample(rng);
                    
                    // Convert to Var
                    let z_vars: Vector<Var> = z.iter().map(|&zi| Var::new(zi)).collect();
                    
                    // Log probability of data
                    let log_p = log_prob(&z_vars);
                    
                    // Log probability of variational distribution
                    let log_q = Var::new(variational.log_prob(&z));
                    
                    elbo = elbo + (log_p - log_q);
                }
                
                elbo * (1.0 / self.n_samples as f64)
            };
            
            // Compute gradient
            let grad = gradient(elbo_fn, &params);
            let current_elbo = {
                let params_vars: Vector<Var> = params.iter().map(|&p| Var::new(p)).collect();
                elbo_fn(&params_vars).value()
            };
            
            elbo_history.push(current_elbo);
            
            // Check convergence
            if iter > 0 && (current_elbo - prev_elbo).abs() < self.tol {
                return VIResult {
                    params,
                    elbo_history: Vector::from_slice(&elbo_history),
                    final_elbo: current_elbo,
                    n_iter: iter + 1,
                    converged: true,
                };
            }
            
            // Gradient ascent update (maximize ELBO)
            for i in 0..params.len() {
                params[i] += self.learning_rate * grad[i];
            }
            
            prev_elbo = current_elbo;
        }
        
        VIResult {
            params,
            elbo_history: Vector::from_slice(&elbo_history),
            final_elbo: prev_elbo,
            n_iter: self.max_iter,
            converged: false,
        }
    }
}

/// Stochastic Variational Inference
pub struct SVI {
    /// Learning rate schedule parameters
    pub learning_rate: f64,
    pub decay_rate: f64,
    
    /// Batch size for stochastic updates
    pub batch_size: usize,
    
    /// Maximum iterations
    pub max_iter: usize,
}

impl SVI {
    pub fn new(batch_size: usize) -> Self {
        SVI {
            learning_rate: 0.01,
            decay_rate: 0.9,
            batch_size,
            max_iter: 1000,
        }
    }
    
    /// Fit using stochastic variational inference
    pub fn fit<F, V>(
        &self,
        log_prob: F,
        data: &[Vector<f64>],
        mut variational: V,
        rng: &mut impl rand::Rng,
    ) -> VIResult
    where 
        F: Fn(&Vector<Var>, &Vector<f64>) -> Var + Clone,
        V: VariationalFamily,
    {
        let n_data = data.len();
        let mut elbo_history = Vec::new();
        let mut params = variational.parameters();
        
        for iter in 0..self.max_iter {
            // Sample mini-batch
            let mut batch_indices = Vec::new();
            for _ in 0..self.batch_size.min(n_data) {
                batch_indices.push(rng.uniform_range(0, n_data));
            }
            
            // Compute stochastic ELBO gradient
            let elbo_fn = |theta: &Vector<Var>| -> Var {
                let theta_vals: Vector<f64> = theta.iter().map(|t| t.value()).collect();
                variational.update_parameters(&theta_vals);
                
                let mut elbo = Var::new(0.0);
                
                for &idx in &batch_indices {
                    let z = variational.sample(rng);
                    let z_vars: Vector<Var> = z.iter().map(|&zi| Var::new(zi)).collect();
                    
                    let log_p = log_prob(&z_vars, &data[idx]);
                    let log_q = Var::new(variational.log_prob(&z));
                    
                    elbo = elbo + (log_p - log_q);
                }
                
                // Scale by data size / batch size
                elbo * (n_data as f64 / batch_indices.len() as f64)
            };
            
            let grad = gradient(elbo_fn, &params);
            let current_elbo = {
                let params_vars: Vector<Var> = params.iter().map(|&p| Var::new(p)).collect();
                elbo_fn(&params_vars).value()
            };
            
            elbo_history.push(current_elbo);
            
            // Adaptive learning rate
            let lr = self.learning_rate / (1.0 + self.decay_rate * iter as f64);
            
            // Update parameters
            for i in 0..params.len() {
                params[i] += lr * grad[i];
            }
        }
        
        VIResult {
            params,
            elbo_history: Vector::from_slice(&elbo_history),
            final_elbo: elbo_history.last().copied().unwrap_or(0.0),
            n_iter: self.max_iter,
            converged: false, // TODO: implement convergence check
        }
    }
}
