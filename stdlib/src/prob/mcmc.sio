//! Markov Chain Monte Carlo samplers

use super::distributions::{Distribution, Normal, MultivariateNormal}
use linalg::{Vector, Matrix}
use autodiff::reverse::{gradient, Var}
use rand::Rng

/// MCMC sampling result
pub struct MCMCSample {
    /// Parameter samples (rows = iterations, cols = parameters)
    pub samples: Matrix<f64>,
    
    /// Log probability at each sample
    pub log_probs: Vector<f64>,
    
    /// Acceptance rate
    pub acceptance_rate: f64,
    
    /// Number of samples
    pub n_samples: usize,
    
    /// Effective sample size estimate
    pub ess: Vector<f64>,
    
    /// R-hat convergence diagnostic
    pub rhat: Vector<f64>,
}

/// Metropolis-Hastings sampler
pub struct MetropolisHastings {
    /// Proposal covariance
    pub proposal_cov: Matrix<f64>,
    
    /// Adaptation parameters
    pub adapt_period: usize,
    pub target_acceptance: f64,
}

impl MetropolisHastings {
    pub fn new(dim: usize) -> Self {
        MetropolisHastings {
            proposal_cov: Matrix::eye(dim) * 0.1,
            adapt_period: 100,
            target_acceptance: 0.44,
        }
    }
    
    /// Sample from target distribution
    pub fn sample<F>(
        &mut self,
        log_prob: F,
        x0: &Vector<f64>,
        n_samples: usize,
        rng: &mut impl Rng,
    ) -> MCMCSample
    where F: Fn(&Vector<f64>) -> f64
    {
        let dim = x0.len();
        let mut samples = Matrix::zeros(n_samples, dim);
        let mut log_probs = Vector::zeros(n_samples);
        
        let mut current_x = x0.clone();
        let mut current_log_prob = log_prob(&current_x);
        let mut n_accepted = 0;
        
        // Proposal distribution
        let proposal = MultivariateNormal::new(
            Vector::zeros(dim),
            self.proposal_cov.clone()
        ).unwrap();
        
        for i in 0..n_samples {
            // Propose new state
            let proposal_step = proposal.sample(rng);
            let proposed_x = &current_x + &proposal_step;
            let proposed_log_prob = log_prob(&proposed_x);
            
            // Acceptance probability
            let log_alpha = proposed_log_prob - current_log_prob;
            let alpha = log_alpha.exp().min(1.0);
            
            // Accept or reject
            if rng.uniform(0.0, 1.0) < alpha {
                current_x = proposed_x;
                current_log_prob = proposed_log_prob;
                n_accepted += 1;
            }
            
            // Store sample
            for j in 0..dim {
                samples[(i, j)] = current_x[j];
            }
            log_probs[i] = current_log_prob;
            
            // Adapt proposal covariance
            if i > 0 && i % self.adapt_period == 0 {
                let acceptance_rate = n_accepted as f64 / (i + 1) as f64;
                let scale_factor = if acceptance_rate > self.target_acceptance {
                    1.1
                } else {
                    0.9
                };
                
                for j in 0..dim {
                    for k in 0..dim {
                        self.proposal_cov[(j, k)] *= scale_factor;
                    }
                }
            }
        }
        
        let acceptance_rate = n_accepted as f64 / n_samples as f64;
        
        MCMCSample {
            samples,
            log_probs,
            acceptance_rate,
            n_samples,
            ess: Vector::zeros(dim), // TODO: compute ESS
            rhat: Vector::zeros(dim), // TODO: compute R-hat
        }
    }
}

/// Hamiltonian Monte Carlo sampler
pub struct HMC {
    /// Step size
    pub epsilon: f64,
    
    /// Number of leapfrog steps
    pub l: usize,
    
    /// Mass matrix (inverse)
    pub mass_matrix_inv: Matrix<f64>,
}

impl HMC {
    pub fn new(dim: usize, epsilon: f64, l: usize) -> Self {
        HMC {
            epsilon,
            l,
            mass_matrix_inv: Matrix::eye(dim),
        }
    }
    
    /// Sample using Hamiltonian dynamics
    pub fn sample<F>(
        &self,
        log_prob: F,
        x0: &Vector<f64>,
        n_samples: usize,
        rng: &mut impl Rng,
    ) -> MCMCSample
    where F: Fn(&Vector<Var>) -> Var + Clone
    {
        let dim = x0.len();
        let mut samples = Matrix::zeros(n_samples, dim);
        let mut log_probs = Vector::zeros(n_samples);
        
        let mut current_x = x0.clone();
        let mut n_accepted = 0;
        
        // Momentum distribution
        let momentum_dist = MultivariateNormal::new(
            Vector::zeros(dim),
            self.mass_matrix_inv.clone()
        ).unwrap();
        
        for i in 0..n_samples {
            // Sample momentum
            let mut p = momentum_dist.sample(rng);
            let p0 = p.clone();
            
            // Current energy
            let current_log_prob = {
                let x_vars: Vector<Var> = current_x.iter().map(|&xi| Var::new(xi)).collect();
                log_prob(&x_vars).value()
            };
            let current_kinetic = 0.5 * p.dot(&(&self.mass_matrix_inv * &p));
            let current_energy = -current_log_prob + current_kinetic;
            
            // Leapfrog integration
            let mut x = current_x.clone();
            
            // Half step for momentum
            let grad = gradient(log_prob.clone(), &x);
            for j in 0..dim {
                p[j] += 0.5 * self.epsilon * grad[j];
            }
            
            // Full steps
            for _ in 0..self.l {
                // Full step for position
                let mass_p = &self.mass_matrix_inv * &p;
                for j in 0..dim {
                    x[j] += self.epsilon * mass_p[j];
                }
                
                // Full step for momentum (except last)
                let grad = gradient(log_prob.clone(), &x);
                for j in 0..dim {
                    p[j] += self.epsilon * grad[j];
                }
            }
            
            // Half step for momentum
            let grad = gradient(log_prob.clone(), &x);
            for j in 0..dim {
                p[j] += 0.5 * self.epsilon * grad[j];
            }
            
            // Proposed energy
            let proposed_log_prob = {
                let x_vars: Vector<Var> = x.iter().map(|&xi| Var::new(xi)).collect();
                log_prob(&x_vars).value()
            };
            let proposed_kinetic = 0.5 * p.dot(&(&self.mass_matrix_inv * &p));
            let proposed_energy = -proposed_log_prob + proposed_kinetic;
            
            // Accept or reject
            let delta_energy = proposed_energy - current_energy;
            let alpha = (-delta_energy).exp().min(1.0);
            
            if rng.uniform(0.0, 1.0) < alpha {
                current_x = x;
                n_accepted += 1;
            }
            
            // Store sample
            for j in 0..dim {
                samples[(i, j)] = current_x[j];
            }
            log_probs[i] = {
                let x_vars: Vector<Var> = current_x.iter().map(|&xi| Var::new(xi)).collect();
                log_prob(&x_vars).value()
            };
        }
        
        let acceptance_rate = n_accepted as f64 / n_samples as f64;
        
        MCMCSample {
            samples,
            log_probs,
            acceptance_rate,
            n_samples,
            ess: Vector::zeros(dim),
            rhat: Vector::zeros(dim),
        }
    }
}

/// No-U-Turn Sampler (NUTS)
pub struct NUTS {
    /// Target acceptance probability
    pub target_accept: f64,
    
    /// Maximum tree depth
    pub max_treedepth: usize,
    
    /// Step size adaptation parameters
    pub gamma: f64,
    pub t0: f64,
    pub kappa: f64,
}

impl NUTS {
    pub fn new() -> Self {
        NUTS {
            target_accept: 0.8,
            max_treedepth: 10,
            gamma: 0.05,
            t0: 10.0,
            kappa: 0.75,
        }
    }
    
    /// Sample using NUTS algorithm (simplified implementation)
    pub fn sample<F>(
        &self,
        log_prob: F,
        x0: &Vector<f64>,
        n_samples: usize,
        rng: &mut impl Rng,
    ) -> MCMCSample
    where F: Fn(&Vector<Var>) -> Var + Clone
    {
        // This is a simplified NUTS implementation
        // A full implementation would include the tree building algorithm
        
        let hmc = HMC::new(x0.len(), 0.1, 10);
        hmc.sample(log_prob, x0, n_samples, rng)
    }
}

/// Compute effective sample size
pub fn effective_sample_size(samples: &Matrix<f64>) -> Vector<f64> {
    let (n_samples, n_params) = samples.shape();
    let mut ess = Vector::zeros(n_params);
    
    for j in 0..n_params {
        let chain: Vec<f64> = (0..n_samples).map(|i| samples[(i, j)]).collect();
        
        // Compute autocorrelation
        let mean: f64 = chain.iter().sum::<f64>() / n_samples as f64;
        let var: f64 = chain.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (n_samples - 1) as f64;
        
        let mut autocorr_sum = 0.0;
        let max_lag = (n_samples / 4).min(200);
        
        for lag in 1..max_lag {
            let mut autocorr = 0.0;
            for i in 0..(n_samples - lag) {
                autocorr += (chain[i] - mean) * (chain[i + lag] - mean);
            }
            autocorr /= (n_samples - lag) as f64 * var;
            
            if autocorr < 0.05 {
                break;
            }
            autocorr_sum += autocorr;
        }
        
        ess[j] = n_samples as f64 / (1.0 + 2.0 * autocorr_sum);
    }
    
    ess
}

/// Compute R-hat convergence diagnostic
pub fn rhat(chains: &[Matrix<f64>]) -> Vector<f64> {
    let n_chains = chains.len();
    let n_samples = chains[0].nrows();
    let n_params = chains[0].ncols();
    
    let mut rhat = Vector::zeros(n_params);
    
    for j in 0..n_params {
        // Chain means
        let mut chain_means = Vec::new();
        let mut chain_vars = Vec::new();
        
        for chain in chains {
            let values: Vec<f64> = (0..n_samples).map(|i| chain[(i, j)]).collect();
            let mean = values.iter().sum::<f64>() / n_samples as f64;
            let var = values.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / (n_samples - 1) as f64;
            
            chain_means.push(mean);
            chain_vars.push(var);
        }
        
        // Overall mean
        let overall_mean = chain_means.iter().sum::<f64>() / n_chains as f64;
        
        // Between-chain variance
        let b = n_samples as f64 * chain_means.iter()
            .map(|m| (m - overall_mean).powi(2))
            .sum::<f64>() / (n_chains - 1) as f64;
        
        // Within-chain variance
        let w = chain_vars.iter().sum::<f64>() / n_chains as f64;
        
        // Marginal posterior variance
        let var_plus = ((n_samples - 1) as f64 * w + b) / n_samples as f64;
        
        // R-hat
        rhat[j] = (var_plus / w).sqrt();
    }
    
    rhat
}
