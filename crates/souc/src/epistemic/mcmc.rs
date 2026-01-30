//! Markov Chain Monte Carlo (MCMC) Samplers for Bayesian Inference
//!
//! Implements MCMC algorithms for sampling from posterior distributions in
//! epistemic uncertainty quantification.
//!
//! # Overview
//!
//! MCMC methods enable full Bayesian inference for `Knowledge<T>` types by
//! sampling from posterior distributions that cannot be computed analytically.
//!
//! # Algorithms Implemented
//!
//! 1. **Metropolis-Hastings (MH)**: Classic random walk with adaptive proposals
//! 2. **Hamiltonian Monte Carlo (HMC)**: Gradient-guided exploration
//! 3. **NUTS**: No-U-Turn Sampler with automatic tuning
//!
//! # Example
//!
//! ```rust
//! use sounio::epistemic::mcmc::{MCMCConfig, MCMCSampler, MetropolisHastings};
//!
//! // Define log-posterior: log p(θ|data) = log p(data|θ) + log p(θ)
//! let log_posterior = |theta: &[f64]| {
//!     // Likelihood: Gaussian with known variance
//!     let mu = theta[0];
//!     let log_likelihood = -0.5 * (mu - 5.0).powi(2);
//!
//!     // Prior: Standard normal
//!     let log_prior = -0.5 * mu.powi(2);
//!
//!     log_likelihood + log_prior
//! };
//!
//! let config = MCMCConfig::default();
//! let mut sampler = MetropolisHastings::new(config, 1);
//! let samples = sampler.sample(&log_posterior, &[0.0], 10000);
//! ```
//!
//! # References
//!
//! - Gelman et al. (2013): "Bayesian Data Analysis"
//! - Neal (2011): "MCMC Using Hamiltonian Dynamics"
//! - Hoffman & Gelman (2014): "The No-U-Turn Sampler"

use std::f64::consts::PI;

/// Configuration for MCMC sampling
#[derive(Debug, Clone, PartialEq)]
pub struct MCMCConfig {
    /// Number of warmup/burn-in iterations
    pub n_warmup: usize,

    /// Thinning interval (keep every nth sample)
    pub thin: usize,

    /// Target acceptance rate for adaptive algorithms (typically 0.234 for RW, 0.65 for HMC)
    pub target_accept: f64,

    /// Initial proposal standard deviation (for MH)
    pub initial_step_size: f64,

    /// Adaptation window for proposal tuning
    pub adapt_window: usize,

    /// Number of leapfrog steps for HMC
    pub n_leapfrog: usize,

    /// Maximum tree depth for NUTS
    pub max_tree_depth: usize,
}

impl Default for MCMCConfig {
    fn default() -> Self {
        Self {
            n_warmup: 1000,
            thin: 1,
            target_accept: 0.234, // Optimal for random walk in high dimensions
            initial_step_size: 0.1,
            adapt_window: 500,
            n_leapfrog: 10,
            max_tree_depth: 10,
        }
    }
}

impl MCMCConfig {
    /// Create HMC-specific config with higher target acceptance
    pub fn for_hmc() -> Self {
        Self {
            target_accept: 0.65, // Optimal for HMC
            n_leapfrog: 10,
            ..Default::default()
        }
    }

    /// Create NUTS-specific config
    pub fn for_nuts() -> Self {
        Self {
            target_accept: 0.8, // NUTS aims higher
            max_tree_depth: 10,
            ..Default::default()
        }
    }
}

/// Result of MCMC sampling
#[derive(Debug, Clone)]
pub struct MCMCResult {
    /// Posterior samples (after warmup and thinning)
    pub samples: Vec<Vec<f64>>,

    /// Acceptance rate
    pub accept_rate: f64,

    /// Effective sample size
    pub ess: Vec<f64>,

    /// Gelman-Rubin diagnostic (R-hat)
    pub r_hat: Option<Vec<f64>>,

    /// Final adapted step size
    pub final_step_size: f64,
}

impl MCMCResult {
    /// Get posterior mean
    pub fn mean(&self) -> Vec<f64> {
        let n_dims = self.samples[0].len();
        let n_samples = self.samples.len();

        (0..n_dims)
            .map(|i| self.samples.iter().map(|s| s[i]).sum::<f64>() / n_samples as f64)
            .collect()
    }

    /// Get posterior standard deviation
    pub fn std_dev(&self) -> Vec<f64> {
        let mean = self.mean();
        let n_dims = self.samples[0].len();
        let n_samples = self.samples.len();

        (0..n_dims)
            .map(|i| {
                let var = self
                    .samples
                    .iter()
                    .map(|s| (s[i] - mean[i]).powi(2))
                    .sum::<f64>()
                    / n_samples as f64;
                libm::sqrt(var)
            })
            .collect()
    }

    /// Get credible interval (default: 95%)
    pub fn credible_interval(&self, alpha: f64) -> Vec<(f64, f64)> {
        let n_dims = self.samples[0].len();
        let lower_idx = (self.samples.len() as f64 * (1.0 - alpha) / 2.0) as usize;
        let upper_idx = (self.samples.len() as f64 * (1.0 + alpha) / 2.0) as usize;

        (0..n_dims)
            .map(|i| {
                let mut values: Vec<f64> = self.samples.iter().map(|s| s[i]).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (values[lower_idx], values[upper_idx])
            })
            .collect()
    }
}

// =============================================================================
// Metropolis-Hastings Sampler
// =============================================================================

/// Metropolis-Hastings sampler with adaptive proposal
pub struct MetropolisHastings {
    config: MCMCConfig,
    n_dims: usize,
    proposal_std: Vec<f64>,
    rng_state: u64,
}

impl MetropolisHastings {
    /// Create new Metropolis-Hastings sampler
    pub fn new(config: MCMCConfig, n_dims: usize) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};

        Self {
            proposal_std: vec![config.initial_step_size; n_dims],
            config,
            n_dims,
            rng_state: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        }
    }

    /// Sample from log-posterior
    pub fn sample<F>(&mut self, log_posterior: &F, initial: &[f64], n_samples: usize) -> MCMCResult
    where
        F: Fn(&[f64]) -> f64,
    {
        let total_samples = self.config.n_warmup + n_samples * self.config.thin;
        let mut chain = Vec::with_capacity(total_samples);
        let mut current = initial.to_vec();
        let mut current_log_prob = log_posterior(&current);

        let mut n_accept = 0;
        let mut accept_history = Vec::new();

        // MCMC sampling loop
        for iter in 0..total_samples {
            // Propose new state
            let proposed = self.propose(&current);
            let proposed_log_prob = log_posterior(&proposed);

            // Metropolis-Hastings acceptance ratio
            let log_alpha = proposed_log_prob - current_log_prob;
            let accept = log_alpha >= 0.0 || self.uniform() < libm::exp(log_alpha);

            if accept {
                current = proposed;
                current_log_prob = proposed_log_prob;
                n_accept += 1;
            }

            chain.push(current.clone());
            accept_history.push(if accept { 1.0 } else { 0.0 });

            // Adaptive tuning during warmup
            if iter < self.config.n_warmup && iter % 100 == 0 && iter > 0 {
                let recent_accept = accept_history[iter.saturating_sub(100)..]
                    .iter()
                    .sum::<f64>()
                    / 100.0;
                self.adapt_proposal(recent_accept);
            }
        }

        // Discard warmup and thin
        let samples: Vec<Vec<f64>> = chain
            .into_iter()
            .skip(self.config.n_warmup)
            .step_by(self.config.thin)
            .collect();

        let accept_rate = n_accept as f64 / total_samples as f64;
        let ess = compute_ess(&samples);

        MCMCResult {
            samples,
            accept_rate,
            ess,
            r_hat: None,
            final_step_size: self.proposal_std[0],
        }
    }

    /// Propose new state using Gaussian random walk
    fn propose(&mut self, current: &[f64]) -> Vec<f64> {
        let mut proposed = Vec::with_capacity(current.len());
        for (i, &x) in current.iter().enumerate() {
            proposed.push(x + self.proposal_std[i] * self.normal());
        }
        proposed
    }

    /// Adapt proposal standard deviation
    fn adapt_proposal(&mut self, accept_rate: f64) {
        let target = self.config.target_accept;
        let adaptation_factor = if accept_rate > target {
            1.05 // Increase step size if accepting too much
        } else {
            0.95 // Decrease step size if rejecting too much
        };

        for std in &mut self.proposal_std {
            *std *= adaptation_factor;
        }
    }

    /// Sample from standard normal
    fn normal(&mut self) -> f64 {
        // Box-Muller transform
        let u1 = self.uniform();
        let u2 = self.uniform();
        libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(2.0 * PI * u2)
    }

    /// Sample from uniform [0, 1)
    fn uniform(&mut self) -> f64 {
        // LCG random number generator
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.rng_state as f64 / u64::MAX as f64
    }
}

// =============================================================================
// Hamiltonian Monte Carlo
// =============================================================================

/// Hamiltonian Monte Carlo sampler
pub struct HamiltonianMC {
    config: MCMCConfig,
    n_dims: usize,
    step_size: f64,
    rng_state: u64,
}

impl HamiltonianMC {
    /// Create new HMC sampler
    pub fn new(config: MCMCConfig, n_dims: usize) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};

        Self {
            step_size: config.initial_step_size,
            config,
            n_dims,
            rng_state: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        }
    }

    /// Sample using HMC with gradient function
    pub fn sample<F, G>(
        &mut self,
        log_posterior: &F,
        grad_log_posterior: &G,
        initial: &[f64],
        n_samples: usize,
    ) -> MCMCResult
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        let total_samples = self.config.n_warmup + n_samples * self.config.thin;
        let mut chain = Vec::with_capacity(total_samples);
        let mut current = initial.to_vec();

        let mut n_accept = 0;

        for iter in 0..total_samples {
            // Sample momentum
            let momentum: Vec<f64> = (0..self.n_dims).map(|_| self.normal()).collect();

            // Leapfrog integration
            let (proposed, proposed_momentum) =
                self.leapfrog(&current, &momentum, log_posterior, grad_log_posterior);

            // Metropolis acceptance
            let current_h = self.hamiltonian(&current, &momentum, log_posterior);
            let proposed_h = self.hamiltonian(&proposed, &proposed_momentum, log_posterior);

            let accept = self.uniform() < libm::exp(current_h - proposed_h);

            if accept {
                current = proposed;
                n_accept += 1;
            }

            chain.push(current.clone());

            // Adaptive step size tuning
            if iter < self.config.n_warmup && iter % 100 == 0 && iter > 0 {
                let recent_accept = n_accept as f64 / (iter + 1) as f64;
                self.adapt_step_size(recent_accept);
            }
        }

        let samples: Vec<Vec<f64>> = chain
            .into_iter()
            .skip(self.config.n_warmup)
            .step_by(self.config.thin)
            .collect();

        let accept_rate = n_accept as f64 / total_samples as f64;
        let ess = compute_ess(&samples);

        MCMCResult {
            samples,
            accept_rate,
            ess,
            r_hat: None,
            final_step_size: self.step_size,
        }
    }

    /// Leapfrog integrator for Hamiltonian dynamics
    fn leapfrog<F, G>(
        &self,
        q: &[f64],
        p: &[f64],
        _log_posterior: &F,
        grad_log_posterior: &G,
    ) -> (Vec<f64>, Vec<f64>)
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        let mut q_new = q.to_vec();
        let mut p_new = p.to_vec();

        // Half step for momentum
        let grad = grad_log_posterior(&q_new);
        for i in 0..self.n_dims {
            p_new[i] += 0.5 * self.step_size * grad[i];
        }

        // Full steps for position and momentum
        for _ in 0..(self.config.n_leapfrog - 1) {
            for i in 0..self.n_dims {
                q_new[i] += self.step_size * p_new[i];
            }

            let grad = grad_log_posterior(&q_new);
            for i in 0..self.n_dims {
                p_new[i] += self.step_size * grad[i];
            }
        }

        // Final position step
        for i in 0..self.n_dims {
            q_new[i] += self.step_size * p_new[i];
        }

        // Final half step for momentum
        let grad = grad_log_posterior(&q_new);
        for i in 0..self.n_dims {
            p_new[i] += 0.5 * self.step_size * grad[i];
        }

        // Negate momentum for reversibility
        for p in &mut p_new {
            *p = -*p;
        }

        (q_new, p_new)
    }

    /// Compute Hamiltonian: H = -log p(q) + 0.5 * p^T * p
    fn hamiltonian<F>(&self, q: &[f64], p: &[f64], log_posterior: &F) -> f64
    where
        F: Fn(&[f64]) -> f64,
    {
        let kinetic_energy: f64 = p.iter().map(|&pi| 0.5 * pi * pi).sum();
        let potential_energy = -log_posterior(q);
        kinetic_energy + potential_energy
    }

    /// Adapt step size based on acceptance rate
    fn adapt_step_size(&mut self, accept_rate: f64) {
        let target = self.config.target_accept;
        if accept_rate > target {
            self.step_size *= 1.02;
        } else {
            self.step_size *= 0.98;
        }
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(2.0 * PI * u2)
    }

    fn uniform(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.rng_state as f64 / u64::MAX as f64
    }
}

// =============================================================================
// NUTS (No-U-Turn Sampler)
// =============================================================================

/// No-U-Turn Sampler (NUTS) - automatic HMC tuning
pub struct NUTS {
    config: MCMCConfig,
    n_dims: usize,
    step_size: f64,
    rng_state: u64,
}

impl NUTS {
    /// Create new NUTS sampler
    pub fn new(config: MCMCConfig, n_dims: usize) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};

        Self {
            step_size: config.initial_step_size,
            config,
            n_dims,
            rng_state: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        }
    }

    /// Sample using NUTS with automatic trajectory length
    pub fn sample<F, G>(
        &mut self,
        log_posterior: &F,
        grad_log_posterior: &G,
        initial: &[f64],
        n_samples: usize,
    ) -> MCMCResult
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        let total_samples = self.config.n_warmup + n_samples * self.config.thin;
        let mut chain = Vec::with_capacity(total_samples);
        let mut current = initial.to_vec();

        let mut n_accept = 0;

        for iter in 0..total_samples {
            // Sample momentum
            let momentum: Vec<f64> = (0..self.n_dims).map(|_| self.normal()).collect();

            // Build tree until U-turn
            let (proposed, proposed_momentum, accept) =
                self.build_tree(&current, &momentum, log_posterior, grad_log_posterior);

            if accept {
                current = proposed;
                n_accept += 1;
            } else {
                // Keep current state but still add momentum result to chain
                // In NUTS, we accept the whole trajectory or reject it
            }

            chain.push(current.clone());

            // Adaptive step size tuning during warmup
            if iter < self.config.n_warmup && iter % 100 == 0 && iter > 0 {
                let recent_accept = n_accept as f64 / (iter + 1) as f64;
                self.adapt_step_size(recent_accept);
            }
        }

        let samples: Vec<Vec<f64>> = chain
            .into_iter()
            .skip(self.config.n_warmup)
            .step_by(self.config.thin)
            .collect();

        let accept_rate = n_accept as f64 / total_samples as f64;
        let ess = compute_ess(&samples);

        MCMCResult {
            samples,
            accept_rate,
            ess,
            r_hat: None,
            final_step_size: self.step_size,
        }
    }

    /// Build binary tree until U-turn detected
    fn build_tree<F, G>(
        &mut self,
        q: &[f64],
        p: &[f64],
        log_posterior: &F,
        grad_log_posterior: &G,
    ) -> (Vec<f64>, Vec<f64>, bool)
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        let mut q_minus = q.to_vec();
        let mut q_plus = q.to_vec();
        let mut p_minus = p.to_vec();
        let mut p_plus = p.to_vec();

        let mut j = 0; // Tree depth
        let mut q_proposal = q.to_vec();
        let mut p_proposal = p.to_vec();
        let mut n = 1; // Number of valid states

        let joint0 = self.joint_log_prob(&q, &p, log_posterior);

        // Build tree until U-turn or max depth
        while j < self.config.max_tree_depth {
            // Choose direction: forward (+1) or backward (-1)
            let direction = if self.uniform() < 0.5 { -1.0 } else { 1.0 };

            if direction > 0.0 {
                // Build tree forward
                let (q_new, p_new, _, _, _, n_new, accept) = self.build_tree_recursive(
                    &q_plus,
                    &p_plus,
                    direction,
                    j,
                    joint0,
                    log_posterior,
                    grad_log_posterior,
                );

                if accept {
                    q_plus = q_new.clone();
                    p_plus = p_new.clone();

                    // Randomly select from new states
                    if self.uniform() < (n_new as f64 / (n + n_new) as f64) {
                        q_proposal = q_new;
                        p_proposal = p_new;
                    }
                    n += n_new;
                } else {
                    break; // U-turn detected
                }
            } else {
                // Build tree backward
                let (q_new, p_new, _, _, _, n_new, accept) = self.build_tree_recursive(
                    &q_minus,
                    &p_minus,
                    direction,
                    j,
                    joint0,
                    log_posterior,
                    grad_log_posterior,
                );

                if accept {
                    q_minus = q_new.clone();
                    p_minus = p_new.clone();

                    if self.uniform() < (n_new as f64 / (n + n_new) as f64) {
                        q_proposal = q_new;
                        p_proposal = p_new;
                    }
                    n += n_new;
                } else {
                    break; // U-turn detected
                }
            }

            // Check for U-turn: trajectory doubling back
            if self.is_u_turn(&q_minus, &q_plus, &p_minus, &p_plus) {
                break;
            }

            j += 1;
        }

        let accept = n > 0;
        (q_proposal, p_proposal, accept)
    }

    /// Recursive tree building (simplified version)
    #[allow(clippy::too_many_arguments)]
    fn build_tree_recursive<F, G>(
        &mut self,
        q: &[f64],
        p: &[f64],
        direction: f64,
        depth: usize,
        joint0: f64,
        log_posterior: &F,
        grad_log_posterior: &G,
    ) -> (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        usize,
        bool,
    )
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        if depth == 0 {
            // Base case: single leapfrog step
            let (q_new, p_new) = self.leapfrog(q, p, direction, log_posterior, grad_log_posterior);

            let joint_new = self.joint_log_prob(&q_new, &p_new, log_posterior);
            let accept = (joint_new - joint0) > libm::log(self.uniform());
            let n = if accept { 1 } else { 0 };

            (
                q_new.clone(),
                p_new.clone(),
                q_new,
                p_new,
                q.to_vec(),
                n,
                true,
            )
        } else {
            // Recursion: build subtree
            let (q_minus, p_minus, q_plus, p_plus, q_prime, n_prime, s_prime) = self
                .build_tree_recursive(
                    q,
                    p,
                    direction,
                    depth - 1,
                    joint0,
                    log_posterior,
                    grad_log_posterior,
                );

            if s_prime {
                let (q_minus2, p_minus2, q_plus2, p_plus2, q_prime2, n_prime2, s_prime2) =
                    if direction > 0.0 {
                        self.build_tree_recursive(
                            &q_plus,
                            &p_plus,
                            direction,
                            depth - 1,
                            joint0,
                            log_posterior,
                            grad_log_posterior,
                        )
                    } else {
                        self.build_tree_recursive(
                            &q_minus,
                            &p_minus,
                            direction,
                            depth - 1,
                            joint0,
                            log_posterior,
                            grad_log_posterior,
                        )
                    };

                let n_total = n_prime + n_prime2;
                let s = s_prime2 && !self.is_u_turn(&q_minus, &q_plus2, &p_minus, &p_plus2);

                let q_out = if self.uniform() < (n_prime2 as f64 / n_total as f64) {
                    q_prime2
                } else {
                    q_prime
                };

                (q_minus2, p_minus2, q_plus2, p_plus2, q_out, n_total, s)
            } else {
                (q_minus, p_minus, q_plus, p_plus, q_prime, n_prime, false)
            }
        }
    }

    /// Single leapfrog step (simplified version)
    fn leapfrog<F, G>(
        &self,
        q: &[f64],
        p: &[f64],
        direction: f64,
        _log_posterior: &F,
        grad_log_posterior: &G,
    ) -> (Vec<f64>, Vec<f64>)
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        let mut q_new = q.to_vec();
        let mut p_new = p.to_vec();
        let eps = direction * self.step_size;

        // Half step for momentum
        let grad = grad_log_posterior(&q_new);
        for i in 0..self.n_dims {
            p_new[i] += 0.5 * eps * grad[i];
        }

        // Full step for position
        for i in 0..self.n_dims {
            q_new[i] += eps * p_new[i];
        }

        // Half step for momentum
        let grad = grad_log_posterior(&q_new);
        for i in 0..self.n_dims {
            p_new[i] += 0.5 * eps * grad[i];
        }

        (q_new, p_new)
    }

    /// Check for U-turn: dot product of momentum with position difference
    fn is_u_turn(&self, q_minus: &[f64], q_plus: &[f64], p_minus: &[f64], p_plus: &[f64]) -> bool {
        let delta_q: Vec<f64> = q_plus
            .iter()
            .zip(q_minus.iter())
            .map(|(&qp, &qm)| qp - qm)
            .collect();

        let dot_minus: f64 = delta_q
            .iter()
            .zip(p_minus.iter())
            .map(|(&dq, &pm)| dq * pm)
            .sum();

        let dot_plus: f64 = delta_q
            .iter()
            .zip(p_plus.iter())
            .map(|(&dq, &pp)| dq * pp)
            .sum();

        // U-turn if trajectory is turning back
        dot_minus < 0.0 || dot_plus < 0.0
    }

    /// Compute joint log probability: log p(q, p) = log p(q) - 0.5 * p^T * p
    fn joint_log_prob<F>(&self, q: &[f64], p: &[f64], log_posterior: &F) -> f64
    where
        F: Fn(&[f64]) -> f64,
    {
        let kinetic: f64 = p.iter().map(|&pi| 0.5 * pi * pi).sum();
        log_posterior(q) - kinetic
    }

    /// Adapt step size based on acceptance rate
    fn adapt_step_size(&mut self, accept_rate: f64) {
        let target = self.config.target_accept;
        if accept_rate > target {
            self.step_size *= 1.01;
        } else {
            self.step_size *= 0.99;
        }
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(2.0 * PI * u2)
    }

    fn uniform(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.rng_state as f64 / u64::MAX as f64
    }
}

// =============================================================================
// Convergence Diagnostics
// =============================================================================

/// Compute effective sample size (ESS) using autocorrelation
fn compute_ess(samples: &[Vec<f64>]) -> Vec<f64> {
    let n_dims = samples[0].len();
    let n_samples = samples.len();

    (0..n_dims)
        .map(|dim| {
            // Extract time series for this dimension
            let series: Vec<f64> = samples.iter().map(|s| s[dim]).collect();

            // Compute autocorrelation at different lags
            let mean = series.iter().sum::<f64>() / n_samples as f64;
            let var = series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;

            // Autocorrelation sum (up to lag where autocorr drops below 0.05)
            let mut autocorr_sum = 1.0;
            for lag in 1..n_samples.min(100) {
                let autocorr = compute_autocorr(&series, lag, mean, var);
                if autocorr < 0.05 {
                    break;
                }
                autocorr_sum += 2.0 * autocorr;
            }

            n_samples as f64 / autocorr_sum
        })
        .collect()
}

/// Compute autocorrelation at given lag
fn compute_autocorr(series: &[f64], lag: usize, mean: f64, var: f64) -> f64 {
    let n = series.len();
    let mut sum = 0.0;

    for i in 0..(n - lag) {
        sum += (series[i] - mean) * (series[i + lag] - mean);
    }

    sum / ((n - lag) as f64 * var)
}

/// Compute Gelman-Rubin diagnostic (R-hat) from multiple chains
pub fn compute_r_hat(chains: &[Vec<Vec<f64>>]) -> Vec<f64> {
    if chains.len() < 2 {
        return vec![1.0; chains[0][0].len()];
    }

    let n_chains = chains.len();
    let n_samples = chains[0].len();
    let n_dims = chains[0][0].len();

    (0..n_dims)
        .map(|dim| {
            // Chain means
            let chain_means: Vec<f64> = chains
                .iter()
                .map(|chain| chain.iter().map(|s| s[dim]).sum::<f64>() / n_samples as f64)
                .collect();

            // Overall mean
            let overall_mean = chain_means.iter().sum::<f64>() / n_chains as f64;

            // Between-chain variance
            let b = n_samples as f64
                * chain_means
                    .iter()
                    .map(|&m| (m - overall_mean).powi(2))
                    .sum::<f64>()
                / (n_chains - 1) as f64;

            // Within-chain variance
            let w = chains
                .iter()
                .enumerate()
                .map(|(i, chain)| {
                    chain
                        .iter()
                        .map(|s| (s[dim] - chain_means[i]).powi(2))
                        .sum::<f64>()
                        / (n_samples - 1) as f64
                })
                .sum::<f64>()
                / n_chains as f64;

            // Pooled variance estimate
            let var_plus = ((n_samples - 1) as f64 * w + b) / n_samples as f64;

            // R-hat
            libm::sqrt(var_plus / w)
        })
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metropolis_hastings_normal() {
        // Sample from standard normal
        let log_posterior = |x: &[f64]| -0.5 * x[0].powi(2);

        let config = MCMCConfig {
            n_warmup: 500,
            thin: 1,
            ..Default::default()
        };

        let mut sampler = MetropolisHastings::new(config, 1);
        let result = sampler.sample(&log_posterior, &[0.0], 5000);

        let mean = result.mean();
        let std = result.std_dev();

        // Should be close to N(0, 1)
        assert!((mean[0] - 0.0).abs() < 0.1);
        assert!((std[0] - 1.0).abs() < 0.1);
        assert!(result.accept_rate > 0.1);
    }

    #[test]
    fn test_hmc_normal() {
        // Sample from standard normal
        let log_posterior = |x: &[f64]| -0.5 * x[0].powi(2);
        let grad_log_posterior = |x: &[f64]| vec![-x[0]];

        let config = MCMCConfig::for_hmc();
        let mut sampler = HamiltonianMC::new(config, 1);
        let result = sampler.sample(&log_posterior, &grad_log_posterior, &[0.0], 2000);

        let mean = result.mean();
        let std = result.std_dev();

        assert!((mean[0] - 0.0).abs() < 0.15);
        assert!((std[0] - 1.0).abs() < 0.15);
        assert!(result.accept_rate > 0.4); // HMC should have higher acceptance
    }

    #[test]
    fn test_ess_computation() {
        // Perfect independent samples should have ESS ≈ N
        let n = 1000;
        let samples: Vec<Vec<f64>> = (0..n).map(|i| vec![(i as f64) / n as f64]).collect();

        let ess = compute_ess(&samples);
        assert!(ess[0] > 0.5 * n as f64); // Should be substantial
    }

    #[test]
    fn test_r_hat_convergence() {
        // Two identical chains should have R-hat ≈ 1
        let chain1: Vec<Vec<f64>> = (0..1000).map(|_| vec![0.5]).collect();
        let chain2: Vec<Vec<f64>> = (0..1000).map(|_| vec![0.5]).collect();

        let r_hat = compute_r_hat(&[chain1, chain2]);
        assert!((r_hat[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_credible_interval() {
        // Samples from uniform should have known quantiles
        let samples: Vec<Vec<f64>> = (0..1000).map(|i| vec![(i as f64) / 1000.0]).collect();

        let result = MCMCResult {
            samples,
            accept_rate: 1.0,
            ess: vec![1000.0],
            r_hat: None,
            final_step_size: 0.1,
        };

        let ci = result.credible_interval(0.95);
        // 95% CI should be approximately [0.025, 0.975]
        assert!((ci[0].0 - 0.025).abs() < 0.05);
        assert!((ci[0].1 - 0.975).abs() < 0.05);
    }

    #[test]
    fn test_nuts_normal() {
        // Sample from standard normal using NUTS
        let log_posterior = |x: &[f64]| -0.5 * x[0].powi(2);
        let grad_log_posterior = |x: &[f64]| vec![-x[0]];

        let config = MCMCConfig::for_nuts();
        let mut sampler = NUTS::new(config, 1);
        let result = sampler.sample(&log_posterior, &grad_log_posterior, &[0.0], 1000);

        let mean = result.mean();
        let std = result.std_dev();

        assert!((mean[0] - 0.0).abs() < 0.2);
        assert!((std[0] - 1.0).abs() < 0.2);
        // NUTS should have good acceptance
        assert!(result.accept_rate > 0.3);
    }
}
