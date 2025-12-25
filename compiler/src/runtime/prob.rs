//! Probabilistic Programming Runtime (Prob Effect)
//!
//! This module provides the runtime for probabilistic programming in Sounio,
//! enabling Bayesian inference through the Prob effect.
//!
//! # Core Operations
//!
//! - `sample(dist)` - Draw a sample from a probability distribution
//! - `observe(dist, value)` - Condition on observed data (likelihood)
//! - `infer(model, data, method)` - Perform posterior inference
//!
//! # Supported Distributions
//!
//! - Normal(mean, std)
//! - Uniform(low, high)
//! - Beta(alpha, beta)
//! - Gamma(shape, rate)
//! - Exponential(rate)
//! - Bernoulli(p)
//! - Poisson(lambda)
//! - Categorical(probs)
//!
//! # Inference Methods
//!
//! - Metropolis-Hastings (MH)
//! - Hamiltonian Monte Carlo (HMC)
//! - Variational Inference (VI) - future
//!
//! # Example
//!
//! ```d
//! fn coin_model() with Prob {
//!     let p = sample(Beta(1.0, 1.0));  // Prior
//!     observe(Bernoulli(p), true);      // Likelihood
//!     observe(Bernoulli(p), true);
//!     observe(Bernoulli(p), false);
//!     p
//! }
//!
//! let posterior = infer(coin_model, MH { samples: 10000 });
//! ```

use std::collections::HashMap;
use std::f64::consts::PI;

/// Random number generator state (xorshift128+)
#[derive(Clone)]
pub struct Rng {
    state: [u64; 2],
}

impl Rng {
    /// Create new RNG with seed
    pub fn new(seed: u64) -> Self {
        Self {
            state: [seed, seed.wrapping_mul(0x9E3779B97F4A7C15)],
        }
    }

    /// Generate random u64
    pub fn next_u64(&mut self) -> u64 {
        let mut s1 = self.state[0];
        let s0 = self.state[1];
        let result = s0.wrapping_add(s1);
        self.state[0] = s0;
        s1 ^= s1 << 23;
        self.state[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
        result
    }

    /// Generate uniform f64 in [0, 1)
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Generate standard normal using Box-Muller
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

impl Default for Rng {
    fn default() -> Self {
        Self::new(42)
    }
}

/// Probability distribution
#[derive(Debug, Clone)]
pub enum Distribution {
    /// Normal distribution: N(mean, std)
    Normal { mean: f64, std: f64 },
    /// Uniform distribution: U(low, high)
    Uniform { low: f64, high: f64 },
    /// Beta distribution: Beta(alpha, beta)
    Beta { alpha: f64, beta: f64 },
    /// Gamma distribution: Gamma(shape, rate)
    Gamma { shape: f64, rate: f64 },
    /// Exponential distribution: Exp(rate)
    Exponential { rate: f64 },
    /// Bernoulli distribution: Bernoulli(p)
    Bernoulli { p: f64 },
    /// Poisson distribution: Poisson(lambda)
    Poisson { lambda: f64 },
    /// Categorical distribution
    Categorical { probs: Vec<f64> },
    /// Dirichlet distribution
    Dirichlet { alpha: Vec<f64> },
    /// Multivariate Normal
    MultivariateNormal { mean: Vec<f64>, cov: Vec<Vec<f64>> },
}

impl Distribution {
    /// Sample from the distribution
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        match self {
            Distribution::Normal { mean, std } => mean + std * rng.next_normal(),
            Distribution::Uniform { low, high } => low + (high - low) * rng.next_f64(),
            Distribution::Beta { alpha, beta } => {
                // Use Gamma sampling: X ~ Gamma(alpha), Y ~ Gamma(beta), X/(X+Y) ~ Beta(alpha, beta)
                let x = sample_gamma(*alpha, 1.0, rng);
                let y = sample_gamma(*beta, 1.0, rng);
                x / (x + y)
            }
            Distribution::Gamma { shape, rate } => sample_gamma(*shape, *rate, rng),
            Distribution::Exponential { rate } => -rng.next_f64().ln() / rate,
            Distribution::Bernoulli { p } => {
                if rng.next_f64() < *p {
                    1.0
                } else {
                    0.0
                }
            }
            Distribution::Poisson { lambda } => {
                let l = (-lambda).exp();
                let mut k = 0.0;
                let mut p = 1.0;
                loop {
                    k += 1.0;
                    p *= rng.next_f64();
                    if p <= l {
                        return k - 1.0;
                    }
                }
            }
            Distribution::Categorical { probs } => {
                let u = rng.next_f64();
                let mut cumsum = 0.0;
                for (i, &p) in probs.iter().enumerate() {
                    cumsum += p;
                    if u < cumsum {
                        return i as f64;
                    }
                }
                (probs.len() - 1) as f64
            }
            Distribution::Dirichlet { alpha } => {
                // Return first component (full vector would need Vec return)
                let samples: Vec<f64> = alpha.iter().map(|&a| sample_gamma(a, 1.0, rng)).collect();
                let sum: f64 = samples.iter().sum();
                samples[0] / sum
            }
            Distribution::MultivariateNormal { mean, .. } => {
                // Return first component (full vector would need Vec return)
                mean[0] + rng.next_normal()
            }
        }
    }

    /// Log probability density at x
    pub fn log_prob(&self, x: f64) -> f64 {
        match self {
            Distribution::Normal { mean, std } => {
                let z = (x - mean) / std;
                -0.5 * z * z - std.ln() - 0.5 * (2.0 * PI).ln()
            }
            Distribution::Uniform { low, high } => {
                if x >= *low && x <= *high {
                    -(high - low).ln()
                } else {
                    f64::NEG_INFINITY
                }
            }
            Distribution::Beta { alpha, beta } => {
                if x <= 0.0 || x >= 1.0 {
                    return f64::NEG_INFINITY;
                }
                (alpha - 1.0) * x.ln() + (beta - 1.0) * (1.0 - x).ln() - log_beta(*alpha, *beta)
            }
            Distribution::Gamma { shape, rate } => {
                if x <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                shape * rate.ln() - log_gamma(*shape) + (shape - 1.0) * x.ln() - rate * x
            }
            Distribution::Exponential { rate } => {
                if x < 0.0 {
                    return f64::NEG_INFINITY;
                }
                rate.ln() - rate * x
            }
            Distribution::Bernoulli { p } => {
                if (x - 1.0).abs() < 0.01 {
                    p.ln()
                } else if x.abs() < 0.01 {
                    (1.0 - p).ln()
                } else {
                    f64::NEG_INFINITY
                }
            }
            Distribution::Poisson { lambda } => {
                let k = x.round() as i64;
                if k < 0 {
                    return f64::NEG_INFINITY;
                }
                k as f64 * lambda.ln() - *lambda - log_factorial(k as u64)
            }
            Distribution::Categorical { probs } => {
                let i = x.round() as usize;
                if i < probs.len() {
                    probs[i].ln()
                } else {
                    f64::NEG_INFINITY
                }
            }
            Distribution::Dirichlet { .. } | Distribution::MultivariateNormal { .. } => {
                // Multivariate - would need vector input
                0.0
            }
        }
    }

    /// Probability density at x
    pub fn prob(&self, x: f64) -> f64 {
        self.log_prob(x).exp()
    }
}

/// Sample from Gamma distribution using Marsaglia and Tsang's method
fn sample_gamma(shape: f64, rate: f64, rng: &mut Rng) -> f64 {
    if shape < 1.0 {
        // Use Ahrens-Dieter for shape < 1
        let u = rng.next_f64();
        sample_gamma(1.0 + shape, rate, rng) * u.powf(1.0 / shape)
    } else {
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let mut x = rng.next_normal();
            let mut v = 1.0 + c * x;
            while v <= 0.0 {
                x = rng.next_normal();
                v = 1.0 + c * x;
            }
            v = v * v * v;
            let u = rng.next_f64();
            if u < 1.0 - 0.0331 * x * x * x * x {
                return d * v / rate;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v / rate;
            }
        }
    }
}

/// Log gamma function (Stirling approximation for large values)
fn log_gamma(x: f64) -> f64 {
    if x < 0.5 {
        // Reflection formula
        PI.ln() - (PI * x).sin().ln() - log_gamma(1.0 - x)
    } else if x < 7.0 {
        // Use recurrence relation to push to larger value
        let mut result = 0.0;
        let mut xx = x;
        while xx < 7.0 {
            result -= xx.ln();
            xx += 1.0;
        }
        result + log_gamma(xx)
    } else {
        // Stirling's approximation
        (x - 0.5) * x.ln() - x + 0.5 * (2.0 * PI).ln() + 1.0 / (12.0 * x)
            - 1.0 / (360.0 * x * x * x)
    }
}

/// Log beta function
fn log_beta(a: f64, b: f64) -> f64 {
    log_gamma(a) + log_gamma(b) - log_gamma(a + b)
}

/// Log factorial
fn log_factorial(n: u64) -> f64 {
    log_gamma(n as f64 + 1.0)
}

/// Inference method configuration
#[derive(Debug, Clone)]
pub enum InferenceMethod {
    /// Metropolis-Hastings
    MetropolisHastings {
        samples: usize,
        burn_in: usize,
        proposal_std: f64,
    },
    /// Hamiltonian Monte Carlo
    HMC {
        samples: usize,
        burn_in: usize,
        step_size: f64,
        num_steps: usize,
    },
    /// Importance Sampling
    ImportanceSampling { samples: usize },
    /// Rejection Sampling
    RejectionSampling {
        samples: usize,
        proposal: Box<Distribution>,
        max_ratio: f64,
    },
}

impl Default for InferenceMethod {
    fn default() -> Self {
        InferenceMethod::MetropolisHastings {
            samples: 10000,
            burn_in: 1000,
            proposal_std: 0.1,
        }
    }
}

/// Trace of sampled values
#[derive(Debug, Clone)]
pub struct Trace {
    /// Variable name -> samples
    pub samples: HashMap<String, Vec<f64>>,
    /// Log probabilities for each sample
    pub log_probs: Vec<f64>,
    /// Acceptance rate (for MCMC)
    pub acceptance_rate: f64,
}

impl Trace {
    pub fn new() -> Self {
        Self {
            samples: HashMap::new(),
            log_probs: Vec::new(),
            acceptance_rate: 0.0,
        }
    }

    /// Get mean of samples for a variable
    pub fn mean(&self, var: &str) -> Option<f64> {
        self.samples
            .get(var)
            .map(|s| s.iter().sum::<f64>() / s.len() as f64)
    }

    /// Get standard deviation of samples
    pub fn std(&self, var: &str) -> Option<f64> {
        self.samples.get(var).map(|s| {
            let mean = s.iter().sum::<f64>() / s.len() as f64;
            let var = s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / s.len() as f64;
            var.sqrt()
        })
    }

    /// Get percentile
    pub fn percentile(&self, var: &str, p: f64) -> Option<f64> {
        self.samples.get(var).map(|s| {
            let mut sorted = s.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let idx = ((p / 100.0) * sorted.len() as f64) as usize;
            sorted[idx.min(sorted.len() - 1)]
        })
    }

    /// Get 95% credible interval
    pub fn credible_interval_95(&self, var: &str) -> Option<(f64, f64)> {
        Some((self.percentile(var, 2.5)?, self.percentile(var, 97.5)?))
    }
}

impl Default for Trace {
    fn default() -> Self {
        Self::new()
    }
}

/// Probabilistic program execution context
pub struct ProbContext {
    /// Random number generator
    rng: Rng,
    /// Current log probability
    log_prob: f64,
    /// Sampled values in current execution
    values: HashMap<String, f64>,
    /// Sample counter for auto-naming
    sample_counter: usize,
}

impl ProbContext {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
            log_prob: 0.0,
            values: HashMap::new(),
            sample_counter: 0,
        }
    }

    /// Sample from distribution
    pub fn sample(&mut self, name: Option<&str>, dist: &Distribution) -> f64 {
        let value = dist.sample(&mut self.rng);
        let var_name = name.map(|s| s.to_string()).unwrap_or_else(|| {
            self.sample_counter += 1;
            format!("__sample_{}", self.sample_counter)
        });
        self.values.insert(var_name, value);
        value
    }

    /// Sample with prior log probability contribution
    pub fn sample_with_prior(&mut self, name: Option<&str>, dist: &Distribution) -> f64 {
        let value = self.sample(name, dist);
        self.log_prob += dist.log_prob(value);
        value
    }

    /// Observe (condition on) a value
    pub fn observe(&mut self, dist: &Distribution, value: f64) {
        self.log_prob += dist.log_prob(value);
    }

    /// Get current log probability
    pub fn get_log_prob(&self) -> f64 {
        self.log_prob
    }

    /// Reset for new execution
    pub fn reset(&mut self) {
        self.log_prob = 0.0;
        self.values.clear();
        self.sample_counter = 0;
    }

    /// Get sampled values
    pub fn get_values(&self) -> &HashMap<String, f64> {
        &self.values
    }
}

/// Probabilistic runtime for inference
pub struct ProbRuntime {
    rng: Rng,
}

impl ProbRuntime {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
        }
    }

    /// Run Metropolis-Hastings inference
    pub fn metropolis_hastings<F>(
        &mut self,
        model: F,
        samples: usize,
        burn_in: usize,
        proposal_std: f64,
    ) -> Trace
    where
        F: Fn(&mut ProbContext) -> f64,
    {
        let mut trace = Trace::new();
        let mut accepted = 0usize;

        // Initialize
        let mut ctx = ProbContext::new(self.rng.next_u64());
        let _current_value = model(&mut ctx);
        let mut current_log_prob = ctx.get_log_prob();
        let mut current_params = ctx.get_values().clone();

        for i in 0..(samples + burn_in) {
            // Propose new state
            ctx.reset();

            // Perturb parameters
            for (name, &value) in &current_params {
                let proposed = value + proposal_std * self.rng.next_normal();
                ctx.values.insert(name.clone(), proposed);
            }

            // Evaluate model at proposed state
            let _proposed_value = model(&mut ctx);
            let proposed_log_prob = ctx.get_log_prob();

            // Accept/reject
            let log_alpha = proposed_log_prob - current_log_prob;
            if log_alpha >= 0.0 || self.rng.next_f64().ln() < log_alpha {
                current_log_prob = proposed_log_prob;
                current_params = ctx.get_values().clone();
                accepted += 1;
            }

            // Record after burn-in
            if i >= burn_in {
                for (name, &value) in &current_params {
                    trace.samples.entry(name.clone()).or_default().push(value);
                }
                trace.log_probs.push(current_log_prob);
            }
        }

        trace.acceptance_rate = accepted as f64 / (samples + burn_in) as f64;
        trace
    }

    /// Run Hamiltonian Monte Carlo
    pub fn hmc<F, G>(
        &mut self,
        log_prob: F,
        grad_log_prob: G,
        initial: &[f64],
        samples: usize,
        burn_in: usize,
        step_size: f64,
        num_steps: usize,
    ) -> Vec<Vec<f64>>
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        let dim = initial.len();
        let mut results = Vec::with_capacity(samples);
        let mut accepted = 0usize;

        let mut q = initial.to_vec();

        for i in 0..(samples + burn_in) {
            // Sample momentum
            let mut p: Vec<f64> = (0..dim).map(|_| self.rng.next_normal()).collect();

            // Save initial state
            let q_init = q.clone();
            let p_init = p.clone();

            // Leapfrog integration
            let mut grad = grad_log_prob(&q);

            // Half step for momentum
            for j in 0..dim {
                p[j] += 0.5 * step_size * grad[j];
            }

            // Full steps
            for _ in 0..num_steps {
                // Full step for position
                for j in 0..dim {
                    q[j] += step_size * p[j];
                }

                // Full step for momentum (except at end)
                grad = grad_log_prob(&q);
                for j in 0..dim {
                    p[j] += step_size * grad[j];
                }
            }

            // Half step for momentum at end
            for j in 0..dim {
                p[j] -= 0.5 * step_size * grad[j];
            }

            // Negate momentum for reversibility
            for j in 0..dim {
                p[j] = -p[j];
            }

            // Compute Hamiltonians
            let kinetic_init: f64 = p_init.iter().map(|x| x * x).sum::<f64>() / 2.0;
            let kinetic_prop: f64 = p.iter().map(|x| x * x).sum::<f64>() / 2.0;
            let h_init = -log_prob(&q_init) + kinetic_init;
            let h_prop = -log_prob(&q) + kinetic_prop;

            // Accept/reject
            let log_alpha = h_init - h_prop;
            if log_alpha >= 0.0 || self.rng.next_f64().ln() < log_alpha {
                accepted += 1;
                // q already updated
            } else {
                q = q_init;
            }

            // Record after burn-in
            if i >= burn_in {
                results.push(q.clone());
            }
        }

        results
    }

    /// Simple importance sampling
    pub fn importance_sampling<F>(&mut self, model: F, samples: usize) -> (Vec<f64>, Vec<f64>)
    where
        F: Fn(&mut ProbContext) -> f64,
    {
        let mut values = Vec::with_capacity(samples);
        let mut weights = Vec::with_capacity(samples);

        for _ in 0..samples {
            let mut ctx = ProbContext::new(self.rng.next_u64());
            let value = model(&mut ctx);
            values.push(value);
            weights.push(ctx.get_log_prob().exp());
        }

        (values, weights)
    }
}

impl Default for ProbRuntime {
    fn default() -> Self {
        Self::new(42)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng() {
        let mut rng = Rng::new(12345);
        let samples: Vec<f64> = (0..1000).map(|_| rng.next_f64()).collect();

        // Check range
        for &s in &samples {
            assert!(s >= 0.0 && s < 1.0);
        }

        // Check mean is approximately 0.5
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((mean - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_normal_distribution() {
        let mut rng = Rng::new(42);
        let dist = Distribution::Normal {
            mean: 0.0,
            std: 1.0,
        };

        let samples: Vec<f64> = (0..10000).map(|_| dist.sample(&mut rng)).collect();
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let var: f64 =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;

        assert!((mean).abs() < 0.05, "mean should be ~0");
        assert!((var - 1.0).abs() < 0.1, "variance should be ~1");
    }

    #[test]
    fn test_beta_distribution() {
        let mut rng = Rng::new(42);
        let dist = Distribution::Beta {
            alpha: 2.0,
            beta: 5.0,
        };

        let samples: Vec<f64> = (0..10000).map(|_| dist.sample(&mut rng)).collect();
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;

        // Mean of Beta(a, b) = a / (a + b) = 2/7 ≈ 0.286
        assert!((mean - 2.0 / 7.0).abs() < 0.02, "mean should be ~0.286");
    }

    #[test]
    fn test_log_prob_normal() {
        let dist = Distribution::Normal {
            mean: 0.0,
            std: 1.0,
        };

        // PDF at x=0 for N(0,1) = 1/sqrt(2π) ≈ 0.399
        let lp = dist.log_prob(0.0);
        assert!((lp.exp() - 0.399).abs() < 0.01);
    }

    #[test]
    fn test_observe() {
        let mut ctx = ProbContext::new(42);

        ctx.observe(
            &Distribution::Normal {
                mean: 0.0,
                std: 1.0,
            },
            0.0,
        );
        ctx.observe(
            &Distribution::Normal {
                mean: 0.0,
                std: 1.0,
            },
            0.0,
        );

        // Two observations at the mode should give log_prob ≈ 2 * log(0.399)
        assert!(ctx.get_log_prob() < 0.0);
        assert!((ctx.get_log_prob() - 2.0 * 0.399_f64.ln()).abs() < 0.1);
    }

    #[test]
    fn test_metropolis_hastings() {
        let mut runtime = ProbRuntime::new(42);

        // Simple model: sample from N(3, 1)
        let trace = runtime.metropolis_hastings(
            |ctx| {
                let x = ctx.sample_with_prior(
                    Some("x"),
                    &Distribution::Normal {
                        mean: 3.0,
                        std: 1.0,
                    },
                );
                x
            },
            5000,
            1000,
            0.5,
        );

        let mean = trace.mean("x").unwrap();
        assert!(
            (mean - 3.0).abs() < 0.2,
            "mean should be ~3.0, got {}",
            mean
        );
    }

    #[test]
    fn test_coin_flip_inference() {
        let mut runtime = ProbRuntime::new(42);

        // Bayesian coin flip: Beta(1,1) prior, observe 7 heads, 3 tails
        let trace = runtime.metropolis_hastings(
            |ctx| {
                let p = ctx.sample_with_prior(
                    Some("p"),
                    &Distribution::Beta {
                        alpha: 1.0,
                        beta: 1.0,
                    },
                );

                // Observe outcomes
                for _ in 0..7 {
                    ctx.observe(&Distribution::Bernoulli { p }, 1.0);
                }
                for _ in 0..3 {
                    ctx.observe(&Distribution::Bernoulli { p }, 0.0);
                }

                p
            },
            10000,
            2000,
            0.1,
        );

        let mean = trace.mean("p").unwrap();
        // Posterior: Beta(1+7, 1+3) = Beta(8, 4), mean = 8/12 = 0.667
        assert!(
            (mean - 0.667).abs() < 0.1,
            "posterior mean should be ~0.667, got {}",
            mean
        );
    }

    #[test]
    fn test_trace_statistics() {
        let mut trace = Trace::new();
        trace
            .samples
            .insert("x".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        assert_eq!(trace.mean("x"), Some(3.0));

        let std = trace.std("x").unwrap();
        // Std of [1,2,3,4,5] = sqrt(2) ≈ 1.414
        assert!((std - 1.414).abs() < 0.01);
    }

    #[test]
    fn test_gamma_sampling() {
        let mut rng = Rng::new(42);
        let dist = Distribution::Gamma {
            shape: 2.0,
            rate: 1.0,
        };

        let samples: Vec<f64> = (0..10000).map(|_| dist.sample(&mut rng)).collect();
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;

        // Mean of Gamma(k, θ) with rate parameterization = shape/rate = 2
        assert!(
            (mean - 2.0).abs() < 0.1,
            "mean should be ~2.0, got {}",
            mean
        );
    }
}
