//! Probabilistic Effect Handler
//!
//! This module implements the `Prob` effect handler for probabilistic programming
//! through algebraic effects. The Prob effect enables Bayesian inference and
//! stochastic computation with explicit effect tracking.
//!
//! # Effect Operations
//!
//! | Operation     | Arguments                        | Return Type | Confidence |
//! |--------------|----------------------------------|-------------|------------|
//! | `sample`     | `(dist: Distribution)`           | `Float`     | 0.9        |
//! | `observe`    | `(dist: Distribution, val: Float)` | `Unit`    | 1.0        |
//! | `condition`  | `(predicate: Bool)`              | `Unit`      | 1.0        |
//! | `bernoulli`  | `(p: Float)`                     | `Bool`      | 0.9        |
//! | `uniform`    | `(a: Float, b: Float)`           | `Float`     | 0.9        |
//! | `normal`     | `(mean: Float, std: Float)`      | `Float`     | 0.9        |
//! | `categorical`| `(probs: Array<Float>)`          | `Int`       | 0.9        |
//! | `beta`       | `(alpha: Float, beta: Float)`    | `Float`     | 0.9        |
//! | `exponential`| `(lambda: Float)`                | `Float`     | 0.9        |
//! | `log_prob`   | `(dist: Distribution, val: Float)` | `Float`   | 1.0        |
//!
//! # Epistemic Impact
//!
//! Sampling operations degrade epistemic confidence because they introduce
//! stochastic uncertainty. The default confidence factor is 0.9 (10% degradation).
//! Observation and conditioning do not degrade confidence - they constrain the
//! probability space but don't introduce uncertainty.
//!
//! # Example
//!
//! ```text
//! // Sounio code using Prob effect:
//! fn coin_flip() -> bool with Prob {
//!     bernoulli(0.5)
//! }
//!
//! fn bayesian_inference() with Prob {
//!     let mu = normal(0.0, 1.0)
//!     observe(Normal(mu, 0.1), 0.5)  // condition on observation
//!     mu
//! }
//! ```

use crate::effects::handler_capability::{
    Continuation, EpistemicImpact, HandlerCapability, HandlerError, HandlerResult, HandlerState,
    OperationSpec,
};
use crate::effects::linearity::Linearity;
use crate::interp::value::Distribution;
use crate::interp::Value;
use std::f64::consts::PI;
use std::sync::OnceLock;

/// Confidence factor for sampling operations (10% degradation)
const SAMPLE_CONFIDENCE_FACTOR: f64 = 0.9;

/// Key for storing the RNG seed in HandlerState
const RNG_SEED_KEY: &str = "__prob_rng_seed";

/// Key for storing log probability accumulator
const LOG_PROB_KEY: &str = "__prob_log_prob";

/// Key for storing observation count
const OBSERVE_COUNT_KEY: &str = "__prob_observe_count";

/// Static operation specifications for ProbHandler
static PROB_OPERATIONS: OnceLock<Vec<OperationSpec>> = OnceLock::new();

fn get_prob_operations() -> &'static [OperationSpec] {
    PROB_OPERATIONS.get_or_init(|| {
        vec![
            OperationSpec::new("sample", "Float")
                .with_params(vec!["Distribution"])
                .with_confidence_factor(SAMPLE_CONFIDENCE_FACTOR),
            OperationSpec::new("observe", "Unit").with_params(vec!["Distribution", "Float"]),
            OperationSpec::new("condition", "Unit").with_params(vec!["Bool"]),
            OperationSpec::new("bernoulli", "Bool")
                .with_params(vec!["Float"])
                .with_confidence_factor(SAMPLE_CONFIDENCE_FACTOR),
            OperationSpec::new("uniform", "Float")
                .with_params(vec!["Float", "Float"])
                .with_confidence_factor(SAMPLE_CONFIDENCE_FACTOR),
            OperationSpec::new("normal", "Float")
                .with_params(vec!["Float", "Float"])
                .with_confidence_factor(SAMPLE_CONFIDENCE_FACTOR),
            OperationSpec::new("categorical", "Int")
                .with_params(vec!["Array"])
                .with_confidence_factor(SAMPLE_CONFIDENCE_FACTOR),
            OperationSpec::new("beta", "Float")
                .with_params(vec!["Float", "Float"])
                .with_confidence_factor(SAMPLE_CONFIDENCE_FACTOR),
            OperationSpec::new("exponential", "Float")
                .with_params(vec!["Float"])
                .with_confidence_factor(SAMPLE_CONFIDENCE_FACTOR),
            OperationSpec::new("log_prob", "Float").with_params(vec!["Distribution", "Float"]),
            OperationSpec::new("score", "Unit").with_params(vec!["Float"]),
            OperationSpec::new("get_log_prob", "Float").with_params(vec![]),
        ]
    })
}

/// Handler for the Prob (probabilistic) effect
///
/// ProbHandler provides probabilistic programming primitives through algebraic
/// effects. It supports sampling from distributions, conditioning on observations,
/// and computing log probabilities.
///
/// # Random Number Generation
///
/// The handler uses a simple LCG (Linear Congruential Generator) for reproducible
/// pseudo-random numbers. The seed can be set via `set_seed` for deterministic
/// execution during testing.
///
/// # Inference Support
///
/// The handler tracks:
/// - Accumulated log probability (for importance sampling)
/// - Observation count (for diagnostics)
///
/// This enables integration with inference algorithms like:
/// - Importance sampling
/// - Metropolis-Hastings
/// - Sequential Monte Carlo
#[derive(Debug)]
pub struct ProbHandler {
    _private: (),
}

impl Default for ProbHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl ProbHandler {
    /// Create a new ProbHandler
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Get or initialize the RNG seed
    fn get_seed(state: &mut HandlerState) -> u64 {
        match state.named_state.get(RNG_SEED_KEY) {
            Some(Value::Int(seed)) => *seed as u64,
            _ => {
                // Initialize with a default seed based on current time-ish
                let seed = 12345u64;
                state
                    .named_state
                    .insert(RNG_SEED_KEY.to_string(), Value::Int(seed as i64));
                seed
            }
        }
    }

    /// Update the RNG seed (LCG step)
    fn next_seed(state: &mut HandlerState) -> u64 {
        let seed = Self::get_seed(state);
        // LCG parameters (same as glibc)
        let new_seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        state
            .named_state
            .insert(RNG_SEED_KEY.to_string(), Value::Int(new_seed as i64));
        new_seed
    }

    /// Generate a uniform random f64 in [0, 1)
    fn random_uniform(state: &mut HandlerState) -> f64 {
        let seed = Self::next_seed(state);
        // Use upper bits for better randomness
        ((seed >> 16) & 0x7FFFFFFF) as f64 / 0x80000000u64 as f64
    }

    /// Sample from a standard normal using Box-Muller transform
    fn random_normal(state: &mut HandlerState) -> f64 {
        let u1 = Self::random_uniform(state);
        let u2 = Self::random_uniform(state);
        // Avoid log(0)
        let u1 = if u1 < 1e-10 { 1e-10 } else { u1 };
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Get accumulated log probability
    fn get_log_prob_value(state: &HandlerState) -> f64 {
        match state.named_state.get(LOG_PROB_KEY) {
            Some(Value::Float(lp)) => *lp,
            _ => 0.0,
        }
    }

    /// Add to accumulated log probability
    fn add_log_prob(state: &mut HandlerState, lp: f64) {
        let current = Self::get_log_prob_value(state);
        state
            .named_state
            .insert(LOG_PROB_KEY.to_string(), Value::Float(current + lp));
    }

    /// Increment observation count
    fn increment_observe_count(state: &mut HandlerState) {
        let current = match state.named_state.get(OBSERVE_COUNT_KEY) {
            Some(Value::Int(n)) => *n,
            _ => 0,
        };
        state
            .named_state
            .insert(OBSERVE_COUNT_KEY.to_string(), Value::Int(current + 1));
    }

    /// Compute log probability of value under distribution
    fn compute_log_prob(dist: &Distribution, value: f64) -> f64 {
        match dist {
            Distribution::Normal { mean, std } => {
                if *std <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                let z = (value - mean) / std;
                -0.5 * z * z - std.ln() - 0.5 * (2.0 * PI).ln()
            }
            Distribution::Uniform { a, b } => {
                if *a >= *b {
                    return f64::NEG_INFINITY;
                }
                if value >= *a && value < *b {
                    -(b - a).ln()
                } else {
                    f64::NEG_INFINITY
                }
            }
            Distribution::Beta { alpha, beta } => {
                if *alpha <= 0.0 || *beta <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                if value <= 0.0 || value >= 1.0 {
                    return f64::NEG_INFINITY;
                }
                // log Beta PDF: (alpha-1)*log(x) + (beta-1)*log(1-x) - log(B(alpha,beta))
                // Using lgamma for log(B(alpha, beta)) = lgamma(alpha) + lgamma(beta) - lgamma(alpha+beta)
                (alpha - 1.0) * value.ln() + (beta - 1.0) * (1.0 - value).ln()
                    - Self::log_beta(*alpha, *beta)
            }
            Distribution::Exponential { lambda } => {
                if *lambda <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                if value < 0.0 {
                    return f64::NEG_INFINITY;
                }
                lambda.ln() - lambda * value
            }
            Distribution::Categorical { probs } => {
                let idx = value.round() as usize;
                if idx >= probs.len() {
                    return f64::NEG_INFINITY;
                }
                let p = probs[idx];
                if p <= 0.0 {
                    f64::NEG_INFINITY
                } else {
                    p.ln()
                }
            }
        }
    }

    /// Approximate log(Beta(a, b)) using Stirling's approximation
    fn log_beta(a: f64, b: f64) -> f64 {
        Self::log_gamma(a) + Self::log_gamma(b) - Self::log_gamma(a + b)
    }

    /// Stirling's approximation for log(Gamma(x))
    fn log_gamma(x: f64) -> f64 {
        if x <= 0.0 {
            return f64::INFINITY;
        }
        // Stirling's approximation: log(Gamma(x)) ≈ (x - 0.5) * ln(x) - x + 0.5 * ln(2π)
        (x - 0.5) * x.ln() - x + 0.5 * (2.0 * PI).ln()
    }

    /// Sample from a distribution
    fn sample_distribution(dist: &Distribution, state: &mut HandlerState) -> f64 {
        match dist {
            Distribution::Normal { mean, std } => mean + std * Self::random_normal(state),
            Distribution::Uniform { a, b } => a + (b - a) * Self::random_uniform(state),
            Distribution::Beta { alpha, beta } => {
                // Use rejection sampling or approximation
                // Simple approximation using normal for large alpha, beta
                if *alpha > 1.0 && *beta > 1.0 {
                    let mean = alpha / (alpha + beta);
                    let var = (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0));
                    let sample = mean + var.sqrt() * Self::random_normal(state);
                    sample.clamp(0.001, 0.999)
                } else {
                    // Fallback: use uniform and accept/reject (simplified)
                    Self::random_uniform(state)
                }
            }
            Distribution::Exponential { lambda } => {
                let u = Self::random_uniform(state);
                let u = if u < 1e-10 { 1e-10 } else { u };
                -u.ln() / lambda
            }
            Distribution::Categorical { probs } => {
                let u = Self::random_uniform(state);
                let mut cumsum = 0.0;
                for (i, p) in probs.iter().enumerate() {
                    cumsum += p;
                    if u < cumsum {
                        return i as f64;
                    }
                }
                (probs.len() - 1) as f64
            }
        }
    }

    /// Handle sample operation
    fn handle_sample(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Prob",
                "sample",
                "sample requires a distribution argument",
            ));
        }

        let dist = match &args[0] {
            Value::Distribution(d) => d.clone(),
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Prob",
                    "sample",
                    format!("expected Distribution, got {:?}", args[0].type_name()),
                ));
            }
        };

        let value = Self::sample_distribution(&dist, state);
        HandlerResult::Resume(Value::Float(value))
    }

    /// Handle observe operation
    fn handle_observe(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Prob",
                "observe",
                "observe requires distribution and value arguments",
            ));
        }

        let dist = match &args[0] {
            Value::Distribution(d) => d.clone(),
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Prob",
                    "observe",
                    format!("expected Distribution, got {:?}", args[0].type_name()),
                ));
            }
        };

        let value = match &args[1] {
            Value::Float(f) => *f,
            Value::Int(i) => *i as f64,
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Prob",
                    "observe",
                    format!("expected Float for value, got {:?}", args[1].type_name()),
                ));
            }
        };

        // Add log probability to accumulator
        let lp = Self::compute_log_prob(&dist, value);
        Self::add_log_prob(state, lp);
        Self::increment_observe_count(state);

        HandlerResult::Resume(Value::Unit)
    }

    /// Handle condition operation
    fn handle_condition(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Prob",
                "condition",
                "condition requires a boolean predicate",
            ));
        }

        let predicate = match &args[0] {
            Value::Bool(b) => *b,
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Prob",
                    "condition",
                    format!("expected Bool, got {:?}", args[0].type_name()),
                ));
            }
        };

        if predicate {
            HandlerResult::Resume(Value::Unit)
        } else {
            // Condition failed - add -infinity to log prob (rejection)
            Self::add_log_prob(state, f64::NEG_INFINITY);
            HandlerResult::Resume(Value::Unit)
        }
    }

    /// Handle bernoulli operation
    fn handle_bernoulli(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Prob",
                "bernoulli",
                "bernoulli requires a probability argument",
            ));
        }

        let p = match &args[0] {
            Value::Float(f) => *f,
            Value::Int(i) => *i as f64,
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Prob",
                    "bernoulli",
                    format!(
                        "expected Float for probability, got {:?}",
                        args[0].type_name()
                    ),
                ));
            }
        };

        if !(0.0..=1.0).contains(&p) {
            return HandlerResult::Abort(HandlerError::new(
                "Prob",
                "bernoulli",
                format!("probability must be in [0, 1], got {}", p),
            ));
        }

        let u = Self::random_uniform(state);
        HandlerResult::Resume(Value::Bool(u < p))
    }

    /// Handle uniform operation
    fn handle_uniform(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Prob",
                "uniform",
                "uniform requires lower and upper bound arguments",
            ));
        }

        let a = match Self::extract_float(&args[0], "Prob", "uniform", "lower bound") {
            Ok(v) => v,
            Err(result) => return result,
        };
        let b = match Self::extract_float(&args[1], "Prob", "uniform", "upper bound") {
            Ok(v) => v,
            Err(result) => return result,
        };

        if a >= b {
            return HandlerResult::Abort(HandlerError::new(
                "Prob",
                "uniform",
                format!("lower bound must be less than upper bound: {} >= {}", a, b),
            ));
        }

        let value = a + (b - a) * Self::random_uniform(state);
        HandlerResult::Resume(Value::Float(value))
    }

    /// Handle normal operation
    fn handle_normal(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Prob",
                "normal",
                "normal requires mean and std arguments",
            ));
        }

        let mean = match Self::extract_float(&args[0], "Prob", "normal", "mean") {
            Ok(v) => v,
            Err(result) => return result,
        };
        let std = match Self::extract_float(&args[1], "Prob", "normal", "std") {
            Ok(v) => v,
            Err(result) => return result,
        };

        if std <= 0.0 {
            return HandlerResult::Abort(HandlerError::new(
                "Prob",
                "normal",
                format!("std must be positive, got {}", std),
            ));
        }

        let value = mean + std * Self::random_normal(state);
        HandlerResult::Resume(Value::Float(value))
    }

    /// Handle categorical operation
    fn handle_categorical(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Prob",
                "categorical",
                "categorical requires a probability array",
            ));
        }

        let probs_rc = match &args[0] {
            Value::Array(arr) => arr.clone(),
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Prob",
                    "categorical",
                    format!(
                        "expected Array of probabilities, got {:?}",
                        args[0].type_name()
                    ),
                ));
            }
        };

        let probs_ref = probs_rc.borrow();
        let mut probs = Vec::with_capacity(probs_ref.len());

        for (i, v) in probs_ref.iter().enumerate() {
            match v {
                Value::Float(f) => probs.push(*f),
                Value::Int(n) => probs.push(*n as f64),
                _ => {
                    return HandlerResult::Abort(HandlerError::new(
                        "Prob",
                        "categorical",
                        format!("expected Float at index {}, got {:?}", i, v.type_name()),
                    ));
                }
            }
        }
        drop(probs_ref);

        if probs.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Prob",
                "categorical",
                "categorical requires at least one probability",
            ));
        }

        // Normalize probabilities
        let sum: f64 = probs.iter().sum();
        if sum <= 0.0 {
            return HandlerResult::Abort(HandlerError::new(
                "Prob",
                "categorical",
                "probabilities must sum to a positive value",
            ));
        }

        let u = Self::random_uniform(state) * sum;
        let mut cumsum = 0.0;
        for (i, p) in probs.iter().enumerate() {
            cumsum += p;
            if u < cumsum {
                return HandlerResult::Resume(Value::Int(i as i64));
            }
        }

        HandlerResult::Resume(Value::Int((probs.len() - 1) as i64))
    }

    /// Handle beta operation
    fn handle_beta(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Prob",
                "beta",
                "beta requires alpha and beta arguments",
            ));
        }

        let alpha = match Self::extract_float(&args[0], "Prob", "beta", "alpha") {
            Ok(v) => v,
            Err(result) => return result,
        };
        let beta_param = match Self::extract_float(&args[1], "Prob", "beta", "beta") {
            Ok(v) => v,
            Err(result) => return result,
        };

        if alpha <= 0.0 || beta_param <= 0.0 {
            return HandlerResult::Abort(HandlerError::new(
                "Prob",
                "beta",
                format!(
                    "alpha and beta must be positive, got alpha={}, beta={}",
                    alpha, beta_param
                ),
            ));
        }

        let dist = Distribution::Beta {
            alpha,
            beta: beta_param,
        };
        let value = Self::sample_distribution(&dist, state);
        HandlerResult::Resume(Value::Float(value))
    }

    /// Handle exponential operation
    fn handle_exponential(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Prob",
                "exponential",
                "exponential requires a rate (lambda) argument",
            ));
        }

        let lambda = match Self::extract_float(&args[0], "Prob", "exponential", "lambda") {
            Ok(v) => v,
            Err(result) => return result,
        };

        if lambda <= 0.0 {
            return HandlerResult::Abort(HandlerError::new(
                "Prob",
                "exponential",
                format!("lambda must be positive, got {}", lambda),
            ));
        }

        let u = Self::random_uniform(state);
        let u = if u < 1e-10 { 1e-10 } else { u };
        let value = -u.ln() / lambda;
        HandlerResult::Resume(Value::Float(value))
    }

    /// Handle log_prob operation
    fn handle_log_prob(&self, args: &[Value]) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Prob",
                "log_prob",
                "log_prob requires distribution and value arguments",
            ));
        }

        let dist = match &args[0] {
            Value::Distribution(d) => d.clone(),
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Prob",
                    "log_prob",
                    format!("expected Distribution, got {:?}", args[0].type_name()),
                ));
            }
        };

        let value = match Self::extract_float(&args[1], "Prob", "log_prob", "value") {
            Ok(v) => v,
            Err(result) => return result,
        };
        let lp = Self::compute_log_prob(&dist, value);
        HandlerResult::Resume(Value::Float(lp))
    }

    /// Handle score operation (add to log probability)
    fn handle_score(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Prob",
                "score",
                "score requires a log probability argument",
            ));
        }

        let lp = match Self::extract_float(&args[0], "Prob", "score", "log_prob") {
            Ok(v) => v,
            Err(result) => return result,
        };
        Self::add_log_prob(state, lp);
        HandlerResult::Resume(Value::Unit)
    }

    /// Handle get_log_prob operation
    fn handle_get_log_prob(&self, state: &HandlerState) -> HandlerResult {
        let lp = Self::get_log_prob_value(state);
        HandlerResult::Resume(Value::Float(lp))
    }

    /// Extract a float from a Value
    fn extract_float(
        value: &Value,
        effect: &str,
        op: &str,
        param: &str,
    ) -> Result<f64, HandlerResult> {
        match value {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            _ => Err(HandlerResult::Abort(HandlerError::new(
                effect,
                op,
                format!("expected Float for {}, got {:?}", param, value.type_name()),
            ))),
        }
    }

    /// Set the RNG seed for reproducible sampling
    pub fn set_seed(state: &mut HandlerState, seed: u64) {
        state
            .named_state
            .insert(RNG_SEED_KEY.to_string(), Value::Int(seed as i64));
    }

    /// Reset the log probability accumulator
    pub fn reset_log_prob(state: &mut HandlerState) {
        state
            .named_state
            .insert(LOG_PROB_KEY.to_string(), Value::Float(0.0));
    }

    /// Get the observation count
    pub fn get_observe_count(state: &HandlerState) -> i64 {
        match state.named_state.get(OBSERVE_COUNT_KEY) {
            Some(Value::Int(n)) => *n,
            _ => 0,
        }
    }
}

impl HandlerCapability for ProbHandler {
    fn effect_name(&self) -> &str {
        "Prob"
    }

    fn handler_name(&self) -> &str {
        "ProbHandler"
    }

    fn operations(&self) -> &[OperationSpec] {
        get_prob_operations()
    }

    fn handle(
        &self,
        op: &str,
        args: &[Value],
        _cont: Continuation,
        state: &mut HandlerState,
    ) -> HandlerResult {
        match op {
            "sample" => self.handle_sample(args, state),
            "observe" => self.handle_observe(args, state),
            "condition" => self.handle_condition(args, state),
            "bernoulli" => self.handle_bernoulli(args, state),
            "uniform" => self.handle_uniform(args, state),
            "normal" => self.handle_normal(args, state),
            "categorical" => self.handle_categorical(args, state),
            "beta" => self.handle_beta(args, state),
            "exponential" => self.handle_exponential(args, state),
            "log_prob" => self.handle_log_prob(args),
            "score" => self.handle_score(args, state),
            "get_log_prob" => self.handle_get_log_prob(state),
            _ => HandlerResult::Abort(HandlerError::new(
                "Prob",
                op,
                format!("unknown operation: {}", op),
            )),
        }
    }

    fn supports_multi_shot(&self) -> bool {
        // Probabilistic effects benefit from multi-shot for enumeration
        true
    }

    fn operation_linearity(&self, operation: &str) -> Linearity {
        match operation {
            // Sampling operations require multi-shot for proper probabilistic inference
            // Multiple resumptions enable exploring different branches of the probability space
            "sample" | "bernoulli" | "uniform" | "normal" | "categorical" | "beta"
            | "exponential" => Linearity::MultiShot,

            // Observation and conditioning create constraints but don't require multi-shot
            // These are typically one-shot operations that filter the probability space
            "observe" | "condition" | "factor" | "score" => Linearity::ExactlyOnce,

            // Log probability queries are pure functions (can be multi-shot)
            "log_prob" | "get_log_prob" => Linearity::MultiShot,

            // Unknown operations default to linear (conservative)
            _ => Linearity::ExactlyOnce,
        }
    }

    fn epistemic_impact(&self, operation: &str) -> EpistemicImpact {
        match operation {
            // Sampling introduces uncertainty
            "sample" | "bernoulli" | "uniform" | "normal" | "categorical" | "beta"
            | "exponential" => EpistemicImpact::with_confidence(SAMPLE_CONFIDENCE_FACTOR),
            // Observation and conditioning don't introduce uncertainty
            _ => EpistemicImpact::none(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn new_state() -> HandlerState {
        HandlerState::new()
    }

    fn cont() -> Continuation {
        Continuation::new()
    }

    fn seeded_state() -> HandlerState {
        let mut state = new_state();
        ProbHandler::set_seed(&mut state, 42);
        state
    }

    #[test]
    fn test_handler_identity() {
        let handler = ProbHandler::new();
        assert_eq!(handler.effect_name(), "Prob");
        assert_eq!(handler.handler_name(), "ProbHandler");
    }

    #[test]
    fn test_operations_list() {
        let handler = ProbHandler::new();
        let ops = handler.operations();
        let names: Vec<_> = ops.iter().map(|o| o.name.as_str()).collect();
        assert!(names.contains(&"sample"));
        assert!(names.contains(&"observe"));
        assert!(names.contains(&"condition"));
        assert!(names.contains(&"bernoulli"));
        assert!(names.contains(&"uniform"));
        assert!(names.contains(&"normal"));
        assert!(names.contains(&"categorical"));
        assert!(names.contains(&"beta"));
        assert!(names.contains(&"exponential"));
        assert!(names.contains(&"log_prob"));
    }

    #[test]
    fn test_supports_multi_shot() {
        let handler = ProbHandler::new();
        assert!(handler.supports_multi_shot());
    }

    #[test]
    fn test_epistemic_impact_sampling() {
        let handler = ProbHandler::new();

        let sample_impact = handler.epistemic_impact("sample");
        assert!((sample_impact.confidence_factor - 0.9).abs() < 0.001);

        let bernoulli_impact = handler.epistemic_impact("bernoulli");
        assert!((bernoulli_impact.confidence_factor - 0.9).abs() < 0.001);

        let normal_impact = handler.epistemic_impact("normal");
        assert!((normal_impact.confidence_factor - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_epistemic_impact_observe() {
        let handler = ProbHandler::new();

        let observe_impact = handler.epistemic_impact("observe");
        assert!((observe_impact.confidence_factor - 1.0).abs() < 0.001);

        let condition_impact = handler.epistemic_impact("condition");
        assert!((condition_impact.confidence_factor - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_bernoulli_basic() {
        let handler = ProbHandler::new();
        let mut state = seeded_state();

        // Sample multiple times and check we get bools
        for _ in 0..10 {
            let result = handler.handle("bernoulli", &[Value::Float(0.5)], cont(), &mut state);
            match result {
                HandlerResult::Resume(Value::Bool(_)) => {}
                other => panic!("expected Resume(Bool), got {:?}", other),
            }
        }
    }

    #[test]
    fn test_bernoulli_deterministic() {
        let handler = ProbHandler::new();

        // With p=0, always false
        let mut state = seeded_state();
        for _ in 0..5 {
            let result = handler.handle("bernoulli", &[Value::Float(0.0)], cont(), &mut state);
            match result {
                HandlerResult::Resume(Value::Bool(false)) => {}
                other => panic!("expected false, got {:?}", other),
            }
        }

        // With p=1, always true
        let mut state = seeded_state();
        for _ in 0..5 {
            let result = handler.handle("bernoulli", &[Value::Float(1.0)], cont(), &mut state);
            match result {
                HandlerResult::Resume(Value::Bool(true)) => {}
                other => panic!("expected true, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_bernoulli_invalid_prob() {
        let handler = ProbHandler::new();
        let mut state = new_state();

        let result = handler.handle("bernoulli", &[Value::Float(1.5)], cont(), &mut state);
        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("[0, 1]"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_uniform_basic() {
        let handler = ProbHandler::new();
        let mut state = seeded_state();

        let result = handler.handle(
            "uniform",
            &[Value::Float(0.0), Value::Float(1.0)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Float(v)) => {
                assert!(v >= 0.0 && v < 1.0, "uniform should be in [0, 1): {}", v);
            }
            other => panic!("expected Resume(Float), got {:?}", other),
        }
    }

    #[test]
    fn test_uniform_range() {
        let handler = ProbHandler::new();
        let mut state = seeded_state();

        for _ in 0..20 {
            let result = handler.handle(
                "uniform",
                &[Value::Float(10.0), Value::Float(20.0)],
                cont(),
                &mut state,
            );

            match result {
                HandlerResult::Resume(Value::Float(v)) => {
                    assert!(
                        v >= 10.0 && v < 20.0,
                        "uniform should be in [10, 20): {}",
                        v
                    );
                }
                _ => panic!("expected Resume(Float)"),
            }
        }
    }

    #[test]
    fn test_uniform_invalid_range() {
        let handler = ProbHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "uniform",
            &[Value::Float(10.0), Value::Float(5.0)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("less than"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_normal_basic() {
        let handler = ProbHandler::new();
        let mut state = seeded_state();

        let result = handler.handle(
            "normal",
            &[Value::Float(0.0), Value::Float(1.0)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Float(_)) => {}
            other => panic!("expected Resume(Float), got {:?}", other),
        }
    }

    #[test]
    fn test_normal_invalid_std() {
        let handler = ProbHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "normal",
            &[Value::Float(0.0), Value::Float(-1.0)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("positive"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_categorical_basic() {
        let handler = ProbHandler::new();
        let mut state = seeded_state();

        let probs = Value::Array(Rc::new(RefCell::new(vec![
            Value::Float(0.2),
            Value::Float(0.3),
            Value::Float(0.5),
        ])));

        for _ in 0..10 {
            let result = handler.handle("categorical", &[probs.clone()], cont(), &mut state);

            match result {
                HandlerResult::Resume(Value::Int(i)) => {
                    assert!(
                        i >= 0 && i < 3,
                        "categorical should return 0, 1, or 2: {}",
                        i
                    );
                }
                other => panic!("expected Resume(Int), got {:?}", other),
            }
        }
    }

    #[test]
    fn test_categorical_deterministic() {
        let handler = ProbHandler::new();
        let mut state = seeded_state();

        // All probability on index 1
        let probs = Value::Array(Rc::new(RefCell::new(vec![
            Value::Float(0.0),
            Value::Float(1.0),
            Value::Float(0.0),
        ])));

        for _ in 0..5 {
            let result = handler.handle("categorical", &[probs.clone()], cont(), &mut state);

            match result {
                HandlerResult::Resume(Value::Int(1)) => {}
                other => panic!("expected index 1, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_exponential_basic() {
        let handler = ProbHandler::new();
        let mut state = seeded_state();

        let result = handler.handle("exponential", &[Value::Float(1.0)], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Float(v)) => {
                assert!(v >= 0.0, "exponential should be non-negative: {}", v);
            }
            other => panic!("expected Resume(Float), got {:?}", other),
        }
    }

    #[test]
    fn test_exponential_invalid_lambda() {
        let handler = ProbHandler::new();
        let mut state = new_state();

        let result = handler.handle("exponential", &[Value::Float(0.0)], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("positive"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_beta_basic() {
        let handler = ProbHandler::new();
        let mut state = seeded_state();

        let result = handler.handle(
            "beta",
            &[Value::Float(2.0), Value::Float(5.0)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Float(v)) => {
                assert!(v > 0.0 && v < 1.0, "beta should be in (0, 1): {}", v);
            }
            other => panic!("expected Resume(Float), got {:?}", other),
        }
    }

    #[test]
    fn test_sample_with_distribution() {
        let handler = ProbHandler::new();
        let mut state = seeded_state();

        let dist = Value::Distribution(Distribution::Normal {
            mean: 0.0,
            std: 1.0,
        });
        let result = handler.handle("sample", &[dist], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Float(_)) => {}
            other => panic!("expected Resume(Float), got {:?}", other),
        }
    }

    #[test]
    fn test_observe_updates_log_prob() {
        let handler = ProbHandler::new();
        let mut state = seeded_state();
        ProbHandler::reset_log_prob(&mut state);

        let dist = Value::Distribution(Distribution::Normal {
            mean: 0.0,
            std: 1.0,
        });
        handler.handle("observe", &[dist, Value::Float(0.0)], cont(), &mut state);

        let lp = ProbHandler::get_log_prob_value(&state);
        // Log prob of 0 under N(0,1) should be about -0.919
        assert!(lp < 0.0, "log prob should be negative: {}", lp);
        assert!(lp > -2.0, "log prob should be reasonable: {}", lp);
    }

    #[test]
    fn test_condition_true() {
        let handler = ProbHandler::new();
        let mut state = seeded_state();
        ProbHandler::reset_log_prob(&mut state);

        let result = handler.handle("condition", &[Value::Bool(true)], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }

        let lp = ProbHandler::get_log_prob_value(&state);
        assert!(
            (lp - 0.0).abs() < 0.001,
            "condition(true) should not change log prob"
        );
    }

    #[test]
    fn test_condition_false() {
        let handler = ProbHandler::new();
        let mut state = seeded_state();
        ProbHandler::reset_log_prob(&mut state);

        handler.handle("condition", &[Value::Bool(false)], cont(), &mut state);

        let lp = ProbHandler::get_log_prob_value(&state);
        assert!(
            lp == f64::NEG_INFINITY,
            "condition(false) should set log prob to -inf"
        );
    }

    #[test]
    fn test_log_prob_operation() {
        let handler = ProbHandler::new();
        let mut state = new_state();

        let dist = Value::Distribution(Distribution::Normal {
            mean: 0.0,
            std: 1.0,
        });
        let result = handler.handle("log_prob", &[dist, Value::Float(0.0)], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Float(lp)) => {
                // log prob of 0 under N(0,1) is -0.5*log(2*pi) ≈ -0.919
                assert!(
                    lp < 0.0 && lp > -2.0,
                    "log prob should be around -0.919: {}",
                    lp
                );
            }
            other => panic!("expected Resume(Float), got {:?}", other),
        }
    }

    #[test]
    fn test_score_operation() {
        let handler = ProbHandler::new();
        let mut state = seeded_state();
        ProbHandler::reset_log_prob(&mut state);

        handler.handle("score", &[Value::Float(-1.5)], cont(), &mut state);

        let lp = ProbHandler::get_log_prob_value(&state);
        assert!((lp - (-1.5)).abs() < 0.001);
    }

    #[test]
    fn test_get_log_prob_operation() {
        let handler = ProbHandler::new();
        let mut state = seeded_state();
        ProbHandler::reset_log_prob(&mut state);
        ProbHandler::add_log_prob(&mut state, -2.5);

        let result = handler.handle("get_log_prob", &[], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Float(lp)) => {
                assert!((lp - (-2.5)).abs() < 0.001);
            }
            other => panic!("expected Resume(Float), got {:?}", other),
        }
    }

    #[test]
    fn test_seed_reproducibility() {
        let handler = ProbHandler::new();

        // First run
        let mut state1 = new_state();
        ProbHandler::set_seed(&mut state1, 12345);
        let mut values1 = Vec::new();
        for _ in 0..5 {
            if let HandlerResult::Resume(Value::Float(v)) = handler.handle(
                "uniform",
                &[Value::Float(0.0), Value::Float(1.0)],
                cont(),
                &mut state1,
            ) {
                values1.push(v);
            }
        }

        // Second run with same seed
        let mut state2 = new_state();
        ProbHandler::set_seed(&mut state2, 12345);
        let mut values2 = Vec::new();
        for _ in 0..5 {
            if let HandlerResult::Resume(Value::Float(v)) = handler.handle(
                "uniform",
                &[Value::Float(0.0), Value::Float(1.0)],
                cont(),
                &mut state2,
            ) {
                values2.push(v);
            }
        }

        assert_eq!(values1, values2, "same seed should produce same sequence");
    }

    #[test]
    fn test_observe_count() {
        let handler = ProbHandler::new();
        let mut state = new_state();

        assert_eq!(ProbHandler::get_observe_count(&state), 0);

        let dist = Value::Distribution(Distribution::Normal {
            mean: 0.0,
            std: 1.0,
        });
        handler.handle(
            "observe",
            &[dist.clone(), Value::Float(0.0)],
            cont(),
            &mut state,
        );
        assert_eq!(ProbHandler::get_observe_count(&state), 1);

        handler.handle(
            "observe",
            &[dist.clone(), Value::Float(1.0)],
            cont(),
            &mut state,
        );
        assert_eq!(ProbHandler::get_observe_count(&state), 2);
    }

    #[test]
    fn test_unknown_operation() {
        let handler = ProbHandler::new();
        let mut state = new_state();

        let result = handler.handle("unknown_op", &[], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("unknown operation"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_default_impl() {
        let handler = ProbHandler::default();
        assert_eq!(handler.effect_name(), "Prob");
    }

    #[test]
    fn test_operation_linearity() {
        let handler = ProbHandler::new();

        // Sampling operations should be multi-shot (can explore multiple branches)
        assert_eq!(handler.operation_linearity("sample"), Linearity::MultiShot);
        assert_eq!(
            handler.operation_linearity("bernoulli"),
            Linearity::MultiShot
        );
        assert_eq!(handler.operation_linearity("uniform"), Linearity::MultiShot);
        assert_eq!(handler.operation_linearity("normal"), Linearity::MultiShot);

        // Observation/conditioning are one-shot (filter probability space)
        assert_eq!(
            handler.operation_linearity("observe"),
            Linearity::ExactlyOnce
        );
        assert_eq!(
            handler.operation_linearity("condition"),
            Linearity::ExactlyOnce
        );
        assert_eq!(
            handler.operation_linearity("factor"),
            Linearity::ExactlyOnce
        );

        // Log probability queries are multi-shot (pure functions)
        assert_eq!(
            handler.operation_linearity("log_prob"),
            Linearity::MultiShot
        );
        assert_eq!(
            handler.operation_linearity("get_log_prob"),
            Linearity::MultiShot
        );

        // Handler should support multi-shot globally
        assert!(handler.supports_multi_shot());
    }

    #[test]
    fn test_linearity_enforcement_integration() {
        use crate::interp::effect_dispatch::{EffectContext, EffectKind};

        let mut ctx = EffectContext::new();
        ctx.push_handler(crate::interp::effect_dispatch::default_prob_handler());

        // Multi-shot operation (sample) - should allow multiple resumes
        let dist = Value::Distribution(Distribution::Normal {
            mean: 0.0,
            std: 1.0,
        });
        let result1 = ctx.dispatch(EffectKind::Prob, "sample", vec![dist.clone()]);
        assert!(result1.is_ok(), "First sample should succeed");

        let result2 = ctx.dispatch(EffectKind::Prob, "sample", vec![dist.clone()]);
        assert!(result2.is_ok(), "Second sample should succeed (multi-shot)");

        // One-shot operation (observe) - verifies linearity tracking is set correctly
        // Note: This tests that the linearity is declared correctly, actual enforcement
        // happens in the continuation resume logic
        let observe_result = ctx.dispatch(
            EffectKind::Prob,
            "observe",
            vec![dist.clone(), Value::Float(0.0)],
        );
        assert!(
            observe_result.is_ok(),
            "Observe should succeed (one-shot linearity correctly set)"
        );
    }
}
