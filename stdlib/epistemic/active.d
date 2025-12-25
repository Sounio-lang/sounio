//! stdlib/epistemic/active.d
//!
//! Active Inference: Uncertainty-Driven Exploration
//!
//! In Active Inference, agents minimize expected free energy by:
//! 1. Reducing uncertainty about hidden states (exploration)
//! 2. Achieving preferred outcomes (exploitation)
//!
//! High variance in Knowledge values signals where to explore next.
//!
//! # Core Concepts
//!
//! - **Expected Free Energy (EFE)**: Combines epistemic value (info gain) and
//!   pragmatic value (goal achievement)
//! - **Precision**: Inverse variance - high precision = confident knowledge
//! - **Information Gain**: How much we expect to learn from an action
//!
//! # Example
//!
//! ```demetrios
//! use epistemic::{Knowledge, active}
//!
//! // Current beliefs about drug efficacy
//! let drug_a = Knowledge::measured(0.65, 0.10, "trial_small")
//! let drug_b = Knowledge::measured(0.60, 0.02, "trial_large")
//!
//! // Which should we investigate further?
//! let priority = active::exploration_priority([drug_a, drug_b])
//! // Returns drug_a - higher variance means more to learn
//! ```

use core::{Option, Vec}
use epistemic::knowledge::{Knowledge, BetaConfidence, Source, Provenance}

// ============================================================================
// PRECISION AND INFORMATION METRICS
// ============================================================================

/// Precision: inverse of variance (how confident are we?)
/// Higher precision = more confident = less need to explore
pub fn precision(k: &Knowledge<f64>) -> f64 {
    1.0 / max_f64(k.variance, 0.0000001)
}

/// Entropy of a Knowledge value (assuming Gaussian)
/// H = 0.5 * ln(2πe * variance)
pub fn entropy(k: &Knowledge<f64>) -> f64 {
    let two_pi_e = 17.0794684453  // 2 * π * e
    0.5 * ln_f64(two_pi_e * max_f64(k.variance, 0.0000001))
}

/// Expected information gain from observing a variable
/// For Gaussian: IG ≈ 0.5 * ln(prior_var / posterior_var)
///
/// If we expect posterior variance to be reduced by factor r:
/// IG = 0.5 * ln(1 / r) = -0.5 * ln(r)
pub fn expected_info_gain(k: &Knowledge<f64>, expected_reduction: f64) -> f64 {
    0.0 - 0.5 * ln_f64(max_f64(min_f64(expected_reduction, 1.0), 0.01))
}

/// Relative uncertainty: variance / value²
/// Useful for comparing uncertainties across different scales
pub fn relative_uncertainty(k: &Knowledge<f64>) -> f64 {
    k.variance / max_f64(k.value * k.value, 0.0000001)
}

/// Coefficient of variation: std / |mean|
pub fn coefficient_of_variation(k: &Knowledge<f64>) -> f64 {
    sqrt_f64(k.variance) / max_f64(abs_f64(k.value), 0.0000001)
}

// ============================================================================
// EXPLORATION PRIORITY
// ============================================================================

/// Exploration score for a single Knowledge value
/// Higher score = should explore this more
///
/// Components:
/// - Variance contribution: high variance → high score
/// - Confidence penalty: low confidence → high score
/// - Value bonus: important values get priority
pub fn exploration_score(k: &Knowledge<f64>, importance: f64) -> f64 {
    let variance_term = ln_f64(1.0 + k.variance)
    let confidence_term = 1.0 - k.confidence.mean()
    let importance_term = importance

    variance_term + confidence_term + importance_term
}

/// Rank Knowledge values by exploration priority
/// Returns indices sorted by priority (highest first)
pub fn exploration_priority(values: &[Knowledge<f64>]) -> Vec<usize> {
    let n = values.len()
    var scores = Vec::new()
    var indices = Vec::new()

    var i = 0
    while i < n {
        scores.push(exploration_score(&values[i], 1.0))
        indices.push(i)
        i = i + 1
    }

    // Simple bubble sort by score (descending)
    i = 0
    while i < n {
        var j = i + 1
        while j < n {
            if scores[j] > scores[i] {
                // Swap
                let tmp_score = scores[i]
                scores[i] = scores[j]
                scores[j] = tmp_score

                let tmp_idx = indices[i]
                indices[i] = indices[j]
                indices[j] = tmp_idx
            }
            j = j + 1
        }
        i = i + 1
    }

    indices
}

/// Get the Knowledge value most worth exploring
pub fn most_uncertain<'a>(values: &'a [Knowledge<f64>]) -> Option<&'a Knowledge<f64>> {
    if values.is_empty() {
        return Option::None
    }

    let priority = exploration_priority(values)
    Option::Some(&values[priority[0]])
}

// ============================================================================
// EXPECTED FREE ENERGY
// ============================================================================

/// Expected Free Energy components
pub struct EFEComponents {
    /// Epistemic value: expected information gain
    epistemic: f64,

    /// Pragmatic value: expected reward/goal achievement
    pragmatic: f64,

    /// Total EFE (lower is better)
    total: f64,
}

/// Compute Expected Free Energy for an action
///
/// EFE = -epistemic_value - pragmatic_value
///     = -E[info_gain] - E[reward]
///
/// We want to MINIMIZE EFE (maximize epistemic + pragmatic value)
pub fn expected_free_energy(
    current: &Knowledge<f64>,
    expected_posterior_var: f64,
    expected_reward: f64,
    reward_weight: f64,
) -> EFEComponents {
    // Epistemic value: how much will we learn?
    let info_gain = 0.5 * ln_f64(current.variance / max_f64(expected_posterior_var, 0.0000001))
    let epistemic = max_f64(info_gain, 0.0)

    // Pragmatic value: how much reward do we expect?
    let pragmatic = expected_reward * reward_weight

    // EFE: we want to minimize this (so negate the values we want to maximize)
    let total = 0.0 - epistemic - pragmatic

    EFEComponents {
        epistemic: epistemic,
        pragmatic: pragmatic,
        total: total,
    }
}

/// Select action that minimizes EFE
pub fn select_action<A: Clone>(
    actions: &[(A, f64, f64)],  // (action, expected_var_reduction, expected_reward)
    current: &Knowledge<f64>,
    reward_weight: f64,
) -> Option<A> {
    if actions.is_empty() {
        return Option::None
    }

    var best_action = actions[0].0.clone()
    var best_efe = 1.0e308  // f64::MAX equivalent

    for (action, var_reduction, reward) in actions {
        let posterior_var = current.variance * var_reduction
        let efe = expected_free_energy(current, posterior_var, *reward, reward_weight)

        if efe.total < best_efe {
            best_efe = efe.total
            best_action = action.clone()
        }
    }

    Option::Some(best_action)
}

// ============================================================================
// BELIEF UPDATING (PREDICTIVE PROCESSING)
// ============================================================================

/// Prediction error: difference between observation and belief
pub fn prediction_error(belief: &Knowledge<f64>, observation: f64) -> f64 {
    observation - belief.value
}

/// Precision-weighted prediction error
/// This is the key quantity in predictive processing
pub fn precision_weighted_error(belief: &Knowledge<f64>, observation: f64) -> f64 {
    let error = observation - belief.value
    let prec = precision(belief)
    prec * error
}

/// Update belief with observation using precision weighting
///
/// posterior_mean = (prior_precision * prior_mean + obs_precision * obs)
///                / (prior_precision + obs_precision)
pub fn update_belief(
    prior: &Knowledge<f64>,
    observation: f64,
    observation_variance: f64,
) -> Knowledge<f64> {
    let prior_prec = precision(prior)
    let obs_prec = 1.0 / max_f64(observation_variance, 0.0000001)
    let total_prec = prior_prec + obs_prec

    let posterior_mean = (prior_prec * prior.value + obs_prec * observation) / total_prec
    let posterior_var = 1.0 / total_prec

    // Update confidence based on evidence accumulation
    let evidence_ratio = obs_prec / prior_prec
    let new_conf = prior.confidence.update(
        (evidence_ratio * 10.0) as i64,  // Pseudo-successes
        1,                                // Pseudo-failures
    )

    Knowledge {
        value: posterior_mean,
        variance: posterior_var,
        confidence: new_conf,
        provenance: prior.provenance.with_step("belief_update"),
    }
}

/// Sequential belief updating with multiple observations
pub fn update_sequential(
    prior: Knowledge<f64>,
    observations: &[(f64, f64)],  // (value, variance) pairs
) -> Knowledge<f64> {
    var current = prior

    for (obs, var) in observations {
        current = update_belief(&current, *obs, *var)
    }

    current
}

// ============================================================================
// EXPLORATION STRATEGIES
// ============================================================================

/// Epsilon-greedy exploration
/// With probability epsilon, explore (pick most uncertain)
/// Otherwise exploit (pick highest value)
pub fn epsilon_greedy(
    values: &[Knowledge<f64>],
    epsilon: f64,
    random_value: f64,  // Should be in [0, 1]
) -> Option<usize> {
    if values.is_empty() {
        return Option::None
    }

    if random_value < epsilon {
        // Explore: pick most uncertain
        let priority = exploration_priority(values)
        Option::Some(priority[0])
    } else {
        // Exploit: pick highest value
        var best_idx = 0
        var best_val = values[0].value

        var i = 1
        while i < values.len() {
            if values[i].value > best_val {
                best_val = values[i].value
                best_idx = i
            }
            i = i + 1
        }

        Option::Some(best_idx)
    }
}

/// Upper Confidence Bound (UCB) selection
/// Balances exploitation (high value) with exploration (high uncertainty)
///
/// UCB = value + c * sqrt(variance)
pub fn ucb_select(values: &[Knowledge<f64>], exploration_constant: f64) -> Option<usize> {
    if values.is_empty() {
        return Option::None
    }

    var best_idx = 0
    var best_ucb = values[0].value + exploration_constant * sqrt_f64(values[0].variance)

    var i = 1
    while i < values.len() {
        let ucb = values[i].value + exploration_constant * sqrt_f64(values[i].variance)
        if ucb > best_ucb {
            best_ucb = ucb
            best_idx = i
        }
        i = i + 1
    }

    Option::Some(best_idx)
}

/// Thompson Sampling: sample from posterior and pick highest
/// Approximation: sample from N(value, variance)
pub fn thompson_select(values: &[Knowledge<f64>], random_normals: &[f64]) -> Option<usize> {
    if values.is_empty() || random_normals.len() < values.len() {
        return Option::None
    }

    var best_idx = 0
    var best_sample = values[0].value + random_normals[0] * sqrt_f64(values[0].variance)

    var i = 1
    while i < values.len() {
        let sample = values[i].value + random_normals[i] * sqrt_f64(values[i].variance)
        if sample > best_sample {
            best_sample = sample
            best_idx = i
        }
        i = i + 1
    }

    Option::Some(best_idx)
}

// ============================================================================
// HELPERS
// ============================================================================

fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    var y = x
    var i = 0
    while i < 10 {
        y = 0.5 * (y + x / y)
        i = i + 1
    }
    y
}

fn ln_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 - 999999.0 }
    if x == 1.0 { return 0.0 }

    let y = (x - 1.0) / (x + 1.0)
    var result = 0.0
    var term = y
    var i = 1
    while i < 30 {
        result = result + term / (i as f64)
        term = term * y * y
        i = i + 2
    }
    2.0 * result
}

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { 0.0 - x } else { x }
}

fn max_f64(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

fn min_f64(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}
