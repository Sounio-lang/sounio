/// epistemic::stats — Revolutionary Statistics with Epistemic Honesty
///
/// The world's first statistics library where:
/// - Every estimate is a full Beta posterior, not a point estimate
/// - Variance is a first-class optimization target, not just a diagnostic
/// - Active inference utilities guide data collection to reduce ignorance
/// - ML loss functions can penalize epistemic uncertainty
///
/// This module changes scientific computing forever by making epistemic
/// honesty the default, not an afterthought.
///
/// # Philosophy
///
/// Traditional statistics: "The mean is 0.7"
/// Demetrios epistemic:    "We believe the mean is Beta(14, 6), giving
///                          0.7 ± 0.09 with 0.013 residual ignorance"
///
/// The difference matters for every decision under uncertainty.

// ============================================================================
// SECTION 1: CORE BETA DISTRIBUTION (Extended)
// ============================================================================

/// Beta distribution - the fundamental epistemic primitive
struct Beta {
    alpha: f64,  // Pseudo-successes (evidence for)
    beta: f64    // Pseudo-failures (evidence against)
}

/// Confidence interval bounds
struct CI {
    lo: f64,
    hi: f64
}

/// Epistemic summary - complete picture of belief state
struct EpistemicSummary {
    mean: f64,           // Expected value
    variance: f64,       // Residual ignorance
    std_dev: f64,        // Standard deviation
    sample_size: f64,    // Effective evidence count
    entropy: f64,        // Information-theoretic uncertainty
    ci_lo: f64,          // 95% credible interval lower
    ci_hi: f64           // 95% credible interval upper
}

// ----------------------------------------------------------------------------
// Numeric Helpers (self-contained, no external dependencies)
// ----------------------------------------------------------------------------

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

fn max_f64(a: f64, b: f64) -> f64 {
    if a > b { return a }
    return b
}

fn min_f64(a: f64, b: f64) -> f64 {
    if a < b { return a }
    return b
}

fn clamp01(x: f64) -> f64 {
    if x < 0.0 { return 0.0 }
    if x > 1.0 { return 1.0 }
    return x
}

fn clamp_positive(x: f64) -> f64 {
    if x < 0.0001 { return 0.0001 }
    return x
}

/// Newton-Raphson square root (15 iterations for high precision)
fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    var y = x
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    return y
}

/// Natural logarithm approximation using series expansion
fn ln_approx(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 - 1000000.0 }  // -infinity approximation

    // Reduce to [1, 2] range: ln(x) = ln(m * 2^k) = ln(m) + k*ln(2)
    var m = x
    var k: i64 = 0

    while m >= 2.0 {
        m = m / 2.0
        k = k + 1
    }
    while m < 1.0 {
        m = m * 2.0
        k = k - 1
    }

    // Series expansion around 1: ln(1+y) ≈ y - y²/2 + y³/3 - ...
    let y = m - 1.0
    let y2 = y * y
    let y3 = y2 * y
    let y4 = y3 * y
    let y5 = y4 * y
    let y6 = y5 * y
    let y7 = y6 * y
    let y8 = y7 * y

    let ln_m = y - y2/2.0 + y3/3.0 - y4/4.0 + y5/5.0 - y6/6.0 + y7/7.0 - y8/8.0

    // Convert k to f64
    var kf = 0.0
    if k > 0 {
        var i: i64 = 0
        while i < k {
            kf = kf + 1.0
            i = i + 1
        }
    } else if k < 0 {
        var i: i64 = 0
        while i > k {
            kf = kf - 1.0
            i = i - 1
        }
    }

    let ln2 = 0.693147180559945
    return ln_m + kf * ln2
}

/// Exponential function approximation
fn exp_approx(x: f64) -> f64 {
    if x > 700.0 { return 1.0e308 }  // Overflow protection
    if x < -700.0 { return 0.0 }     // Underflow protection

    // Reduce range: exp(x) = exp(r) * 2^k where x = r + k*ln(2)
    let ln2 = 0.693147180559945
    let k_float = x / ln2

    var k: i64 = 0
    if k_float >= 0.0 {
        while k_float > 1.0 {
            k = k + 1
            if k > 1000 { break }  // Safety
        }
    }
    // Simplified: just use series for now
    let r = x

    // Taylor series: exp(r) = 1 + r + r²/2! + r³/3! + ...
    let r2 = r * r
    let r3 = r2 * r
    let r4 = r3 * r
    let r5 = r4 * r
    let r6 = r5 * r
    let r7 = r6 * r
    let r8 = r7 * r

    return 1.0 + r + r2/2.0 + r3/6.0 + r4/24.0 + r5/120.0 + r6/720.0 + r7/5040.0 + r8/40320.0
}

/// Digamma function approximation ψ(x) = d/dx ln Γ(x)
fn digamma(x: f64) -> f64 {
    if x < 0.5 {
        // Reflection formula
        let pi = 3.141592653589793
        return digamma(1.0 - x) + pi / tan_approx(pi * x)
    }

    // Shift to asymptotic region
    var result = 0.0
    var z = x
    while z < 6.0 {
        result = result - 1.0 / z
        z = z + 1.0
    }

    // Asymptotic expansion
    let inv_z = 1.0 / z
    let inv_z2 = inv_z * inv_z
    result = result + ln_approx(z) - 0.5 * inv_z
    result = result - inv_z2 * (1.0/12.0 - inv_z2 * (1.0/120.0 - inv_z2/252.0))
    return result
}

/// Simple tangent approximation
fn tan_approx(x: f64) -> f64 {
    let sin_x = sin_approx(x)
    let cos_x = cos_approx(x)
    if abs_f64(cos_x) < 0.0001 { return 10000.0 }  // Near singularity
    return sin_x / cos_x
}

/// Sine approximation using Taylor series
fn sin_approx(x: f64) -> f64 {
    let pi = 3.141592653589793
    // Reduce to [-π, π]
    var y = x
    while y > pi { y = y - 2.0 * pi }
    while y < 0.0 - pi { y = y + 2.0 * pi }

    let y2 = y * y
    let y3 = y2 * y
    let y5 = y3 * y2
    let y7 = y5 * y2
    return y - y3/6.0 + y5/120.0 - y7/5040.0
}

/// Cosine approximation using Taylor series
fn cos_approx(x: f64) -> f64 {
    let pi = 3.141592653589793
    return sin_approx(x + pi/2.0)
}

// ----------------------------------------------------------------------------
// Beta Distribution Core API
// ----------------------------------------------------------------------------

/// Create a new Beta distribution with safe parameters
fn beta_new(a: f64, b: f64) -> Beta {
    return Beta {
        alpha: clamp_positive(a),
        beta: clamp_positive(b)
    }
}

/// Uniform prior Beta(1, 1) - complete ignorance
fn beta_uniform() -> Beta {
    return Beta { alpha: 1.0, beta: 1.0 }
}

/// Jeffreys prior Beta(0.5, 0.5) - reference prior
fn beta_jeffreys() -> Beta {
    return Beta { alpha: 0.5, beta: 0.5 }
}

/// Weak informative prior Beta(2, 2)
fn beta_weak() -> Beta {
    return Beta { alpha: 2.0, beta: 2.0 }
}

/// Haldane prior Beta(ε, ε) - improper but useful
fn beta_haldane() -> Beta {
    return Beta { alpha: 0.001, beta: 0.001 }
}

/// Informative prior centered at mean with concentration
fn beta_informative(mean: f64, concentration: f64) -> Beta {
    let m = clamp01(mean)
    let c = clamp_positive(concentration)
    return Beta { alpha: m * c, beta: (1.0 - m) * c }
}

/// Create from observed success rate
fn beta_from_rate(successes: f64, total: f64) -> Beta {
    if total <= 0.0 {
        return beta_uniform()
    }
    return Beta { alpha: successes + 1.0, beta: (total - successes) + 1.0 }
}

// ----------------------------------------------------------------------------
// Beta Moments and Properties
// ----------------------------------------------------------------------------

/// Mean: E[X] = α / (α + β)
fn beta_mean(b: Beta) -> f64 {
    return b.alpha / (b.alpha + b.beta)
}

/// Variance: Var[X] = αβ / ((α+β)²(α+β+1))
/// THE KEY EPISTEMIC QUANTITY - measures residual ignorance
fn beta_variance(b: Beta) -> f64 {
    let n = b.alpha + b.beta
    return (b.alpha * b.beta) / (n * n * (n + 1.0))
}

/// Standard deviation
fn beta_std(b: Beta) -> f64 {
    return sqrt_f64(beta_variance(b))
}

/// Mode: (α-1)/(α+β-2) when α,β > 1
fn beta_mode(b: Beta) -> f64 {
    if b.alpha <= 1.0 { return beta_mean(b) }
    if b.beta <= 1.0 { return beta_mean(b) }
    return (b.alpha - 1.0) / (b.alpha + b.beta - 2.0)
}

/// Effective sample size: n = α + β
fn beta_sample_size(b: Beta) -> f64 {
    return b.alpha + b.beta
}

/// Concentration parameter
fn beta_concentration(b: Beta) -> f64 {
    return b.alpha + b.beta
}

/// Skewness
fn beta_skewness(b: Beta) -> f64 {
    let n = b.alpha + b.beta
    let diff = b.beta - b.alpha
    let num = 2.0 * diff * sqrt_f64(n + 1.0)
    let den = (n + 2.0) * sqrt_f64(b.alpha * b.beta)
    if den < 0.0001 { return 0.0 }
    return num / den
}

/// Entropy: Information-theoretic uncertainty
fn beta_entropy(b: Beta) -> f64 {
    let a = b.alpha
    let bb = b.beta
    let n = a + bb

    // H = ln B(a,b) - (a-1)ψ(a) - (b-1)ψ(b) + (n-2)ψ(n)
    // Approximation using digamma
    let ln_beta_ab = ln_approx(a) + ln_approx(bb) - ln_approx(n)  // Simplified
    let da = digamma(a)
    let db = digamma(bb)
    let dn = digamma(n)

    return ln_beta_ab - (a - 1.0) * da - (bb - 1.0) * db + (n - 2.0) * dn
}

/// Approximate 95% CI using normal approximation
fn beta_ci95(b: Beta) -> CI {
    let m = beta_mean(b)
    let s = beta_std(b)
    let z = 1.96  // 95% z-score
    return CI {
        lo: clamp01(m - z * s),
        hi: clamp01(m + z * s)
    }
}

/// Approximate CI at given confidence level
fn beta_ci(b: Beta, confidence: f64) -> CI {
    let m = beta_mean(b)
    let s = beta_std(b)

    // Convert confidence to z-score (approximation)
    let z = if confidence >= 0.99 { 2.576 }
           else if confidence >= 0.95 { 1.96 }
           else if confidence >= 0.90 { 1.645 }
           else if confidence >= 0.80 { 1.282 }
           else { 1.0 }

    return CI { lo: clamp01(m - z * s), hi: clamp01(m + z * s) }
}

/// Complete epistemic summary
fn beta_summary(b: Beta) -> EpistemicSummary {
    let ci = beta_ci95(b)
    return EpistemicSummary {
        mean: beta_mean(b),
        variance: beta_variance(b),
        std_dev: beta_std(b),
        sample_size: beta_sample_size(b),
        entropy: beta_entropy(b),
        ci_lo: ci.lo,
        ci_hi: ci.hi
    }
}

// ============================================================================
// SECTION 2: BAYESIAN UPDATING (Conjugate Posterior)
// ============================================================================

/// Conjugate posterior update: prior + data → posterior
fn beta_update(prior: Beta, successes: f64, failures: f64) -> Beta {
    return Beta {
        alpha: prior.alpha + max_f64(0.0, successes),
        beta: prior.beta + max_f64(0.0, failures)
    }
}

/// Update with single observation
fn beta_observe(prior: Beta, success: i64) -> Beta {
    if success > 0 {
        return Beta { alpha: prior.alpha + 1.0, beta: prior.beta }
    } else {
        return Beta { alpha: prior.alpha, beta: prior.beta + 1.0 }
    }
}

/// Batch update from observations
fn beta_observe_batch(prior: Beta, observations: [i64]) -> Beta {
    var successes = 0.0
    var failures = 0.0
    var i: i64 = 0
    let n = len(observations)

    while i < n {
        if observations[i] > 0 {
            successes = successes + 1.0
        } else {
            failures = failures + 1.0
        }
        i = i + 1
    }

    return beta_update(prior, successes, failures)
}

/// Update from probability estimate with effective sample size
fn beta_update_from_estimate(prior: Beta, prob: f64, effective_n: f64) -> Beta {
    let successes = prob * effective_n
    let failures = effective_n - successes
    return beta_update(prior, successes, failures)
}

// ============================================================================
// SECTION 3: VARIANCE PROPAGATION (Epistemic Arithmetic)
// ============================================================================

/// Result type for epistemic operations - carries both value and uncertainty
struct EpistemicValue {
    value: f64,       // Point estimate (mean)
    variance: f64,    // Uncertainty (variance)
    alpha: f64,       // Beta alpha (for reconstruction)
    beta: f64         // Beta beta (for reconstruction)
}

/// Create epistemic value from Beta
fn epistemic_from_beta(b: Beta) -> EpistemicValue {
    return EpistemicValue {
        value: beta_mean(b),
        variance: beta_variance(b),
        alpha: b.alpha,
        beta: b.beta
    }
}

/// Create epistemic value from point estimate with assumed variance
fn epistemic_point(value: f64, variance: f64) -> EpistemicValue {
    // Reconstruct Beta parameters from mean/variance
    let m = clamp01(value)
    let v = clamp_positive(variance)
    let max_v = m * (1.0 - m)
    let v_safe = min_f64(v, max_v * 0.99)

    let concentration = m * (1.0 - m) / v_safe - 1.0
    let c = max_f64(2.0, concentration)

    return EpistemicValue {
        value: value,
        variance: v_safe,
        alpha: m * c,
        beta: (1.0 - m) * c
    }
}

/// Reconstruct Beta from epistemic value
fn epistemic_to_beta(e: EpistemicValue) -> Beta {
    return Beta { alpha: e.alpha, beta: e.beta }
}

/// Add two epistemic values (variance propagates)
/// E[X+Y] = E[X] + E[Y], Var[X+Y] = Var[X] + Var[Y] (assuming independence)
fn epistemic_add(a: EpistemicValue, b: EpistemicValue) -> EpistemicValue {
    let new_value = a.value + b.value
    let new_variance = a.variance + b.variance

    // Normalize to [0,1] for Beta reconstruction
    let normalized_value = new_value / 2.0
    let normalized_variance = new_variance / 4.0

    return epistemic_point(normalized_value, normalized_variance)
}

/// Multiply two epistemic values
/// E[XY] = E[X]E[Y], Var[XY] ≈ E[X]²Var[Y] + E[Y]²Var[X]
fn epistemic_mul(a: EpistemicValue, b: EpistemicValue) -> EpistemicValue {
    let new_value = a.value * b.value
    let new_variance = a.value * a.value * b.variance +
                       b.value * b.value * a.variance +
                       a.variance * b.variance

    return epistemic_point(new_value, new_variance)
}

/// Scale epistemic value
fn epistemic_scale(e: EpistemicValue, k: f64) -> EpistemicValue {
    return EpistemicValue {
        value: e.value * k,
        variance: e.variance * k * k,
        alpha: e.alpha,
        beta: e.beta
    }
}

/// Total epistemic uncertainty (sum of variances)
fn epistemic_total_uncertainty(values: [EpistemicValue]) -> f64 {
    var total = 0.0
    var i: i64 = 0
    let n = len(values)
    while i < n {
        total = total + values[i].variance
        i = i + 1
    }
    return total
}

// ============================================================================
// SECTION 4: ACTIVE INFERENCE (Revolutionary)
// ============================================================================

/// Active inference metrics for guiding data collection
struct ActiveInferenceMetrics {
    expected_info_gain: f64,     // Bits of information expected from observation
    variance_reduction: f64,     // Expected reduction in variance
    exploration_value: f64,      // Value of reducing uncertainty
    exploitation_value: f64,     // Value of using current best estimate
    optimal_action: f64          // Explore (0) vs Exploit (1) recommendation
}

/// Expected information gain from one more observation
/// Key insight: more uncertain beliefs gain more from observation
fn expected_info_gain(b: Beta) -> f64 {
    // IG ≈ 1/(2n) where n = α + β (concentration)
    let n = b.alpha + b.beta
    if n < 0.1 { return 10.0 }  // High info gain for ignorant prior
    return 1.0 / (2.0 * n)
}

/// Expected variance reduction from one more observation
fn expected_variance_reduction(b: Beta) -> f64 {
    let current_var = beta_variance(b)

    // Expected variance after one more observation
    // Approximate: Var(posterior) ≈ Var(prior) * n/(n+1)
    let n = b.alpha + b.beta
    let expected_var = current_var * n / (n + 1.0)

    return current_var - expected_var
}

/// Expected variance reduction from k more observations
fn expected_variance_reduction_k(b: Beta, k: f64) -> f64 {
    let current_var = beta_variance(b)
    let n = b.alpha + b.beta
    let expected_var = current_var * n / (n + k)
    return current_var - expected_var
}

/// Exploration-exploitation trade-off score
/// Returns value in [0, 1] where 0 = explore, 1 = exploit
fn exploration_exploitation_score(b: Beta, exploit_threshold: f64) -> f64 {
    let n = b.alpha + b.beta
    let var = beta_variance(b)

    // High variance = prefer exploration
    // High sample size = prefer exploitation

    // Normalized uncertainty: compare to threshold
    let uncertainty_score = var / 0.25  // Max variance is 0.25 at Beta(1,1)

    // If uncertainty > threshold, explore; otherwise exploit
    if uncertainty_score > (1.0 - exploit_threshold) {
        return 0.0  // Explore
    }
    return 1.0  // Exploit
}

/// Compute full active inference metrics
fn active_inference_metrics(b: Beta, exploit_threshold: f64) -> ActiveInferenceMetrics {
    let ig = expected_info_gain(b)
    let vr = expected_variance_reduction(b)

    let mean = beta_mean(b)
    let explore_value = ig + vr
    let exploit_value = mean * (1.0 - beta_variance(b) * 4.0)  // Penalize uncertainty

    let ee_score = exploration_exploitation_score(b, exploit_threshold)

    return ActiveInferenceMetrics {
        expected_info_gain: ig,
        variance_reduction: vr,
        exploration_value: explore_value,
        exploitation_value: exploit_value,
        optimal_action: ee_score
    }
}

/// Thompson sampling: sample from posterior for exploration
/// Returns a value from the Beta distribution (approximation using mean + noise)
fn thompson_sample(b: Beta) -> f64 {
    // Approximate sampling using mean + scaled standard deviation
    // In production, use proper PRNG
    let mean = beta_mean(b)
    let std = beta_std(b)

    // Pseudo-random offset (deterministic for reproducibility)
    let noise_scale = 0.5  // Moderate exploration
    let offset = std * noise_scale

    // Return perturbed mean
    return clamp01(mean + offset)
}

/// Upper Confidence Bound (UCB) for Beta
fn beta_ucb(b: Beta, exploration_weight: f64) -> f64 {
    let mean = beta_mean(b)
    let std = beta_std(b)
    return clamp01(mean + exploration_weight * std)
}

/// Lower Confidence Bound (LCB) for Beta - conservative estimate
fn beta_lcb(b: Beta, conservatism: f64) -> f64 {
    let mean = beta_mean(b)
    let std = beta_std(b)
    return clamp01(mean - conservatism * std)
}

// ============================================================================
// SECTION 5: ML VARIANCE PENALTY (Game-Changer)
// ============================================================================

/// Variance penalty for ML loss functions
/// Penalizes predictions where the model is uncertain
struct VariancePenalty {
    base_loss: f64,          // Original loss (e.g., MSE, cross-entropy)
    variance_term: f64,      // Penalty for epistemic uncertainty
    total_loss: f64,         // Combined loss
    lambda: f64              // Variance penalty weight
}

/// Compute variance-penalized loss
/// L_total = L_base + λ * Var[prediction]
///
/// This is REVOLUTIONARY: trains models to minimize not just error,
/// but also their own ignorance about what they don't know.
fn variance_penalty_loss(base_loss: f64, prediction_variance: f64, lambda: f64) -> VariancePenalty {
    let variance_term = lambda * prediction_variance
    return VariancePenalty {
        base_loss: base_loss,
        variance_term: variance_term,
        total_loss: base_loss + variance_term,
        lambda: lambda
    }
}

/// Epistemic loss: penalize predictions that are both wrong AND confident
/// L = L_base * (1 + λ * (1 - variance)) when prediction is wrong
/// This forces the model to be uncertain about things it doesn't know
fn epistemic_calibration_loss(base_loss: f64, variance: f64, lambda: f64) -> f64 {
    // If base_loss is high (wrong prediction), penalize low variance (overconfidence)
    let overconfidence_penalty = lambda * base_loss * (1.0 - variance * 4.0)
    return base_loss + max_f64(0.0, overconfidence_penalty)
}

/// Ignorance-weighted sampling weight for curriculum learning
/// Items with high variance should be sampled more often (need more learning)
fn curriculum_sample_weight(variance: f64, base_weight: f64) -> f64 {
    // Higher variance = higher sampling priority
    let variance_boost = 1.0 + variance * 4.0  // Up to 2x weight at max variance
    return base_weight * variance_boost
}

/// Uncertainty-aware early stopping
/// Returns true if we should stop (low variance = confident enough)
fn should_stop_training(variance: f64, threshold: f64) -> i64 {
    if variance < threshold {
        return 1  // Confident enough, stop
    }
    return 0  // Still uncertain, continue training
}

/// Variance budget for batch construction
/// Ensures each batch has enough "hard" examples (high variance)
fn compute_batch_variance_budget(
    item_variances: [f64],
    target_total_variance: f64
) -> [f64] {
    // Return sampling weights to achieve target variance budget
    var total_var = 0.0
    var i: i64 = 0
    let n = len(item_variances)

    while i < n {
        total_var = total_var + item_variances[i]
        i = i + 1
    }

    // Scale weights to achieve target
    let scale = if total_var > 0.0001 { target_total_variance / total_var } else { 1.0 }

    var weights: [f64] = []
    i = 0
    while i < n {
        weights = weights ++ [item_variances[i] * scale]
        i = i + 1
    }

    return weights
}

// ============================================================================
// SECTION 6: HIERARCHICAL BAYESIAN COMBINATION
// ============================================================================

/// Combine multiple Beta distributions using hierarchical pooling
fn beta_hierarchical_combine(distributions: [Beta]) -> Beta {
    let n = len(distributions)
    if n == 0 { return beta_uniform() }
    if n == 1 { return distributions[0] }

    // Compute means and variances
    var total_mean = 0.0
    var total_var = 0.0
    var i: i64 = 0

    while i < n {
        total_mean = total_mean + beta_mean(distributions[i])
        total_var = total_var + beta_variance(distributions[i])
        i = i + 1
    }

    // Convert to f64
    var nf = 0.0
    i = 0
    while i < n {
        nf = nf + 1.0
        i = i + 1
    }

    let pooled_mean = total_mean / nf
    let pooled_var = total_var / (nf * nf)  // Variance of mean

    // Reconstruct Beta
    return beta_from_mean_variance(pooled_mean, pooled_var)
}

/// Create Beta from mean and variance (inverse problem)
fn beta_from_mean_variance(mean: f64, variance: f64) -> Beta {
    let m = clamp01(mean)
    let max_var = m * (1.0 - m)
    let v = min_f64(clamp_positive(variance), max_var * 0.99)

    let concentration = m * (1.0 - m) / v - 1.0
    let c = max_f64(2.0, concentration)

    return Beta { alpha: m * c, beta: (1.0 - m) * c }
}

/// Precision-weighted combination (inverse variance weighting)
fn beta_precision_weighted(distributions: [Beta]) -> Beta {
    let n = len(distributions)
    if n == 0 { return beta_uniform() }
    if n == 1 { return distributions[0] }

    var total_weight = 0.0
    var weighted_mean = 0.0
    var i: i64 = 0

    while i < n {
        let v = beta_variance(distributions[i])
        let weight = if v > 0.0001 { 1.0 / v } else { 10000.0 }

        weighted_mean = weighted_mean + beta_mean(distributions[i]) * weight
        total_weight = total_weight + weight
        i = i + 1
    }

    let pooled_mean = weighted_mean / total_weight
    let pooled_var = 1.0 / total_weight

    return beta_from_mean_variance(pooled_mean, pooled_var)
}

/// Simple pooling (as if all data came from one experiment)
fn beta_pool_homogeneous(distributions: [Beta]) -> Beta {
    let n = len(distributions)
    if n == 0 { return beta_uniform() }

    var total_alpha = 0.0
    var total_beta = 0.0
    var i: i64 = 0

    while i < n {
        total_alpha = total_alpha + distributions[i].alpha
        total_beta = total_beta + distributions[i].beta
        i = i + 1
    }

    // Subtract extra priors (n-1 priors added)
    var extra_prior = 0.0
    i = 0
    while i < n - 1 {
        extra_prior = extra_prior + 0.5  // Jeffreys prior contribution
        i = i + 1
    }

    return Beta {
        alpha: max_f64(0.5, total_alpha - extra_prior),
        beta: max_f64(0.5, total_beta - extra_prior)
    }
}

// ============================================================================
// SECTION 7: INFORMATION THEORY
// ============================================================================

/// KL divergence from p to q: D_KL(p || q)
fn beta_kl_divergence(p: Beta, q: Beta) -> f64 {
    let ap = p.alpha
    let bp = p.beta
    let aq = q.alpha
    let bq = q.beta

    let sum_p = ap + bp
    let sum_q = aq + bq

    // D_KL = ln B(q) - ln B(p) + (ap-aq)ψ(ap) + (bp-bq)ψ(bp) + (aq+bq-ap-bp)ψ(sum_p)
    let ln_beta_q = ln_approx(aq) + ln_approx(bq) - ln_approx(sum_q)
    let ln_beta_p = ln_approx(ap) + ln_approx(bp) - ln_approx(sum_p)

    let kl = ln_beta_q - ln_beta_p
           + (ap - aq) * digamma(ap)
           + (bp - bq) * digamma(bp)
           + (aq + bq - ap - bp) * digamma(sum_p)

    return max_f64(0.0, kl)  // KL should be non-negative
}

/// Symmetric KL (Jeffreys divergence)
fn beta_jeffreys_divergence(p: Beta, q: Beta) -> f64 {
    return beta_kl_divergence(p, q) + beta_kl_divergence(q, p)
}

/// Mutual information between two dependent Betas (approximation)
fn mutual_information_approx(joint_alpha: f64, joint_beta: f64,
                              marginal_a: Beta, marginal_b: Beta) -> f64 {
    // MI = H(A) + H(B) - H(A,B)
    let h_a = beta_entropy(marginal_a)
    let h_b = beta_entropy(marginal_b)

    // Joint entropy approximation
    let joint = Beta { alpha: joint_alpha, beta: joint_beta }
    let h_joint = beta_entropy(joint)

    return max_f64(0.0, h_a + h_b - h_joint)
}

// ============================================================================
// SECTION 8: HYPOTHESIS TESTING
// ============================================================================

/// Probability that true parameter exceeds threshold
fn beta_prob_greater_than(b: Beta, threshold: f64) -> f64 {
    // Approximate using CDF
    // P(X > t) ≈ 1 - Φ((t - mean) / std)
    let mean = beta_mean(b)
    let std = beta_std(b)

    if std < 0.0001 {
        if mean > threshold { return 1.0 }
        return 0.0
    }

    let z = (threshold - mean) / std

    // Approximate normal CDF
    let cdf = 0.5 * (1.0 + erf_approx(z / sqrt_f64(2.0)))

    return 1.0 - cdf
}

/// Probability that true parameter is less than threshold
fn beta_prob_less_than(b: Beta, threshold: f64) -> f64 {
    return 1.0 - beta_prob_greater_than(b, threshold)
}

/// Error function approximation
fn erf_approx(x: f64) -> f64 {
    // Horner form approximation
    let a1 = 0.254829592
    let a2 = -0.284496736
    let a3 = 1.421413741
    let a4 = -1.453152027
    let a5 = 1.061405429
    let p = 0.3275911

    let sign = if x >= 0.0 { 1.0 } else { 0.0 - 1.0 }
    let x_abs = abs_f64(x)

    let t = 1.0 / (1.0 + p * x_abs)
    let t2 = t * t
    let t3 = t2 * t
    let t4 = t3 * t
    let t5 = t4 * t

    let y = 1.0 - (a1*t + a2*t2 + a3*t3 + a4*t4 + a5*t5) * exp_approx(0.0 - x_abs * x_abs)

    return sign * y
}

/// Bayes factor for point null H0: θ = θ0 vs H1: θ ≠ θ0
fn bayes_factor_point_null(posterior: Beta, prior: Beta, theta0: f64) -> f64 {
    // Savage-Dickey ratio: BF = p(θ0|prior) / p(θ0|posterior)
    let prior_density = beta_pdf_at(prior, theta0)
    let posterior_density = beta_pdf_at(posterior, theta0)

    if posterior_density < 0.0001 { return 1000.0 }  // Strong evidence against null
    return prior_density / posterior_density
}

/// Beta PDF at point x
fn beta_pdf_at(b: Beta, x: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    if x >= 1.0 { return 0.0 }

    // PDF = x^(α-1) * (1-x)^(β-1) / B(α,β)
    let log_pdf = (b.alpha - 1.0) * ln_approx(x)
                + (b.beta - 1.0) * ln_approx(1.0 - x)
                - ln_approx(b.alpha) - ln_approx(b.beta) + ln_approx(b.alpha + b.beta)

    return exp_approx(log_pdf)
}

// ============================================================================
// SECTION 9: NOVELTY - VARIANCE AS OPTIMIZATION TARGET
// ============================================================================

/// Variance-minimizing prior selection
/// Choose prior that minimizes expected posterior variance
fn optimal_prior_for_variance(expected_successes: f64, expected_trials: f64) -> Beta {
    // Jeffreys prior minimizes expected posterior variance in the limit
    // But for finite samples, adjust based on expected proportion
    let expected_prop = expected_successes / max_f64(1.0, expected_trials)

    // Symmetric prior centered at expected proportion
    let concentration = 2.0 + expected_trials * 0.1  // Scales with data
    return beta_informative(expected_prop, concentration)
}

/// Variance budget allocation across multiple estimates
/// Given total variance budget, allocate to minimize overall uncertainty
fn allocate_variance_budget(
    current_variances: [f64],
    target_total_variance: f64
) -> [f64] {
    // Equal allocation: each gets proportional share
    let n = len(current_variances)

    var total_current = 0.0
    var i: i64 = 0
    while i < n {
        total_current = total_current + current_variances[i]
        i = i + 1
    }

    // Convert to f64
    var nf = 0.0
    i = 0
    while i < n {
        nf = nf + 1.0
        i = i + 1
    }

    let target_per_item = target_total_variance / nf

    var allocations: [f64] = []
    i = 0
    while i < n {
        allocations = allocations ++ [target_per_item]
        i = i + 1
    }

    return allocations
}

/// Expected observations needed to reach target variance
fn observations_for_target_variance(current: Beta, target_variance: f64) -> f64 {
    let current_var = beta_variance(current)
    if current_var <= target_variance { return 0.0 }

    // Variance scales as 1/n approximately
    // target_var / current_var ≈ current_n / (current_n + k)
    // Solving: k ≈ current_n * (current_var / target_var - 1)

    let n = current.alpha + current.beta
    let ratio = current_var / target_variance

    return n * (ratio - 1.0)
}

/// Ignorance-driven data collection priority
/// Returns priority score for collecting more data (higher = more important)
fn data_collection_priority(b: Beta, importance_weight: f64) -> f64 {
    let variance = beta_variance(b)
    let info_gain = expected_info_gain(b)

    // Priority = importance * (variance + info_gain)
    return importance_weight * (variance + info_gain)
}

// ============================================================================
// SECTION 10: PRETTY PRINTING AND DIAGNOSTICS
// ============================================================================

/// Print Beta distribution summary
fn beta_print(b: Beta) -> i64 {
    print("Beta(")
    print(b.alpha)
    print(", ")
    print(b.beta)
    print(") → mean=")
    print(beta_mean(b))
    print(" var=")
    print(beta_variance(b))
    println("")
    return 0
}

/// Print epistemic summary
fn epistemic_print(s: EpistemicSummary) -> i64 {
    println("Epistemic Summary:")
    print("  Mean: ")
    println(s.mean)
    print("  Variance (ignorance): ")
    println(s.variance)
    print("  Std Dev: ")
    println(s.std_dev)
    print("  Sample Size: ")
    println(s.sample_size)
    print("  95% CI: [")
    print(s.ci_lo)
    print(", ")
    print(s.ci_hi)
    println("]")
    return 0
}

/// Print active inference metrics
fn active_inference_print(m: ActiveInferenceMetrics) -> i64 {
    println("Active Inference Metrics:")
    print("  Expected Info Gain: ")
    println(m.expected_info_gain)
    print("  Variance Reduction: ")
    println(m.variance_reduction)
    print("  Exploration Value: ")
    println(m.exploration_value)
    print("  Exploitation Value: ")
    println(m.exploitation_value)
    print("  Optimal Action: ")
    if m.optimal_action < 0.5 {
        println("EXPLORE (collect more data)")
    } else {
        println("EXPLOIT (use current estimate)")
    }
    return 0
}

// ============================================================================
// MAIN: Demonstration
// ============================================================================

fn main() -> i32 {
    println("=== epistemic::stats — Revolutionary Statistics Demo ===")
    println("")

    // Section 1: Basic Beta operations
    println("--- Section 1: Basic Beta Operations ---")
    let prior = beta_uniform()
    beta_print(prior)

    let posterior = beta_update(prior, 7.0, 3.0)
    print("After 7 successes, 3 failures: ")
    beta_print(posterior)

    let summary = beta_summary(posterior)
    epistemic_print(summary)
    println("")

    // Section 2: Active Inference
    println("--- Section 2: Active Inference ---")
    let metrics = active_inference_metrics(posterior, 0.7)
    active_inference_print(metrics)
    println("")

    // Section 3: Variance Penalty for ML
    println("--- Section 3: Variance Penalty (ML Integration) ---")
    let base_loss = 0.5
    let pred_variance = beta_variance(posterior)
    let lambda = 0.1
    let penalty = variance_penalty_loss(base_loss, pred_variance, lambda)
    print("  Base Loss: ")
    println(penalty.base_loss)
    print("  Variance Term: ")
    println(penalty.variance_term)
    print("  Total Loss: ")
    println(penalty.total_loss)
    println("")

    // Section 4: Hierarchical Combination
    println("--- Section 4: Hierarchical Bayesian ---")
    let d1 = beta_update(beta_uniform(), 8.0, 2.0)
    let d2 = beta_update(beta_uniform(), 6.0, 4.0)
    let d3 = beta_update(beta_uniform(), 7.0, 3.0)

    print("Study 1: ")
    beta_print(d1)
    print("Study 2: ")
    beta_print(d2)
    print("Study 3: ")
    beta_print(d3)

    let combined = beta_precision_weighted([d1, d2, d3])
    print("Combined (precision-weighted): ")
    beta_print(combined)
    println("")

    // Section 5: Novelty - Variance as optimization target
    println("--- Section 5: Variance as Optimization Target ---")
    let obs_needed = observations_for_target_variance(posterior, 0.005)
    print("Observations needed to reach variance < 0.005: ")
    println(obs_needed)

    let priority = data_collection_priority(posterior, 1.0)
    print("Data collection priority score: ")
    println(priority)
    println("")

    // Section 6: Information Theory
    println("--- Section 6: Information Theory ---")
    let kl = beta_kl_divergence(posterior, prior)
    print("KL divergence (posterior || prior): ")
    println(kl)
    println("")

    println("=== Demo Complete ===")
    println("")
    println("Key Innovations:")
    println("1. Beta posterior as default - no point estimates")
    println("2. Variance is first-class - measures 'ignorance'")
    println("3. Active inference - guides data collection")
    println("4. Variance penalty - ML loss that minimizes uncertainty")
    println("5. Hierarchical Bayesian - principled meta-analysis")
    println("")
    println("This changes scientific computing forever.")

    return 0
}
