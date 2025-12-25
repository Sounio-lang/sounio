// combine.d - Epistemic Combination Calculus
//
// Every combination of beliefs follows principled rules.
// Every fusion of sources respects conflict.
// Every propagation knows its assumptions.
//
// Philosophy:
// When two sources say different things, averaging is a lie.
// Dempster-Shafer respects conflict. Bayesian fusion respects priors.
// Interval arithmetic respects ignorance. This module implements all three.
//
// The choice of combination rule is itself epistemic metadata.
//
// References:
// - Shafer (1976): "A Mathematical Theory of Evidence"
// - Dezert & Smarandache: "Advances in DSmT for Information Fusion"
// - Ferson et al.: "Constructing Probability Boxes"
// - Walley: "Statistical Reasoning with Imprecise Probabilities"

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

fn min_f64(a: f64, b: f64) -> f64 {
    if a < b { return a }
    return b
}

fn max_f64(a: f64, b: f64) -> f64 {
    if a > b { return a }
    return b
}

fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    var y = x
    y = (y + x / y) / 2.0
    y = (y + x / y) / 2.0
    y = (y + x / y) / 2.0
    y = (y + x / y) / 2.0
    y = (y + x / y) / 2.0
    y = (y + x / y) / 2.0
    y = (y + x / y) / 2.0
    y = (y + x / y) / 2.0
    return y
}

fn ln_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 - 1000000.0 }
    var y = x - 1.0
    if abs_f64(y) < 1.0 {
        var result = y
        var term = y
        var i: i64 = 2
        while i < 20 {
            term = 0.0 - term * y
            result = result + term / (i as f64)
            i = i + 1
        }
        return result
    }
    var guess = 1.0
    var i: i64 = 0
    while i < 30 {
        let exp_guess = exp_f64(guess)
        guess = guess + (x - exp_guess) / exp_guess
        i = i + 1
    }
    return guess
}

fn exp_f64(x: f64) -> f64 {
    var result = 1.0
    var term = 1.0
    var i: i64 = 1
    while i < 30 {
        term = term * x / (i as f64)
        result = result + term
        if abs_f64(term) < 0.000000000000001 { i = 30 }
        i = i + 1
    }
    return result
}

// ============================================================================
// BETA CONFIDENCE (Epistemic Uncertainty)
// ============================================================================

// Beta distribution for representing confidence
// Alpha = successes + 1, Beta = failures + 1
struct BetaConfidence {
    alpha: f64,
    beta: f64
}

fn beta_confidence_new(alpha: f64, beta: f64) -> BetaConfidence {
    return BetaConfidence { alpha: alpha, beta: beta }
}

fn beta_confidence_from_evidence(successes: i64, failures: i64) -> BetaConfidence {
    return BetaConfidence {
        alpha: (successes + 1) as f64,
        beta: (failures + 1) as f64
    }
}

fn beta_confidence_strong(mean: f64) -> BetaConfidence {
    // High confidence at the given mean
    let alpha = mean * 20.0 + 1.0
    let beta = (1.0 - mean) * 20.0 + 1.0
    return BetaConfidence { alpha: alpha, beta: beta }
}

fn beta_confidence_weak(mean: f64) -> BetaConfidence {
    // Low confidence at the given mean
    let alpha = mean * 2.0 + 1.0
    let beta = (1.0 - mean) * 2.0 + 1.0
    return BetaConfidence { alpha: alpha, beta: beta }
}

fn beta_confidence_uniform() -> BetaConfidence {
    // Complete uncertainty
    return BetaConfidence { alpha: 1.0, beta: 1.0 }
}

fn beta_mean(c: BetaConfidence) -> f64 {
    return c.alpha / (c.alpha + c.beta)
}

fn beta_variance(c: BetaConfidence) -> f64 {
    let ab = c.alpha + c.beta
    return (c.alpha * c.beta) / (ab * ab * (ab + 1.0))
}

fn beta_entropy(c: BetaConfidence) -> f64 {
    // Simplified entropy approximation
    let p = beta_mean(c)
    if p <= 0.0 || p >= 1.0 { return 0.0 }
    return 0.0 - p * ln_f64(p) - (1.0 - p) * ln_f64(1.0 - p)
}

// ============================================================================
// PROBABILITY INTERVALS (Imprecise Probabilities)
// ============================================================================

// Probability interval [lower, upper] for imprecise probabilities
struct ProbabilityInterval {
    lower: f64,
    upper: f64
}

fn prob_interval_new(lower: f64, upper: f64) -> ProbabilityInterval {
    var l = lower
    var u = upper
    if l < 0.0 { l = 0.0 }
    if u > 1.0 { u = 1.0 }
    if l > u { l = u }
    return ProbabilityInterval { lower: l, upper: u }
}

fn prob_interval_precise(p: f64) -> ProbabilityInterval {
    return prob_interval_new(p, p)
}

fn prob_interval_vacuous() -> ProbabilityInterval {
    return ProbabilityInterval { lower: 0.0, upper: 1.0 }
}

fn prob_interval_width(p: ProbabilityInterval) -> f64 {
    return p.upper - p.lower
}

fn prob_interval_midpoint(p: ProbabilityInterval) -> f64 {
    return (p.lower + p.upper) / 2.0
}

fn prob_interval_is_precise(p: ProbabilityInterval) -> i64 {
    if prob_interval_width(p) < 0.0000000001 { return 1 }
    return 0
}

// Conjunction: P(A AND B) under independence
fn prob_interval_and_independent(p1: ProbabilityInterval, p2: ProbabilityInterval) -> ProbabilityInterval {
    return ProbabilityInterval {
        lower: p1.lower * p2.lower,
        upper: p1.upper * p2.upper
    }
}

// Disjunction: P(A OR B) under independence
fn prob_interval_or_independent(p1: ProbabilityInterval, p2: ProbabilityInterval) -> ProbabilityInterval {
    var lower = p1.lower + p2.lower - p1.upper * p2.upper
    var upper = p1.upper + p2.upper - p1.lower * p2.lower
    if lower < 0.0 { lower = 0.0 }
    if upper > 1.0 { upper = 1.0 }
    return ProbabilityInterval { lower: lower, upper: upper }
}

// Negation: P(NOT A)
fn prob_interval_not(p: ProbabilityInterval) -> ProbabilityInterval {
    return ProbabilityInterval {
        lower: 1.0 - p.upper,
        upper: 1.0 - p.lower
    }
}

// Freshet bounds for conjunction (no independence assumption)
fn prob_interval_and_frechet(p1: ProbabilityInterval, p2: ProbabilityInterval) -> ProbabilityInterval {
    var lower = p1.lower + p2.lower - 1.0
    if lower < 0.0 { lower = 0.0 }
    let upper = min_f64(p1.upper, p2.upper)
    return ProbabilityInterval { lower: lower, upper: upper }
}

// Freshet bounds for disjunction
fn prob_interval_or_frechet(p1: ProbabilityInterval, p2: ProbabilityInterval) -> ProbabilityInterval {
    let lower = max_f64(p1.lower, p2.lower)
    var upper = p1.upper + p2.upper
    if upper > 1.0 { upper = 1.0 }
    return ProbabilityInterval { lower: lower, upper: upper }
}

// ============================================================================
// DEMPSTER-SHAFER BELIEF FUNCTIONS (Simplified)
// ============================================================================

// For a frame of discernment with up to 4 hypotheses
// We use a fixed array for masses on each subset

// Belief mass for a simple 2-hypothesis frame (e.g., True/False)
struct BinaryMass {
    m_empty: f64,      // Should be 0
    m_true: f64,       // Mass on {True}
    m_false: f64,      // Mass on {False}
    m_both: f64        // Mass on {True, False} = ignorance
}

fn binary_mass_new() -> BinaryMass {
    return BinaryMass {
        m_empty: 0.0,
        m_true: 0.0,
        m_false: 0.0,
        m_both: 1.0  // Start with total ignorance
    }
}

fn binary_mass_from_evidence(true_mass: f64, false_mass: f64, ignorance: f64) -> BinaryMass {
    let total = true_mass + false_mass + ignorance
    var t = true_mass
    var f = false_mass
    var i = ignorance
    if total > 0.0 {
        t = true_mass / total
        f = false_mass / total
        i = ignorance / total
    }
    return BinaryMass {
        m_empty: 0.0,
        m_true: t,
        m_false: f,
        m_both: i
    }
}

// Belief: lower bound on probability
fn binary_mass_belief_true(m: BinaryMass) -> f64 {
    return m.m_true
}

fn binary_mass_belief_false(m: BinaryMass) -> f64 {
    return m.m_false
}

// Plausibility: upper bound on probability
fn binary_mass_plausibility_true(m: BinaryMass) -> f64 {
    return m.m_true + m.m_both
}

fn binary_mass_plausibility_false(m: BinaryMass) -> f64 {
    return m.m_false + m.m_both
}

// Dempster's rule of combination
fn dempster_combine_binary(m1: BinaryMass, m2: BinaryMass) -> BinaryMass {
    // Compute unnormalized masses
    var new_true = 0.0
    var new_false = 0.0
    var new_both = 0.0
    var conflict = 0.0

    // m1_true AND m2_true -> true
    new_true = new_true + m1.m_true * m2.m_true
    // m1_true AND m2_both -> true
    new_true = new_true + m1.m_true * m2.m_both
    // m1_both AND m2_true -> true
    new_true = new_true + m1.m_both * m2.m_true

    // m1_false AND m2_false -> false
    new_false = new_false + m1.m_false * m2.m_false
    // m1_false AND m2_both -> false
    new_false = new_false + m1.m_false * m2.m_both
    // m1_both AND m2_false -> false
    new_false = new_false + m1.m_both * m2.m_false

    // m1_both AND m2_both -> both
    new_both = m1.m_both * m2.m_both

    // Conflict: m1_true AND m2_false OR m1_false AND m2_true
    conflict = m1.m_true * m2.m_false + m1.m_false * m2.m_true

    // Normalize
    let norm = 1.0 - conflict
    if norm > 0.0001 {
        new_true = new_true / norm
        new_false = new_false / norm
        new_both = new_both / norm
    }

    return BinaryMass {
        m_empty: 0.0,
        m_true: new_true,
        m_false: new_false,
        m_both: new_both
    }
}

// Conflict measure
fn dempster_conflict_binary(m1: BinaryMass, m2: BinaryMass) -> f64 {
    return m1.m_true * m2.m_false + m1.m_false * m2.m_true
}

// ============================================================================
// BAYESIAN FUSION (Simplified)
// ============================================================================

// Bayesian belief state for a binary hypothesis
struct BayesianBelief {
    prior_true: f64,
    likelihood_true: f64,
    likelihood_false: f64
}

fn bayesian_belief_new(prior: f64) -> BayesianBelief {
    return BayesianBelief {
        prior_true: prior,
        likelihood_true: 1.0,
        likelihood_false: 1.0
    }
}

fn bayesian_update(belief: BayesianBelief, likelihood_if_true: f64, likelihood_if_false: f64) -> BayesianBelief {
    let new_lt = belief.likelihood_true * likelihood_if_true
    let new_lf = belief.likelihood_false * likelihood_if_false
    return BayesianBelief {
        prior_true: belief.prior_true,
        likelihood_true: new_lt,
        likelihood_false: new_lf
    }
}

fn bayesian_posterior(belief: BayesianBelief) -> f64 {
    let p_true = belief.prior_true
    let p_false = 1.0 - belief.prior_true
    let numerator = belief.likelihood_true * p_true
    let denominator = belief.likelihood_true * p_true + belief.likelihood_false * p_false
    if denominator > 0.0 {
        return numerator / denominator
    }
    return belief.prior_true
}

fn bayesian_odds_ratio(belief: BayesianBelief) -> f64 {
    let post = bayesian_posterior(belief)
    if post >= 1.0 { return 1000000.0 }
    if post <= 0.0 { return 0.0 }
    return post / (1.0 - post)
}

// ============================================================================
// CONFIDENCE COMBINATION RULES
// ============================================================================

// Combination rule types
fn rule_multiplicative() -> i64 { return 0 }
fn rule_quadrature() -> i64 { return 1 }
fn rule_minimum() -> i64 { return 2 }
fn rule_dempster() -> i64 { return 3 }

// Combine two confidences using specified rule
fn combine_confidence(c1: BetaConfidence, c2: BetaConfidence, rule: i64) -> BetaConfidence {
    let m1 = beta_mean(c1)
    let m2 = beta_mean(c2)

    if rule == rule_multiplicative() {
        // Independent errors: multiply means
        let combined = m1 * m2
        return beta_confidence_strong(combined)
    }

    if rule == rule_quadrature() {
        // Correlated errors: use quadrature
        let v1 = 1.0 - m1
        let v2 = 1.0 - m2
        let combined_uncertainty = sqrt_f64(v1 * v1 + v2 * v2)
        var combined = 1.0 - combined_uncertainty
        if combined < 0.0 { combined = 0.0 }
        return beta_confidence_strong(combined)
    }

    if rule == rule_minimum() {
        // Conservative: take minimum
        let combined = min_f64(m1, m2)
        return beta_confidence_strong(combined)
    }

    if rule == rule_dempster() {
        // Dempster-Shafer style
        let conflict = (1.0 - m1) * m2 + m1 * (1.0 - m2)
        let combined = m1 * m2
        var norm = 1.0 - conflict
        if norm < 0.01 { norm = 0.01 }
        return beta_confidence_strong(combined / norm)
    }

    // Default: multiplicative
    return beta_confidence_strong(m1 * m2)
}

// Combine multiple confidences
fn combine_confidence_many(confidences: [BetaConfidence], n: i64, rule: i64) -> BetaConfidence {
    if n <= 0 { return beta_confidence_uniform() }
    if n == 1 { return confidences[0] }

    var result = confidences[0]
    var i: i64 = 1
    while i < n {
        result = combine_confidence(result, confidences[i], rule)
        i = i + 1
    }
    return result
}

// ============================================================================
// CONFIDENCE PROPAGATION
// ============================================================================

// Propagate confidence through arithmetic
fn propagate_arithmetic(c1: BetaConfidence, c2: BetaConfidence, independent: i64) -> BetaConfidence {
    if independent > 0 {
        return combine_confidence(c1, c2, rule_multiplicative())
    }
    return combine_confidence(c1, c2, rule_minimum())
}

// Propagate confidence through comparison
fn propagate_comparison(c1: BetaConfidence, c2: BetaConfidence, value_distance: f64, combined_std: f64) -> BetaConfidence {
    if combined_std <= 0.0 {
        return combine_confidence(c1, c2, rule_minimum())
    }

    // How many standard deviations apart?
    let z = abs_f64(value_distance) / combined_std

    // Higher z = more confident in comparison
    var comparison_confidence = 1.0 - exp_f64(0.0 - z * z / 2.0)

    // Combined with input confidences
    let total = comparison_confidence * beta_mean(c1) * beta_mean(c2)
    return beta_confidence_strong(total)
}

// Propagate confidence through aggregation
fn propagate_aggregation(confidences: [BetaConfidence], n: i64, independent: i64) -> BetaConfidence {
    if n <= 0 { return beta_confidence_uniform() }

    if independent > 0 {
        // For independent sources, confidence grows with sqrt(n)
        var product = 1.0
        var i: i64 = 0
        while i < n {
            product = product * beta_mean(confidences[i])
            i = i + 1
        }
        let boost = sqrt_f64(n as f64) / (n as f64)
        var result = exp_f64(boost * ln_f64(product))
        if result > 0.999 { result = 0.999 }
        return beta_confidence_strong(result)
    }

    // For correlated, take minimum
    var min_conf = 1.0
    var i: i64 = 0
    while i < n {
        let m = beta_mean(confidences[i])
        if m < min_conf { min_conf = m }
        i = i + 1
    }
    return beta_confidence_strong(min_conf)
}

// ============================================================================
// PRINTING UTILITIES
// ============================================================================

fn print_beta_confidence(name: [u8], c: BetaConfidence) -> i64 {
    print("  ")
    print(name)
    print(": mean=")
    print(beta_mean(c))
    print(", var=")
    println(beta_variance(c))
    return 0
}

fn print_prob_interval(name: [u8], p: ProbabilityInterval) -> i64 {
    print("  ")
    print(name)
    print(": [")
    print(p.lower)
    print(", ")
    print(p.upper)
    println("]")
    return 0
}

fn print_binary_mass(name: [u8], m: BinaryMass) -> i64 {
    print("  ")
    print(name)
    print(": m(T)=")
    print(m.m_true)
    print(", m(F)=")
    print(m.m_false)
    print(", m(TF)=")
    println(m.m_both)
    return 0
}

// ============================================================================
// DEMONSTRATION
// ============================================================================

fn main() -> i32 {
    println("=== epistemic::combine — Combination Calculus Demo ===")
    println("")

    // Part 1: Beta Confidence
    println("--- Part 1: Beta Confidence ---")
    let c1 = beta_confidence_strong(0.9)
    let c2 = beta_confidence_strong(0.8)
    print_beta_confidence("Source 1", c1)
    print_beta_confidence("Source 2", c2)

    let c_mult = combine_confidence(c1, c2, rule_multiplicative())
    let c_quad = combine_confidence(c1, c2, rule_quadrature())
    let c_min = combine_confidence(c1, c2, rule_minimum())
    println("Combined (multiplicative):")
    print_beta_confidence("  Result", c_mult)
    println("Combined (quadrature):")
    print_beta_confidence("  Result", c_quad)
    println("Combined (minimum):")
    print_beta_confidence("  Result", c_min)
    println("")

    // Part 2: Probability Intervals
    println("--- Part 2: Probability Intervals ---")
    let p1 = prob_interval_new(0.3, 0.5)
    let p2 = prob_interval_new(0.4, 0.6)
    print_prob_interval("P1", p1)
    print_prob_interval("P2", p2)

    let p_and = prob_interval_and_frechet(p1, p2)
    let p_or = prob_interval_or_frechet(p1, p2)
    print_prob_interval("P1 AND P2 (Frechet)", p_and)
    print_prob_interval("P1 OR P2 (Frechet)", p_or)
    println("")

    // Part 3: Dempster-Shafer
    println("--- Part 3: Dempster-Shafer ---")
    let m1 = binary_mass_from_evidence(0.6, 0.2, 0.2)
    let m2 = binary_mass_from_evidence(0.5, 0.3, 0.2)
    print_binary_mass("Evidence 1", m1)
    print_binary_mass("Evidence 2", m2)

    let m_combined = dempster_combine_binary(m1, m2)
    print_binary_mass("Combined (Dempster)", m_combined)

    print("Conflict: ")
    println(dempster_conflict_binary(m1, m2))

    print("Bel(True): ")
    println(binary_mass_belief_true(m_combined))
    print("Pl(True): ")
    println(binary_mass_plausibility_true(m_combined))
    println("")

    // Part 4: Bayesian Fusion
    println("--- Part 4: Bayesian Fusion ---")
    let b0 = bayesian_belief_new(0.5)
    let b1 = bayesian_update(b0, 0.9, 0.1)
    let b2 = bayesian_update(b1, 0.8, 0.2)

    print("Prior: 0.5")
    println("")
    print("After evidence 1 (L_T=0.9, L_F=0.1): ")
    println(bayesian_posterior(b1))
    print("After evidence 2 (L_T=0.8, L_F=0.2): ")
    println(bayesian_posterior(b2))
    println("")

    println("=== Demo Complete ===")

    return 0
}
