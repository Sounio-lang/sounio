//! Evidence Fusion Module
//!
//! This module provides rigorous evidence combination for epistemic computing.
//!
//! TWO DISTINCT FUSION PARADIGMS:
//!
//! 1. DEMPSTER-SHAFER (for propositions/hypotheses)
//!    - Combines belief masses over discrete hypotheses
//!    - Explicit conflict handling (K factor)
//!    - Appropriate for: binary outcomes, categorical decisions
//!
//! 2. BAYESIAN (for continuous beliefs)
//!    - Likelihood-ratio updates
//!    - Proper posterior computation
//!    - Appropriate for: parameter estimation, probability updates
//!
//! CRITICAL: DS is NOT for combining numeric measurements.
//! Use inverse-variance weighting (see core.d fuse_measurements) for that.

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn exp(x: f64) -> f64;
}

fn sqrt_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 }
    return sqrt(x)
}

fn log_f64(x: f64) -> f64 {
    if x <= 0.0 { return -1.0e308 }
    return log(x)
}

fn exp_f64(x: f64) -> f64 {
    return exp(x)
}

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

fn min_f64(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

fn max_f64(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

fn clamp_f64(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo { return lo }
    if x > hi { return hi }
    return x
}

// ============================================================================
// DEMPSTER-SHAFER FOR BINARY PROPOSITIONS
// ============================================================================

// Binary mass function: masses on {H}, {¬H}, {H,¬H} (uncertainty)
// INVARIANT: m_h + m_not_h + m_uncertain = 1.0
// INVARIANT: all masses >= 0
struct BinaryMass {
    m_h: f64,           // mass on hypothesis H
    m_not_h: f64,       // mass on ¬H
    m_uncertain: f64,   // mass on {H, ¬H} (ignorance)
}

// Create a binary mass function from belief and plausibility
fn binary_mass_new(belief_h: f64, plausibility_h: f64) -> BinaryMass {
    // Bel(H) <= Pl(H), both in [0,1]
    let bel = clamp_f64(belief_h, 0.0, 1.0)
    let pl = clamp_f64(plausibility_h, bel, 1.0)

    // m(H) = Bel(H)
    // m(¬H) = 1 - Pl(H)
    // m({H,¬H}) = Pl(H) - Bel(H)
    return BinaryMass {
        m_h: bel,
        m_not_h: 1.0 - pl,
        m_uncertain: pl - bel,
    }
}

// Create from direct mass assignment
fn binary_mass_direct(m_h: f64, m_not_h: f64, m_uncertain: f64) -> BinaryMass {
    // Normalize to ensure sum = 1
    let total = m_h + m_not_h + m_uncertain
    if total < 1.0e-15 {
        // Complete ignorance
        return BinaryMass { m_h: 0.0, m_not_h: 0.0, m_uncertain: 1.0 }
    }

    return BinaryMass {
        m_h: max_f64(0.0, m_h / total),
        m_not_h: max_f64(0.0, m_not_h / total),
        m_uncertain: max_f64(0.0, m_uncertain / total),
    }
}

// Complete ignorance (vacuous belief)
fn binary_mass_vacuous() -> BinaryMass {
    return BinaryMass { m_h: 0.0, m_not_h: 0.0, m_uncertain: 1.0 }
}

// Complete belief in H
fn binary_mass_certain_h() -> BinaryMass {
    return BinaryMass { m_h: 1.0, m_not_h: 0.0, m_uncertain: 0.0 }
}

// Complete belief in ¬H
fn binary_mass_certain_not_h() -> BinaryMass {
    return BinaryMass { m_h: 0.0, m_not_h: 1.0, m_uncertain: 0.0 }
}

// Get belief in H: Bel(H) = m(H)
fn binary_belief_h(m: BinaryMass) -> f64 {
    return m.m_h
}

// Get plausibility of H: Pl(H) = m(H) + m({H,¬H}) = 1 - m(¬H)
fn binary_plausibility_h(m: BinaryMass) -> f64 {
    return m.m_h + m.m_uncertain
}

// Get uncertainty interval width: Pl(H) - Bel(H)
fn binary_uncertainty_width(m: BinaryMass) -> f64 {
    return m.m_uncertain
}

// ============================================================================
// DEMPSTER'S RULE OF COMBINATION
// ============================================================================

// Result of DS combination includes conflict measure
struct DSCombineResult {
    combined: BinaryMass,
    conflict_k: f64,    // K ∈ [0,1), higher = more conflict
    is_valid: bool,     // false if K = 1 (total conflict)
}

// Combine two binary mass functions using Dempster's rule
// WARNING: High conflict (K near 1) indicates inconsistent evidence
fn ds_combine_binary(m1: BinaryMass, m2: BinaryMass) -> DSCombineResult {
    // Compute unnormalized combined masses
    // m12(H) comes from: m1(H)·m2(H) + m1(H)·m2(Θ) + m1(Θ)·m2(H)
    // m12(¬H) comes from: m1(¬H)·m2(¬H) + m1(¬H)·m2(Θ) + m1(Θ)·m2(¬H)
    // m12(Θ) comes from: m1(Θ)·m2(Θ)

    let raw_h = m1.m_h * m2.m_h +
                m1.m_h * m2.m_uncertain +
                m1.m_uncertain * m2.m_h

    let raw_not_h = m1.m_not_h * m2.m_not_h +
                    m1.m_not_h * m2.m_uncertain +
                    m1.m_uncertain * m2.m_not_h

    let raw_uncertain = m1.m_uncertain * m2.m_uncertain

    // Conflict: mass assigned to empty set
    // K = m1(H)·m2(¬H) + m1(¬H)·m2(H)
    let conflict_k = m1.m_h * m2.m_not_h + m1.m_not_h * m2.m_h

    // Check for total conflict
    if conflict_k >= 1.0 - 1.0e-10 {
        return DSCombineResult {
            combined: binary_mass_vacuous(),
            conflict_k: 1.0,
            is_valid: false,
        }
    }

    // Normalize by (1 - K)
    let norm = 1.0 / (1.0 - conflict_k)

    return DSCombineResult {
        combined: BinaryMass {
            m_h: raw_h * norm,
            m_not_h: raw_not_h * norm,
            m_uncertain: raw_uncertain * norm,
        },
        conflict_k: conflict_k,
        is_valid: true,
    }
}

// Combine multiple evidence sources
fn ds_combine_multiple(masses: &[BinaryMass], n: i64) -> DSCombineResult {
    if n <= 0 {
        return DSCombineResult {
            combined: binary_mass_vacuous(),
            conflict_k: 0.0,
            is_valid: true,
        }
    }
    if n == 1 {
        return DSCombineResult {
            combined: masses[0],
            conflict_k: 0.0,
            is_valid: true,
        }
    }

    var result = masses[0]
    var total_conflict = 0.0

    for i in 1..n {
        let combined = ds_combine_binary(result, masses[i as i64])
        if !combined.is_valid {
            return combined
        }
        result = combined.combined
        // Accumulate conflict (approximately)
        total_conflict = total_conflict + combined.conflict_k * (1.0 - total_conflict)
    }

    return DSCombineResult {
        combined: result,
        conflict_k: total_conflict,
        is_valid: true,
    }
}

// ============================================================================
// BAYESIAN BELIEF (for probability updates)
// ============================================================================

// Beta distribution parameters (conjugate prior for binomial)
struct BetaBelief {
    alpha: f64,     // successes + 1
    beta: f64,      // failures + 1
}

// Create uniform prior (complete ignorance)
fn beta_uniform() -> BetaBelief {
    return BetaBelief { alpha: 1.0, beta: 1.0 }
}

// Create from observed data
fn beta_from_data(successes: f64, failures: f64) -> BetaBelief {
    return BetaBelief {
        alpha: max_f64(0.0, successes) + 1.0,
        beta: max_f64(0.0, failures) + 1.0,
    }
}

// Create informative prior
fn beta_prior(alpha: f64, beta: f64) -> BetaBelief {
    return BetaBelief {
        alpha: max_f64(0.001, alpha),
        beta: max_f64(0.001, beta),
    }
}

// Get mean of Beta distribution
fn beta_mean(b: BetaBelief) -> f64 {
    return b.alpha / (b.alpha + b.beta)
}

// Get variance of Beta distribution
fn beta_variance(b: BetaBelief) -> f64 {
    let total = b.alpha + b.beta
    return (b.alpha * b.beta) / (total * total * (total + 1.0))
}

// Get standard deviation
fn beta_std(b: BetaBelief) -> f64 {
    return sqrt_f64(beta_variance(b))
}

// Get effective sample size (measure of certainty)
fn beta_sample_size(b: BetaBelief) -> f64 {
    return b.alpha + b.beta - 2.0  // subtract prior counts
}

// ============================================================================
// BAYESIAN UPDATES
// ============================================================================

// Update Beta belief with new observations
fn beta_update(prior: BetaBelief, successes: f64, failures: f64) -> BetaBelief {
    return BetaBelief {
        alpha: prior.alpha + max_f64(0.0, successes),
        beta: prior.beta + max_f64(0.0, failures),
    }
}

// Combine two independent Beta beliefs (product of experts)
fn beta_combine_independent(b1: BetaBelief, b2: BetaBelief) -> BetaBelief {
    // Approximate: add pseudo-counts
    return BetaBelief {
        alpha: b1.alpha + b2.alpha - 1.0,  // subtract one prior
        beta: b1.beta + b2.beta - 1.0,
    }
}

// Convert Beta belief to confidence score
// Uses the probability that the true rate is above 0.5 (for H)
fn beta_to_confidence(b: BetaBelief) -> f64 {
    // Approximate using normal approximation for large samples
    let mean = beta_mean(b)
    let std = beta_std(b)

    if std < 1.0e-10 {
        return if mean > 0.5 { 1.0 } else { 0.0 }
    }

    // z-score for mean being above 0.5
    let z = (mean - 0.5) / std

    // Approximate CDF using logistic approximation
    let conf = 1.0 / (1.0 + exp_f64(-1.7 * z))
    return clamp_f64(conf, 0.0, 1.0)
}

// ============================================================================
// LIKELIHOOD RATIO UPDATES
// ============================================================================

// Odds form of probability
struct OddsBelief {
    log_odds: f64,    // log(p / (1-p))
}

// Convert probability to odds
fn odds_from_prob(p: f64) -> OddsBelief {
    let prob = clamp_f64(p, 1.0e-10, 1.0 - 1.0e-10)
    return OddsBelief {
        log_odds: log_f64(prob / (1.0 - prob)),
    }
}

// Convert odds to probability
fn odds_to_prob(o: OddsBelief) -> f64 {
    let odds = exp_f64(o.log_odds)
    return odds / (1.0 + odds)
}

// Update odds with likelihood ratio
// LR = P(evidence | H) / P(evidence | ¬H)
fn odds_update(prior: OddsBelief, log_likelihood_ratio: f64) -> OddsBelief {
    return OddsBelief {
        log_odds: prior.log_odds + log_likelihood_ratio,
    }
}

// Combine independent odds (in log space, just add)
fn odds_combine(o1: OddsBelief, o2: OddsBelief) -> OddsBelief {
    // This assumes the evidence is independent
    // Posterior odds = prior odds × LR1 × LR2 × ...
    // In log space: log(posterior) = log(prior) + log(LR1) + log(LR2)
    return OddsBelief {
        log_odds: o1.log_odds + o2.log_odds,
    }
}

// ============================================================================
// BRIDGE: DS ↔ BAYESIAN
// ============================================================================

// Convert DS belief to probability (pignistic transformation)
fn ds_to_probability(m: BinaryMass) -> f64 {
    // Pignistic probability: P(H) = m(H) + m(Θ)/2
    return m.m_h + m.m_uncertain / 2.0
}

// Convert probability + confidence to DS mass
fn probability_to_ds(prob: f64, confidence: f64) -> BinaryMass {
    let p = clamp_f64(prob, 0.0, 1.0)
    let c = clamp_f64(confidence, 0.0, 1.0)

    // Higher confidence = more mass on singletons
    // Lower confidence = more mass on Θ
    return BinaryMass {
        m_h: p * c,
        m_not_h: (1.0 - p) * c,
        m_uncertain: 1.0 - c,
    }
}

// Convert Beta to DS mass
fn beta_to_ds(b: BetaBelief) -> BinaryMass {
    let mean = beta_mean(b)
    let confidence = beta_to_confidence(b)
    return probability_to_ds(mean, confidence)
}

// ============================================================================
// INVARIANT CHECKS
// ============================================================================

fn is_valid_binary_mass(m: BinaryMass) -> bool {
    // All masses non-negative
    if m.m_h < -1.0e-10 || m.m_not_h < -1.0e-10 || m.m_uncertain < -1.0e-10 {
        return false
    }
    // Sum to 1
    let sum = m.m_h + m.m_not_h + m.m_uncertain
    if abs_f64(sum - 1.0) > 1.0e-6 {
        return false
    }
    return true
}

fn is_valid_beta(b: BetaBelief) -> bool {
    return b.alpha > 0.0 && b.beta > 0.0
}

// ============================================================================
// TESTS
// ============================================================================

fn test_ds_vacuous_combination() -> bool {
    // Combining with vacuous belief should not change the other
    let m1 = binary_mass_new(0.6, 0.8)
    let m2 = binary_mass_vacuous()
    let result = ds_combine_binary(m1, m2)

    if !result.is_valid { return false }
    if abs_f64(result.combined.m_h - m1.m_h) > 0.01 { return false }
    return true
}

fn test_ds_conflict_detection() -> bool {
    // Completely contradictory evidence should have high conflict
    let m1 = binary_mass_certain_h()
    let m2 = binary_mass_certain_not_h()
    let result = ds_combine_binary(m1, m2)

    // Should be invalid (K = 1)
    return !result.is_valid && result.conflict_k > 0.99
}

fn test_beta_update_increases_certainty() -> bool {
    let prior = beta_uniform()
    let posterior = beta_update(prior, 10.0, 2.0)

    // More data = less variance
    return beta_variance(posterior) < beta_variance(prior)
}

fn test_ds_mass_validity() -> bool {
    let m = binary_mass_new(0.3, 0.7)
    return is_valid_binary_mass(m)
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> i32 {
    // Test DS combination
    if !test_ds_vacuous_combination() { return 1 }
    if !test_ds_conflict_detection() { return 2 }
    if !test_ds_mass_validity() { return 3 }

    // Test Bayesian updates
    if !test_beta_update_increases_certainty() { return 4 }

    // Test DS combination with partial beliefs
    let evidence1 = binary_mass_new(0.6, 0.8)  // mild belief in H
    let evidence2 = binary_mass_new(0.5, 0.9)  // weaker, but more uncertain
    let combined = ds_combine_binary(evidence1, evidence2)

    if !combined.is_valid { return 5 }
    if !is_valid_binary_mass(combined.combined) { return 6 }

    // Conflict should be low for compatible evidence
    if combined.conflict_k > 0.5 { return 7 }

    // Test Beta updates
    let prior = beta_uniform()
    let post1 = beta_update(prior, 7.0, 3.0)
    let mean1 = beta_mean(post1)

    // Mean should be around 0.667 (alpha=8, beta=4 -> 8/12)
    if abs_f64(mean1 - 0.667) > 0.05 { return 8 }

    // Test pignistic transformation roundtrip
    let m = binary_mass_new(0.4, 0.7)
    let prob = ds_to_probability(m)
    // prob should be between belief and plausibility
    if prob < 0.4 || prob > 0.7 { return 9 }

    return 0
}
