//! MCMC Convergence Diagnostics
//!
//! Implements Effective Sample Size (ESS) and split-R-hat diagnostics
//! following Vehtari et al. (2021) and Geyer (1992).
//!
//! Upgrades over `mcmc.rs` basic diagnostics:
//! - Geyer's initial positive sequence ESS estimator (monotone)
//! - Rank-normalized R-hat for bulk and tail convergence
//! - Chain folding for tail behavior assessment
//! - Moro algorithm for inverse normal CDF
//!
//! # References
//!
//! - Geyer, C. J. (1992). Practical Markov chain Monte Carlo.
//! - Vehtari, A., et al. (2021). Rank-normalization, folding, and localization.
//!   Bayesian Analysis, 16(2), 667-718.

/// Comprehensive MCMC diagnostics
#[derive(Debug, Clone, PartialEq)]
pub struct MCMCDiagnostics {
    /// Effective Sample Size
    pub ess: f64,
    /// ESS per second (if runtime provided)
    pub ess_per_sec: Option<f64>,
    /// Split-R-hat statistic
    pub rhat: f64,
    /// Rank-normalized R-hat for bulk convergence
    pub rhat_bulk: f64,
    /// Rank-normalized R-hat for tail convergence
    pub rhat_tail: f64,
    /// Whether chains appear converged (R-hat < 1.01 and ESS >= 100)
    pub converged: bool,
}

/// Compute Effective Sample Size using initial positive sequence estimator (Geyer 1992)
///
/// Uses consecutive pairs of autocovariance to guarantee a monotone estimator.
/// Stops summing when the first negative pair is encountered.
pub fn effective_sample_size(chain: &[f64]) -> f64 {
    if chain.len() < 4 {
        return chain.len() as f64;
    }

    let n = chain.len() as f64;
    let mean = chain.iter().sum::<f64>() / n;

    let variance = chain.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    if variance.abs() < 1e-12 {
        return n;
    }

    let mut autocov_sum = 0.0;
    let mut k = 0;

    while 2 * k + 1 < chain.len() {
        let lag1 = 2 * k;
        let lag2 = 2 * k + 1;

        let gamma1 = autocovariance(chain, mean, lag1) / variance;
        let gamma2 = autocovariance(chain, mean, lag2) / variance;

        let pair_sum = gamma1 + gamma2;
        if pair_sum <= 0.0 {
            break;
        }

        autocov_sum += pair_sum;
        k += 1;
    }

    n / (1.0 + 2.0 * autocov_sum)
}

/// Compute autocovariance at given lag
fn autocovariance(chain: &[f64], mean: f64, lag: usize) -> f64 {
    if lag >= chain.len() {
        return 0.0;
    }

    let n = chain.len() as f64;
    let mut sum = 0.0;

    for i in 0..(chain.len() - lag) {
        sum += (chain[i] - mean) * (chain[i + lag] - mean);
    }

    sum / n
}

/// Compute split-chain ESS across multiple chains
///
/// Splits each chain in half and computes ESS on the concatenated result.
pub fn effective_sample_size_multi(chains: &[Vec<f64>]) -> f64 {
    if chains.is_empty() {
        return 0.0;
    }

    let mut split_chains = Vec::new();
    for chain in chains {
        if chain.len() >= 2 {
            let mid = chain.len() / 2;
            split_chains.push(chain[..mid].to_vec());
            split_chains.push(chain[mid..].to_vec());
        }
    }

    if split_chains.is_empty() {
        return 0.0;
    }

    let concatenated: Vec<f64> = split_chains.into_iter().flatten().collect();
    effective_sample_size(&concatenated)
}

/// Compute split-R-hat statistic (Gelman-Rubin diagnostic)
///
/// Splits each chain in half, then computes between-chain and within-chain
/// variance. R-hat should approach 1.0 for converged chains (< 1.01 is good).
pub fn rhat(chains: &[Vec<f64>]) -> f64 {
    if chains.len() < 2 {
        return 1.0;
    }

    let split = split_chains_in_half(chains);
    compute_split_rhat(&split)
}

/// Rank-normalized split-R-hat for bulk convergence (Vehtari et al. 2021)
pub fn rhat_bulk(chains: &[Vec<f64>]) -> f64 {
    if chains.len() < 2 {
        return 1.0;
    }

    let split = split_chains_in_half(chains);
    let ranked = rank_normalize_chains(&split);
    compute_split_rhat(&ranked)
}

/// Rank-normalized split-R-hat for tail convergence (Vehtari et al. 2021)
///
/// Folds chains by taking absolute deviation from median, then rank-normalizes.
pub fn rhat_tail(chains: &[Vec<f64>]) -> f64 {
    if chains.len() < 2 {
        return 1.0;
    }

    let split = split_chains_in_half(chains);
    let folded = fold_chains(&split);
    let ranked = rank_normalize_chains(&folded);
    compute_split_rhat(&ranked)
}

/// Compute comprehensive diagnostics for MCMC chains
pub fn diagnose_chains(chains: &[Vec<f64>], runtime_secs: Option<f64>) -> MCMCDiagnostics {
    let ess = effective_sample_size_multi(chains);
    let ess_per_sec = runtime_secs.map(|t| ess / t);
    let rhat_value = rhat(chains);
    let rhat_bulk_value = rhat_bulk(chains);
    let rhat_tail_value = rhat_tail(chains);

    let converged = rhat_value < 1.01 && ess >= 100.0;

    MCMCDiagnostics {
        ess,
        ess_per_sec,
        rhat: rhat_value,
        rhat_bulk: rhat_bulk_value,
        rhat_tail: rhat_tail_value,
        converged,
    }
}

// =============================================================================
// Internal helpers
// =============================================================================

fn split_chains_in_half(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut split = Vec::with_capacity(chains.len() * 2);

    for chain in chains {
        if chain.len() >= 2 {
            let mid = chain.len() / 2;
            split.push(chain[..mid].to_vec());
            split.push(chain[mid..].to_vec());
        }
    }

    split
}

fn compute_split_rhat(chains: &[Vec<f64>]) -> f64 {
    if chains.len() < 2 {
        return 1.0;
    }

    let m = chains.len() as f64;
    let n = chains[0].len() as f64;

    let chain_means: Vec<f64> = chains
        .iter()
        .map(|chain| chain.iter().sum::<f64>() / n)
        .collect();

    let overall_mean = chain_means.iter().sum::<f64>() / m;

    // Between-chain variance
    let b = n
        * chain_means
            .iter()
            .map(|&mu| (mu - overall_mean).powi(2))
            .sum::<f64>()
        / (m - 1.0);

    // Within-chain variance
    let w = chains
        .iter()
        .map(|chain| {
            let mu = chain.iter().sum::<f64>() / n;
            chain.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (n - 1.0)
        })
        .sum::<f64>()
        / m;

    let var_plus = ((n - 1.0) / n) * w + (1.0 / n) * b;

    (var_plus / w).sqrt()
}

fn rank_normalize_chains(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let total_len: usize = chains.iter().map(|c| c.len()).sum();
    let mut all_values: Vec<(f64, usize, usize)> = Vec::with_capacity(total_len);

    for (chain_idx, chain) in chains.iter().enumerate() {
        for (sample_idx, &value) in chain.iter().enumerate() {
            all_values.push((value, chain_idx, sample_idx));
        }
    }

    all_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks (averaging ties)
    let mut ranks = vec![0.0; total_len];
    let mut i = 0;
    while i < total_len {
        let mut j = i;
        while j < total_len && (all_values[j].0 - all_values[i].0).abs() < 1e-12 {
            j += 1;
        }

        let avg_rank = ((i + j - 1) as f64 + 2.0) / 2.0;

        for k in i..j {
            ranks[k] = avg_rank;
        }

        i = j;
    }

    let total_len_f = total_len as f64;
    let mut normalized = vec![vec![0.0; chains[0].len()]; chains.len()];

    for (idx, (_, chain_idx, sample_idx)) in all_values.into_iter().enumerate() {
        let rank = ranks[idx];
        let p = rank / (total_len_f + 1.0);
        normalized[chain_idx][sample_idx] = inverse_normal_cdf(p);
    }

    normalized
}

fn fold_chains(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    chains
        .iter()
        .map(|chain| {
            let mut sorted = chain.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = sorted[sorted.len() / 2];
            chain.iter().map(|&x| (x - median).abs()).collect()
        })
        .collect()
}

/// Approximate inverse normal CDF (Moro 1995 rational approximation)
fn inverse_normal_cdf(p: f64) -> f64 {
    let p = p.max(1e-16).min(1.0 - 1e-16);

    let a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ];

    let b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];

    let c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ];

    let d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ];

    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.425 * 0.425 - q * q;
        let num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q;
        let den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0;
        num / den
    } else {
        let r = if q <= 0.0 { p } else { 1.0 - p };
        let r = (-2.0 * r.ln()).sqrt();

        let num = ((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5];
        let den = (((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1.0;

        let x = num / den;
        if q < 0.0 { x } else { -x }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{E, PI};

    fn pseudo_iid(n: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(n);
        let mut x = 0.5;
        for _ in 0..n {
            x = (x * PI + E).fract();
            result.push(x);
        }
        result
    }

    fn correlated_chain(n: usize, rho: f64) -> Vec<f64> {
        let noise = pseudo_iid(n);
        let mut chain = Vec::with_capacity(n);
        let mut x = 0.0;
        for i in 0..n {
            x = rho * x + (1.0 - rho.powi(2)).sqrt() * noise[i];
            chain.push(x);
        }
        chain
    }

    #[test]
    fn test_ess_iid() {
        let n = 1000;
        let chain = pseudo_iid(n);
        let ess = effective_sample_size(&chain);
        // Pseudo-random sequence has some autocorrelation; ESS > 20% is reasonable
        assert!(
            ess > n as f64 * 0.2,
            "ESS for pseudo-IID should be nontrivial. Got: {ess}, expected > {}",
            n as f64 * 0.2
        );
    }

    #[test]
    fn test_ess_correlated() {
        let n = 1000;
        let chain = correlated_chain(n, 0.95);
        let ess = effective_sample_size(&chain);
        assert!(
            ess < n as f64 * 0.3,
            "ESS for highly correlated chain should be low. Got: {ess}"
        );
    }

    #[test]
    fn test_rhat_converged() {
        let n = 1000;
        let base = pseudo_iid(n);
        let chains = vec![base.clone(), base.clone(), base.clone()];
        let r = rhat(&chains);
        assert!(
            (r - 1.0).abs() < 0.02,
            "R-hat for identical chains should be ~1.0. Got: {r}"
        );
    }

    #[test]
    fn test_rhat_diverged() {
        let n = 1000;
        let chains = vec![
            pseudo_iid(n).iter().map(|x| x + 0.0).collect(),
            pseudo_iid(n).iter().map(|x| x + 5.0).collect(),
            pseudo_iid(n).iter().map(|x| x + 10.0).collect(),
        ];
        let r = rhat(&chains);
        assert!(r > 1.1, "R-hat for divergent chains should be > 1.1. Got: {r}");
    }

    #[test]
    fn test_diagnose_basic() {
        let n = 1000;
        let base = pseudo_iid(n);
        let chains = vec![
            base.iter().map(|x| x + 0.0).collect(),
            base.iter().map(|x| x + 0.1).collect(),
            base.iter().map(|x| x + 0.2).collect(),
        ];

        let diag = diagnose_chains(&chains, Some(1.0));
        assert!(diag.ess > 0.0);
        assert!(diag.ess_per_sec.unwrap() > 0.0);
        assert!(diag.rhat >= 1.0);
    }

    #[test]
    fn test_inverse_normal_cdf_median() {
        assert!((inverse_normal_cdf(0.5) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_inverse_normal_cdf_tails() {
        assert!(inverse_normal_cdf(0.975) > 1.9);
        assert!(inverse_normal_cdf(0.025) < -1.9);
    }

    #[test]
    fn test_fold_chains_basic() {
        let chain = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let folded = fold_chains(&[chain]);
        let expected = vec![2.0, 1.0, 0.0, 1.0, 2.0];
        for (i, &val) in folded[0].iter().enumerate() {
            assert!(
                (val - expected[i]).abs() < 1e-12,
                "fold[{i}]: got {val}, expected {}",
                expected[i]
            );
        }
    }
}
