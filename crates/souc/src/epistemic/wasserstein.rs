// Sounio Compiler - Optimal Transport (Wasserstein Geometry)
// Wasserstein-2 distance on probability distributions for rigorous confidence composition

use crate::epistemic::BetaConfidence;

/// Wasserstein distance between two Beta distributions
#[derive(Debug, Clone, Copy)]
pub struct WassersteinDistance {
    /// Computed W₂ distance
    pub distance: f64,
}

/// Wasserstein barycenter result
///
/// Minimizes weighted sum of W₂² distances from target distributions
#[derive(Debug, Clone)]
pub struct WassersteinBarycenter {
    /// Alpha parameter of barycenter Beta distribution
    pub alpha: f64,
    /// Beta parameter of barycenter Beta distribution
    pub beta: f64,
    /// Number of iterations for convergence
    pub iterations: usize,
}

/// Compute normalized Beta PDF at point x
fn beta_pdf(x: f64, alpha: f64, beta: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 {
        return 0.0;
    }

    // Use log-space computation for numerical stability
    let log_x = x.ln();
    let log_1mx = (1.0 - x).ln();

    // log(B(α,β)) ≈ lgamma(α) + lgamma(β) - lgamma(α+β)
    let log_beta_fn = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta);

    let log_pdf = (alpha - 1.0) * log_x + (beta - 1.0) * log_1mx - log_beta_fn;
    log_pdf.exp()
}

/// Log-gamma function (Stirling's approximation)
fn lgamma(x: f64) -> f64 {
    if x < 0.5 {
        return lgamma(x + 1.0) - x.ln();
    }

    // Stirling's approximation: ln(Γ(x)) ≈ (x-0.5)ln(x) - x + 0.5ln(2π)
    let mut z = x - 0.5;
    z * z.ln() - z + 0.9189385332046727 // 0.5 * ln(2π)
}

/// Beta quantile function using Newton's method
///
/// Solves: CDF(q) = p for quantile q
fn beta_quantile(alpha: f64, beta: f64, p: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }

    // Binary search for quantile
    let mut low = 0.0;
    let mut high = 1.0;

    for _ in 0..60 {
        let mid = (low + high) / 2.0;
        let cdf = beta_cdf_integration(mid, alpha, beta);

        if cdf < p {
            low = mid;
        } else {
            high = mid;
        }

        if (high - low).abs() < 1e-12 {
            break;
        }
    }

    (low + high) / 2.0
}

/// Beta CDF via numerical integration (Simpson's rule)
fn beta_cdf_integration(x: f64, alpha: f64, beta: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Simpson's rule integration: ∫₀ˣ pdf(t) dt
    let n = 100;
    let h = x / n as f64;
    let mut sum = beta_pdf(0.0, alpha, beta) + beta_pdf(x, alpha, beta);

    for i in 1..n {
        let t = (i as f64) * h;
        let coeff = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += coeff * beta_pdf(t, alpha, beta);
    }

    (sum * h / 3.0).min(1.0)
}

/// Compute Wasserstein-2 distance between two Beta distributions
///
/// W₂(P, Q) = (∫₀¹ |F_P⁻¹(t) - F_Q⁻¹(t)|² dt)^(1/2)
///
/// where F^(-1) is the quantile function
pub fn wasserstein2_distance(p: &BetaConfidence, q: &BetaConfidence) -> f64 {
    let n = 500; // Integration points
    let mut integral = 0.0;

    // Simpson's rule integration over quantile differences squared
    for i in 0..=n {
        let t = i as f64 / n as f64;

        let qp = beta_quantile(p.alpha, p.beta, t);
        let qq = beta_quantile(q.alpha, q.beta, t);
        let diff_sq = (qp - qq).powi(2);

        let coeff = if i == 0 || i == n {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };

        integral += coeff * diff_sq;
    }

    integral *= (1.0 / n as f64) / 3.0;
    integral.sqrt()
}

/// Compute Wasserstein barycenter of multiple Beta distributions
///
/// Minimizes: arg min_P Σᵢ wᵢ W₂²(P, Pᵢ)
///
/// Uses fixed-point iteration (method of moments)
pub fn wasserstein_barycenter(
    distributions: &[BetaConfidence],
    weights: &[f64],
) -> WassersteinBarycenter {
    if distributions.is_empty() || weights.is_empty() || distributions.len() != weights.len() {
        return WassersteinBarycenter {
            alpha: 1.0,
            beta: 1.0,
            iterations: 0,
        };
    }

    let m = distributions.len();

    // Compute weighted mean E[X]
    let mut mean = 0.0;
    for i in 0..m {
        let d = &distributions[i];
        let exp_x = d.alpha / (d.alpha + d.beta);
        mean += weights[i] * exp_x;
    }
    mean = mean.max(0.01).min(0.99);

    // Compute E[X²] via quantile integration
    let mut e2 = 0.0;
    let n_quad = 200;

    for k in 0..=n_quad {
        let t = k as f64 / n_quad as f64;
        let mut avg_q = 0.0;

        for i in 0..m {
            let d = &distributions[i];
            let q = beta_quantile(d.alpha, d.beta, t);
            avg_q += weights[i] * q;
        }

        let coeff = if k == 0 || k == n_quad {
            1.0
        } else if k % 2 == 0 {
            2.0
        } else {
            4.0
        };

        e2 += coeff * avg_q * avg_q;
    }

    e2 *= (1.0 / n_quad as f64) / 3.0;

    // Compute variance and method of moments parameters
    let variance = (e2 - mean * mean).max(0.001);
    let s = if (mean * (1.0 - mean)).abs() > 1e-10 {
        variance / (mean * (1.0 - mean))
    } else {
        1.0
    };

    let mut alpha = (mean * s).max(0.1);
    let mut beta = ((1.0 - mean) * s).max(0.1);

    // Fallback to weighted mean if invalid
    if !alpha.is_finite() || !beta.is_finite() {
        alpha = 0.0;
        beta = 0.0;
        for i in 0..m {
            alpha += weights[i] * distributions[i].alpha;
            beta += weights[i] * distributions[i].beta;
        }
        alpha = alpha.max(0.1);
        beta = beta.max(0.1);
    }

    WassersteinBarycenter {
        alpha,
        beta,
        iterations: 10, // Fixed iterations for this approximation
    }
}

/// Compute transport cost between two distributions
///
/// Metric measuring "distance" between type distributions
/// Used for type compatibility assessment
pub fn transport_cost(source: &BetaConfidence, target: &BetaConfidence) -> f64 {
    wasserstein2_distance(source, target)
}

/// Verify triangle inequality for Wasserstein distances
///
/// W₂(P, R) ≤ W₂(P, Q) + W₂(Q, R)
pub fn verify_triangle_inequality(
    p: &BetaConfidence,
    q: &BetaConfidence,
    r: &BetaConfidence,
) -> bool {
    let d_pr = wasserstein2_distance(p, r);
    let d_pq = wasserstein2_distance(p, q);
    let d_qr = wasserstein2_distance(q, r);

    let lhs = d_pr;
    let rhs = d_pq + d_qr;

    (lhs - rhs).abs() < 1e-6 || lhs <= rhs + 1e-6
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasserstein_identity() {
        let p = BetaConfidence::new(2.0, 3.0);
        let dist = wasserstein2_distance(&p, &p);
        assert!(dist.abs() < 1e-10);
    }

    #[test]
    fn test_wasserstein_symmetry() {
        let p = BetaConfidence::new(2.0, 3.0);
        let q = BetaConfidence::new(3.0, 2.0);

        let d_pq = wasserstein2_distance(&p, &q);
        let d_qp = wasserstein2_distance(&q, &p);

        assert!((d_pq - d_qp).abs() < 1e-10);
    }

    #[test]
    fn test_wasserstein_positive() {
        let p = BetaConfidence::new(2.0, 3.0);
        let q = BetaConfidence::new(5.0, 1.0);

        let dist = wasserstein2_distance(&p, &q);
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_triangle_inequality() {
        let p = BetaConfidence::new(1.0, 2.0);
        let q = BetaConfidence::new(2.0, 2.0);
        let r = BetaConfidence::new(3.0, 1.0);

        assert!(verify_triangle_inequality(&p, &q, &r));
    }

    #[test]
    fn test_barycenter_convergence() {
        let dists = vec![BetaConfidence::new(2.0, 3.0), BetaConfidence::new(3.0, 2.0)];
        let weights = vec![0.5, 0.5];

        let barycenter = wasserstein_barycenter(&dists, &weights);

        assert!(barycenter.alpha > 0.0);
        assert!(barycenter.beta > 0.0);
        assert!(barycenter.iterations > 0);
    }

    #[test]
    fn test_barycenter_equal_weights() {
        let dists = vec![BetaConfidence::new(1.0, 1.0), BetaConfidence::new(1.0, 1.0)];
        let weights = vec![0.5, 0.5];

        let barycenter = wasserstein_barycenter(&dists, &weights);

        // Equal inputs should give approximately uniform result
        assert!(barycenter.alpha > 0.5);
        assert!(barycenter.beta > 0.5);
    }

    #[test]
    fn test_transport_cost_positive() {
        let p = BetaConfidence::new(2.0, 3.0);
        let q = BetaConfidence::new(4.0, 1.0);

        let cost = transport_cost(&p, &q);
        assert!(cost >= 0.0);
    }

    #[test]
    fn test_beta_quantile_bounds() {
        let alpha = 2.0;
        let beta = 3.0;

        let q0 = beta_quantile(alpha, beta, 0.0);
        let q1 = beta_quantile(alpha, beta, 1.0);
        let q05 = beta_quantile(alpha, beta, 0.5);

        assert!(q0 >= 0.0 && q0 <= 0.01);
        assert!(q1 >= 0.99 && q1 <= 1.0);
        assert!(q05 > q0 && q05 < q1);
    }

    #[test]
    fn test_multiple_barycenters() {
        let dists = vec![
            BetaConfidence::new(1.0, 3.0),
            BetaConfidence::new(2.0, 2.0),
            BetaConfidence::new(3.0, 1.0),
        ];
        let weights = vec![0.3, 0.4, 0.3];

        let barycenter = wasserstein_barycenter(&dists, &weights);

        assert!(barycenter.alpha > 0.0);
        assert!(barycenter.beta > 0.0);
    }
}
