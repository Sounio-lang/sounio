//! Numerical integration routines

use std::math::{abs, sqrt, pow}

/// Integration result
pub struct IntegrationResult {
    /// Integral value
    pub value: f64,
    
    /// Error estimate
    pub error: f64,
    
    /// Number of function evaluations
    pub neval: usize,
    
    /// Whether integration converged
    pub success: bool,
    
    /// Termination message
    pub message: string,
}

/// Integration configuration
pub struct IntegrationConfig {
    /// Absolute tolerance
    pub epsabs: f64,
    
    /// Relative tolerance
    pub epsrel: f64,
    
    /// Maximum number of subdivisions
    pub limit: usize,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        IntegrationConfig {
            epsabs: 1e-8,
            epsrel: 1e-8,
            limit: 50,
        }
    }
}

/// Adaptive Gauss-Kronrod quadrature
pub fn quad<F>(f: F, a: f64, b: f64, config: IntegrationConfig) -> IntegrationResult
where F: Fn(f64) -> f64
{
    // 15-point Gauss-Kronrod rule
    let xgk = [
        0.0, 0.2077849550078853, 0.4058451513773972, 0.5860872354676911,
        0.7415311855993944, 0.8648644233597691, 0.9491079123427585,
        0.9914553711208126
    ];
    
    let wg = [
        0.4179591836734694, 0.3818300505051189, 0.2797053914892767,
        0.1294849661688697, 0.0307532419961173
    ];
    
    let wgk = [
        0.2094821410847278, 0.2044329400752989, 0.1903505780647854,
        0.1690047266392679, 0.1406532597155259, 0.1047900103222502,
        0.0630920926299785, 0.0229353220105292
    ];
    
    let mut result = 0.0;
    let mut error = 0.0;
    let mut neval = 0;
    
    // Transform to [-1, 1]
    let center = 0.5 * (a + b);
    let half_length = 0.5 * (b - a);
    
    // Gauss quadrature
    let mut resg = 0.0;
    for i in 0..4 {
        let x1 = center + half_length * xgk[2*i + 1];
        let x2 = center - half_length * xgk[2*i + 1];
        let f1 = f(x1);
        let f2 = f(x2);
        resg += wg[i] * (f1 + f2);
        neval += 2;
    }
    resg *= half_length;
    
    // Kronrod quadrature
    let mut resk = 0.0;
    resk += wgk[0] * f(center);
    neval += 1;
    
    for i in 0..7 {
        let x1 = center + half_length * xgk[i + 1];
        let x2 = center - half_length * xgk[i + 1];
        let f1 = f(x1);
        let f2 = f(x2);
        resk += wgk[i + 1] * (f1 + f2);
        neval += 2;
    }
    resk *= half_length;
    
    // Error estimate
    let err = abs(resk - resg);
    
    IntegrationResult {
        value: resk,
        error: err,
        neval,
        success: err <= config.epsabs || err <= config.epsrel * abs(resk),
        message: if err <= config.epsabs || err <= config.epsrel * abs(resk) {
            "convergence achieved".to_string()
        } else {
            "tolerance not met".to_string()
        },
    }
}

/// Adaptive Simpson's rule
pub fn simpson<F>(f: F, a: f64, b: f64, tol: f64) -> IntegrationResult
where F: Fn(f64) -> f64
{
    fn simpson_recursive<F>(
        f: &F, 
        a: f64, 
        b: f64, 
        tol: f64, 
        s: f64, 
        fa: f64, 
        fb: f64, 
        fc: f64,
        neval: &mut usize
    ) -> f64
    where F: Fn(f64) -> f64
    {
        let c = (a + b) / 2.0;
        let h = b - a;
        let d = (a + c) / 2.0;
        let e = (c + b) / 2.0;
        let fd = f(d);
        let fe = f(e);
        *neval += 2;
        
        let s1 = (h / 12.0) * (fa + 4.0 * fd + fc);
        let s2 = (h / 12.0) * (fc + 4.0 * fe + fb);
        let s_new = s1 + s2;
        
        if abs(s_new - s) <= 15.0 * tol {
            s_new + (s_new - s) / 15.0
        } else {
            simpson_recursive(f, a, c, tol / 2.0, s1, fa, fc, fd, neval) +
            simpson_recursive(f, c, b, tol / 2.0, s2, fc, fb, fe, neval)
        }
    }
    
    let mut neval = 3;
    let fa = f(a);
    let fb = f(b);
    let fc = f((a + b) / 2.0);
    
    let h = b - a;
    let s = (h / 6.0) * (fa + 4.0 * fc + fb);
    
    let result = simpson_recursive(&f, a, b, tol, s, fa, fb, fc, &mut neval);
    
    IntegrationResult {
        value: result,
        error: tol, // Approximate
        neval,
        success: true,
        message: "adaptive Simpson completed".to_string(),
    }
}

/// Monte Carlo integration
pub fn monte_carlo<F>(
    f: F, 
    bounds: &[(f64, f64)], 
    n_samples: usize,
    rng: &mut impl rand::Rng
) -> IntegrationResult
where F: Fn(&[f64]) -> f64
{
    let dim = bounds.len();
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    
    // Calculate volume
    let volume: f64 = bounds.iter()
        .map(|(a, b)| b - a)
        .product();
    
    for _ in 0..n_samples {
        let mut x = vec![0.0; dim];
        for i in 0..dim {
            x[i] = bounds[i].0 + rng.uniform(0.0, 1.0) * (bounds[i].1 - bounds[i].0);
        }
        
        let fx = f(&x);
        sum += fx;
        sum_sq += fx * fx;
    }
    
    let mean = sum / n_samples as f64;
    let variance = (sum_sq / n_samples as f64) - mean * mean;
    let std_error = sqrt(variance / n_samples as f64);
    
    IntegrationResult {
        value: volume * mean,
        error: volume * std_error,
        neval: n_samples,
        success: true,
        message: "Monte Carlo integration completed".to_string(),
    }
}
